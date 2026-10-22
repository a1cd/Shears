import Foundation
import MLX

/// The `shears-serve` command line: `serve` runs the chat server, `bench` measures throughput.
///
/// ```
/// shears-serve serve --model DIR (--socket PATH | --port N [--host ADDR]) [--alias NAME] [--dtype bf16|float16|float32]
///                    [--tokenizer DIR] [--max-tokens N]
/// shears-serve bench --model DIR [--prompt-tokens N] [--decode-tokens N] [--repeats N] [--dtype ...] [--json]
/// ```
public enum ServeCommand {
    struct UsageError: Error { var message: String }

    public static let usage = """
        usage:
          shears-serve serve --model DIR (--socket PATH | --port N [--host ADDR]) [--alias NAME]
                             [--dtype bf16|float16|float32] [--tokenizer DIR] [--max-tokens N]
          shears-serve bench --model DIR [--prompt-tokens N] [--decode-tokens N] [--repeats N]
                             [--dtype bf16|float16|float32] [--json]
        """

    public static func main(arguments: [String]) -> Int32 {
        var args = Array(arguments.dropFirst())
        guard let command = args.first else {
            FileHandle.standardError.write(Data((usage + "\n").utf8))
            return 2
        }
        args.removeFirst()
        do {
            var options: [String: String] = [:]
            var flags = Set<String>()
            var i = 0
            while i < args.count {
                let a = args[i]
                guard a.hasPrefix("--") else { throw UsageError(message: "unexpected argument \(a)") }
                let key = String(a.dropFirst(2))
                if key == "json" { flags.insert(key); i += 1; continue }
                guard i + 1 < args.count else { throw UsageError(message: "--\(key) needs a value") }
                options[key] = args[i + 1]
                i += 2
            }
            switch command {
            case "serve": return try serve(options)
            case "bench": return try bench(options, json: flags.contains("json"))
            case "-h", "--help", "help":
                print(usage)
                return 0
            default: throw UsageError(message: "unknown command \(command)")
            }
        } catch let e as UsageError {
            FileHandle.standardError.write(Data("error: \(e.message)\n\(usage)\n".utf8))
            return 2
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }

    private static func dtype(_ s: String?) throws -> DType? {
        switch s {
        case nil: return nil
        case "bf16", "bfloat16": return .bfloat16
        case "float16", "f16": return .float16
        case "float32", "f32": return .float32
        default: throw UsageError(message: "unknown dtype \(s!)")
        }
    }

    private static func load(_ o: [String: String]) throws -> (InferenceRuntime, URL) {
        guard let m = o["model"] else { throw UsageError(message: "--model is required") }
        let dir = URL(fileURLWithPath: m)
        let rt = try InferenceRuntime.load(
            directory: dir, dtype: try dtype(o["dtype"]), tokenizerDirectory: o["tokenizer"].map { URL(fileURLWithPath: $0) })
        return (rt, dir)
    }

    private static func serve(_ o: [String: String]) throws -> Int32 {
        let endpoint: HTTPServer.Endpoint
        if let s = o["socket"] {
            endpoint = .unixSocket(path: s)
        } else if let p = o["port"] {
            guard let port = UInt16(p) else { throw UsageError(message: "bad --port") }
            endpoint = .tcp(host: o["host"] ?? "127.0.0.1", port: port)
        } else {
            throw UsageError(message: "one of --socket or --port is required")
        }
        let (rt, dir) = try load(o)
        var config = InferenceServer.Configuration(alias: o["alias"] ?? dir.lastPathComponent, modelPath: dir.path)
        if let m = o["max-tokens"] {
            guard let n = Int(m), n > 0 else { throw UsageError(message: "bad --max-tokens") }
            config.defaultMaxTokens = n
        }
        let server = InferenceServer(runtime: rt, configuration: config)
        try server.start(endpoint)
        switch endpoint {
        case .unixSocket(let p): FileHandle.standardError.write(Data("listening on unix socket \(p)\n".utf8))
        case .tcp(let h, _): FileHandle.standardError.write(Data("listening on http://\(h):\(server.port)\n".utf8))
        }
        let done = DispatchSemaphore(value: 0)
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let sources = [SIGINT, SIGTERM].map { sig -> DispatchSourceSignal in
            let s = DispatchSource.makeSignalSource(signal: sig, queue: .global())
            s.setEventHandler { done.signal() }
            s.resume()
            return s
        }
        done.wait()
        sources.forEach { $0.cancel() }
        server.stop()
        return 0
    }

    private static func bench(_ o: [String: String], json: Bool) throws -> Int32 {
        func int(_ k: String, _ d: Int) throws -> Int {
            guard let v = o[k] else { return d }
            guard let n = Int(v), n > 0 else { throw UsageError(message: "bad --\(k)") }
            return n
        }
        let (rt, _) = try load(o)
        let r = try InferenceBenchmark.run(
            runtime: rt, promptTokens: try int("prompt-tokens", 512), decodeTokens: try int("decode-tokens", 128),
            repeats: try int("repeats", 3))
        if json {
            let e = JSONEncoder()
            e.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try e.encode(r), as: UTF8.self))
        } else {
            func mb(_ b: Int) -> String { String(format: "%.1f MB", Double(b) / 1_048_576) }
            print(String(format: "prefill   %9.1f tokens/s (%d-token prompt)", r.prefillTokensPerSecond, r.promptTokens))
            print(String(format: "decode    %9.1f tokens/s (%d tokens)", r.decodeTokensPerSecond, r.decodeTokens))
            print(String(format: "first token %7.3f s", r.timeToFirstTokenSeconds))
            print("peak MLX memory      \(mb(r.peakMLXMemoryBytes))")
            print("peak phys footprint  \(mb(r.peakPhysFootprintBytes))")
            print("final phys footprint \(mb(r.finalPhysFootprintBytes))")
        }
        return 0
    }
}

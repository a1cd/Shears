import Foundation
import MLX

/// The `shears-serve` command line: `serve` runs the chat server, `bench` measures throughput.
///
/// ```
/// shears-serve serve --model DIR (--socket PATH | --port N [--host ADDR]) [--alias NAME] [--dtype bf16|float16|float32]
///                    [--tokenizer DIR] [--max-tokens N]
/// shears-serve bench --model DIR [--prompt-tokens N] [--decode-tokens N] [--repeats N] [--dtype ...] [--json]
/// ```
///
/// Every command that loads a model also takes `--vocab FILE` (run on a vocabulary window: the embedding table stays on
/// disk, see `docs/vocabulary-subset.md`) and `--cache-limit-mb N` (a cap on MLX's buffer cache).
public enum ServeCommand {
    struct UsageError: Error { var message: String }

    public static let usage = """
        usage:
          shears-serve serve --model DIR (--socket PATH | --port N [--host ADDR]) [--alias NAME]
                             [--dtype bf16|float16|float32] [--tokenizer DIR] [--max-tokens N]
          shears-serve bench --model DIR [--prompt-tokens N] [--decode-tokens N] [--repeats N]
                             [--dtype bf16|float16|float32] [--json]
          shears-serve decode-bench --model DIR --prompts FILE [--index N] [--decode-tokens N] [--repeats N] [--json]
          shears-serve ab-bench --model DIR --vocab FILE --prompts FILE [--index N] [--decode-tokens N] [--rounds N]
          shears-serve verdict-bench --model DIR --prompts FILE [--index N] [--repeats N]
          shears-serve generate --model DIR (--prompt TEXT | --prompts FILE [--index N]) [--max-tokens N]
          shears-serve quantize --model DIR --out DIR [--bits 4] [--group-size 64]
          shears-serve vocab-eval --model DIR --vocab FILE --prompts FILE [--out REPORT.json] [--max-new N] [--limit N]

        serve, bench, decode-bench and generate also take
          --vocab FILE          run on a vocabulary window (static list from `shears-cli vocab build`): the embedding table
                                stays on disk, and only the rows of the static list and of each request's context are read
          --cache-limit-mb N    cap MLX's cache of freed buffers (it otherwise grows without limit and counts in the footprint)
          --prefill-chunk N     prompt tokens per forward pass during prefill (default 512); smaller lowers the transient peak
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
            case "decode-bench": return try decodeBench(options, json: flags.contains("json"))
            case "generate": return try generate(options)
            case "verdict-bench": return try verdictBench(options)
            case "ab-bench": return try abBench(options)
            case "quantize": return try quantizeCommand(options)
            case "vocab-eval": return try vocabEval(options)
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
        if let c = o["cache-limit-mb"] {
            guard let n = Int(c), n >= 0 else { throw UsageError(message: "bad --cache-limit-mb") }
            Memory.cacheLimit = n << 20
        }
        let vocabulary = try o["vocab"].map { try StaticVocabulary.load(URL(fileURLWithPath: $0)) }
        let rt = try InferenceRuntime.load(
            directory: dir, dtype: try dtype(o["dtype"]), tokenizerDirectory: o["tokenizer"].map { URL(fileURLWithPath: $0) },
            vocabulary: vocabulary)
        if let c = o["prefill-chunk"] {
            guard let n = Int(c), n > 0 else { throw UsageError(message: "bad --prefill-chunk") }
            rt.generator.prefillChunk = n
        }
        return (rt, dir)
    }

    private static func int(_ o: [String: String], _ k: String, _ d: Int) throws -> Int {
        guard let v = o[k] else { return d }
        guard let n = Int(v), n >= 0 else { throw UsageError(message: "bad --\(k)") }
        return n
    }

    /// The `index`-th prompt of a prompts file, as token ids.
    private static func promptTokens(_ rt: InferenceRuntime, _ o: [String: String]) throws -> [Int] {
        guard let f = o["prompts"] else { throw UsageError(message: "--prompts is required") }
        let prompts = try VocabularyEvaluation.loadPrompts(URL(fileURLWithPath: f))
        let i = try int(o, "index", 0)
        guard i < prompts.count else { throw UsageError(message: "--index \(i) is past the \(prompts.count) prompts") }
        return try rt.promptTokens(messages: prompts[i].messages)
    }

    private static func decodeBench(_ o: [String: String], json: Bool) throws -> Int32 {
        let loadStart = Date()
        let (rt, _) = try load(o)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let prompt = try promptTokens(rt, o)
        let r = try InferenceBenchmark.run(
            runtime: rt, prompt: prompt, decodeTokens: try int(o, "decode-tokens", 80), repeats: try int(o, "repeats", 3))
        report(r, json: json, loadSeconds: loadSeconds, window: rt.model.vocabularyWindow)
        return 0
    }

    /// The same model with its whole table (A) and on a vocabulary window (B), in one process, alternating, so both see the
    /// same machine load: prefill and decode speed of each, the median over `rounds` rounds.
    private static func abBench(_ o: [String: String]) throws -> Int32 {
        guard o["vocab"] != nil else { throw UsageError(message: "--vocab is required") }
        var plain = o
        plain["vocab"] = nil
        let (a, _) = try load(plain)
        let (b, _) = try load(o)
        let prompt = try promptTokens(a, o)
        let decode = try int(o, "decode-tokens", 80), rounds = try int(o, "rounds", 7)
        func once(_ rt: InferenceRuntime) throws -> (prefill: Double, decode: Double, ttft: Double) {
            rt.generator.reusesPromptCache = false
            rt.generator.resetCache()
            var first: Date?
            let start = Date()
            let r = try rt.generator.generate(GenerationRequest(promptTokens: prompt, maxTokens: decode)) { _ in
                if first == nil { first = Date() }
                return true
            }
            return (Double(prompt.count) / r.promptSeconds, Double(max(1, r.completionTokenCount - 1)) / r.decodeSeconds,
                    (first ?? start).timeIntervalSince(start))
        }
        _ = try once(a); _ = try once(b)
        var sa: [(Double, Double, Double)] = [], sb: [(Double, Double, Double)] = []
        for i in 0 ..< rounds {
            if i % 2 == 0 { let x = try once(a); sa.append((x.prefill, x.decode, x.ttft)); let y = try once(b); sb.append((y.prefill, y.decode, y.ttft)) }
            else { let y = try once(b); sb.append((y.prefill, y.decode, y.ttft)); let x = try once(a); sa.append((x.prefill, x.decode, x.ttft)) }
        }
        func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
        func line(_ name: String, _ s: [(Double, Double, Double)]) {
            print(name + String(format: ": prefill %.0f tok/s (%.0f to %.0f), decode %.1f tok/s (%.1f to %.1f), time to first token %.2f s", median(s.map { $0.0 }),
                s.map { $0.0 }.min()!, s.map { $0.0 }.max()!, median(s.map { $0.1 }), s.map { $0.1 }.min()!, s.map { $0.1 }.max()!, median(s.map { $0.2 })))
        }
        print("\(prompt.count)-token prompt, \(decode) tokens, median of \(rounds) alternating rounds (min to max)")
        line("whole table      ", sa)
        line("vocabulary window", sb)
        return 0
    }

    /// Time to a verdict: prefill and the first token's logits over the whole vocabulary (or the window), against prefill
    /// and the scores of just the four verdict tokens.
    private static func verdictBench(_ o: [String: String]) throws -> Int32 {
        let (rt, _) = try load(o)
        let prompt = try promptTokens(rt, o)
        let repeats = try int(o, "repeats", 5)
        let tok = rt.tokenizer
        let fulls = ["ON_TASK", "NOT_ALIGNED", "IDLE", "NO_TASK"].map { tok.encode("{\"state\": \"" + $0) }
        var common = 0
        while common < fulls[0].count, fulls.allSatisfy({ common < $0.count && $0[common] == fulls[0][common] }) { common += 1 }
        let candidates = fulls.map { $0[common] }
        let start = prompt + Array(fulls[0][..<common])
        let g = rt.generator
        func time(_ body: () throws -> Void) rethrows -> Double {
            var best = Double.infinity, total = 0.0
            try body() // warm-up
            for _ in 0 ..< repeats {
                let t = Date()
                try body()
                let d = Date().timeIntervalSince(t)
                best = min(best, d); total += d
            }
            return total / Double(repeats)
        }
        Memory.clearCache()
        let full = try time {
            try rt.model.prepareVocabulary(promptTokens: start)
            let cache = rt.model.newCache()
            let l = try g.prefill(start, cache: cache)
            eval(l)
            _ = argMax(l[0], axis: -1).item(Int.self)
        }
        let four = try time { _ = try g.candidateScores(promptTokens: start, candidates: candidates) }
        print("prompt \(start.count) tokens; logits over the \(rt.model.vocabularyWindow.map { "\($0.rowCount)-row window" } ?? "whole vocabulary"): "
              + String(format: "%.3f s", full) + "; the four verdict tokens only: " + String(format: "%.3f s", four))
        print("peak MLX \(Memory.peakMemory >> 20) MB, footprint \(ProcessMemory.snapshot().footprint >> 20) MB")
        return 0
    }

    private static func generate(_ o: [String: String]) throws -> Int32 {
        let (rt, _) = try load(o)
        let tokens: [Int]
        if let text = o["prompt"] {
            tokens = try rt.promptTokens(messages: [["role": "user", "content": text]])
        } else {
            tokens = try promptTokens(rt, o)
        }
        let r = try rt.generator.generate(rt.request(promptTokens: tokens, maxTokens: try int(o, "max-tokens", 120))) { _ in true }
        print(r.text)
        let rate = r.decodeSeconds > 0 ? Double(max(0, r.completionTokenCount - 1)) / r.decodeSeconds : 0
        FileHandle.standardError.write(Data((
            "\n[\(r.promptTokenCount) prompt tokens in " + String(format: "%.2f", r.promptSeconds)
                + "s, \(r.completionTokenCount) new tokens in " + String(format: "%.2f", r.decodeSeconds)
                + "s (" + String(format: "%.1f", rate) + " tok/s), finish \(r.finishReason.rawValue)]\n").utf8))
        return 0
    }

    private static func quantizeCommand(_ o: [String: String]) throws -> Int32 {
        guard let m = o["model"], let out = o["out"] else { throw UsageError(message: "--model and --out are required") }
        try CheckpointQuantizer.write(
            from: URL(fileURLWithPath: m), to: URL(fileURLWithPath: out), bits: try int(o, "bits", 4),
            groupSize: try int(o, "group-size", 64)
        ) { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
        return 0
    }

    private static func vocabEval(_ o: [String: String]) throws -> Int32 {
        guard let m = o["model"], let v = o["vocab"], let p = o["prompts"] else {
            throw UsageError(message: "--model, --vocab and --prompts are required")
        }
        var o2 = o
        o2["vocab"] = nil // the reference holds its whole table
        let (rt, dir) = try load(o2)
        let vocabulary = try StaticVocabulary.load(URL(fileURLWithPath: v))
        var prompts = try VocabularyEvaluation.loadPrompts(URL(fileURLWithPath: p))
        if let limit = o["limit"], let n = Int(limit), n < prompts.count { prompts = Array(prompts.prefix(n)) }
        var options = VocabularyEvaluation.Options()
        options.maxNewTokens = try int(o, "max-new", options.maxNewTokens)
        _ = m
        let summary = try VocabularyEvaluation.run(
            runtime: rt, directory: dir, vocabulary: vocabulary, prompts: prompts, options: options
        ) { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try e.encode(summary)
        if let out = o["out"] { try data.write(to: URL(fileURLWithPath: out)) }
        var brief = summary
        brief.perPrompt = []
        print(String(decoding: try e.encode(brief), as: UTF8.self))
        return 0
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
        let loadStart = Date()
        let (rt, _) = try load(o)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let r = try InferenceBenchmark.run(
            runtime: rt, promptTokens: try int("prompt-tokens", 512), decodeTokens: try int("decode-tokens", 128),
            repeats: try int("repeats", 3))
        report(r, json: json, loadSeconds: loadSeconds, window: rt.model.vocabularyWindow)
        return 0
    }

    private static func report(_ r: BenchmarkResult, json: Bool, loadSeconds: Double, window: VocabularyWindow?) {
        if json {
            let e = JSONEncoder()
            e.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try! e.encode(r), as: UTF8.self))
            return
        }
        func mb(_ b: Int?) -> String { b.map { String(format: "%.1f MB", Double($0) / 1_048_576) } ?? "-" }
        print(String(format: "load      %9.1f s", loadSeconds))
        print(String(format: "prefill   %9.1f tokens/s (%d-token prompt)", r.prefillTokensPerSecond, r.promptTokens))
        print(String(format: "decode    %9.1f tokens/s (%d tokens)", r.decodeTokensPerSecond, r.decodeTokens))
        let cold = r.coldTimeToFirstTokenSeconds.map { String(format: "%.3f", $0) } ?? "-"
        print(String(format: "first token %7.3f s", r.timeToFirstTokenSeconds) + " (first request \(cold) s)")
        if let w = window {
            print("window    \(w.rowCount) rows (\(w.staticIDs.count) static + \(w.extraIDs.count) from the last request), \(mb(w.residentBytes))")
        }
        print("as loaded: footprint \(mb(r.loadedPhysFootprintBytes)), resident set \(mb(r.loadedResidentBytes)), MLX active \(mb(r.loadedMLXActiveBytes))")
        print("peak MLX memory      \(mb(r.peakMLXMemoryBytes))")
        print("peak phys footprint  \(mb(r.peakPhysFootprintBytes))")
        print("final phys footprint \(mb(r.finalPhysFootprintBytes)) (resident set \(mb(r.finalResidentBytes)), MLX cache \(mb(r.finalMLXCacheBytes)))")
    }
}

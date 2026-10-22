import Foundation
import MLX
import XCTest

@testable import Shears

/// Spins up the server on a temporary Unix socket (and on a loopback port) and talks HTTP to it.
final class InferenceServerTests: ShearsTestCase {
    struct Response {
        var status: Int
        var headers: [String: String]
        var body: Data
        var json: [String: Any] { (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:] }
        var text: String { String(decoding: body, as: UTF8.self) }
    }

    var server: InferenceServer!
    var socketPath = ""
    var modelDirectory: URL!
    var runtime: InferenceRuntime!

    override func setUpWithError() throws {
        try super.setUpWithError()
        let tokDir = try InferenceFixture.tempDirectory("tok")
        modelDirectory = try InferenceFixture.tempDirectory("srv")
        addTeardownBlock { [dir = modelDirectory!] in try? FileManager.default.removeItem(at: dir); try? FileManager.default.removeItem(at: tokDir) }
        try InferenceFixture.writeTokenizerFiles(to: tokDir)
        let model = try InferenceFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let healer = try HealingFixture.healer(model: model, keep: keep)
        HealingFixture.randomizeAdapters(healer.adapters, scale: 0.1)
        try InferenceFixture.exportDirectory(model: model, keep: keep, adapters: healer.adapters, tokenizerDirectory: tokDir, in: modelDirectory)
        runtime = try InferenceRuntime.load(directory: modelDirectory, dtype: .float32)
        var config = InferenceServer.Configuration(alias: "tiny", modelPath: modelDirectory.path)
        config.defaultMaxTokens = 12
        server = InferenceServer(runtime: runtime, configuration: config)
        // Unix socket paths are limited to about 100 bytes, so keep the name short.
        socketPath = FileManager.default.temporaryDirectory.appendingPathComponent("s\(UUID().uuidString.prefix(6)).sock").path
        try server.start(.unixSocket(path: socketPath))
        addTeardownBlock { [server = server!] in server.stop() }
    }

    // MARK: Client

    func connect() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &addr.sun_path) { buf in
            for (i, b) in socketPath.utf8.enumerated() { buf[i] = b }
            buf[socketPath.utf8.count] = 0
        }
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        XCTAssertEqual(rc, 0, "connect \(socketPath)")
        return fd
    }

    /// One request on a fresh connection, read to the end (`Connection: close`).
    func request(_ method: String, _ path: String, body: String? = nil, raw: Data? = nil, fd given: Int32? = nil) throws -> Response {
        let fd = try given ?? connect()
        defer { Darwin.close(fd) }
        let payload = raw ?? Data((body ?? "").utf8)
        var head = "\(method) \(path) HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
        if body != nil || raw != nil { head += "Content-Type: application/json\r\nContent-Length: \(payload.count)\r\n" }
        head += "\r\n"
        let data = Data(head.utf8) + payload
        _ = data.withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, 0) }
        return Self.readResponse(fd)
    }

    static func readResponse(_ fd: Int32) -> Response {
        var all = Data()
        var buf = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = read(fd, &buf, buf.count)
            if n <= 0 { break }
            all.append(contentsOf: buf[0 ..< n])
        }
        let sep = Data("\r\n\r\n".utf8)
        guard let r = all.range(of: sep) else { return Response(status: 0, headers: [:], body: all) }
        let head = String(decoding: all[..<r.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
        let status = Int(head[0].split(separator: " ")[1]) ?? 0
        var headers: [String: String] = [:]
        for l in head.dropFirst() {
            if let c = l.firstIndex(of: ":") { headers[l[..<c].lowercased()] = l[l.index(after: c)...].trimmingCharacters(in: .whitespaces) }
        }
        var body = Data(all[r.upperBound...])
        if headers["transfer-encoding"] == "chunked" {
            var out = Data()
            var rest = body
            while let line = rest.range(of: Data("\r\n".utf8)) {
                let size = Int(String(decoding: rest[..<line.lowerBound], as: UTF8.self), radix: 16) ?? 0
                rest = Data(rest[line.upperBound...])
                if size == 0 { break }
                out.append(rest.prefix(size))
                rest = Data(rest.dropFirst(size + 2))
            }
            body = out
        }
        return Response(status: status, headers: headers, body: body)
    }

    func chat(_ fields: [String: Any]) throws -> Response {
        var f = fields
        if f["messages"] == nil { f["messages"] = [["role": "user", "content": "the cat in the hat"]] }
        return try request("POST", "/v1/chat/completions", body: String(decoding: try JSONSerialization.data(withJSONObject: f), as: UTF8.self))
    }

    // MARK: Tests

    func testHealthPropsAndModels() throws {
        let h = try request("GET", "/health")
        XCTAssertEqual(h.status, 200)
        XCTAssertEqual(h.json["status"] as? String, "ok")
        XCTAssertTrue(h.headers["content-type"]?.hasPrefix("application/json") == true)

        let p = try request("GET", "/props")
        XCTAssertEqual(p.status, 200)
        XCTAssertEqual(p.json["model_path"] as? String, modelDirectory.path)
        XCTAssertEqual(p.json["total_slots"] as? Int, 1)
        XCTAssertEqual((p.json["default_generation_settings"] as? [String: Any])?["n_ctx"] as? Int, 512)
        XCTAssertEqual(p.json["bos_token"] as? String, "<bos>")
        XCTAssertEqual((p.json["chat_template"] as? String)?.contains("<|turn>"), true)

        let m = try request("GET", "/v1/models")
        XCTAssertEqual(m.status, 200)
        XCTAssertEqual(m.json["object"] as? String, "list")
        let data = try XCTUnwrap(m.json["data"] as? [[String: Any]])
        XCTAssertEqual(data.count, 1)
        XCTAssertEqual(data[0]["id"] as? String, "tiny")
        XCTAssertEqual(data[0]["object"] as? String, "model")

        let missing = try request("GET", "/nope")
        XCTAssertEqual(missing.status, 404)
        XCTAssertNotNil((missing.json["error"] as? [String: Any])?["message"])
        XCTAssertEqual(try request("POST", "/health", body: "{}").status, 405)
    }

    func testChatCompletionShapeAndDeterminism() throws {
        let r = try chat(["temperature": 0, "max_tokens": 10, "model": "ignored", "unknown_field": [1, 2]])
        XCTAssertEqual(r.status, 200, r.text)
        let j = r.json
        XCTAssertTrue((j["id"] as? String)?.hasPrefix("chatcmpl-") == true)
        XCTAssertEqual(j["object"] as? String, "chat.completion")
        XCTAssertEqual(j["model"] as? String, "tiny")
        XCTAssertNotNil(j["created"] as? Int)
        let choice = try XCTUnwrap((j["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["index"] as? Int, 0)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertEqual(message["role"] as? String, "assistant")
        let content = try XCTUnwrap(message["content"] as? String)
        XCTAssertEqual(choice["finish_reason"] as? String, "length")
        let usage = try XCTUnwrap(j["usage"] as? [String: Any])
        XCTAssertEqual(usage["completion_tokens"] as? Int, 10)
        XCTAssertEqual((usage["prompt_tokens"] as? Int ?? 0) + 10, usage["total_tokens"] as? Int)
        XCTAssertNotNil(j["timings"] as? [String: Any])

        // The same request through the library gives the same text.
        let tokens = try runtime.promptTokens(messages: [["role": "user", "content": "the cat in the hat"]])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, tokens.count)
        let direct = try runtime.generator.generate(runtime.request(promptTokens: tokens, maxTokens: 10, sampling: .greedy))
        XCTAssertEqual(content, direct.text)
        // And so does the same request again.
        XCTAssertEqual((try chat(["temperature": 0, "max_tokens": 10]).json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any] == nil, false)
        let second = try chat(["temperature": 0, "max_tokens": 10])
        XCTAssertEqual(((second.json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String, content)
    }

    func testMaxTokensAndStop() throws {
        let short = try chat(["temperature": 0, "max_tokens": 3]).json
        XCTAssertEqual((short["usage"] as? [String: Any])?["completion_tokens"] as? Int, 3)
        let full = ((try chat(["temperature": 0, "max_tokens": 12]).json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
        XCTAssertGreaterThan(full.count, 4)
        let marker = String(full.dropFirst(3).prefix(2))
        let stopped = try chat(["temperature": 0, "max_tokens": 12, "stop": [marker]])
        let choice = try XCTUnwrap((stopped.json["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "stop")
        let text = (choice["message"] as? [String: Any])?["content"] as? String ?? ""
        XCTAssertEqual(text, full.components(separatedBy: marker).first!)
        // A bare string is accepted for stop too.
        let one = try chat(["temperature": 0, "max_tokens": 12, "stop": marker])
        XCTAssertEqual(((one.json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String, text)
    }

    func testLogprobsShape() throws {
        let r = try chat(["temperature": 0, "max_tokens": 5, "logprobs": true, "top_logprobs": 3])
        XCTAssertEqual(r.status, 200, r.text)
        let choice = try XCTUnwrap((r.json["choices"] as? [[String: Any]])?.first)
        let content = try XCTUnwrap((choice["logprobs"] as? [String: Any])?["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 5)
        for entry in content {
            XCTAssertNotNil(entry["token"] as? String)
            XCTAssertLessThanOrEqual(entry["logprob"] as? Double ?? 1, 0)
            XCTAssertNotNil(entry["bytes"] as? [Int])
            let top = try XCTUnwrap(entry["top_logprobs"] as? [[String: Any]])
            XCTAssertEqual(top.count, 3)
            XCTAssertEqual(top[0]["logprob"] as? Double, entry["logprob"] as? Double)
        }
        // Not asked for: null.
        let plain = try XCTUnwrap((try chat(["max_tokens": 2]).json["choices"] as? [[String: Any]])?.first)
        XCTAssertTrue(plain["logprobs"] is NSNull)
    }

    func testStreamingMatchesNonStreaming() throws {
        let whole = try chat(["temperature": 0, "max_tokens": 10])
        let wholeText = ((whole.json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
        let s = try chat(["temperature": 0, "max_tokens": 10, "stream": true, "stream_options": ["include_usage": true]])
        XCTAssertEqual(s.status, 200)
        XCTAssertEqual(s.headers["content-type"], "text/event-stream")
        let events = s.text.components(separatedBy: "\n\n").filter { !$0.isEmpty }
        XCTAssertEqual(events.last, "data: [DONE]")
        var content = ""
        var finish: String?
        var sawRole = false
        var usage: [String: Any]?
        for e in events.dropLast() {
            XCTAssertTrue(e.hasPrefix("data: "), e)
            let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(e.dropFirst(6).utf8)) as? [String: Any])
            XCTAssertEqual(obj["object"] as? String, "chat.completion.chunk")
            if let u = obj["usage"] as? [String: Any] { usage = u }
            guard let choice = (obj["choices"] as? [[String: Any]])?.first else { continue }
            let delta = choice["delta"] as? [String: Any] ?? [:]
            if delta["role"] as? String == "assistant" { sawRole = true }
            content += delta["content"] as? String ?? ""
            if let f = choice["finish_reason"] as? String { finish = f }
        }
        XCTAssertTrue(sawRole)
        XCTAssertEqual(content, wholeText)
        XCTAssertEqual(finish, "length")
        XCTAssertEqual(usage?["completion_tokens"] as? Int, 10)
    }

    func testStreamingLogprobs() throws {
        let s = try chat(["temperature": 0, "max_tokens": 4, "stream": true, "logprobs": true, "top_logprobs": 2])
        var entries = 0
        for e in s.text.components(separatedBy: "\n\n") where e.hasPrefix("data: {") {
            let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(e.dropFirst(6).utf8)) as? [String: Any])
            if let lp = ((obj["choices"] as? [[String: Any]])?.first?["logprobs"] as? [String: Any])?["content"] as? [[String: Any]] {
                entries += lp.count
            }
        }
        XCTAssertEqual(entries, 4)
    }

    func testJSONObjectResponseFormat() throws {
        let r = try chat(["temperature": 1, "seed": 3, "max_tokens": 200, "response_format": ["type": "json_object"]])
        XCTAssertEqual(r.status, 200, r.text)
        let choice = try XCTUnwrap((r.json["choices"] as? [[String: Any]])?.first)
        let text = (choice["message"] as? [String: Any])?["content"] as? String ?? ""
        var a = JSONObjectAutomaton()
        for b in text.utf8 { XCTAssertTrue(a.accept(b), text) }
        if choice["finish_reason"] as? String == "stop" {
            XCTAssertNotNil(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any], text)
        }
        // Plain text format is accepted and changes nothing.
        XCTAssertEqual(try chat(["max_tokens": 2, "response_format": ["type": "text"]]).status, 200)
    }

    func testErrors() throws {
        func error(_ r: Response) -> [String: Any]? { r.json["error"] as? [String: Any] }
        let bad = try request("POST", "/v1/chat/completions", body: "{not json")
        XCTAssertEqual(bad.status, 400)
        XCTAssertEqual(error(bad)?["type"] as? String, "invalid_request_error")
        XCTAssertEqual(error(bad)?["code"] as? Int, 400)
        XCTAssertNotNil(error(bad)?["message"] as? String)
        XCTAssertEqual(try chat(["messages": [Any]()]).status, 400)
        XCTAssertEqual(try chat(["messages": [["content": "no role"]]]).status, 400)
        XCTAssertEqual(try chat(["n": 2]).status, 400)
        let schema = try chat(["response_format": ["type": "json_schema", "json_schema": ["schema": ["type": "object"]]]])
        XCTAssertEqual(schema.status, 400)
        XCTAssertTrue((error(schema)?["message"] as? String)?.contains("json_schema") == true)
        XCTAssertEqual(try chat(["grammar": "root ::= \"a\""]).status, 400)
        let big = try chat(["messages": [["role": "user", "content": String(repeating: "the ", count: 700)]]])
        XCTAssertEqual(big.status, 400)
        XCTAssertEqual(error(big)?["type"] as? String, "exceed_context_size_error")
    }

    func testRequestsQueueOneAtATime() throws {
        let group = DispatchGroup()
        let lock = NSLock()
        var bodies: [String] = []
        for i in 0 ..< 4 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                if let r = try? self.chat(["temperature": 0, "max_tokens": 6, "messages": [["role": "user", "content": "the \(i % 2 == 0 ? "cat" : "hat")"]]]) {
                    lock.lock(); bodies.append(r.text); lock.unlock()
                }
            }
        }
        // /health answers while requests are generating.
        XCTAssertEqual(try request("GET", "/health").status, 200)
        group.wait()
        XCTAssertEqual(bodies.count, 4)
        for b in bodies { XCTAssertTrue(b.contains("\"finish_reason\""), b) }
    }

    func testApplyTemplate() throws {
        let r = try request("POST", "/apply-template", body: #"{"messages":[{"role":"user","content":"hi"}]}"#)
        XCTAssertEqual(r.status, 200)
        XCTAssertEqual(r.json["prompt"] as? String, "<bos><|turn>user\nhi<turn|>\n<|turn>model\n")
    }

    func testChunkedRequestBodyAndKeepAlive() throws {
        let fd = try connect()
        defer { Darwin.close(fd) }
        let body = Data(#"{"max_tokens":2,"temperature":0,"messages":[{"role":"user","content":"a"}]}"#.utf8)
        let head = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
        let chunked = Data(head.utf8) + Data(String(body.count, radix: 16).utf8) + Data("\r\n".utf8) + body + Data("\r\n0\r\n\r\n".utf8)
        _ = chunked.withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, 0) }
        // Two requests on one connection: read the first response by its Content-Length, then ask again.
        var got = Data()
        var buf = [UInt8](repeating: 0, count: 65536)
        while !String(decoding: got, as: UTF8.self).contains("\"usage\"") {
            let n = read(fd, &buf, buf.count)
            if n <= 0 { break }
            got.append(contentsOf: buf[0 ..< n])
        }
        XCTAssertTrue(String(decoding: got, as: UTF8.self).hasPrefix("HTTP/1.1 200"))
        let second = "GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
        _ = Data(second.utf8).withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, 0) }
        let r = Self.readResponse(fd)
        XCTAssertEqual(r.status, 200)
    }

    func testTCPListenerAndSocketPathLimit() throws {
        let tcp = InferenceServer(runtime: runtime, configuration: .init(alias: "tiny", modelPath: modelDirectory.path))
        try tcp.start(.tcp(host: "127.0.0.1", port: 0))
        defer { tcp.stop() }
        XCTAssertNotEqual(tcp.port, 0)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(tcp.port)/health")!)
        request.timeoutInterval = 10
        let exp = expectation(description: "get")
        var status = 0
        URLSession.shared.dataTask(with: request) { _, response, _ in
            status = (response as? HTTPURLResponse)?.statusCode ?? 0
            exp.fulfill()
        }.resume()
        wait(for: [exp], timeout: 15)
        XCTAssertEqual(status, 200)

        let long = InferenceServer(runtime: runtime, configuration: .init(alias: "x", modelPath: "x"))
        XCTAssertThrowsError(try long.start(.unixSocket(path: "/tmp/" + String(repeating: "a", count: 120))))
    }

    func testStoppingRemovesTheSocket() throws {
        XCTAssertTrue(FileManager.default.fileExists(atPath: socketPath))
        server.stop()
        XCTAssertFalse(FileManager.default.fileExists(atPath: socketPath))
    }

    func testBenchmarkOnTheTinyModel() throws {
        let r = try InferenceBenchmark.run(runtime: runtime, promptTokens: 24, decodeTokens: 16, repeats: 2)
        XCTAssertGreaterThan(r.prefillTokensPerSecond, 0)
        XCTAssertGreaterThan(r.decodeTokensPerSecond, 0)
        XCTAssertGreaterThan(r.timeToFirstTokenSeconds, 0)
        XCTAssertGreaterThan(r.peakMLXMemoryBytes, 0)
        XCTAssertGreaterThanOrEqual(r.peakPhysFootprintBytes, r.finalPhysFootprintBytes / 2)
        XCTAssertGreaterThan(r.finalPhysFootprintBytes, 0)
        let prompt = InferenceBenchmark.syntheticPrompt(length: 50, tokenizer: runtime.tokenizer, vocabularySize: InferenceFixture.vocab)
        XCTAssertEqual(prompt.count, 50)
        XCTAssertTrue(prompt.allSatisfy { !runtime.tokenizer.isSpecial($0) })
    }
}

import Foundation
import MLX

/// Splits a model's reasoning from its answer in a stream of text: what lies between `start` and `end`
/// is reasoning, the rest content. A marker that arrives in pieces is held back until it is complete.
struct ReasoningSplitter {
    let start: String
    let end: String
    private var inside = false
    private var pending = ""

    init(start: String, end: String) {
        self.start = start
        self.end = end
    }

    mutating func feed(_ text: String) -> (content: String, reasoning: String) {
        pending += text
        var content = "", reasoning = ""
        while true {
            let marker = inside ? end : start
            if let r = pending.range(of: marker) {
                let before = String(pending[..<r.lowerBound])
                if inside { reasoning += before } else { content += before }
                pending = String(pending[r.upperBound...])
                inside.toggle()
                continue
            }
            let hold = Self.partialSuffix(pending, of: marker)
            let cut = pending.index(pending.endIndex, offsetBy: -hold)
            let ready = String(pending[..<cut])
            if inside { reasoning += ready } else { content += ready }
            pending = String(pending[cut...])
            return (content, reasoning)
        }
    }

    /// What is held at the end of the stream (an unfinished marker is ordinary text then).
    mutating func finish() -> (content: String, reasoning: String) {
        defer { pending = "" }
        return inside ? ("", pending) : (pending, "")
    }

    private static func partialSuffix(_ text: String, of marker: String) -> Int {
        let maxLen = min(marker.count - 1, text.count)
        guard maxLen > 0 else { return 0 }
        for len in stride(from: maxLen, through: 1, by: -1) where marker.hasPrefix(String(text.suffix(len))) { return len }
        return 0
    }
}

/// An OpenAI-compatible chat server over a ``Generator``, shaped after `llama-server`: the same
/// paths, request fields, response fields and error body, so a client written for it can talk to this.
///
/// Endpoints: `GET /health`, `GET /props`, `GET /v1/models` (and `/models`), `POST /v1/chat/completions`
/// (and `/chat/completions`), `POST /apply-template`.
///
/// `POST /v1/chat/completions` reads `messages`, `max_tokens` / `max_completion_tokens`, `temperature`,
/// `top_p`, `top_k`, `seed`, `stop`, `stream` (with `stream_options.include_usage`), `logprobs` with
/// `top_logprobs`, `response_format` (`text` or `json_object`), `tools` (rendered into the prompt by the
/// model's template; calls the model writes come back as plain content) and
/// `chat_template_kwargs`. Unknown fields are ignored. Not supported, and refused with a 400 rather than
/// ignored: `n` other than 1, `response_format` of `json_schema`, and a `grammar`.
///
/// One generation runs at a time; requests queue.
public final class InferenceServer: @unchecked Sendable {
    public struct Configuration {
        /// The name reported as the model's id.
        public var alias: String
        /// Reported by `/props` as `model_path`.
        public var modelPath: String
        /// `max_tokens` when a request names none.
        public var defaultMaxTokens = 512
        /// Marks of a reasoning section in the model's output, split into `reasoning_content`.
        public var reasoningStart: String? = "<|channel>thought\n"
        public var reasoningEnd: String? = "<channel|>"

        public init(alias: String, modelPath: String) {
            self.alias = alias
            self.modelPath = modelPath
        }
    }

    public let runtime: InferenceRuntime
    public let configuration: Configuration
    private var http: HTTPServer!
    private let generationLock = NSLock()

    public init(runtime: InferenceRuntime, configuration: Configuration) {
        self.runtime = runtime
        self.configuration = configuration
        http = HTTPServer { [weak self] request, writer in self?.route(request, writer) }
    }

    public func start(_ endpoint: HTTPServer.Endpoint) throws { try http.start(endpoint) }
    public func stop() { http.stop() }
    public var port: UInt16 { http.port }

    // MARK: Routing

    func route(_ request: HTTPServer.Request, _ w: HTTPServer.ResponseWriter) {
        switch (request.method, request.path) {
        case ("GET", "/health"), ("GET", "/v1/health"):
            json(w, 200, ["status": "ok"])
        case ("GET", "/props"):
            json(w, 200, props())
        case ("GET", "/v1/models"), ("GET", "/models"):
            json(w, 200, [
                "object": "list",
                "data": [[
                    "id": configuration.alias, "object": "model", "created": Int(Date().timeIntervalSince1970),
                    "owned_by": "shears",
                    "meta": ["n_ctx_train": runtime.generator.contextLength, "n_vocab": runtime.model.configuration.vocabSize],
                ]],
            ])
        case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
            chatCompletions(request, w)
        case ("POST", "/apply-template"):
            applyTemplate(request, w)
        case ("OPTIONS", _):
            w.send(status: 204, body: Data())
        case (_, "/health"), (_, "/props"), (_, "/v1/models"), (_, "/v1/chat/completions"):
            error(w, 405, "method not allowed", "invalid_request_error")
        default:
            error(w, 404, "File Not Found", "not_found_error")
        }
    }

    private func props() -> [String: Any] {
        let d = runtime.defaults.sampling
        return [
            "default_generation_settings": [
                "id": 0, "n_ctx": runtime.generator.contextLength,
                "params": ["temperature": Double(d.temperature), "top_k": d.topK, "top_p": Double(d.topP), "n_predict": -1],
            ] as [String: Any],
            "total_slots": 1,
            "model_path": configuration.modelPath,
            "model_alias": configuration.alias,
            "chat_template": runtime.chatTemplate?.source ?? "",
            "bos_token": runtime.tokenizer.bosToken ?? "",
            "eos_token": runtime.tokenizer.eosToken ?? "",
            "build_info": "shears-serve",
            "is_sleeping": false,
            "modalities": ["vision": false, "audio": false],
        ]
    }

    // MARK: Responses

    private func jsonData(_ object: Any) -> Data {
        (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
    }

    private func json(_ w: HTTPServer.ResponseWriter, _ status: Int, _ object: Any) {
        w.send(status: status, body: jsonData(object))
    }

    private func error(_ w: HTTPServer.ResponseWriter, _ status: Int, _ message: String, _ type: String) {
        json(w, status, ["error": ["code": status, "message": message, "type": type]])
    }

    // MARK: Request parsing

    private struct ChatParameters {
        var messages: [[String: Any]]
        var tools: [[String: Any]]?
        var maxTokens: Int
        var sampling: SamplingParameters
        var stop: [String]
        var stream: Bool
        var includeUsage: Bool
        var logprobs: Int?
        var jsonMode: Bool
        var templateArguments: [String: Any]
    }

    private struct BadRequest: Error {
        var message: String
        var type = "invalid_request_error"
    }

    private func parse(_ body: Data) throws -> ChatParameters {
        guard let text = String(data: body, encoding: .utf8), let root = try? GGUFJSON.parse(text), root.object != nil else {
            throw BadRequest(message: "request body is not a JSON object")
        }
        guard let msgs = root["messages"]?.array, !msgs.isEmpty else { throw BadRequest(message: "\"messages\" is required and must be a non-empty array") }
        let messages = try msgs.map { m -> [String: Any] in
            guard let o = m.anyValue as? [String: Any], o["role"] is String else { throw BadRequest(message: "each message needs a \"role\"") }
            return o
        }
        if let n = root["n"]?.int, n != 1 { throw BadRequest(message: "only n = 1 is supported") }
        if root["grammar"] != nil, root["grammar"]?.string?.isEmpty == false { throw BadRequest(message: "\"grammar\" is not supported") }
        var jsonMode = false
        if let rf = root["response_format"], rf.object != nil {
            switch rf["type"]?.string {
            case "json_object": jsonMode = true
            case "text", nil: break
            default: throw BadRequest(message: "response_format \"\(rf["type"]?.string ?? "")\" is not supported; use text or json_object")
            }
        }
        var sampling = runtime.defaults.sampling
        if let t = root["temperature"]?.double { sampling.temperature = Float(t) }
        if let p = root["top_p"]?.double { sampling.topP = Float(p) }
        if let k = root["top_k"]?.int { sampling.topK = max(0, k) }
        if let s = root["seed"]?.int, s >= 0 { sampling.seed = UInt64(s) }
        var stops: [String] = []
        if let s = root["stop"]?.string { stops = [s] } else if let a = root["stop"]?.array { stops = a.compactMap { $0.string } }
        let maxTokens = root["max_completion_tokens"]?.int ?? root["max_tokens"]?.int ?? root["n_predict"]?.int ?? configuration.defaultMaxTokens
        var logprobs: Int?
        if root["logprobs"]?.bool == true { logprobs = max(0, min(20, root["top_logprobs"]?.int ?? 0)) }
        var kwargs: [String: Any] = [:]
        if let k = root["chat_template_kwargs"]?.anyValue as? [String: Any] { kwargs = k }
        let tools = (root["tools"]?.anyValue as? [Any])?.compactMap { $0 as? [String: Any] }
        return ChatParameters(
            messages: messages, tools: tools, maxTokens: maxTokens < 0 ? configuration.defaultMaxTokens : maxTokens,
            sampling: sampling, stop: stops, stream: root["stream"]?.bool ?? false,
            includeUsage: root["stream_options"]?["include_usage"]?.bool ?? false, logprobs: logprobs,
            jsonMode: jsonMode, templateArguments: kwargs)
    }

    private func applyTemplate(_ request: HTTPServer.Request, _ w: HTTPServer.ResponseWriter) {
        do {
            let p = try parse(request.body)
            guard let t = runtime.chatTemplate else { throw BadRequest(message: "this model has no chat template") }
            let prompt = try t.render(messages: p.messages, tools: p.tools, addGenerationPrompt: true, extra: p.templateArguments)
            json(w, 200, ["prompt": prompt])
        } catch let e as BadRequest {
            error(w, 400, e.message, e.type)
        } catch {
            self.error(w, 400, "\(error)", "invalid_request_error")
        }
    }

    // MARK: Chat completions

    private func chatCompletions(_ request: HTTPServer.Request, _ w: HTTPServer.ResponseWriter) {
        let p: ChatParameters
        let tokens: [Int]
        do {
            p = try parse(request.body)
            tokens = try runtime.promptTokens(
                messages: p.messages, tools: p.tools, addGenerationPrompt: true, templateArguments: p.templateArguments)
            if tokens.count + 1 > runtime.generator.contextLength {
                throw BadRequest(
                    message: "the request (\(tokens.count) tokens) exceeds the available context size (\(runtime.generator.contextLength) tokens)",
                    type: "exceed_context_size_error")
            }
        } catch let e as BadRequest {
            error(w, 400, e.message, e.type)
            return
        } catch {
            self.error(w, 400, "\(error)", "invalid_request_error")
            return
        }

        generationLock.lock()
        defer { generationLock.unlock() }

        var constraint: GenerationConstraint?
        if p.jsonMode {
            constraint = JSONObjectConstraint(
                tokenizer: runtime.tokenizer, stopTokenIDs: runtime.defaults.eosTokenIDs,
                logitsVocabularySize: runtime.model.configuration.vocabSize)
        }
        let genRequest = runtime.request(
            promptTokens: tokens, maxTokens: p.maxTokens, sampling: p.sampling, stopStrings: p.stop, logprobs: p.logprobs,
            constraint: constraint)
        let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let created = Int(Date().timeIntervalSince1970)
        var splitter = Self.splitter(configuration)

        if p.stream {
            guard w.beginStream() else { return }
            func sse(_ object: [String: Any]) -> Bool {
                w.chunk(Data("data: ".utf8) + jsonData(object) + Data("\n\n".utf8))
            }
            func chunk(delta: [String: Any], finish: String? = nil, logprobs: Any? = nil) -> [String: Any] {
                var choice: [String: Any] = ["index": 0, "delta": delta, "finish_reason": finish ?? NSNull()]
                if let logprobs { choice["logprobs"] = logprobs }
                return [
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": configuration.alias,
                    "system_fingerprint": "shears", "choices": [choice],
                ]
            }
            var alive = sse(chunk(delta: ["role": "assistant", "content": ""]))
            func emit(_ pieces: (content: String, reasoning: String), logprobs: Any?) {
                guard alive else { return }
                if pieces.content.isEmpty, pieces.reasoning.isEmpty, logprobs == nil { return }
                var delta: [String: Any] = [:]
                if !pieces.content.isEmpty { delta["content"] = pieces.content }
                if !pieces.reasoning.isEmpty { delta["reasoning_content"] = pieces.reasoning }
                alive = sse(chunk(delta: delta, logprobs: logprobs))
            }
            let result: GenerationResult
            do {
                result = try runtime.generator.generate(genRequest) { event in
                    let lp = event.isTail ? nil : Self.logprobContent(event, self.runtime.tokenizer, requested: p.logprobs != nil)
                    if var s = splitter {
                        let pieces = s.feed(event.text)
                        splitter = s
                        emit(pieces, logprobs: lp)
                    } else {
                        emit((event.text, ""), logprobs: lp)
                    }
                    return alive
                }
            } catch {
                _ = sse(["error": ["code": 500, "message": "\(error)", "type": "server_error"]])
                w.end()
                return
            }
            if var s = splitter { emit(s.finish(), logprobs: nil); splitter = s }
            if alive {
                alive = sse(chunk(delta: [:], finish: result.finishReason == .length ? "length" : "stop"))
            }
            if alive, p.includeUsage {
                var usage = chunk(delta: [:])
                usage["choices"] = [Any]()
                usage["usage"] = Self.usage(result)
                usage["timings"] = Self.timings(result)
                alive = sse(usage)
            }
            if alive { _ = w.chunk(Data("data: [DONE]\n\n".utf8)) }
            w.end()
            return
        }

        do {
            var logprobEntries: [[String: Any]] = []
            let result = try runtime.generator.generate(genRequest) { event in
                if !event.isTail, let e = Self.logprobContent(event, self.runtime.tokenizer, requested: p.logprobs != nil) {
                    logprobEntries.append(contentsOf: e["content"] as? [[String: Any]] ?? [])
                }
                return true
            }
            var content = result.text
            var reasoning = ""
            if var s = splitter {
                var a = s.feed(result.text)
                let b = s.finish()
                a.content += b.content
                a.reasoning += b.reasoning
                content = a.content
                reasoning = a.reasoning
            }
            var message: [String: Any] = ["role": "assistant", "content": content]
            if !reasoning.isEmpty { message["reasoning_content"] = reasoning }
            var choice: [String: Any] = [
                "index": 0, "finish_reason": result.finishReason == .length ? "length" : "stop", "message": message,
                "logprobs": p.logprobs != nil ? ["content": logprobEntries] as [String: Any] : NSNull(),
            ]
            if p.logprobs == nil { choice["logprobs"] = NSNull() }
            json(w, 200, [
                "id": id, "object": "chat.completion", "created": created, "model": configuration.alias,
                "system_fingerprint": "shears", "choices": [choice], "usage": Self.usage(result),
                "timings": Self.timings(result),
            ])
        } catch {
            self.error(w, 500, "\(error)", "server_error")
        }
    }

    private static func splitter(_ c: Configuration) -> ReasoningSplitter? {
        guard let s = c.reasoningStart, let e = c.reasoningEnd, !s.isEmpty, !e.isEmpty else { return nil }
        return ReasoningSplitter(start: s, end: e)
    }

    private static func usage(_ r: GenerationResult) -> [String: Any] {
        ["prompt_tokens": r.promptTokenCount, "completion_tokens": r.completionTokenCount, "total_tokens": r.promptTokenCount + r.completionTokenCount]
    }

    private static func timings(_ r: GenerationResult) -> [String: Any] {
        let computed = max(r.promptTokenCount - r.cachedPromptTokenCount, 0)
        func rate(_ n: Int, _ s: Double) -> Double { s > 0 ? Double(n) / s : 0 }
        return [
            "cache_n": r.cachedPromptTokenCount, "prompt_n": computed, "prompt_ms": r.promptSeconds * 1000,
            "prompt_per_second": rate(computed, r.promptSeconds), "predicted_n": r.completionTokenCount,
            "predicted_ms": r.decodeSeconds * 1000, "predicted_per_second": rate(r.completionTokenCount, r.decodeSeconds),
        ]
    }

    /// The `logprobs.content` entry for one token: `{"content": [entry]}`, or nil when none was asked for.
    private static func logprobContent(_ e: GenerationEvent, _ tok: BPETokenizer, requested: Bool) -> [String: Any]? {
        guard requested, let lp = e.logprob else { return nil }
        func entry(_ t: TokenLogprob, top: [[String: Any]]? = nil) -> [String: Any] {
            let text = tok.decode([t.id])
            var d: [String: Any] = [
                "token": text, "logprob": Double(max(t.logprob, -9999)), "bytes": Array(tok.decodedBytes(of: t.id)).map(Int.init),
            ]
            if let top { d["top_logprobs"] = top }
            return d
        }
        return ["content": [entry(lp, top: e.topLogprobs.map { entry($0) })]]
    }
}

extension GGUFJSON {
    /// A native Swift value: dictionary, array, String, Bool, Int, Double or NSNull.
    var anyValue: Any {
        switch self {
        case .null: NSNull()
        case .bool(let b): b
        case .int(let i): i
        case .double(let d): d
        case .string(let s): s
        case .array(let a): a.map { $0.anyValue }
        case .object(let o):
            Dictionary(o.map { ($0.key, $0.value.anyValue) }, uniquingKeysWith: { $1 })
        }
    }
}

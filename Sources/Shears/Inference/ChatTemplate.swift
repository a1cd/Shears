import Foundation
import Jinja

/// A model's Jinja chat template (`chat_template.jinja`, or the `chat_template` of
/// `tokenizer_config.json` / `chat_template.json`), rendered with swift-jinja.
///
/// The variables are the ones Hugging Face's `apply_chat_template` passes: `messages`, `tools`,
/// `add_generation_prompt`, `bos_token`, `eos_token`, plus whatever extra keys the caller gives
/// (`enable_thinking`, ...).
///
/// swift-jinja 2.5 does not apply whitespace control (`{{-`, `-}}`, `{%-`, `-%}`) the way Jinja
/// does when the tag sits next to text that ends or starts with a brace, so a template that relies
/// on it (Gemma 4's tool declarations do) renders with stray spaces. ``normalizeWhitespaceControl(_:)``
/// applies the control to the source first and the result matches Python's output byte for byte on
/// the cases in the tests.
public struct ChatTemplate: Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public var description: String
        init(_ d: String) { description = d }
    }

    public let source: String
    public var bosToken: String?
    public var eosToken: String?
    private let normalized: String
    /// Render the template as written, without ``normalizeWhitespaceControl(_:)``. For tests.
    var rendersUnnormalized = false

    public init(source: String, bosToken: String? = nil, eosToken: String? = nil) {
        self.source = source
        self.bosToken = bosToken
        self.eosToken = eosToken
        normalized = Self.normalizeWhitespaceControl(source)
    }

    /// Looks in `directory` for `chat_template.jinja`, then `tokenizer_config.json`'s `chat_template` (a
    /// string, or a list of `{name, template}` where `default` is used), then `chat_template.json`.
    public static func load(directory: URL, bosToken: String?, eosToken: String?) throws -> ChatTemplate? {
        let jinja = directory.appendingPathComponent("chat_template.jinja")
        if let s = try? String(contentsOf: jinja, encoding: .utf8) {
            return ChatTemplate(source: s, bosToken: bosToken, eosToken: eosToken)
        }
        func extract(_ v: GGUFJSON?) -> String? {
            if let s = v?.string { return s }
            for item in v?.array ?? [] where item["name"]?.string == "default" { return item["template"]?.string }
            return nil
        }
        for name in ["tokenizer_config.json", "chat_template.json"] {
            if let json = try GGUFJSON.load(directory.appendingPathComponent(name)), let s = extract(json["chat_template"]) {
                return ChatTemplate(source: s, bosToken: bosToken, eosToken: eosToken)
            }
        }
        return nil
    }

    /// Renders the prompt text for `messages` (each `["role": ..., "content": ...]`, content a string or a
    /// list of `["type": "text", "text": ...]` parts; assistant messages may carry `tool_calls`).
    public func render(
        messages: [[String: Any]], tools: [[String: Any]]? = nil, addGenerationPrompt: Bool = true,
        extra: [String: Any] = [:]
    ) throws -> String {
        var context: [String: Value] = [:]
        do {
            for (k, v) in extra { context[k] = try Value(any: Self.native(v)) }
            context["messages"] = try Value(any: Self.native(messages))
            if let tools, !tools.isEmpty { context["tools"] = try Value(any: Self.native(tools)) }
            context["add_generation_prompt"] = .boolean(addGenerationPrompt)
            if let bosToken { context["bos_token"] = .string(bosToken) }
            if let eosToken { context["eos_token"] = .string(eosToken) }
            return try Template(rendersUnnormalized ? source : normalized).render(context)
        } catch {
            throw Failure("chat template failed: \(error)")
        }
    }

    /// `value` with Foundation's boolean numbers (what `JSONSerialization` returns for `true` and `false`)
    /// turned into Swift `Bool`s, so a template sees `true` and not `1`.
    static func native(_ value: Any) -> Any {
        switch value {
        case let n as NSNumber:
            if n.isBoolean { return n.boolValue }
            return n.isFloatingPoint ? n.doubleValue : n.intValue
        case let d as [String: Any]: return d.mapValues(native)
        case let a as [Any]: return a.map(native)
        default: return value
        }
    }

    // MARK: Whitespace control

    /// Applies `-` whitespace control to the template source: for a tag opened with `-`, trailing
    /// whitespace of the text before it is dropped; for one closed with `-`, leading whitespace of the text
    /// after it. The dashes are then removed so the engine does not apply them a second time. Text and
    /// tags are told apart with string literals respected (`'}}'` inside an expression is not a tag end).
    ///
    /// A `{` left at the end of the text before a tag would fuse with the tag's own opening brace once the
    /// space between them is gone (`{ {{- x -}}`), so it is emitted as an expression.
    public static func normalizeWhitespaceControl(_ source: String) -> String {
        let s = Array(source.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        var trimNext = false
        func isSpace(_ c: Unicode.Scalar) -> Bool { c == " " || c == "\t" || c == "\n" || c == "\r" }
        while i < s.count {
            var j = i
            while j < s.count, !(s[j] == "{" && j + 1 < s.count && (s[j + 1] == "{" || s[j + 1] == "%" || s[j + 1] == "#")) { j += 1 }
            var text = Array(s[i ..< j])
            if trimNext { while let f = text.first, isSpace(f) { text.removeFirst() } }
            trimNext = false
            guard j < s.count else { out.append(contentsOf: text); break }
            let kind = s[j + 1]
            let dash = j + 2 < s.count && s[j + 2] == "-"
            if dash { while let l = text.last, isSpace(l) { text.removeLast() } }
            if dash, text.last == "{" {
                text.removeLast()
                text += Array("{{ '{' }}".unicodeScalars)
            }
            out.append(contentsOf: text)
            var k = j + 2
            var quote: Unicode.Scalar?
            let closer: Unicode.Scalar = kind == "{" ? "}" : (kind == "%" ? "%" : "#")
            while k < s.count {
                let c = s[k]
                if let q = quote {
                    if c == "\\" { k += 1 } else if c == q { quote = nil }
                } else if kind != "#" && (c == "'" || c == "\"") {
                    quote = c
                } else if c == closer && k + 1 < s.count && s[k + 1] == "}" {
                    break
                }
                k += 1
            }
            guard k < s.count else { out.append(contentsOf: s[j...]); break }
            var tag = Array(s[j ... k + 1])
            if dash { tag.remove(at: 2) }
            if tag.count >= 4, tag[tag.count - 3] == "-" {
                tag.remove(at: tag.count - 3)
                trimNext = true
            }
            out.append(contentsOf: tag)
            i = k + 2
        }
        return String(out)
    }
}

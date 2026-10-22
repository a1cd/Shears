import Foundation
import XCTest

@testable import Shears

final class InferenceTemplateTests: XCTestCase {
    struct Case {
        var name: String
        var messages: [[String: Any]]
        var tools: [[String: Any]]?
        var generationPrompt: Bool
        var enableThinking: Bool
        var expected: String
    }

    func cases() throws -> [Case] {
        let root = try JSONSerialization.jsonObject(with: Data(InferenceTemplateFixtures.casesJSON.utf8)) as! [[String: Any]]
        return root.map { c in
            Case(
                name: c["name"] as! String, messages: c["messages"] as! [[String: Any]], tools: c["tools"] as? [[String: Any]],
                generationPrompt: c["gen"] as! Bool, enableThinking: c["enable_thinking"] as? Bool ?? false,
                expected: c["expected"] as! String)
        }
    }

    /// The Gemma 4 template renders byte for byte what Python's jinja2 renders, across system prompts,
    /// thinking, content parts, multi-turn with a stripped thinking channel, tool declarations, tool calls and
    /// tool replies.
    func testGemma4TemplateMatchesJinja2() throws {
        let template = ChatTemplate(source: InferenceTemplateFixtures.template, bosToken: "<bos>", eosToken: "<eos>")
        let all = try cases()
        XCTAssertGreaterThanOrEqual(all.count, 10)
        for c in all {
            let out = try template.render(
                messages: c.messages, tools: c.tools, addGenerationPrompt: c.generationPrompt,
                extra: c.enableThinking ? ["enable_thinking": true] : [:])
            XCTAssertEqual(out, c.expected, "case \(c.name)")
        }
    }

    /// Without whitespace control applied first, swift-jinja renders the tool declarations differently from
    /// jinja2: the normalization earns its place.
    func testWhitespaceNormalizationIsNeeded() throws {
        let tools = try XCTUnwrap(cases().first { $0.name == "tools" })
        var raw = ChatTemplate(source: InferenceTemplateFixtures.template, bosToken: "<bos>")
        raw.rendersUnnormalized = true
        let out = try raw.render(messages: tools.messages, tools: tools.tools, addGenerationPrompt: true)
        XCTAssertNotEqual(out, tools.expected)
    }

    func testNormalizerHonorsDashes() {
        XCTAssertEqual(ChatTemplate.normalizeWhitespaceControl("a  {{- x -}}  b"), "a{{ x }}b")
        XCTAssertEqual(ChatTemplate.normalizeWhitespaceControl("a  {%- if x -%}  b {%- endif %}"), "a{% if x %}b{% endif %}")
        XCTAssertEqual(ChatTemplate.normalizeWhitespaceControl("{ {{- x }}"), "{{ '{' }}{{ x }}")
        // A dash or tag-looking text inside a string literal is left alone.
        XCTAssertEqual(ChatTemplate.normalizeWhitespaceControl("{{ '-}}' }} y"), "{{ '-}}' }} y")
        XCTAssertEqual(ChatTemplate.normalizeWhitespaceControl("{# c -#}\n x"), "{# c #}x")
    }

    func testLoadsTemplateFromFilesInOrder() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(try ChatTemplate.load(directory: dir, bosToken: nil, eosToken: nil))
        try Data(#"{"chat_template": [{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D"}]}"#.utf8)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        XCTAssertEqual(try ChatTemplate.load(directory: dir, bosToken: nil, eosToken: nil)?.source, "D")
        try "J".write(to: dir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        XCTAssertEqual(try ChatTemplate.load(directory: dir, bosToken: nil, eosToken: nil)?.source, "J")
    }

    func testRenderErrorIsReported() {
        let t = ChatTemplate(source: "{{ messages[0]['x'].y.z() }}")
        XCTAssertThrowsError(try t.render(messages: [["role": "user"]]))
    }
}

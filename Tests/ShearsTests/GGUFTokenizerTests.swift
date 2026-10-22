import Foundation
import XCTest

@testable import Shears

final class GGUFTokenizerTests: ShearsTestCase {
    func load(_ dir: URL) throws -> GGUFTokenizerMetadata {
        try GGUFTokenizer.load(
            directory: dir, model: "gemma4", userDefinedTokens: GGUFExport.gemma4VisibleTokens)
    }

    func testTinyTokenizerMetadata() throws {
        let dir = try makeGGUFTempDirectory()
        try GGUFTinyGemma4.writeTokenizer(to: dir)
        // config.json supplies ids that tokenizer_config.json doesn't name.
        try JSONSerialization.data(withJSONObject: ["text_config": ["sep_token_id": 4, "pad_token_id": 0]])
            .write(to: dir.appendingPathComponent("config.json"))
        let m = try load(dir)

        XCTAssertEqual(m.tokenCount, GGUFTinyGemma4.vocabCount)
        XCTAssertEqual(m["tokenizer.ggml.model"], .string("gemma4"))

        let tokens = try XCTUnwrap(m["tokenizer.ggml.tokens"]?.arrayValue).compactMap { $0.stringValue }
        XCTAssertEqual(tokens.count, 293)
        XCTAssertEqual(Array(tokens[0 ..< 8]), ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "▁", "a", "b"])
        XCTAssertEqual(tokens[32], "▁a")
        XCTAssertEqual(tokens[34], "<0x00>")
        XCTAssertEqual(tokens[289], "<0xFF>")
        XCTAssertEqual(Array(tokens[290...]), ["<|turn>", "<turn|>", "<|channel>"])

        // Token types: named specials are CONTROL, byte tokens BYTE, added tokens that aren't
        // special USER_DEFINED, and the chat-parser tokens are forced visible.
        let types = try XCTUnwrap(m["tokenizer.ggml.token_type"]?.arrayValue).compactMap { $0.intValue }
        XCTAssertEqual(types.count, 293)
        XCTAssertEqual(Array(types[0 ..< 6]), [3, 3, 3, 3, 3, 1])
        XCTAssertEqual(types[34], 6)
        XCTAssertEqual(types[289], 6)
        XCTAssertEqual(types[290], 3, "<|turn> is named in extra_special_tokens")
        XCTAssertEqual(types[291], 3, "<turn|> is eot_token")
        XCTAssertEqual(types[292], 4, "<|channel> is forced to USER_DEFINED even though it is named")
        XCTAssertEqual(m["tokenizer.ggml.token_type"]?.arrayValue?.first?.type, .int32)

        XCTAssertEqual(m["tokenizer.ggml.scores"], .float32s([Float](repeating: -1000, count: 293)))
        XCTAssertEqual(m["tokenizer.ggml.merges"], .strings(["▁ a", "a b"]))

        XCTAssertEqual(m["tokenizer.ggml.bos_token_id"], .uint32(2))
        XCTAssertEqual(m["tokenizer.ggml.eos_token_id"], .uint32(1))
        XCTAssertEqual(m["tokenizer.ggml.unknown_token_id"], .uint32(3))
        XCTAssertEqual(m["tokenizer.ggml.padding_token_id"], .uint32(0))
        XCTAssertEqual(m["tokenizer.ggml.mask_token_id"], .uint32(4))
        XCTAssertEqual(m["tokenizer.ggml.seperator_token_id"], .uint32(4), "from config.json when not named")
        XCTAssertEqual(m["tokenizer.ggml.add_eos_token"], .bool(false))
        XCTAssertNil(m["tokenizer.ggml.add_bos_token"], "the architecture sets this, not the tokenizer")
        XCTAssertEqual(m["tokenizer.chat_template"], .string(GGUFTinyGemma4.chatTemplate))
        XCTAssertNil(m["tokenizer.ggml.pre"])
    }

    func testPairMergesWithSpacesAreEscaped() throws {
        let pairs = try GGUFJSON.parse(#"[["a b","c"],["x","y"]]"#)
        XCTAssertEqual(try GGUFTokenizer.mergeStrings(pairs), ["a\u{0120}b c", "x y"])
        let strings = try GGUFJSON.parse(#"["a b","c d"]"#)
        XCTAssertEqual(try GGUFTokenizer.mergeStrings(strings), ["a b", "c d"], "string form is kept")
        XCTAssertNil(try GGUFTokenizer.mergeStrings(try GGUFJSON.parse("[]")))
        XCTAssertThrowsError(try GGUFTokenizer.mergeStrings(try GGUFJSON.parse("[[1, 2]]")))
        // A space followed by a combining mark is still a space code point.
        XCTAssertEqual(GGUFTokenizer.escapeSpaces(" \u{0301}x"), "\u{0120}\u{0301}x")
    }

    func testByteTokenDetection() {
        for t in ["<0x00>", "<0xFF>", "<0xaB>"] { XCTAssertTrue(GGUFTokenizer.isByteToken(t), t) }
        for t in ["<0x0>", "<0x000>", "<0xGG>", "0x00", "<0x00", "a<0x00>", ""] {
            XCTAssertFalse(GGUFTokenizer.isByteToken(t), t)
        }
    }

    func testGapsInTheVocabularyArePadded() throws {
        let dir = try makeGGUFTempDirectory()
        let json: [String: Any] = [
            "added_tokens": [],
            "model": ["type": "BPE", "vocab": ["a": 0, "b": 3], "merges": []],
        ]
        try JSONSerialization.data(withJSONObject: json).write(to: dir.appendingPathComponent("tokenizer.json"))
        let m = try GGUFTokenizer.load(directory: dir, model: "gemma4")
        XCTAssertEqual(m["tokenizer.ggml.tokens"], .strings(["a", "[PAD1]", "[PAD2]", "b"]))
        XCTAssertEqual(m["tokenizer.ggml.token_type"], .int32s([1, 5, 5, 1]))
        XCTAssertNil(m["tokenizer.ggml.merges"], "an empty merge list is not written")
    }

    func testSpecialFlagsAreTheFallbackWithoutAConfig() throws {
        let dir = try makeGGUFTempDirectory()
        let json: [String: Any] = [
            "added_tokens": [
                ["id": 2, "content": "<s>", "special": true], ["id": 3, "content": "<x>", "special": false],
            ],
            "model": ["type": "BPE", "vocab": ["a": 0, "b": 1], "merges": []],
        ]
        try JSONSerialization.data(withJSONObject: json).write(to: dir.appendingPathComponent("tokenizer.json"))
        let m = try GGUFTokenizer.load(directory: dir, model: "gemma4")
        XCTAssertEqual(m["tokenizer.ggml.token_type"], .int32s([1, 1, 3, 4]))
        XCTAssertNil(m["tokenizer.chat_template"])
    }

    func testRejectsNonBPE() throws {
        let dir = try makeGGUFTempDirectory()
        let json: [String: Any] = ["model": ["type": "Unigram", "vocab": []]]
        try JSONSerialization.data(withJSONObject: json).write(to: dir.appendingPathComponent("tokenizer.json"))
        XCTAssertThrowsError(try GGUFTokenizer.load(directory: dir, model: "gemma4"))
        XCTAssertThrowsError(try GGUFTokenizer.load(directory: try makeGGUFTempDirectory(), model: "gemma4"))
    }

    func testChatTemplateForms() throws {
        let dir = try makeGGUFTempDirectory()
        try GGUFTinyGemma4.writeTokenizer(to: dir)
        // A list of named templates in tokenizer_config.json.
        var cfg = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("tokenizer_config.json")))
                as? [String: Any])
        cfg["chat_template"] = [
            ["name": "default", "template": "D"], ["name": "tool use", "template": "T"],
        ]
        try JSONSerialization.data(withJSONObject: cfg).write(to: dir.appendingPathComponent("tokenizer_config.json"))
        let m = try load(dir)
        XCTAssertEqual(m["tokenizer.chat_template"], .string("D"))
        XCTAssertEqual(m["tokenizer.chat_template.tool_use"], .string("T"))
        XCTAssertEqual(m["tokenizer.chat_templates"], .strings(["tool_use"]))

        // A chat_template.json beside the config, when the config has none.
        cfg["chat_template"] = nil
        try JSONSerialization.data(withJSONObject: cfg).write(to: dir.appendingPathComponent("tokenizer_config.json"))
        try FileManager.default.removeItem(at: dir.appendingPathComponent("chat_template.jinja"))
        try JSONSerialization.data(withJSONObject: ["chat_template": "J"])
            .write(to: dir.appendingPathComponent("chat_template.json"))
        XCTAssertEqual(try load(dir)["tokenizer.chat_template"], .string("J"))
    }

    func testNormalizerFlags() throws {
        var n = GGUFTokenizer.Normalizer()
        n.parse(
            try GGUFJSON.parse(
                #"{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Lowercase"},{"type":"BertNormalizer","strip_accents":true}]}"#
            ))
        XCTAssertEqual(n.nfc, true)
        XCTAssertEqual(n.lowercase, true)
        XCTAssertEqual(n.stripAccents, true)
        var r = GGUFTokenizer.Normalizer()
        r.parse(try GGUFJSON.parse(#"{"type":"Replace","content":"\u2581"}"#))
        XCTAssertNil(r.nfc)
        XCTAssertNil(r.lowercase)
    }

    func testTokensThatDifferOnlyInUnicodeFormOrByteOrderMarkStayDistinct() throws {
        // Swift treats "e\u{301}" and "\u{e9}" as equal, and Foundation's JSON reader drops a
        // leading U+FEFF. Real vocabularies contain both patterns.
        let dir = try makeGGUFTempDirectory()
        let json = """
            {"added_tokens":[],"model":{"type":"BPE","vocab":{
              "a":0, "\u{e9}":1, "e\u{301}":2, "\u{FEFF}x":3, "x":4, "\u{FEFF}":5, "\u{FEFF}\u{FEFF}":6
            },"merges":[["\u{e9}","a"],["e\u{301}","a"]]}}
            """
        try json.write(to: dir.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        let m = try GGUFTokenizer.load(directory: dir, model: "gemma4")
        let tokens = try XCTUnwrap(m["tokenizer.ggml.tokens"]?.arrayValue).compactMap { $0.stringValue }
        XCTAssertEqual(m.tokenCount, 7)
        let bytes = tokens.map { Array($0.utf8) }
        XCTAssertEqual(bytes[1], [0xC3, 0xA9])
        XCTAssertEqual(bytes[2], [0x65, 0xCC, 0x81])
        XCTAssertEqual(bytes[3], [0xEF, 0xBB, 0xBF, 0x78])
        XCTAssertEqual(bytes[5], [0xEF, 0xBB, 0xBF])
        XCTAssertEqual(bytes[6], [0xEF, 0xBB, 0xBF, 0xEF, 0xBB, 0xBF])
        let merges = try XCTUnwrap(m["tokenizer.ggml.merges"]?.arrayValue).compactMap { $0.stringValue }
        XCTAssertEqual(merges.map { Array($0.utf8) }, [[0xC3, 0xA9, 0x20, 0x61], [0x65, 0xCC, 0x81, 0x20, 0x61]])

        // The file survives being written and read back.
        let out = dir.appendingPathComponent("t.gguf")
        let w = try GGUFWriter()
        for (k, v) in m.entries { try w.set(k, v) }
        try w.write(to: out)
        let back = try XCTUnwrap(GGUFReader(url: out)["tokenizer.ggml.tokens"]?.arrayValue).compactMap { $0.stringValue }
        XCTAssertEqual(back.map { Array($0.utf8) }, bytes)
    }

    func testJSONReader() throws {
        let v = try GGUFJSON.parse(
            #" {"a": [1, -2, 3.5, 1e3, true, false, null], "s": "q\"\\\/\b\f\n\r\t\u00e9\ud83d\ude00", "o": {}, "e": []} "#)
        XCTAssertEqual(v["a"]?.array?.compactMap { $0.int }, [1, -2, 1000])
        XCTAssertEqual(v["a"]?.array?[2].double, 3.5)
        XCTAssertEqual(v["a"]?.array?[4].bool, true)
        XCTAssertEqual(v["s"]?.string, "q\"\\/\u{8}\u{c}\n\r\t\u{e9}\u{1F600}")
        XCTAssertEqual(v["o"]?.object?.count, 0)
        XCTAssertEqual(v["e"]?.array?.count, 0)
        XCTAssertNil(v["missing"])
        for bad in ["", "{", "[1,]", #"{"a" 1}"#, #""abc"#, "tru", "[1] x", #""\q""#, #""\u12""#] {
            XCTAssertThrowsError(try GGUFJSON.parse(bad), bad)
        }
        // A byte-order mark at the start of the file is skipped; inside strings it is kept.
        XCTAssertEqual(try GGUFJSON.parse(Data([0xEF, 0xBB, 0xBF] + Array(#"["\#u{FEFF}"]"#.utf8))).array?[0].string, "\u{FEFF}")
        // Duplicate keys are kept in order; lookup returns the first.
        XCTAssertEqual(try GGUFJSON.parse(#"{"k":1,"k":2}"#).object?.count, 2)
    }

    // MARK: Oracle: llama.cpp's own vocabulary file for Gemma 4

    /// Compares the metadata built from the real Gemma 4 `tokenizer.json` with the vocabulary-only
    /// GGUF that ships in llama.cpp's `models/` directory. Set
    /// `TEST_RUNNER_GGUF_ORACLE_VOCAB` to that file and `TEST_RUNNER_GGUF_ORACLE_TOKENIZER_DIR` to
    /// a directory holding the model's tokenizer files (not committed; ~32 MB).
    func testMatchesLlamaCppReferenceVocabulary() throws {
        let env = ProcessInfo.processInfo.environment
        guard let vocabPath = env["GGUF_ORACLE_VOCAB"], let tokDir = env["GGUF_ORACLE_TOKENIZER_DIR"] else {
            throw XCTSkip("set TEST_RUNNER_GGUF_ORACLE_VOCAB and TEST_RUNNER_GGUF_ORACLE_TOKENIZER_DIR")
        }
        let oracle = try GGUFReader(url: URL(fileURLWithPath: vocabPath))
        let mine = try load(URL(fileURLWithPath: tokDir))
        XCTAssertEqual(mine.tokenCount, 262144)
        for key in [
            "tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.mask_token_id",
        ] {
            let expected = try XCTUnwrap(oracle[key], key)
            let got = try XCTUnwrap(mine[key], key)
            if let a = expected.arrayValue, let b = got.arrayValue {
                XCTAssertEqual(a.count, b.count, "\(key) length")
                // Compare strings by their bytes: Swift's == treats canonically equivalent
                // Unicode as equal, which would hide exactly the bugs this test is for.
                func same(_ x: GGUFValue, _ y: GGUFValue) -> Bool {
                    if let xs = x.stringValue, let ys = y.stringValue { return Array(xs.utf8) == Array(ys.utf8) }
                    return x == y
                }
                let diffs = zip(a, b).enumerated().filter { !same($0.element.0, $0.element.1) }
                let sample = diffs.prefix(5).map { "\($0.offset): ref \($0.element.0) mine \($0.element.1)" }
                XCTAssertEqual(diffs.count, 0, "\(key) differs: \(sample)")
            } else if let x = expected.stringValue, let y = got.stringValue {
                XCTAssertTrue(Array(x.utf8) == Array(y.utf8), "\(key) differs (\(x.utf8.count) vs \(y.utf8.count) bytes)")
            } else {
                XCTAssertEqual(expected, got, key)
            }
        }
        // The reference file was made from an earlier revision of the chat template, so the
        // template can only be checked against the file it came from.
        let template = try String(contentsOf: URL(fileURLWithPath: tokDir).appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        XCTAssertEqual(mine["tokenizer.chat_template"], .string(template))
        // Keys the reference has that mine leaves to the architecture's own step.
        XCTAssertEqual(oracle["tokenizer.ggml.add_bos_token"], .bool(true))
        XCTAssertEqual(oracle["tokenizer.ggml.add_space_prefix"], .bool(false))
        XCTAssertNil(mine["tokenizer.ggml.add_eos_token"])
    }
}

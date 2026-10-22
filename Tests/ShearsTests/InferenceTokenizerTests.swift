import Foundation
import XCTest

@testable import Shears

final class InferenceTokenizerTests: XCTestCase {
    func makeTokenizer() throws -> BPETokenizer {
        try BPETokenizer(
            tokenizerJSON: Data(InferenceTokenizerFixtures.tokenizerJSON.utf8),
            configuration: Data(InferenceTokenizerFixtures.configJSON.utf8))
    }

    struct Case: Decodable {
        var text: String
        var ids: [Int]
        var decoded: String
    }

    func cases() throws -> [Case] {
        try JSONDecoder().decode([Case].self, from: Data(InferenceTokenizerFixtures.casesJSON.utf8))
    }

    /// Ids equal what Hugging Face `tokenizers` produces from the same tokenizer.json.
    func testEncodeMatchesReference() throws {
        let t = try makeTokenizer()
        for c in try cases() {
            XCTAssertEqual(t.encode(c.text), c.ids, "encode \(c.text.debugDescription)")
        }
    }

    func testDecodeMatchesReferenceAndRoundTrips() throws {
        let t = try makeTokenizer()
        for c in try cases() {
            XCTAssertEqual(t.decode(c.ids), c.decoded, "decode \(c.text.debugDescription)")
            // Round trip: byte fallback restores every character exactly (the reference's decode is
            // lossless for text with no specials in it, and `▁` in the input decodes as a space).
            if !c.text.contains("▁") { XCTAssertEqual(t.decode(t.encode(c.text)), c.text, "round trip \(c.text.debugDescription)") }
        }
    }

    func testAddedTokensAreSingleIds() throws {
        let t = try makeTokenizer()
        let bos = try XCTUnwrap(t.tokenID(for: "<bos>"))
        let turn = try XCTUnwrap(t.tokenID(for: "<|turn>"))
        let end = try XCTUnwrap(t.tokenID(for: "<turn|>"))
        XCTAssertEqual(t.encode("<bos><|turn>user\nhi<turn|>").first, bos)
        XCTAssertEqual(t.encode("<|turn>"), [turn])
        XCTAssertEqual(t.encode("<turn|>"), [end])
        XCTAssertEqual(t.bosTokenID, bos)
        XCTAssertEqual(t.eosToken, "<eos>")
        // Gemma's post-processor adds nothing, so an encode with special tokens equals one without.
        XCTAssertEqual(t.encode("hi", addSpecialTokens: true), t.encode("hi"))
        // Special ids are skipped on request.
        XCTAssertEqual(t.decode([bos] + t.encode("hi"), skipSpecialTokens: true), "hi")
        XCTAssertEqual(t.decode([bos] + t.encode("hi")), "<bos>hi")
    }

    /// A special token's string, once escaped, is encoded as the characters it spells; the tokenizer's own
    /// template markup around it still becomes special ids. `<` is byte 0x3C (id 154), `|` 0x7C (218), `>` 0x3E (156).
    func testEscapedSpecialTokensEncodeAsPlainText() throws {
        let t = try makeTokenizer()
        let turn = try XCTUnwrap(t.tokenID(for: "<|turn>"))
        let end = try XCTUnwrap(t.tokenID(for: "<turn|>"))
        let plainEnd = [154, 24, 25, 22, 18, 218, 156]  // < t u r n | >
        XCTAssertEqual(t.encode(templateOutput: t.escapingSpecialTokens(in: "<turn|>")), plainEnd)
        XCTAssertEqual(t.encode("<turn|>"), [end])
        // Surrounded by real markup and by ordinary text.
        let text = "<|turn>user\n" + t.escapingSpecialTokens(in: "the <turn|><|turn>x") + "<turn|>"
        let ids = t.encode(templateOutput: text)
        XCTAssertEqual(ids.first, turn)
        XCTAssertEqual(ids.last, end)
        XCTAssertEqual(ids.filter { $0 == turn }.count, 1)
        XCTAssertEqual(ids.filter { $0 == end }.count, 1)
        XCTAssertEqual(t.decode(ids), "<|turn>user\nthe <turn|><|turn>x<turn|>")
        // Text with no special token in it is unchanged, so an ordinary prompt encodes as before.
        XCTAssertEqual(t.escapingSpecialTokens(in: "the cat"), "the cat")
        for c in try cases() where !c.text.contains("<") {
            XCTAssertEqual(t.encode(templateOutput: t.escapingSpecialTokens(in: c.text)), c.ids, "\(c.text.debugDescription)")
        }
        // A forged marker is dropped, so input cannot smuggle one in.
        XCTAssertEqual(t.escapingSpecialTokens(in: "a\(String(Unicode.Scalar(0xFDD0)!))b"), "ab")
    }

    /// Tokens that differ only in Unicode normalization are different tokens (a `[String: Int]` vocabulary
    /// would merge them).
    func testNormalizationDistinctTokens() throws {
        let t = try makeTokenizer()
        let nfc = t.encode("\u{00E9}")
        let nfd = t.encode("e\u{0301}")
        XCTAssertEqual(nfc.count, 1)
        XCTAssertEqual(nfd.count, 1)
        XCTAssertNotEqual(nfc, nfd)
        XCTAssertEqual(t.decode(nfc).unicodeScalars.map(\.value), [0xE9])
        XCTAssertEqual(t.decode(nfd).unicodeScalars.map(\.value), [0x65, 0x301])
    }

    func testStreamingDecoderHoldsIncompleteCharacters() throws {
        let t = try makeTokenizer()
        let ids = t.encode("a 😀 b 你")
        var d = t.makeStreamingDecoder()
        var pieces: [String] = []
        for id in ids { pieces.append(d.append(id)) }
        pieces.append(d.finish())
        XCTAssertEqual(pieces.joined(), "a 😀 b 你")
        // Every piece is whole characters: re-encoding the pieces one by one never sees a replacement character.
        XCTAssertFalse(pieces.joined().contains("\u{FFFD}"))
        // The emoji is four byte tokens; nothing is released until the fourth.
        let emojiBytes = ids.filter { t.token(for: $0)?.hasPrefix("<0x") == true }
        XCTAssertGreaterThanOrEqual(emojiBytes.count, 4)
        var e = t.makeStreamingDecoder()
        let firstThree = Array(t.encode("😀").prefix(3))
        for id in firstThree { XCTAssertEqual(e.append(id), "") }
        XCTAssertEqual(e.append(t.encode("😀")[3]), "😀")
    }

    func testLongInputIsFast() throws {
        let t = try makeTokenizer()
        let text = String(repeating: "the in and ", count: 20000)
        let start = Date()
        let ids = t.encode(text)
        XCTAssertLessThan(Date().timeIntervalSince(start), 5)
        XCTAssertEqual(t.decode(ids), text)
    }

    func testUnsupportedTokenizersAreRefused() {
        let unigram = Data(#"{"model":{"type":"Unigram","vocab":[]}}"#.utf8)
        XCTAssertThrowsError(try BPETokenizer(tokenizerJSON: unigram))
        let odd = Data(#"{"normalizer":{"type":"NFKC"},"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#.utf8)
        XCTAssertThrowsError(try BPETokenizer(tokenizerJSON: odd))
    }

    /// The real Gemma 4 tokenizer, when a copy is in the local Hugging Face cache (no weights are read).
    func testRealTokenizerMatchesReference() throws {
        guard let dir = Self.findGemma4TokenizerDirectory() else {
            throw XCTSkip("no Gemma 4 tokenizer.json in the Hugging Face cache")
        }
        let t = try BPETokenizer(directory: dir)
        // Ids from Hugging Face `tokenizers` on the same tokenizer.json.
        let samples: [(String, [Int])] = [
            ("<bos><|turn>user\nHello, world!<turn|>\n<|turn>model\n", [2, 105, 2364, 107, 9259, 236764, 1902, 236888, 106, 107, 105, 4368, 107]),
            ("  two  spaces\n\n\nand\ttabs 😀 你好 café", [138, 13498, 138, 35220, 109, 624, 255968, 39218, 163543, 43758, 237389, 33443]),
            ("The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. ", [818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761, 669, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761, 669, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761, 236743]),
            ("def f(x):\n    return x**2  # square\n", [2063, 517, 236769, 236781, 1473, 107, 140, 2060, 1123, 1018, 236778, 138, 236865, 6281, 107]),
            ("é é ﬁ ½ Ω 𝔘𝔫𝔦", [236744, 238288, 1559, 97940, 47041, 76858, 236743, 478, 395, 386, 390, 478, 395, 386, 409, 254865]),
            ("{\"a\": [1, 2.5e3, null]}", [14937, 236746, 1083, 870, 236770, 236764, 236743, 236778, 236761, 236810, 236744, 236800, 236764, 2303, 21111]),
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", [128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 128820, 50354]),
        ]
        for (i, (text, ids)) in samples.enumerated() {
            XCTAssertEqual(t.encode(text), ids, "encode \(text.debugDescription)")
            if i > 0 { XCTAssertEqual(t.decode(ids), text, "decode \(text.debugDescription)") }
        }
        XCTAssertEqual(t.tokenID(for: "<|turn>"), 105)
        XCTAssertEqual(t.tokenID(for: "<turn|>"), 106)
    }

    static func findGemma4TokenizerDirectory() -> URL? {
        let fm = FileManager.default
        if let p = ProcessInfo.processInfo.environment["SHEARS_TEST_TOKENIZER_DIR"] { return URL(fileURLWithPath: p) }
        let hf = ProcessInfo.processInfo.environment["HF_HOME"].map { URL(fileURLWithPath: $0) }
            ?? fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
        let hub = hf.appendingPathComponent("hub")
        guard let models = try? fm.contentsOfDirectory(atPath: hub.path) else { return nil }
        for m in models.sorted() where m.contains("gemma-4") {
            let snaps = hub.appendingPathComponent(m).appendingPathComponent("snapshots")
            for s in (try? fm.contentsOfDirectory(atPath: snaps.path)) ?? [] {
                let d = snaps.appendingPathComponent(s)
                if fm.fileExists(atPath: d.appendingPathComponent("tokenizer.json").path) { return d }
            }
        }
        return nil
    }
}

import Foundation
import MLX
import XCTest

@testable import Shears

final class InferenceJSONTests: ShearsTestCase {
    func accepts(_ s: String) -> (valid: Bool, complete: Bool) {
        var a = JSONObjectAutomaton()
        for b in s.utf8 where !a.accept(b) { return (false, false) }
        return (true, a.isComplete)
    }

    func testAcceptsValidObjectsCharacterByCharacter() {
        let docs = [
            "{}", " {\n}\n", #"{"a": 1}"#, #"{"a":[1,2,{"b":null}],"c":"x\"y\\zé\n","d":-12.5e+3,"e":true,"f":false}"#,
            #"{"": {"": []}}"#, "{\"k\":0}", "{\"k\":0.5}", "{\"k\":-0}", "{\"k\":1E5}", "{\"k\": [ ] , \"j\" : { } }",
        ]
        for d in docs {
            let r = accepts(d)
            XCTAssertTrue(r.valid && r.complete, d)
            XCTAssertNotNil(try? JSONSerialization.jsonObject(with: Data(d.utf8)) as? [String: Any], d)
        }
    }

    func testRejectsInvalidDocuments() {
        let bad = [
            "[]", "1", "\"a\"", "{a:1}", "{'a':1}", #"{"a":01}"#, #"{"a":1,}"#, #"{"a":.5}"#, #"{"a":1.}"#, #"{"a":tru}"#,
            #"{"a" 1}"#, #"{"a":1 "b":2}"#, "{\"a\":\"\n\"}", #"{"a":"\x"}"#, #"{"a":"\u12G4"}"#, #"{"a":[1,]}"#, "{}}", "{} x", #"{"a":+1}"#,
            #"{"a":1e}"#, #"{"a":-}"#,
        ]
        for d in bad {
            let r = accepts(d)
            XCTAssertFalse(r.valid && r.complete, d)
        }
    }

    func testPrefixesAreValidButIncomplete() {
        let doc = #"{"a": [1, 2.5, {"b": "c"}], "d": null}"#
        var a = JSONObjectAutomaton()
        for (i, b) in doc.utf8.enumerated() {
            XCTAssertTrue(a.accept(b), "byte \(i)")
            XCTAssertEqual(a.isComplete, i == doc.utf8.count - 1)
        }
        // `accepts` does not move the state.
        var fresh = JSONObjectAutomaton()
        XCTAssertTrue(fresh.accepts(Array("{\"a\"".utf8)))
        XCTAssertTrue(fresh.accepts(Array("{".utf8)))
        XCTAssertFalse(fresh.accepts(Array("[".utf8)))
    }

    func testGeneratedValuesAreAlwaysAccepted() throws {
        var rng = SplitMix64(seed: 5)
        func value(_ depth: Int) -> Any {
            switch Int.random(in: 0 ..< (depth > 3 ? 5 : 7), using: &rng) {
            case 0: return Int.random(in: -1000 ... 1000, using: &rng)
            case 1: return Double.random(in: -1e6 ... 1e6, using: &rng)
            case 2: return "s\(Int.random(in: 0 ..< 99, using: &rng)) \"q\" \\ é\n"
            case 3: return Bool.random(using: &rng)
            case 4: return NSNull()
            case 5: return (0 ..< Int.random(in: 0 ... 3, using: &rng)).map { _ in value(depth + 1) }
            default: return Dictionary(uniqueKeysWithValues: (0 ..< Int.random(in: 0 ... 3, using: &rng)).map { ("k\($0)", value(depth + 1)) })
            }
        }
        for _ in 0 ..< 200 {
            let obj = ["root": value(0), "n": 1] as [String: Any]
            for opts: JSONSerialization.WritingOptions in [[], [.prettyPrinted], [.sortedKeys]] {
                let data = try JSONSerialization.data(withJSONObject: obj, options: opts)
                let s = String(decoding: data, as: UTF8.self)
                let r = accepts(s)
                XCTAssertTrue(r.valid && r.complete, s)
            }
        }
    }

    /// Generation constrained to a JSON object always produces a prefix of one, and a complete one when it
    /// stops on its own.
    func testConstrainedGenerationProducesJSON() throws {
        let tok = try InferenceFixture.tokenizer()
        let model = try InferenceModel(model: InferenceFixture.model())
        let gen = Generator(model: model, tokenizer: tok)
        gen.reusesPromptCache = false
        var completed = 0
        for seed in 0 ..< 12 {
            let c = JSONObjectConstraint(tokenizer: tok, stopTokenIDs: [InferenceFixture.endOfSequence, InferenceFixture.endOfTurn], logitsVocabularySize: InferenceFixture.vocab)
            c.candidateCount = 64
            let r = try gen.generate(GenerationRequest(
                promptTokens: InferenceFixture.prompt(6, seed: UInt64(seed)), maxTokens: 120,
                sampling: SamplingParameters(temperature: 2, seed: UInt64(seed)),
                stopTokenIDs: [InferenceFixture.endOfSequence, InferenceFixture.endOfTurn], constraint: c))
            let v = accepts(r.text)
            XCTAssertTrue(v.valid, "seed \(seed): \(r.text.debugDescription)")
            if r.finishReason == .stop {
                completed += 1
                XCTAssertTrue(v.complete, "stopped early: \(r.text.debugDescription)")
                XCTAssertNotNil(try JSONSerialization.jsonObject(with: Data(r.text.utf8)) as? [String: Any])
            } else {
                XCTAssertFalse(c.isComplete)
            }
        }
        XCTAssertGreaterThan(completed, 0, "no constrained run finished an object; the constraint may not be steering")
    }
}

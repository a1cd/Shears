import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The vocabulary window: a model that does not hold its embedding table, against the same model that does.
final class VocabularyWindowTests: ShearsTestCase {
    /// An exported toy model (HF layout, float32) with its tokenizer files, and the directories to clean up.
    struct Export {
        var directory: URL
        var tokenizerDirectory: URL
        var tokenizer: BPETokenizer
        func cleanUp() {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(at: tokenizerDirectory)
        }
    }

    func makeExport() throws -> Export {
        let model = try InferenceFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let healer = try HealingFixture.healer(model: model, keep: keep)
        let tokDir = try InferenceFixture.tempDirectory("tok")
        let dir = try InferenceFixture.tempDirectory("exp")
        try InferenceFixture.writeTokenizerFiles(to: tokDir)
        try InferenceFixture.exportDirectory(model: model, keep: keep, adapters: healer.adapters, tokenizerDirectory: tokDir, in: dir)
        return Export(directory: dir, tokenizerDirectory: tokDir, tokenizer: try InferenceFixture.tokenizer())
    }

    func vocabulary(_ ids: [Int], for tokenizer: BPETokenizer) -> StaticVocabulary {
        StaticVocabulary(
            vocabularySize: tokenizer.vocabularySize, tokenizerFingerprint: tokenizer.vocabularyFingerprint(), ids: ids,
            sections: [:])
    }

    /// With every token in the window, the model is the full model: same logits, same greedy text.
    func testAWindowOverTheWholeVocabularyIsTheFullModel() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let full = try InferenceRuntime.load(directory: e.directory, dtype: .float32)
        let windowed = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(Array(0 ..< InferenceFixture.vocab), for: e.tokenizer))
        XCTAssertNotNil(windowed.model.vocabularyWindow)
        // The stand-in table is one row, not 352.
        XCTAssertEqual(windowed.model.model.model.embedTokens.weight.shape, [1, windowed.model.configuration.hiddenSize])

        let tokens = InferenceFixture.prompt(26)
        try windowed.model.prepareVocabulary(promptTokens: tokens)
        let input = MLXArray(tokens.map(Int32.init))[.newAxis]
        assertClose(windowed.model.fullLogits(input), full.model.fullLogits(input), atol: 1e-4, rtol: 1e-4, "full logits")

        let a = try full.generator.generate(GenerationRequest(promptTokens: tokens, maxTokens: 20))
        let b = try windowed.generator.generate(GenerationRequest(promptTokens: tokens, maxTokens: 20))
        XCTAssertEqual(a.tokenIDs, b.tokenIDs)
    }

    /// A small static list: logits are the full model's on the window's tokens and -inf elsewhere, and the prompt's
    /// own tokens are embedded from rows read for the request.
    func testSmallWindowScoresOnlyItsTokensAndEmbedsThePromptFromRows() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let staticIDs = Array(0 ..< 40)
        let full = try InferenceRuntime.load(directory: e.directory, dtype: .float32)
        let windowed = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(staticIDs, for: e.tokenizer))
        let window = try XCTUnwrap(windowed.model.vocabularyWindow)
        XCTAssertEqual(window.rowCount, 40)

        let tokens = InferenceFixture.prompt(30) // ids 6..<100: many are outside the static list
        let outside = Set(tokens).filter { $0 >= 40 }
        XCTAssertFalse(outside.isEmpty)
        try windowed.model.prepareVocabulary(promptTokens: tokens)
        XCTAssertEqual(Set(window.extraIDs), outside)
        XCTAssertEqual(window.rowCount, 40 + outside.count)

        let input = MLXArray(tokens.map(Int32.init))[.newAxis]
        let reference = full.model.fullLogits(input)
        let got = windowed.model.fullLogits(input)
        let inWindow = Set(staticIDs).union(outside)
        let mask = MLXArray((0 ..< InferenceFixture.vocab).map { inWindow.contains($0) })
        // On the window's tokens: equal. Elsewhere: -inf.
        assertClose(MLX.where(mask, got, MLXArray(Float(0))), MLX.where(mask, reference, MLXArray(Float(0))), atol: 1e-4, rtol: 1e-4, "inside")
        let outsideValues = MLX.where(mask, MLXArray(-Float.infinity), got)
        XCTAssertTrue((outsideValues .== MLXArray(-Float.infinity)).all().item(Bool.self))
    }

    /// Greedy generation over the window equals greedy generation over everything with the same allowed set as a
    /// mask, and it never leaves the window.
    func testGenerationEqualsMaskedFullGenerationAndStaysInsideTheWindow() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let staticIDs = Array(0 ..< 30) + [InferenceFixture.endOfSequence, InferenceFixture.endOfTurn]
        let full = try InferenceRuntime.load(directory: e.directory, dtype: .float32)
        let windowed = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(staticIDs, for: e.tokenizer))
        let prompt = InferenceFixture.prompt(18, seed: 11)
        let allowed = Set(staticIDs).union(prompt)

        final class Mask: GenerationConstraint {
            let mask: MLXArray
            init(_ allowed: Set<Int>, vocab: Int) { mask = MLXArray((0 ..< vocab).map { allowed.contains($0) }) }
            func allowedMask(logits: MLXArray) -> MLXArray? { mask }
            func accept(_ token: Int) {}
        }
        let want = try full.generator.generate(
            GenerationRequest(promptTokens: prompt, maxTokens: 30, constraint: Mask(allowed, vocab: InferenceFixture.vocab)))
        let got = try windowed.generator.generate(GenerationRequest(promptTokens: prompt, maxTokens: 30))
        XCTAssertEqual(got.tokenIDs.count, 30)
        XCTAssertEqual(got.tokenIDs, want.tokenIDs)
        XCTAssertTrue(got.tokenIDs.allSatisfy { allowed.contains($0) })

        // Sampling too: only window tokens come out.
        let sampled = try windowed.generator.generate(
            GenerationRequest(promptTokens: prompt, maxTokens: 40, sampling: SamplingParameters(temperature: 1.5, seed: 3)))
        XCTAssertTrue(sampled.tokenIDs.allSatisfy { allowed.contains($0) })
    }

    /// The extra rows belong to one request: the next request's context replaces them.
    func testExtraRowsFollowTheRequest() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let rt = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(Array(0 ..< 20), for: e.tokenizer))
        let w = try XCTUnwrap(rt.model.vocabularyWindow)
        try rt.model.prepareVocabulary(promptTokens: [60, 61, 5, 61])
        XCTAssertEqual(w.extraIDs, [60, 61])
        XCTAssertTrue(w.contains(60) && w.contains(5) && !w.contains(70))
        try rt.model.prepareVocabulary(promptTokens: [70, 5])
        XCTAssertEqual(w.extraIDs, [70])
        XCTAssertFalse(w.contains(60))
        try rt.model.prepareVocabulary(promptTokens: [5, 6])
        XCTAssertEqual(w.extraIDs, [])
        XCTAssertEqual(w.rowCount, 20)
        XCTAssertThrowsError(try rt.model.prepareVocabulary(promptTokens: [9999]))
        XCTAssertThrowsError(try rt.model.prepareVocabulary(promptTokens: [-1]))
    }

    /// Only the rows asked for are read from the table: the window never holds the rest.
    func testTheWindowHoldsOnlyItsRows() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let rt = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(Array(0 ..< 25), for: e.tokenizer))
        let w = try XCTUnwrap(rt.model.vocabularyWindow)
        let hidden = rt.model.configuration.hiddenSize
        XCTAssertEqual(w.residentBytes, 25 * hidden * 4 + InferenceFixture.vocab * 4)
        XCTAssertEqual(w.source.bytesPerRow, hidden * 4)
    }

    /// A list made for another tokenizer is refused.
    func testAListForAnotherTokenizerIsRefused() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        var v = vocabulary([1, 2, 3], for: e.tokenizer)
        v.tokenizerFingerprint = "0000"
        XCTAssertThrowsError(try InferenceRuntime.load(directory: e.directory, vocabulary: v))
        v = vocabulary([1, 2, 3], for: e.tokenizer)
        v.vocabularySize += 1
        XCTAssertThrowsError(try v.validate(for: e.tokenizer))
    }

    /// Scoring a few candidates: the same numbers as the full logits at those tokens, with and without a window.
    func testCandidateScoresEqualFullLogits() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let full = try InferenceRuntime.load(directory: e.directory, dtype: .float32)
        let windowed = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(Array(0 ..< 50), for: e.tokenizer))
        let prompt = InferenceFixture.prompt(22, seed: 4)
        let candidates = [7, 12, 31, 44]
        let cache = full.model.newCache()
        let logits = try full.generator.prefill(prompt, cache: cache)[0]
        let want = candidates.map { logits[$0].item(Float.self) }
        let a = try full.generator.candidateScores(promptTokens: prompt, candidates: candidates)
        let b = try windowed.generator.candidateScores(promptTokens: prompt, candidates: candidates)
        for i in candidates.indices {
            XCTAssertEqual(a[i], want[i], accuracy: 1e-3)
            XCTAssertEqual(b[i], want[i], accuracy: 1e-3)
        }
        XCTAssertThrowsError(try windowed.generator.candidateScores(promptTokens: prompt, candidates: [200]))
    }

    /// A 4-bit checkpoint: the quantized rows are read and scored quantized, and agree with the full quantized model.
    func testQuantizedCheckpointWindow() throws {
        let e = try makeExport()
        let q = try InferenceFixture.tempDirectory("q4")
        defer { e.cleanUp(); try? FileManager.default.removeItem(at: q) }
        try CheckpointQuantizer.write(from: e.directory, to: q, bits: 4, groupSize: 32)

        let full = try InferenceRuntime.load(directory: q)
        XCTAssertTrue(full.model.model.model.embedTokens is QuantizedEmbedding)
        let windowed = try InferenceRuntime.load(directory: q, vocabulary: vocabulary(Array(0 ..< 60), for: e.tokenizer))
        let w = try XCTUnwrap(windowed.model.vocabularyWindow)
        XCTAssertNotNil(w.source.quantization)
        // Quantized rows are a fraction of full-precision ones.
        XCTAssertLessThan(w.source.bytesPerRow, windowed.model.configuration.hiddenSize * 2)

        let tokens = InferenceFixture.prompt(20, seed: 8)
        try windowed.model.prepareVocabulary(promptTokens: tokens)
        let input = MLXArray(tokens.map(Int32.init))[.newAxis]
        let reference = full.model.fullLogits(input)
        let got = windowed.model.fullLogits(input)
        let inWindow = Set(0 ..< 60).union(tokens)
        let mask = MLXArray((0 ..< InferenceFixture.vocab).map { inWindow.contains($0) })
        assertClose(
            MLX.where(mask, got, MLXArray(Float(0))).asType(.float32), MLX.where(mask, reference, MLXArray(Float(0))).asType(.float32),
            atol: 5e-3, rtol: 5e-3, "quantized")
    }

    // MARK: Evaluation

    /// With every token in the list, nothing is ever outside it and both decodes agree.
    func testEvaluationOverTheWholeVocabulary() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let full = try InferenceRuntime.load(directory: e.directory, dtype: .float32)
        let all = vocabulary(Array(0 ..< InferenceFixture.vocab), for: e.tokenizer)
        let prompt = VocabularyEvaluation.Prompt(
            id: "a", messages: [["role": "user", "content": "the cat in the thin hen"]], reference: "the cat")
        var options = VocabularyEvaluation.Options()
        options.maxNewTokens = 12
        let s = try VocabularyEvaluation.run(runtime: full, directory: e.directory, vocabulary: all, prompts: [prompt], options: options)
        XCTAssertEqual(s.prompts, 1)
        XCTAssertEqual(s.greedyStepsOutsideSet, 0)
        XCTAssertEqual(s.greedyStepsOutsideStatic, 0)
        XCTAssertEqual(s.constrainedEqualsUnconstrained, 1)
        XCTAssertEqual(s.meanMassOutsideSet, 0, accuracy: 1e-6)
        XCTAssertEqual(s.referenceOutsideSet, 0)
        XCTAssertNil(full.model.vocabularyWindow, "the reference model is left as it was")
    }

    /// A tiny list leaves things out, and the evaluation counts exactly the greedy tokens outside it.
    func testEvaluationCountsWhatTheWindowLacks() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let full = try InferenceRuntime.load(directory: e.directory, dtype: .float32)
        let tiny = vocabulary([InferenceFixture.endOfSequence, InferenceFixture.endOfTurn, 6, 7], for: e.tokenizer)
        let prompt = VocabularyEvaluation.Prompt(id: "b", messages: [["role": "user", "content": "the cat in the thin hen"]], reference: nil)
        var options = VocabularyEvaluation.Options()
        options.maxNewTokens = 25
        let s = try VocabularyEvaluation.run(runtime: full, directory: e.directory, vocabulary: tiny, prompts: [prompt], options: options)
        let r = s.perPrompt[0]
        XCTAssertEqual(r.stepsOutsideStatic, s.greedyStepsOutsideStatic)
        XCTAssertGreaterThanOrEqual(r.stepsOutsideStatic, r.stepsOutsideSet)
        XCTAssertEqual(r.outsideTokens.count, r.stepsOutsideSet)
        XCTAssertGreaterThan(r.meanMassOutsideStatic, 0)
        XCTAssertLessThanOrEqual(r.chanceReplyLeavesSet, 1)
        XCTAssertGreaterThanOrEqual(r.chanceReplyLeavesStatic, r.chanceReplyLeavesSet - 1e-9)
    }
}

/// The static list and its builder.
final class StaticVocabularyTests: ShearsTestCase {
    func tokenizer() throws -> BPETokenizer { try InferenceFixture.tokenizer() }

    func testFileRoundTripAndValidation() throws {
        let t = try tokenizer()
        let v = StaticVocabulary(
            vocabularySize: t.vocabularySize, tokenizerFingerprint: t.vocabularyFingerprint(), ids: [9, 3, 3, 7],
            sections: ["bytes": 2])
        XCTAssertEqual(v.ids, [3, 7, 9])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vocab-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try v.write(to: url)
        XCTAssertEqual(try StaticVocabulary.load(url), v)
        XCTAssertNoThrow(try v.validate(for: t))
        try Data(#"{"format": "other"}"#.utf8).write(to: url)
        XCTAssertThrowsError(try StaticVocabulary.load(url))
    }

    func testBuilderTakesTheAlwaysIncludedTokensThenTheMostFrequent() throws {
        let t = try tokenizer()
        let b = StaticVocabularyBuilder(tokenizer: t)
        let sections = b.mandatorySections()
        let mandatory = sections.reduce(into: Set<Int>()) { $0.formUnion($1.ids) }
        XCTAssertEqual(Set(t.byteFallbackTokenIDs).count, 256)
        XCTAssertTrue(Set(t.byteFallbackTokenIDs).isSubset(of: mandatory))
        XCTAssertTrue(Set(t.specialTokenIDs).isSubset(of: mandatory))

        // A corpus of one repeated phrase and one rarer one.
        let dir = try InferenceFixture.tempDirectory("corpus")
        defer { try? FileManager.default.removeItem(at: dir) }
        for i in 0 ..< 10 {
            try (String(repeating: "the cat in the thin hen\n", count: 30) + (i == 0 ? "hen\n" : "")).write(
                to: dir.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }
        let files = StaticVocabularyBuilder.corpusFiles(under: [dir])
        XCTAssertEqual(files.count, 10)
        b.addCorpus(files: files)
        XCTAssertEqual(b.corpusFiles, 10)
        XCTAssertGreaterThan(b.heldOutCounts.reduce(0, +), 0, "every tenth file is held out")

        XCTAssertThrowsError(try b.select(size: mandatory.count - 1))
        let ranked = b.ranking(excluding: mandatory)
        XCTAssertGreaterThanOrEqual(ranked.count, 1)
        let extra = min(3, ranked.count)
        let sel = try b.select(size: mandatory.count + 3)
        XCTAssertEqual(sel.vocabulary.ids.count, mandatory.count + extra)
        XCTAssertEqual(sel.shortfall, 3 - extra)
        XCTAssertTrue(mandatory.isSubset(of: Set(sel.vocabulary.ids)))
        // The extra tokens are the best-ranked outside the always-included set, whatever they are.
        XCTAssertEqual(Set(sel.vocabulary.ids).subtracting(mandatory), Set(ranked.prefix(extra)))
        XCTAssertNoThrow(try sel.vocabulary.validate(for: t))

        let coverage = b.coverage(sizes: [mandatory.count, mandatory.count + 40])
        XCTAssertLessThanOrEqual(coverage[0].training, coverage[1].training)
        XCTAssertEqual(try XCTUnwrap(coverage[1].heldOut), 1.0, accuracy: 1e-9, "the held-out text is the same phrase")
    }

    func testDictionaryFillsBeyondTheCorpus() throws {
        let t = try tokenizer()
        let b = StaticVocabularyBuilder(tokenizer: t)
        b.addDictionary(words: ["cat", "hen", "thin"])
        XCTAssertEqual(b.dictionaryWords, 3)
        let base = b.mandatorySections().reduce(into: Set<Int>()) { $0.formUnion($1.ids) }
        let sel = try b.select(size: base.count + 2)
        XCTAssertGreaterThan(b.dictionaryCredit.reduce(0, +), 0)
        // Nothing was counted from a corpus: whatever the dictionary credits fills the list.
        XCTAssertEqual(sel.vocabulary.ids.count, base.count + 2)
        XCTAssertEqual((sel.vocabulary.sections["english"] ?? 0) + (sel.vocabulary.sections["corpus"] ?? 0), 2)
    }
}

extension VocabularyWindowTests {
    /// A constraint that allows nothing the window holds is an error, not token 0.
    func testAConstraintOutsideTheWindowIsAnError() throws {
        let e = try makeExport()
        defer { e.cleanUp() }
        let windowed = try InferenceRuntime.load(
            directory: e.directory, dtype: .float32, vocabulary: vocabulary(Array(0 ..< 30), for: e.tokenizer))
        final class Only: GenerationConstraint {
            func allowedMask(logits: MLXArray) -> MLXArray? { MLXArray((0 ..< InferenceFixture.vocab).map { $0 == 200 }) }
            func accept(_ token: Int) {}
        }
        XCTAssertThrowsError(
            try windowed.generator.generate(GenerationRequest(promptTokens: InferenceFixture.prompt(8), maxTokens: 4, constraint: Only())))
    }
}

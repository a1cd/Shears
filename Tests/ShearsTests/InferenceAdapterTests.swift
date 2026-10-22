import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Generation from a healed model with its adapters left unmerged, against the healer's own forward pass
/// and against the merge-and-slice export.
final class InferenceAdapterTests: ShearsTestCase {
    func fixture(keepCount: Int = 4) throws -> (Gemma4TextModel, [[Int]], PruneHealer) {
        let model = try InferenceFixture.model()
        let keep = HealingFixture.keepSets(count: keepCount)
        let healer = try HealingFixture.healer(model: model, keep: keep)
        HealingFixture.randomizeAdapters(healer.adapters, scale: 0.1)
        return (model, keep, healer)
    }

    /// Anchor: the unmerged, uncached path is the training-time student's forward pass.
    func testUnmergedLogitsEqualTheHealersStudent() throws {
        let (model, _, healer) = try fixture()
        let inference = try InferenceModel(model: model, overlay: AdapterOverlay(healer: healer))
        let tokens = MLXArray(InferenceFixture.prompt(26).map(Int32.init))[.newAxis]
        assertClose(inference.fullLogits(tokens), healer.logits(tokens: tokens, student: true), atol: 1e-4, rtol: 1e-4, "student")
        // Not vacuous: the adapters and the pruning both move the output a lot.
        let teacher = healer.logits(tokens: tokens, student: false)
        XCTAssertGreaterThan(abs(inference.fullLogits(tokens) - teacher).max().item(Float.self), 0.05)
        let bare = try InferenceModel(model: model)
        XCTAssertGreaterThan(abs(inference.fullLogits(tokens) - bare.fullLogits(tokens)).max().item(Float.self), 0.05)
    }

    /// Unmerged adapters over a KV cache: every step equals the uncached student.
    func testUnmergedCachedDecodeEqualsUncached() throws {
        let (model, _, healer) = try fixture()
        let inference = try InferenceModel(model: model, overlay: AdapterOverlay(healer: healer))
        let tokens = InferenceFixture.prompt(30)
        let full = inference.fullLogits(MLXArray(tokens.map(Int32.init))[.newAxis])
        let cache = inference.newCache()
        var out = inference.nextLogits(MLXArray(tokens[0 ..< 10].map(Int32.init))[.newAxis], cache: cache)
        assertClose(out[0], full[0, 9], atol: 2e-3, rtol: 2e-3, "prefill")
        for i in 10 ..< tokens.count {
            out = inference.nextLogits(MLXArray([Int32(tokens[i])])[.newAxis], cache: cache)
            assertClose(out[0], full[0, i], atol: 2e-3, rtol: 2e-3, "position \(i)")
        }
    }

    /// Generation with unmerged adapters equals generation from the merged, sliced export.
    func testUnmergedGenerationEqualsMergedExportGeneration() throws {
        let (model, keep, healer) = try fixture()
        let tokDir = try InferenceFixture.tempDirectory("tok")
        let dir = try InferenceFixture.tempDirectory("exp")
        defer { try? FileManager.default.removeItem(at: tokDir); try? FileManager.default.removeItem(at: dir) }
        try InferenceFixture.writeTokenizerFiles(to: tokDir)
        try InferenceFixture.exportDirectory(model: model, keep: keep, adapters: healer.adapters, tokenizerDirectory: tokDir, in: dir)

        let unmerged = Generator(
            model: try InferenceModel(model: model, overlay: AdapterOverlay(healer: healer)), tokenizer: try InferenceFixture.tokenizer())
        let merged = try InferenceRuntime.load(directory: dir, dtype: .float32)
        let prompt = InferenceFixture.prompt(12)
        let a = try unmerged.generate(GenerationRequest(promptTokens: prompt, maxTokens: 24))
        let b = try merged.generator.generate(GenerationRequest(promptTokens: prompt, maxTokens: 24))
        XCTAssertEqual(a.tokenIDs.count, 24)
        // Teacher-forced on the unmerged path's own tokens, the two models' logits agree at every step (the
        // export is exact to float32 rounding, so a token may differ only where the top two are that close).
        var seq = prompt
        for (i, t) in a.tokenIDs.enumerated() {
            let x = unmerged.model.fullLogits(MLXArray(seq.map(Int32.init))[.newAxis])[0, seq.count - 1]
            let y = merged.model.fullLogits(MLXArray(seq.map(Int32.init))[.newAxis])[0, seq.count - 1]
            assertClose(x, y, atol: 3e-3, rtol: 3e-3, "step \(i)")
            seq.append(t)
        }
        XCTAssertEqual(a.tokenIDs, b.tokenIDs, "unmerged and merged-export greedy generations differ")
        // The model is really different from the unadapted, unpruned one.
        let plain = Generator(model: try InferenceModel(model: model), tokenizer: try InferenceFixture.tokenizer())
        XCTAssertNotEqual(try plain.generate(GenerationRequest(promptTokens: prompt, maxTokens: 24)).tokenIDs, a.tokenIDs)
    }

    /// A model already cut to its kept experts, with the same adapters, behaves as the masked one.
    func testSlicedStudentEqualsMaskedStudent() throws {
        let (model, keep, healer) = try fixture()
        let masked = try InferenceModel(model: model, overlay: AdapterOverlay(healer: healer))

        let cut = try InferenceFixture.model()
        for (l, k) in zip(cut.layers, keep) {
            sliceToKeptExperts(l as! Gemma4MoEDecoderLayer, set: ExpertSet(keep: k, numExperts: HealingFixture.experts))
        }
        // Same adapters and kept sets (the adapters keep one format whatever the model's form), layers cut.
        let overlay = AdapterOverlay(adapters: healer.adapters, sets: healer.network.sets, sliced: true)
        let sliced = try InferenceModel(model: cut, overlay: overlay)
        let tokens = MLXArray(InferenceFixture.prompt(20).map(Int32.init))[.newAxis]
        assertClose(sliced.fullLogits(tokens), masked.fullLogits(tokens), atol: 2e-3, rtol: 2e-3, "sliced vs masked")

        // And generation over the cache agrees.
        let g1 = Generator(model: masked, tokenizer: try InferenceFixture.tokenizer())
        let g2 = Generator(model: sliced, tokenizer: try InferenceFixture.tokenizer())
        let prompt = InferenceFixture.prompt(11)
        XCTAssertEqual(
            try g1.generate(GenerationRequest(promptTokens: prompt, maxTokens: 16)).tokenIDs,
            try g2.generate(GenerationRequest(promptTokens: prompt, maxTokens: 16)).tokenIDs)
    }

    func testOverlayMustCoverEveryLayer() throws {
        let (model, _, healer) = try fixture()
        var overlay = AdapterOverlay(healer: healer)
        overlay.sets.removeLast()
        XCTAssertThrowsError(try InferenceModel(model: model, overlay: overlay))
    }
}

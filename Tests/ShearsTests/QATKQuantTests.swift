import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Healing against the Q4_K_M mix: the student's forward pass uses the weights a Q4_K_M GGUF of the
/// healed model will hold.
final class QATKQuantTests: ShearsTestCase {
    func tempDir(_ name: String) throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    /// 6 layers, hidden 256, 16 experts (8 will be kept), every tensor a K-quant can hold except the
    /// two `ffn_down` kinds whose rows are 384 and 128 wide.
    func model() throws -> (Gemma4TextModel, KQuantFixture) {
        var fx = KQuantFixture()
        fx.layers = 6
        fx.experts = 16
        let text = fx.config()["text_config"] as! [String: Any]
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: text))
        try c.validate()
        let m = Gemma4TextModel(c)
        MLXRandom.seed(11)
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.05 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        HealingFixture.makeBF16Exact(m)
        return (m, fx)
    }

    func testQ4KMPlanMatchesTheGGUFPlan() throws {
        let (m, _) = try model()
        let qat = QATConfiguration.q4KM(for: m, keptExperts: 8)
        // 6 layers: ffn_down gets more bits (Q6_K) in layers 2 and 5; the rows (384, 128) are not
        // whole Q4_K / Q6_K super-blocks, so those fall back to Q5_0 / Q8_0. attn_output is Q5_K
        // with exactly 8 experts.
        XCTAssertEqual(qat.formats["blk.0.attn_output.weight"], .gguf(.q5_k))
        XCTAssertEqual(qat.formats["blk.1.ffn_gate.weight"], .gguf(.q4_k))
        XCTAssertEqual(qat.formats["blk.1.ffn_down.weight"], .gguf(.q5_0))  // rows of 384 do not hold Q4_K
        XCTAssertEqual(qat.formats["blk.2.ffn_down.weight"], .q8_0)  // Q6_K (more bits), same rows: Q8_0
        XCTAssertEqual(qat.formats["blk.2.ffn_gate_up_exps.weight"], .gguf(.q4_k))
        XCTAssertEqual(qat.formats["blk.1.ffn_down_exps.weight"], .gguf(.q5_0))
        XCTAssertEqual(qat.formats["blk.5.ffn_down_exps.weight"], .q8_0)
    }

    func testExportedQ4KMReproducesTheHealedForward() throws {
        let (m, fx) = try model()
        let keep = (0 ..< fx.layers).map { l in Array(0 ..< 16).filter { ($0 + l) % 2 == 0 } }
        let qat = QATConfiguration.q4KM(for: m, keptExperts: 8)
        let ad = AdapterConfiguration(expertRank: 4, denseRank: 4, routerRank: 2, qat: qat)
        let h = try PruneHealer(model: m, keep: keep, adapterConfiguration: ad, configuration: HealingConfiguration())
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.02)
        let tokens = MLXArray((0 ..< 24).map { Int32($0 * 11 % 200) })[.newAxis]
        let healed = h.logits(tokens: tokens, student: true)

        // The rounding is real: the same adapters without QAT give different logits.
        let plain = try PruneHealer(
            model: m, keep: keep, adapterConfiguration: AdapterConfiguration(expertRank: 4, denseRank: 4, routerRank: 2),
            configuration: HealingConfiguration(), initialAdapters: HealingFixture.copyTensors(h.adapters.tensors()))
        XCTAssertGreaterThan(abs(plain.logits(tokens: tokens, student: true) - healed).max().item(Float.self), 1e-3)

        // Gradients reach every adapter family through the rounding.
        let row = HealingFixture.row(length: 24, seed: 3, key: "a")
        let step = h.stepFor(.freerun, row, step: 0)
        XCTAssertTrue(step.loss.isFinite)
        for family in ["mlp_gate", "mlp_up", "mlp_down", "attn_out", "experts.gate_a", "experts.down_b", "router.rt_a"] {
            let g = step.grads.filter { $0.key.contains(family) }
            XCTAssertFalse(g.isEmpty, family)
            XCTAssertGreaterThan(g.values.map { abs($0).max().item(Float.self) }.max() ?? 0, 0, "no gradient for \(family)")
        }

        // Export the merged model with the DEFAULT dtype (bfloat16), make the Q4_K_M GGUF, read the
        // covered tensors back. The router correction is left out of this half: the export rounds
        // the merged router to bfloat16, which healing (rightly) does not model.
        let ad2 = AdapterConfiguration(expertRank: 4, denseRank: 4, routerRank: 0, qat: qat)
        let h2 = try PruneHealer(model: m, keep: keep, adapterConfiguration: ad2, configuration: HealingConfiguration())
        HealingFixture.randomizeAdapters(h2.adapters, scale: 0.02)
        let healed2 = h2.logits(tokens: tokens, student: true)

        let source = try tempDir("src")
        try fx.write(to: source)  // config and tokenizer only are used
        let hf = try tempDir("hf")
        var o = MergeAndSliceExport.Options()
        XCTAssertEqual(o.dtype, .bfloat16, "the default export dtype is what is tested")
        o.sourceDirectory = source
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: h2.adapters, to: hf, options: o)
        let gguf = hf.appendingPathComponent("q4km.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: hf, output: gguf, type: .q4_k_m)
        XCTAssertNotNil(summary.tensorsByType["Q4_K"])
        XCTAssertNotNil(summary.tensorsByType["Q5_K"])
        let reader = try GGUFReader(url: gguf)

        var tensors: [String: MLXArray] = [:]
        var replaced = 0
        for file in try FileManager.default.contentsOfDirectory(at: hf, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            for (k, v) in try loadArrays(url: file) {
                var value = v
                if case .mapped(let g) = GGUFGemma4Name.map(k),
                    ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "attn_output.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight"]
                        .contains(where: { g.hasSuffix($0) }),
                    reader.tensorInfo(g) != nil
                {
                    value = MLXArray(try reader.floats(g), v.shape)
                    replaced += 1
                }
                tensors[k] = value
            }
        }
        XCTAssertEqual(replaced, fx.layers * 6)
        let shippedDir = try tempDir("shipped")
        try save(arrays: tensors, url: shippedDir.appendingPathComponent("model.safetensors"))
        try FileManager.default.copyItem(at: hf.appendingPathComponent("config.json"), to: shippedDir.appendingPathComponent("config.json"))
        let shipped = try Gemma4Checkpoint(directory: shippedDir).loadModel(dtype: .float32)
        shipped.train(false)
        let gap = abs(shipped(tokens) - healed2).max().item(Float.self)
        print("QATKQ max |shipped - healed| logits: \(gap)")
        assertClose(shipped(tokens), healed2, atol: 1e-4, rtol: 1e-4, "forward of the Q4_K_M weights vs the QAT student")
    }
}

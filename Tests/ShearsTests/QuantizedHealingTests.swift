import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A tiny Gemma 4 MoE quantized with `MLX.quantized` and saved in the mlx-community layout
/// (`language_model.model.*` names; `weight`/`scales`/`biases` per quantized module; a config
/// `quantization` entry with per-module overrides: 8-bit dense MLPs and routers, 4-bit the rest,
/// as in the published 26B-A4B QAT 4-bit conversion).
enum QuantizedFixture {
    static let groupSize = 32

    static func bits(for key: String) -> Int? {
        if key.hasSuffix(".mlp.gate_proj.weight") || key.hasSuffix(".mlp.up_proj.weight")
            || key.hasSuffix(".mlp.down_proj.weight") || key.hasSuffix(".router.proj.weight")
        { return 8 }
        if key.hasSuffix("_proj.weight") && (key.contains("self_attn.") || key.contains("switch_glu."))
        { return 4 }
        if key == "model.embed_tokens.weight" { return 4 }
        return nil
    }

    /// Writes the quantized checkpoint of `model` to `dir` and returns its dequantized twin
    /// (an fp32 model with exactly the weights the checkpoint represents).
    @discardableResult
    static func write(_ model: Gemma4TextModel, to dir: URL, scalesDType: DType = .float32) throws -> Gemma4TextModel {
        var tensors: [String: MLXArray] = [:]
        var twin: [(String, MLXArray)] = []
        var overrides: [String: Any] = [:]
        for (key, w) in model.parameters().flattened() {
            let name = "language_model." + key
            guard let b = bits(for: key) else {
                tensors[name] = w
                twin.append((key, w))
                continue
            }
            let (wq, s, bi) = MLX.quantized(w, groupSize: groupSize, bits: b)
            let s2 = s.asType(scalesDType), bi2 = bi!.asType(scalesDType)
            let base = String(name.dropLast(".weight".count))
            tensors[name] = wq
            tensors[base + ".scales"] = s2
            tensors[base + ".biases"] = bi2
            twin.append((key, dequantized(wq, scales: s2, biases: bi2, groupSize: groupSize, bits: b).asType(.float32)))
            if b == 8 { overrides[base] = ["group_size": groupSize, "bits": 8] }
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try save(arrays: tensors, url: dir.appendingPathComponent("model.safetensors"))
        var quant: [String: Any] = ["group_size": groupSize, "bits": 4, "mode": "affine"]
        quant.merge(overrides) { $1 }
        let root: [String: Any] = [
            "model_type": "gemma4", "text_config": HealingFixture.configJSON(), "quantization": quant,
        ]
        try JSONSerialization.data(withJSONObject: root).write(to: dir.appendingPathComponent("config.json"))
        let t = Gemma4TextModel(try HealingFixture.config())
        t.update(parameters: ModuleParameters.unflattened(twin))
        t.train(false)
        eval(t)
        return t
    }
}

final class QuantizedHealingTests: ShearsTestCase {
    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("qheal-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func loaded(dtype: DType? = nil) throws -> (quantized: Gemma4TextModel, twin: Gemma4TextModel, dir: URL) {
        let dir = try tempDir()
        let twin = try QuantizedFixture.write(try HealingFixture.model(), to: dir)
        let q = try Gemma4Checkpoint(directory: dir).loadModel(dtype: dtype)
        q.train(false)
        return (q, twin, dir)
    }

    func testConfigAndKeyNamesMatchThePublishedLayout() throws {
        // Names as in mlx-community/gemma-4-26B-A4B-it-qat-4bit's index and config.
        for (src, key) in [
            ("language_model.model.layers.0.experts.switch_glu.gate_proj.scales", "model.layers.0.experts.switch_glu.gate_proj.scales"),
            ("language_model.model.layers.0.router.proj.biases", "model.layers.0.router.proj.biases"),
            ("language_model.model.embed_tokens.scales", "model.embed_tokens.scales"),
        ] { XCTAssertEqual(Gemma4Checkpoint.moduleKey(forSourceName: src), key) }
        XCTAssertNil(Gemma4Checkpoint.moduleKey(forSourceName: "vision_tower.encoder.layers.5.self_attn.v_proj.linear.weight"))
        let dir = try tempDir()
        let cfg: [String: Any] = [
            "model_type": "gemma4", "text_config": HealingFixture.configJSON(),
            "quantization": [
                "group_size": 64, "bits": 4, "mode": "affine",
                "language_model.model.layers.0.mlp.gate_proj": ["group_size": 64, "bits": 8],
                "language_model.model.layers.1.router.proj": ["group_size": 32, "bits": 8],
            ] as [String: Any],
        ]
        let url = dir.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: cfg).write(to: url)
        let q = try XCTUnwrap(Gemma4Quantization.load(from: url))
        XCTAssertEqual(q.parameters(for: "model.layers.7.mlp.gate_proj"), .init(groupSize: 64, bits: 4, mode: .affine))
        XCTAssertEqual(q.parameters(for: "model.layers.0.mlp.gate_proj").bits, 8)
        XCTAssertEqual(q.parameters(for: "model.layers.1.router.proj"), .init(groupSize: 32, bits: 8, mode: .affine))
    }

    func testQuantizedCheckpointLoadsAndMatchesItsDequantizedTwin() throws {
        let (q, twin, _) = try loaded()
        let l = q.layers[1] as! Gemma4MoEDecoderLayer
        XCTAssertEqual((l.selfAttn.qProj as? QuantizedLinear)?.bits, 4)
        XCTAssertEqual((l.mlp.gateProj as? QuantizedLinear)?.bits, 8)
        XCTAssertEqual((l.router.proj as? QuantizedLinear)?.bits, 8)
        XCTAssertTrue(l.experts.switchGLU.upProj is QuantizedSwitchLinear)
        XCTAssertTrue(q.model.embedTokens is QuantizedEmbedding)
        XCTAssertEqual(l.selfAttn.oProj.weight.dtype, .uint32)
        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        assertClose(q(t), twin(t), atol: 2e-3, rtol: 2e-3, "quantized model vs dequantized twin")
        // And it really is quantized: it differs from the original fp model by more than that.
        let original = try HealingFixture.model()
        XCTAssertGreaterThan(abs(q(t) - original(t)).max().item(Float.self), 20 * abs(q(t) - twin(t)).max().item(Float.self))
    }

    func testDTypeCastNeverTouchesPackedIntegers() throws {
        let (q, twin, _) = try loaded(dtype: .bfloat16)
        let l = q.layers[0] as! Gemma4MoEDecoderLayer
        XCTAssertEqual(l.selfAttn.qProj.weight.dtype, .uint32)
        XCTAssertEqual(l.router.perExpertScale.dtype, .bfloat16)
        XCTAssertEqual((l.mlp.gateProj as! QuantizedLinear).scales.dtype, .bfloat16)
        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        // One layer in bf16 stays close to the same layer in fp32 (whole random models amplify
        // bf16 rounding too much to compare end to end).
        let x = twin.embed(t)
        let want = twin.layers[0](x)
        let got = (q.layers[0] as! Gemma4MoEDecoderLayer)(x.asType(.bfloat16))
        XCTAssertTrue(isFinite(got.asType(.float32)).all().item(Bool.self))
        XCTAssertLessThan((abs(got.asType(.float32) - want).mean() / abs(want).mean()).item(Float.self), 0.1)
    }

    func testQuantizedModelHealsWithAdaptersSizedFromLogicalWidths() throws {
        let (q, _, _) = try loaded()
        let h = try HealingFixture.healer(model: q, keep: HealingFixture.keepSets(count: 4))
        // o_proj input is heads * head_dim (64 on sliding layers, 128 on global), not the packed width.
        let a0 = h.adapters.layers[0].attentionOutput!.a, a1 = h.adapters.layers[1].attentionOutput!.a
        XCTAssertEqual(a0.dim(0), 64)
        XCTAssertEqual(a1.dim(0), 128)
        XCTAssertEqual(h.adapters.layers[0].mlpDown!.a.dim(0), 96)

        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        let all = try HealingFixture.healer(model: q, keep: Array(repeating: Array(0 ..< 8), count: 4))
        XCTAssertEqual(abs(all.logits(tokens: t, student: true) - all.logits(tokens: t, student: false)).max().item(Float.self), 0)
        assertClose(h.logits(tokens: t, student: false), q(t), atol: 1e-4)

        for phase in [HealingPhase.block, .end] {
            let hh = try HealingFixture.healer(model: q, keep: HealingFixture.keepSets(count: 4))
            let row = HealingFixture.row()
            let before = hh.probeLoss(phase, row: row)
            try hh.run(rows: ShuffledRows(rows: [row]), plan: PhasePlan(entries: [(phase, 0)], total: 30))
            XCTAssertLessThan(hh.probeLoss(phase, row: row), before * 0.8, "\(phase)")
        }
    }

    func testExportDequantizesEveryTensorAndMatchesTheMaskedModel() throws {
        let (q, _, _) = try loaded()
        let keep = HealingFixture.keepSets(count: 4)
        let h = try HealingFixture.healer(model: q, keep: keep)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        let masked = h.logits(tokens: t, student: true)

        // fp32: the sliced, merged model reproduces the masked + adapters quantized model.
        let dir32 = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: q, keep: keep, adapters: h.adapters, to: dir32, options: o)
        let reloaded = try Gemma4Checkpoint(directory: dir32).loadModel(dtype: .float32)
        reloaded.train(false)
        assertClose(masked, reloaded(t), atol: 3e-3, rtol: 3e-3, "export vs masked+adapters")

        // bf16 + GGUF: no quantization remnants anywhere, and the GGUF writer accepts it.
        let src = try tempDir()
        try GGUFTinyGemma4.writeTokenizer(to: src)
        let dir = try tempDir()
        o.dtype = .bfloat16
        o.sourceDirectory = src
        try MergeAndSliceExport.write(model: q, keep: keep, adapters: h.adapters, to: dir, options: o)
        let idx = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
        let map = idx["weight_map"] as! [String: String]
        XCTAssertFalse(map.keys.contains { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") })
        var seen = Set<String>()
        for f in Set(map.values) {
            for (k, v) in try loadArrays(url: dir.appendingPathComponent(f)) {
                XCTAssertEqual(v.dtype, .bfloat16, k)
                seen.insert(k)
            }
        }
        XCTAssertEqual(seen, Set(map.keys))
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: dir.appendingPathComponent("p.gguf"), type: .bf16)
        XCTAssertEqual(summary.expertCount, 4)
        XCTAssertEqual(summary.warnings, [])
        let bf = try Gemma4Checkpoint(directory: dir).loadModel(dtype: .float32)
        bf.train(false)
        let agree = (bf(t).argMax(axis: -1) .== masked.argMax(axis: -1)).asType(.float32).mean().item(Float.self)
        XCTAssertGreaterThan(agree, 0.8, "bf16 export strays from the masked model")
    }

    func testStreamedBlockPhaseReadsQuantizedLayers() throws {
        let (q, _, dir) = try loaded()
        let keep = HealingFixture.keepSets(count: 4)
        let seed = try HealingFixture.healer(model: q, keep: keep)
        let initial = HealingFixture.copyTensors(seed.adapters.tensors())
        let rows = [HealingFixture.row(seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        let source = ShuffledRows(rows: rows, seed: 9)
        var c = HealingFixture.tinyConfiguration()
        let main = try HealingFixture.healer(model: q, keep: keep, configuration: c, initial: initial)
        try main.run(rows: source, plan: PhasePlan(entries: [(.block, 0)], total: 5))
        c.checkpointBlocks = false
        let cp = try Gemma4Checkpoint(directory: dir)
        let s = StreamedBlockHealer(
            source: Gemma4LayerSource(checkpoint: cp, dtype: nil), keep: keep, blockEnds: q.defaultBlockEnds,
            adapterConfiguration: HealingFixture.smallAdapters, configuration: c, initialAdapters: initial)
        let store = InMemoryBoundaryStore()
        try s.precomputeBoundaries(rows: rows, store: store)
        var steps = 0
        let report = try s.train(rows: source, steps: 5, store: store, onStep: { _, r in
            steps += 1
            XCTAssertTrue(r.applied)
        })
        XCTAssertEqual(steps, 2 * 5)
        var worst: Float = 0, moved: Float = 0
        for (k, v) in main.adapters.tensors() {
            worst = max(worst, abs(v - report.adapters[k]!).max().item(Float.self))
            moved = max(moved, abs(v - initial[k]!).max().item(Float.self))
        }
        XCTAssertGreaterThan(moved, 0.01)
        XCTAssertLessThan(worst, 2e-3)
    }

    // MARK: loader strictness

    func editConfig(_ dir: URL, _ edit: (inout [String: Any]) -> Void) throws {
        let url = dir.appendingPathComponent("config.json")
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        edit(&json)
        try JSONSerialization.data(withJSONObject: json).write(to: url)
    }

    func testGroupSizeAndBitsAreInferredWhenTheConfigHasNoQuantizationEntry() throws {
        let (_, twin, dir) = try loaded()
        try editConfig(dir) { $0["quantization"] = nil }
        let q = try Gemma4Checkpoint(directory: dir).loadModel()
        let l = q.layers[1] as! Gemma4MoEDecoderLayer
        XCTAssertEqual((l.selfAttn.qProj as? QuantizedLinear)?.bits, 4)
        XCTAssertEqual((l.selfAttn.qProj as? QuantizedLinear)?.groupSize, 32, "not an assumed 64")
        XCTAssertEqual((l.mlp.gateProj as? QuantizedLinear)?.bits, 8)
        q.train(false)
        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        assertClose(q(t), twin(t), atol: 2e-3, rtol: 2e-3, "inferred quantization loads the same model")
        // Layer by layer and the embedding table too.
        let layer = try Gemma4Checkpoint(directory: dir).loadDecoderLayer(1) as! Gemma4MoEDecoderLayer
        XCTAssertEqual((layer.selfAttn.qProj as? QuantizedLinear)?.groupSize, 32)
        XCTAssertEqual((try Gemma4Checkpoint(directory: dir).loadEmbedding() as? QuantizedEmbedding)?.groupSize, 32)
    }

    func testAConfigThatDisagreesWithTheShapesIsRefused() throws {
        let (_, _, dir) = try loaded()
        try editConfig(dir) { $0["quantization"] = ["group_size": 64, "bits": 4, "mode": "affine"] }
        XCTAssertThrowsError(try Gemma4Checkpoint(directory: dir).loadModel()) { XCTAssertTrue("\($0)".contains("group size"), "\($0)") }
    }

    func testShapesThatFitNoAffineQuantizationAreRefused() throws {
        let (_, _, dir) = try loaded()
        try editConfig(dir) { $0["quantization"] = nil }
        // Give one module twice the scale columns it should have: the group would be 16.
        let file = dir.appendingPathComponent("model.safetensors")
        var t = try loadArrays(url: file)
        let key = "language_model.model.layers.0.self_attn.q_proj.scales"
        t[key] = concatenated([t[key]!, t[key]!], axis: -1)
        eval(Array(t.values))  // loaded lazily from the file about to be overwritten
        try save(arrays: t, url: file)
        XCTAssertThrowsError(try Gemma4Checkpoint(directory: dir).loadDecoderLayer(0)) { XCTAssertTrue("\($0)".contains("not a supported"), "\($0)") }
    }

    func testLoadModelKeepingChecksTheGlobalsAsStrictlyAsLoadModel() throws {
        let keep = HealingFixture.keepSets(count: 4)
        // A missing global (removed from the shard and the files it is named in).
        let dir = try tempDir()
        try QuantizedFixture.write(try HealingFixture.model(), to: dir)
        let file = dir.appendingPathComponent("model.safetensors")
        var t = try loadArrays(url: file)
        t["language_model.model.norm.weight"] = nil
        eval(Array(t.values))  // loaded lazily from the file about to be overwritten
        try save(arrays: t, url: file)
        XCTAssertThrowsError(try Gemma4Checkpoint(directory: dir).loadModel()) { _ in }
        XCTAssertThrowsError(try Gemma4Checkpoint(directory: dir).loadModel(keeping: keep)) { e in
            XCTAssertTrue("\(e)".contains("norm.weight"), "\(e)")
        }
        // A global of the wrong shape.
        let dir2 = try tempDir()
        try QuantizedFixture.write(try HealingFixture.model(), to: dir2)
        let file2 = dir2.appendingPathComponent("model.safetensors")
        var t2 = try loadArrays(url: file2)
        t2["language_model.model.norm.weight"] = MLXArray.ones([65])
        eval(Array(t2.values))
        try save(arrays: t2, url: file2)
        XCTAssertThrowsError(try Gemma4Checkpoint(directory: dir2).loadModel(keeping: keep))
        // The intact checkpoint still loads.
        let dir3 = try tempDir()
        try QuantizedFixture.write(try HealingFixture.model(), to: dir3)
        XCTAssertNoThrow(try Gemma4Checkpoint(directory: dir3).loadModel(keeping: keep))
    }
}

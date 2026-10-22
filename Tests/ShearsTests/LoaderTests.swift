import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Writes tiny synthetic Qwen3-MoE checkpoints in the Hugging Face and mlx-lm layouts
/// and checks they map onto ``MoEBlock`` with the same behaviour as the source weights.
final class LoaderTests: ShearsTestCase {
    let d = 64, h = 128, e = 4, k = 2

    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory
            .appendingPathComponent("shears-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func writeConfig(_ dir: URL, extra: [String: Any] = [:]) throws {
        var c: [String: Any] = [
            "hidden_size": d, "moe_intermediate_size": h, "num_experts": e,
            "num_experts_per_tok": k, "norm_topk_prob": true, "num_hidden_layers": 2,
            "rms_norm_eps": 1e-6, "decoder_sparse_step": 1, "mlp_only_layers": [Int](),
        ]
        c.merge(extra) { $1 }
        try JSONSerialization.data(withJSONObject: c).write(to: dir.appendingPathComponent("config.json"))
    }

    func testHuggingFaceLayoutWithShardIndex() throws {
        let dir = try tempDir()
        try writeConfig(dir)
        let source = MoEResidualLayer(dims: d, hiddenDims: h, numExperts: e, topK: k)
        eval(source)

        let p = "model.layers.1."
        var shardA: [String: MLXArray] = [
            p + "mlp.gate.weight": source.moe.router.gateWeight,
            p + "post_attention_layernorm.weight": source.norm.weight,
        ]
        var shardB: [String: MLXArray] = [:]
        for i in 0 ..< e {
            let target = i < 2 ? { (k: String, v: MLXArray) in shardA[k] = v } : { shardB[$0] = $1 }
            target(p + "mlp.experts.\(i).gate_proj.weight", source.moe.experts.gateProj.weight[i])
            target(p + "mlp.experts.\(i).up_proj.weight", source.moe.experts.upProj.weight[i])
            target(p + "mlp.experts.\(i).down_proj.weight", source.moe.experts.downProj.weight[i])
        }
        try save(arrays: shardA, url: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try save(arrays: shardB, url: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        var map: [String: String] = [:]
        for key in shardA.keys { map[key] = "model-00001-of-00002.safetensors" }
        for key in shardB.keys { map[key] = "model-00002-of-00002.safetensors" }
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let ckpt = try Qwen3MoECheckpoint(directory: dir)
        XCTAssertTrue(ckpt.configuration.isMoELayer(1))
        let loaded = try ckpt.loadResidualLayer(1)
        XCTAssertEqual(loaded.moe.experts.gateProj.weight.shape, [e, h, d])
        XCTAssertEqual(loaded.moe.router.topK, k)

        let x = MLXRandom.normal([1, 16, d])
        assertClose(loaded(x), source(x))
        XCTAssertThrowsError(try ckpt.loadMoEBlock(layer: 0))
    }

    func testMLXQuantizedLayoutWithPerModuleOverride() throws {
        let dir = try tempDir()
        try writeConfig(dir, extra: [
            "quantization": [
                "group_size": 64, "bits": 4,
                "model.layers.0.mlp.gate": ["group_size": 64, "bits": 8],
            ]
        ])
        let source = MoEBlock(dims: d, hiddenDims: h, numExperts: e, topK: k)
        eval(source)
        let p = "model.layers.0.mlp."
        var arrays: [String: MLXArray] = [:]
        let (gq, gs, gb) = MLX.quantized(source.router.gateWeight, groupSize: 64, bits: 8)
        arrays[p + "gate.weight"] = gq
        arrays[p + "gate.scales"] = gs
        arrays[p + "gate.biases"] = gb!
        for (name, proj) in [
            ("gate_proj", source.experts.gateProj), ("up_proj", source.experts.upProj),
            ("down_proj", source.experts.downProj),
        ] {
            let (w, s, b) = MLX.quantized(proj.weight, groupSize: 64, bits: 4)
            arrays[p + "switch_mlp.\(name).weight"] = w
            arrays[p + "switch_mlp.\(name).scales"] = s
            arrays[p + "switch_mlp.\(name).biases"] = b!
        }
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let block = try Qwen3MoECheckpoint(directory: dir).loadMoEBlock(layer: 0)
        XCTAssertTrue(block.experts.gateProj is QuantizedSwitchLinear)
        XCTAssertEqual(block.router.gateWeight.shape, [e, d], "router is dequantized")
        XCTAssertEqual(block.router.gateWeight.dtype, .float32)
        // The base is frozen; only the router is trainable after loading.
        MoETraining.trainRouterAndExperts(block)
        XCTAssertEqual(Set(block.trainableParameters().flattened().map(\.0)), ["gate.weight"])

        let x = MLXRandom.normal([1, 32, d])
        // 4-bit experts and an 8-bit router: close to the float source, not equal.
        let rel = (abs(block(x) - source(x)).mean() / abs(source(x)).mean()).item(Float.self)
        XCTAssertLessThan(rel, 0.25)
    }

    /// mlx-community conversions (checked against Qwen3-30B-A3B-4bit's index) quantize the
    /// router too and store bf16 scales and biases; training then feeds fp32 activations.
    func testBFloat16QuantizedCheckpointTrainsInFloat32() throws {
        let dir = try tempDir()
        try writeConfig(dir, extra: ["quantization": ["group_size": 64, "bits": 4]])
        let source = MoEBlock(dims: d, hiddenDims: h, numExperts: e, topK: k)
        eval(source)
        let p = "model.layers.0.mlp."
        var arrays: [String: MLXArray] = [:]
        func put(_ name: String, _ w: MLXArray) {
            let (q, s, b) = MLX.quantized(w.asType(.bfloat16), groupSize: 64, bits: 4)
            XCTAssertEqual(s.dtype, .bfloat16)
            arrays[p + name + ".weight"] = q
            arrays[p + name + ".scales"] = s
            arrays[p + name + ".biases"] = b!
        }
        put("gate", source.router.gateWeight)
        put("switch_mlp.gate_proj", source.experts.gateProj.weight)
        put("switch_mlp.up_proj", source.experts.upProj.weight)
        put("switch_mlp.down_proj", source.experts.downProj.weight)
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let block = try Qwen3MoECheckpoint(directory: dir).loadMoEBlock(layer: 0, dtype: .float32)
        XCTAssertEqual(block.router.gateWeight.dtype, .float32)
        let x = MLXRandom.normal([1, 32, d])
        let y = block(x)
        XCTAssertEqual(y.dtype, .float32)
        let rel = (abs(y - source(x)).mean() / abs(source(x)).mean()).item(Float.self)
        XCTAssertLessThan(rel, 0.25)

        // Router training through frozen bf16-scaled experts produces finite gradients.
        MoETraining.trainRouterAndExperts(block)
        let (_, g) = valueAndGrad(model: block) { (m: MoEBlock, x, t) in
            mseLoss(predictions: m(x), targets: t)
        }(block, x, source(x))
        let gw = g.flattened().first { $0.0 == "gate.weight" }!.1
        XCTAssertTrue(isNaN(gw).any().item(Bool.self) == false)
        XCTAssertGreaterThan(gw.abs().sum().item(Float.self), 0)
    }

    /// Teacher and student loaded from one checkpoint must not share parameter objects:
    /// `Module.update` rewrites them in place, so pruning one would prune the other.
    func testTwoLoadsAreIndependent() throws {
        let dir = try tempDir()
        try writeConfig(dir, extra: ["quantization": ["group_size": 64, "bits": 4]])
        let source = MoEBlock(dims: d, hiddenDims: h, numExperts: e, topK: k)
        eval(source)
        let p = "model.layers.0.mlp."
        var arrays: [String: MLXArray] = [p + "gate.weight": source.router.gateWeight]
        for (name, proj) in [
            ("gate_proj", source.experts.gateProj), ("up_proj", source.experts.upProj),
            ("down_proj", source.experts.downProj),
        ] {
            let (w, s, b) = MLX.quantized(proj.weight, groupSize: 64, bits: 4)
            arrays[p + "switch_mlp.\(name).weight"] = w
            arrays[p + "switch_mlp.\(name).scales"] = s
            arrays[p + "switch_mlp.\(name).biases"] = b!
        }
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let ckpt = try Qwen3MoECheckpoint(directory: dir)
        let teacher = try ckpt.loadMoEBlock(layer: 0)
        let student = try ckpt.loadMoEBlock(layer: 0)
        let x = MLXRandom.normal([1, 16, d])
        let before = teacher(x)
        ExpertPruner.prune(student, keep: [0, 2])
        XCTAssertEqual(teacher.numExperts, e)
        XCTAssertEqual(teacher.experts.gateProj.weight.dim(0), e)
        assertClose(teacher(x), before, atol: 0, rtol: 0)
        XCTAssertEqual(try ckpt.loadMoEBlock(layer: 0).numExperts, e, "reload is unpruned")
    }

    func testQuantizedTensorsWithoutQuantizationConfigThrow() throws {
        let dir = try tempDir()
        try writeConfig(dir)
        let w = MLXRandom.normal([e, d])
        let (gq, gs, gb) = MLX.quantized(w, groupSize: 64, bits: 4)
        try save(
            arrays: [
                "model.layers.0.mlp.gate.weight": gq, "model.layers.0.mlp.gate.scales": gs,
                "model.layers.0.mlp.gate.biases": gb!,
            ], url: dir.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try Qwen3MoECheckpoint(directory: dir).loadMoEBlock(layer: 0))
    }
}

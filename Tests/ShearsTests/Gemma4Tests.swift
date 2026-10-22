import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A tiny random Gemma 4 MoE model: forward, cache, Hugging Face round trip, pruning
/// alignment, calibration, healing and LoRA on experts.
final class Gemma4Tests: ShearsTestCase {
    static let vocab = 128, d = 64, experts = 8, topK = 2, moeHidden = 32

    /// Two layers (one sliding, one full), k = v on the full layer with fewer KV heads,
    /// partial RoPE, softcapping, and a sliding window shorter than the test sequences.
    func configJSON(tied: Bool = true) -> [String: Any] {
        [
            "model_type": "gemma4_text",
            "hidden_size": Self.d, "num_hidden_layers": 2, "intermediate_size": 96,
            "num_attention_heads": 4, "head_dim": 16, "global_head_dim": 32,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1,
            "attention_k_eq_v": true, "rms_norm_eps": 1e-6, "vocab_size": Self.vocab,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 8,
            "final_logit_softcapping": 30.0, "enable_moe_block": true,
            "num_experts": Self.experts, "top_k_experts": Self.topK,
            "moe_intermediate_size": Self.moeHidden,
            "layer_types": ["sliding_attention", "full_attention"],
            "tie_word_embeddings": tied,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention": [
                    "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                ],
            ],
        ]
    }

    func config(tied: Bool = true) throws -> Gemma4Configuration {
        let data = try JSONSerialization.data(withJSONObject: configJSON(tied: tied))
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        try c.validate()
        return c
    }

    /// Perturb every parameter, so weights that initialise to ones (norms, router scales,
    /// `per_expert_scale`, `layer_scalar`) actually matter to every check below.
    func randomize(_ m: Module, seed: UInt64 = 7) {
        MLXRandom.seed(seed)
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.2 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        eval(m)
    }

    func model(tied: Bool = true) throws -> Gemma4TextModel {
        let m = Gemma4TextModel(try config(tied: tied))
        randomize(m)
        m.train(false)
        return m
    }

    func tokens(_ b: Int, _ l: Int, seed: UInt64 = 3) -> MLXArray {
        MLXRandom.seed(seed)
        return MLXRandom.randInt(0 ..< Int32(Self.vocab), [b, l])
    }

    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory
            .appendingPathComponent("shears-gemma4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func moeLayer(_ m: Gemma4TextModel, _ i: Int) -> Gemma4MoEDecoderLayer {
        m.layers[i] as! Gemma4MoEDecoderLayer
    }

    func clone(_ layer: Gemma4DecoderLayer, _ c: Gemma4Configuration) -> Gemma4DecoderLayer {
        let s = Gemma4TextModel.makeLayer(c, layer: layer.layerIndex)
        layer.copyParameters(into: s)
        s.train(layer.training)
        return s
    }

    // MARK: Structure and forward

    func testConfigurationDecodesNestedTextConfig() throws {
        let dir = try tempDir()
        let outer: [String: Any] = ["model_type": "gemma4", "text_config": configJSON()]
        let url = dir.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: outer).write(to: url)
        let c = try Gemma4Configuration.load(from: url)
        XCTAssertEqual(c.hiddenSize, Self.d)
        XCTAssertEqual(c.layerTypes, ["sliding_attention", "full_attention"])
        XCTAssertEqual(c.fullPartialRotaryFactor, 0.25)
        XCTAssertEqual(c.numExperts, Self.experts)

        var withPLE = configJSON()
        withPLE["hidden_size_per_layer_input"] = 16
        let data = try JSONSerialization.data(withJSONObject: withPLE)
        let bad = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        XCTAssertThrowsError(try bad.validate(), "per-layer embeddings are rejected, not ignored")
    }

    func testModuleTreeMatchesExpectedKeys() throws {
        let m = try model()
        let keys = Set(m.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.v_proj.weight"))
        XCTAssertFalse(keys.contains("model.layers.1.self_attn.v_proj.weight"), "k = v on full layers")
        for k in [
            "model.layers.1.router.proj.weight", "model.layers.1.router.scale",
            "model.layers.1.router.per_expert_scale", "model.layers.1.layer_scalar",
            "model.layers.1.experts.switch_glu.gate_proj.weight",
            "model.layers.1.pre_feedforward_layernorm_2.weight", "model.embed_tokens.weight",
        ] {
            XCTAssertTrue(keys.contains(k), k)
        }
        XCTAssertFalse(keys.contains { $0.hasPrefix("lm_head") })

        // The text tensor names of the published 26B-A4B checkpoint index, layer numbers
        // normalised. (Its full-attention layers have no v_proj; sliding ones do.)
        let published: Set<String> = [
            "model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
            "experts.down_proj", "experts.gate_up_proj", "input_layernorm.weight", "layer_scalar",
            "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "post_attention_layernorm.weight", "post_feedforward_layernorm.weight",
            "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
            "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight",
            "router.per_expert_scale", "router.proj.weight", "router.scale",
            "self_attn.k_norm.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
        ]
        let exported = Set(Gemma4Checkpoint.huggingFaceWeights(of: m).keys.map { k -> String in
            guard let r = k.range(of: #"layers\.\d+\."#, options: .regularExpression) else { return k }
            return String(k[r.upperBound...])
        })
        XCTAssertEqual(exported, published)
        XCTAssertEqual(m.parameters().flattened().first { $0.0 == "model.layers.1.self_attn.k_proj.weight" }?.1.shape, [32, Self.d])
    }

    func testForwardShapesAndSoftcap() throws {
        let m = try model()
        let logits = m(tokens(2, 24))
        XCTAssertEqual(logits.shape, [2, 24, Self.vocab])
        XCTAssertLessThanOrEqual(abs(logits).max().item(Float.self), 30)
        XCTAssertFalse(isNaN(logits).any().item(Bool.self))
    }

    func testDecoderLayersAsUnaryBlocksReproduceTheModel() throws {
        let m = try model()
        let t = tokens(2, 24)
        var h = m.embed(t)
        for layer in m.layers { h = layer(h) }
        assertClose(m.logits(fromHidden: h), m(t), atol: 1e-5)
    }

    func testSlidingWindowMasksDistantTokens() throws {
        // Changing token 0 must not change the sliding layer's output beyond the window,
        // but must change the full layer's.
        let m = try model()
        var a = tokens(1, 20).asArray(Int32.self)
        let x1 = m.embed(MLXArray(a).reshaped(1, 20))
        a[0] = (a[0] + 1) % Int32(Self.vocab)
        let x2 = m.embed(MLXArray(a).reshaped(1, 20))
        let s1 = m.layers[0](x1), s2 = m.layers[0](x2)
        assertClose(s1[0, 8...], s2[0, 8...], atol: 1e-6)
        let f1 = m.layers[1](x1), f2 = m.layers[1](x2)
        XCTAssertGreaterThan(abs(f1[0, 19] - f2[0, 19]).max().item(Float.self), 1e-4)
    }

    func testKVCacheMatchesPrefill() throws {
        let m = try model()
        let t = tokens(1, 20)
        let full = m(t)
        let cache = m.newCache()
        var parts = [m(t[0..., ..<12], cache: cache)]
        for i in 12 ..< 20 { parts.append(m(t[0..., i ..< (i + 1)], cache: cache)) }
        XCTAssertEqual(cache[0].offset, 20)
        assertClose(concatenated(parts, axis: 1), full, atol: 1e-4)
    }

    // MARK: Checkpoint

    /// Write `model` as a sharded Hugging Face multimodal checkpoint (with a vision tensor
    /// that must be skipped) and load it back.
    func roundTrip(_ m: Gemma4TextModel, tied: Bool) throws -> Gemma4Checkpoint {
        let dir = try tempDir()
        let outer: [String: Any] = ["model_type": "gemma4", "text_config": configJSON(tied: tied)]
        try JSONSerialization.data(withJSONObject: outer)
            .write(to: dir.appendingPathComponent("config.json"))

        var hf = Gemma4Checkpoint.huggingFaceWeights(of: m)
        XCTAssertEqual(hf["model.language_model.layers.1.experts.gate_up_proj"]?.shape,
                       [Self.experts, 2 * Self.moeHidden, Self.d])
        XCTAssertEqual(hf["model.language_model.layers.1.experts.down_proj"]?.shape,
                       [Self.experts, Self.d, Self.moeHidden])
        XCTAssertNil(hf["model.language_model.layers.1.experts.switch_glu.gate_proj.weight"])
        XCTAssertNotNil(hf["model.language_model.layers.0.router.per_expert_scale"])
        XCTAssertEqual(hf["lm_head.weight"] != nil, !tied)
        hf["model.vision_tower.encoder.weight"] = MLXRandom.normal([4, 4])

        let names = hf.keys.sorted()
        var map: [String: String] = [:]
        var shards: [[String: MLXArray]] = [[:], [:]]
        for (i, k) in names.enumerated() {
            let s = i % 2
            shards[s][k] = hf[k]
            map[k] = "model-0000\(s + 1)-of-00002.safetensors"
        }
        for s in 0 ..< 2 {
            try save(arrays: shards[s], url: dir.appendingPathComponent("model-0000\(s + 1)-of-00002.safetensors"))
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        return try Gemma4Checkpoint(directory: dir)
    }

    func testHuggingFaceRoundTripGivesIdenticalLogits() throws {
        for tied in [true, false] {
            let m = try model(tied: tied)
            let ckpt = try roundTrip(m, tied: tied)
            let loaded = try ckpt.loadModel()
            loaded.train(false)
            assertClose(loaded.parameters(), m.parameters(), atol: 0, rtol: 0)
            let t = tokens(2, 24)
            assertClose(loaded(t), m(t), atol: 0, rtol: 0, "tied=\(tied)")

            let layer = try ckpt.loadDecoderLayer(1)
            XCTAssertTrue(layer is Gemma4MoEDecoderLayer)
            assertClose(layer.parameters(), m.layers[1].parameters(), atol: 0, rtol: 0)
            let e = try ckpt.loadEmbedding()
            assertClose(e.weight, m.model.embedTokens.weight, atol: 0, rtol: 0)
        }
    }

    func testLoaderRejectsMissingTensors() throws {
        let dir = try tempDir()
        try JSONSerialization.data(withJSONObject: configJSON())
            .write(to: dir.appendingPathComponent("config.json"))
        var hf = Gemma4Checkpoint.huggingFaceWeights(of: try model(), prefix: "model.")
        hf["model.layers.0.router.per_expert_scale"] = nil
        try save(arrays: hf, url: dir.appendingPathComponent("model.safetensors"))
        let ckpt = try Gemma4Checkpoint(directory: dir)
        XCTAssertThrowsError(try ckpt.loadModel())
        XCTAssertNoThrow(try ckpt.loadDecoderLayer(1))
    }

    // MARK: Pruning

    func testPruningSlicesRouterRowsAndPerExpertScale() throws {
        let layer = moeLayer(try model(), 1)
        let proj = layer.router.proj.weight * 1
        let pes = layer.router.perExpertScale * 1
        let scale = layer.router.scale * 1
        let down = layer.expertBank.downProj.weight * 1
        eval(proj, pes, scale, down)

        ExpertPruner.prune(layer, keep: [7, 0, 2, 5])
        let keep = MLXArray([0, 2, 5, 7] as [Int32])
        XCTAssertEqual(layer.numExperts, 4)
        XCTAssertEqual(layer.router.topK, 2)
        assertClose(layer.router.proj.weight, proj.take(keep, axis: 0), atol: 0, rtol: 0)
        assertClose(layer.router.perExpertScale, pes.take(keep, axis: 0), atol: 0, rtol: 0)
        assertClose(layer.router.scale, scale, atol: 0, rtol: 0)
        assertClose(layer.expertBank.downProj.weight, down.take(keep, axis: 0), atol: 0, rtol: 0)
        XCTAssertEqual(layer.expertBank.gateProj.weight.shape, [4, Self.moeHidden, Self.d])
    }

    func testTokensWhoseExpertsSurviveAreUnchanged() throws {
        // Gemma's softmax runs over the selected experts only, so a token whose whole
        // top-k is kept must give exactly the same layer output after pruning. A router,
        // per_expert_scale or expert misalignment breaks this while shapes stay right.
        let layer = moeLayer(try model(), 1)
        MLXRandom.seed(11)
        let n = 256
        let x = MLXRandom.normal([1, n, Self.d])
        let before = layer(x)
        let h = layer.attentionResidual(x, mask: layer.mask(n: n), cache: nil)
        let (indices, _) = layer.router.route(h)

        let keep = [0, 2, 5, 7]
        ExpertPruner.prune(layer, keep: keep)
        let after = layer(x)

        let idx = indices.reshaped([-1, Self.topK]).asArray(UInt32.self)
        let keepSet = Set(keep.map(UInt32.init))
        var surviving: [Int32] = []
        var dropped: [Int32] = []
        for t in 0 ..< n {
            if keepSet.contains(idx[2 * t]) && keepSet.contains(idx[2 * t + 1]) {
                surviving.append(Int32(t))
            } else {
                dropped.append(Int32(t))
            }
        }
        XCTAssertGreaterThan(surviving.count, 10, "test needs some fully-kept tokens")
        let sel = MLXArray(surviving)
        assertClose(after[0].take(sel, axis: 0), before[0].take(sel, axis: 0), atol: 1e-5)
        // And pruning did change something.
        let other = MLXArray(dropped)
        XCTAssertGreaterThan(
            abs(after[0].take(other, axis: 0) - before[0].take(other, axis: 0)).max().item(Float.self),
            1e-3)
    }

    func testExpertStatsCalibrateAndPrune() throws {
        let m = try model()
        let layer = moeLayer(m, 0)
        let stats = ExpertStats(numExperts: Self.experts)
        layer.stats = stats
        let x = m.embed(tokens(2, 24))
        _ = layer(x)
        XCTAssertEqual(stats.tokens, 48)
        XCTAssertEqual(stats.pickCount.sum().item(Float.self), Float(48 * Self.topK))
        XCTAssertGreaterThan(stats.saliencySum.sum().item(Float.self), 0)
        ExpertPruner.prune(layer, using: stats, keeping: 4, by: .reap)
        XCTAssertEqual(layer.numExperts, 4)
        XCTAssertNil(layer.stats)
        XCTAssertEqual(layer(x).shape, x.shape)
    }

    // MARK: Training

    func testPrepareClosuresSelectRouterAndExperts() throws {
        let layer = moeLayer(try model(), 1)
        XCTAssertEqual(MoETraining.mixtures(in: layer).count, 1)
        MoETraining.trainRouterAndExperts(layer)
        XCTAssertEqual(Set(layer.trainableParameters().flattened().map(\.0)), [
            "router.proj.weight", "router.scale", "router.per_expert_scale",
            "experts.switch_glu.gate_proj.weight", "experts.switch_glu.up_proj.weight",
            "experts.switch_glu.down_proj.weight",
        ])
        MoETraining.trainRouterOnly(layer)
        XCTAssertEqual(Set(layer.trainableParameters().flattened().map(\.0)), [
            "router.proj.weight", "router.scale", "router.per_expert_scale",
        ])
    }

    func testQuantizedExpertsWithLoRA() throws {
        let m = try model()
        let layer = moeLayer(m, 1)
        let x = m.embed(tokens(1, 16))
        let before = layer(x)
        ExpertQuantizer.quantizeExperts(layer, groupSize: 32, bits: 8, loraRank: 4)
        XCTAssertTrue(layer.expertBank.gateProj is QuantizedSwitchLinear)
        // LoRA B starts at zero, so only quantization error separates the outputs.
        assertClose(layer(x), before, atol: 5e-2, rtol: 5e-2)

        MoETraining.trainRouterAndLoRA(layer)
        var expected: Set<String> = ["router.proj.weight", "router.scale", "router.per_expert_scale"]
        for p in ["gate_proj", "up_proj", "down_proj"] {
            expected.insert("experts.switch_glu.\(p).lora_a")
            expected.insert("experts.switch_glu.\(p).lora_b")
        }
        XCTAssertEqual(Set(layer.trainableParameters().flattened().map(\.0)), expected)

        ExpertPruner.prune(layer, keep: [1, 3, 4, 6])
        let q = layer.expertBank.upProj as! QuantizedSwitchLinear
        XCTAssertEqual(q.loraA?.dim(0), 4)
        XCTAssertEqual(layer(x).shape, x.shape)
    }

    func testLayerWiseMSEHealingOverDecoderLayersReducesLoss() throws {
        let c = try config()
        let m = try model()
        m.train(true)
        let inputs: [MLXArray] = (0 ..< 6).map { m.embed(tokens(2, 24, seed: 100 + UInt64($0))) }
        eval(inputs)

        // Calibrate each teacher on what reaches it, keep half its experts.
        let teachers = m.layers
        var students: [Gemma4DecoderLayer] = []
        var xs = inputs
        for t in teachers {
            let moe = t as! Gemma4MoEDecoderLayer
            let stats = ExpertStats(numExperts: moe.numExperts)
            moe.stats = stats
            xs = xs.map { let y = t($0); eval(y); return y }
            moe.stats = nil
            let s = clone(t, c)
            ExpertPruner.prune(s as! Gemma4MoEDecoderLayer, using: stats, keeping: 4)
            students.append(s)
        }
        let snapshot = teachers.map { $0.parameters().flattened().map { ($0.0, $0.1 * 1) } }

        var activations = inputs
        var reports: [(initial: Float, final: Float)] = []
        for (t, s) in zip(teachers, students) {
            reports.append(mseHeal(
                student: s, teacher: t, inputs: activations, prepare: MoETraining.trainRouterAndExperts,
                epochs: 8))
            activations = activations.map { let y = stopGradient(s($0)); eval(y); return y }
        }

        XCTAssertEqual(reports.count, 2)
        for (i, r) in reports.enumerated() {
            XCTAssertGreaterThan(r.initial, 0)
            XCTAssertLessThan(r.final, r.initial * 0.8, "layer \(i): \(r)")
        }
        for (t, snap) in zip(teachers, snapshot) {
            assertClose(t.parameters(), ModuleParameters.unflattened(snap), atol: 0, rtol: 0)
        }
        // A healed student still writes back to Hugging Face names at its pruned size.
        let hf = Gemma4Checkpoint.huggingFaceWeights(of: students[1], index: 1)
        XCTAssertEqual(hf["model.language_model.layers.1.experts.gate_up_proj"]?.shape,
                       [4, 2 * Self.moeHidden, Self.d])
        XCTAssertEqual(hf["model.language_model.layers.1.router.per_expert_scale"]?.shape, [4])
    }
}

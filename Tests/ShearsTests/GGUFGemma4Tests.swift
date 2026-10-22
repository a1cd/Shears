import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Shears

final class GGUFGemma4Tests: ShearsTestCase {
    // MARK: Name mapping

    func testTensorNameMapping() {
        let p = "model.language_model."
        let table: [(String, String)] = [
            (p + "embed_tokens.weight", "token_embd.weight"),
            (p + "norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            (p + "embed_tokens_per_layer.weight", "per_layer_token_embd.weight"),
            (p + "per_layer_model_projection.weight", "per_layer_model_proj.weight"),
            (p + "per_layer_projection_norm.weight", "per_layer_proj_norm.weight"),
            (p + "layers.3.input_layernorm.weight", "blk.3.attn_norm.weight"),
            (p + "layers.3.post_attention_layernorm.weight", "blk.3.post_attention_norm.weight"),
            (p + "layers.3.pre_feedforward_layernorm.weight", "blk.3.ffn_norm.weight"),
            (p + "layers.3.post_feedforward_layernorm.weight", "blk.3.post_ffw_norm.weight"),
            (p + "layers.3.pre_feedforward_layernorm_2.weight", "blk.3.pre_ffw_norm_2.weight"),
            (p + "layers.3.post_feedforward_layernorm_1.weight", "blk.3.post_ffw_norm_1.weight"),
            (p + "layers.3.post_feedforward_layernorm_2.weight", "blk.3.post_ffw_norm_2.weight"),
            (p + "layers.12.self_attn.q_proj.weight", "blk.12.attn_q.weight"),
            (p + "layers.12.self_attn.k_proj.weight", "blk.12.attn_k.weight"),
            (p + "layers.12.self_attn.v_proj.weight", "blk.12.attn_v.weight"),
            (p + "layers.12.self_attn.o_proj.weight", "blk.12.attn_output.weight"),
            (p + "layers.12.self_attn.q_norm.weight", "blk.12.attn_q_norm.weight"),
            (p + "layers.12.self_attn.k_norm.weight", "blk.12.attn_k_norm.weight"),
            (p + "layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
            (p + "layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
            (p + "layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
            (p + "layers.0.experts.gate_up_proj", "blk.0.ffn_gate_up_exps.weight"),
            (p + "layers.0.experts.down_proj", "blk.0.ffn_down_exps.weight"),
            (p + "layers.0.router.proj.weight", "blk.0.ffn_gate_inp.weight"),
            (p + "layers.0.router.scale", "blk.0.ffn_gate_inp.scale"),
            (p + "layers.0.router.per_expert_scale", "blk.0.ffn_down_exps.scale"),
            (p + "layers.0.layer_scalar", "blk.0.layer_output_scale.weight"),
            (p + "layers.5.per_layer_input_gate.weight", "blk.5.inp_gate.weight"),
            (p + "layers.5.per_layer_projection.weight", "blk.5.proj.weight"),
            (p + "layers.5.post_per_layer_input_norm.weight", "blk.5.post_norm.weight"),
            // The same tensors without the language_model level, and the older nesting.
            ("model.layers.3.input_layernorm.weight", "blk.3.attn_norm.weight"),
            ("language_model.model.layers.3.input_layernorm.weight", "blk.3.attn_norm.weight"),
            ("model.embed_tokens.weight", "token_embd.weight"),
        ]
        for (hf, gguf) in table {
            XCTAssertEqual(GGUFGemma4Name.map(hf), .mapped(gguf), hf)
        }
        for skipped in [
            "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
            "model.embed_vision.embedding_projection.weight", "model.audio_tower.layers.0.x.weight",
            "model.vision_tower.std_bias", "model.language_model.layers.0.self_attn.rotary_emb.inv_freq",
        ] {
            XCTAssertEqual(GGUFGemma4Name.map(skipped), .skip, skipped)
        }
        XCTAssertEqual(GGUFGemma4Name.map(p + "layers.0.surprise.weight"), .unknown)
        XCTAssertEqual(GGUFGemma4Name.map(p + "surprise.weight"), .unknown)
    }

    func testNormsAreNotShifted() throws {
        // Gemma 3 stores norm weights minus one and the converter adds it back. Gemma 4 does not.
        let dir = try makeGGUFTempDirectory()
        let source = try GGUFTinyGemma4().write(to: dir)
        let out = dir.appendingPathComponent("out.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        let r = try GGUFReader(url: out)
        let want = source["model.language_model.layers.0.input_layernorm.weight"]!.asType(.float32).asArray(Float.self)
        XCTAssertEqual(try r.floats("blk.0.attn_norm.weight"), want)
        let scalar = source["model.language_model.layers.1.layer_scalar"]!.asType(.float32).asArray(Float.self)
        XCTAssertEqual(try r.floats("blk.1.layer_output_scale.weight"), scalar)
    }

    // MARK: Whole export

    static let expectedTensorNames: Set<String> = {
        var n: Set<String> = ["token_embd.weight", "output_norm.weight", "rope_freqs.weight"]
        for i in 0 ..< 2 {
            for s in [
                "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_ffw_norm_1",
                "post_ffw_norm_2", "pre_ffw_norm_2", "attn_q", "attn_k", "attn_output", "attn_q_norm",
                "attn_k_norm", "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_up_exps", "ffn_down_exps",
                "ffn_gate_inp", "layer_output_scale",
            ] { n.insert("blk.\(i).\(s).weight") }
            n.insert("blk.\(i).ffn_gate_inp.scale")
            n.insert("blk.\(i).ffn_down_exps.scale")
        }
        n.insert("blk.0.attn_v.weight")  // the full-attention layer has no value projection
        return n
    }()

    func testExportBF16WritesTheExpectedFile() throws {
        let dir = try makeGGUFTempDirectory()
        let source = try GGUFTinyGemma4().write(to: dir)
        let out = dir.appendingPathComponent("tiny.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)

        XCTAssertEqual(summary.expertCount, 8)
        XCTAssertEqual(summary.expertsUsed, 2)
        XCTAssertEqual(summary.expertFeedForwardLength, 48)
        XCTAssertEqual(summary.tokenCount, GGUFTinyGemma4.vocabCount)
        XCTAssertEqual(summary.warnings, [])
        XCTAssertEqual(summary.fileType, .bf16)

        let r = try GGUFReader(url: out)
        // The pinned hyperparameter table: every key with its type.
        let expected: [String: GGUFValue] = [
            "general.architecture": .string("gemma4"),
            "general.type": .string("model"),
            "general.file_type": .uint32(32),
            "general.quantization_version": .uint32(2),
            "gemma4.block_count": .uint32(2),
            "gemma4.context_length": .uint32(512),
            "gemma4.embedding_length": .uint32(64),
            "gemma4.feed_forward_length": .uint32(128),
            "gemma4.attention.head_count": .uint32(4),
            "gemma4.attention.head_count_kv": .int32s([2, 1]),
            "gemma4.rope.freq_base": .float32(1_000_000),
            "gemma4.rope.freq_base_swa": .float32(10_000),
            "gemma4.attention.layer_norm_rms_epsilon": .float32(1e-6),
            "gemma4.expert_count": .uint32(8),
            "gemma4.expert_used_count": .uint32(2),
            "gemma4.attention.key_length": .uint32(64),
            "gemma4.attention.value_length": .uint32(64),
            "gemma4.final_logit_softcapping": .float32(30),
            "gemma4.attention.sliding_window": .uint32(16),
            "gemma4.attention.shared_kv_layers": .uint32(0),
            "gemma4.embedding_length_per_layer_input": .uint32(0),
            "gemma4.attention.sliding_window_pattern": .bools([true, false]),
            "gemma4.attention.key_length_swa": .uint32(32),
            "gemma4.attention.value_length_swa": .uint32(32),
            "gemma4.expert_feed_forward_length": .uint32(48),
            "gemma4.rope.dimension_count": .uint32(64),
            "gemma4.rope.dimension_count_swa": .uint32(32),
        ]
        for (k, v) in expected {
            XCTAssertEqual(r[k], v, k)
        }
        let nonTokenizer = r.metadata.map { $0.key }.filter { !$0.hasPrefix("tokenizer.") && $0 != "general.name" }
        XCTAssertEqual(Set(nonTokenizer), Set(expected.keys), "no unexpected hyperparameter keys")
        XCTAssertEqual(r["general.name"], .string(dir.lastPathComponent))

        // Tensor set: nothing extra (vision dropped), nothing missing.
        XCTAssertEqual(Set(r.tensors.map { $0.name }), Self.expectedTensorNames)
        XCTAssertNil(r.tensorInfo("blk.1.attn_v.weight"))
        XCTAssertNotNil(r.tensorInfo("blk.0.attn_v.weight"))
        XCTAssertNil(r.tensorInfo("output.weight"), "tied embeddings: no separate output tensor")

        // Shapes (row-major here, ne reversed on disk) and types.
        func check(_ name: String, _ shape: [Int], _ type: GGUFTensorType, line: UInt = #line) {
            let t = r.tensorInfo(name)
            XCTAssertEqual(t?.shape, shape, name, line: line)
            XCTAssertEqual(t?.ne, shape.reversed(), name, line: line)
            XCTAssertEqual(t?.type, type, name, line: line)
        }
        let n = GGUFTinyGemma4.vocabCount
        check("token_embd.weight", [n, 64], .bf16)
        check("output_norm.weight", [64], .f32)
        check("rope_freqs.weight", [32], .f32)
        check("blk.0.attn_q.weight", [128, 64], .bf16)
        check("blk.0.attn_k.weight", [64, 64], .bf16)
        check("blk.0.attn_v.weight", [64, 64], .bf16)
        check("blk.0.attn_output.weight", [64, 128], .bf16)
        check("blk.0.attn_q_norm.weight", [32], .f32)
        check("blk.1.attn_q.weight", [256, 64], .bf16)
        check("blk.1.attn_k.weight", [64, 64], .bf16)
        check("blk.1.attn_output.weight", [64, 256], .bf16)
        check("blk.1.attn_k_norm.weight", [64], .f32)
        check("blk.1.ffn_gate.weight", [128, 64], .bf16)
        check("blk.1.ffn_down.weight", [64, 128], .bf16)
        check("blk.1.ffn_gate_up_exps.weight", [8, 96, 64], .bf16)
        check("blk.1.ffn_down_exps.weight", [8, 64, 48], .bf16)
        check("blk.1.ffn_gate_inp.weight", [8, 64], .f32)
        check("blk.1.ffn_gate_inp.scale", [64], .f32)
        check("blk.1.ffn_down_exps.scale", [8], .f32)
        check("blk.1.layer_output_scale.weight", [1], .f32)
        check("blk.1.post_ffw_norm_2.weight", [64], .f32)

        // Values. The experts stay fused, gate rows first: nothing is split or reordered.
        func hf(_ name: String) -> [Float] {
            source["model.language_model." + name]!.asType(.float32).asArray(Float.self)
        }
        XCTAssertEqual(try r.floats("blk.1.ffn_gate_up_exps.weight"), hf("layers.1.experts.gate_up_proj"))
        XCTAssertEqual(try r.floats("blk.0.ffn_down_exps.weight"), hf("layers.0.experts.down_proj"))
        XCTAssertEqual(try r.floats("blk.1.ffn_gate_inp.weight"), hf("layers.1.router.proj.weight"))
        XCTAssertEqual(try r.floats("blk.0.ffn_down_exps.scale"), hf("layers.0.router.per_expert_scale"))
        XCTAssertEqual(try r.floats("blk.0.ffn_gate_inp.scale"), hf("layers.0.router.scale"))
        XCTAssertEqual(try r.floats("blk.1.attn_q.weight"), hf("layers.1.self_attn.q_proj.weight"))
        // The embedding keeps only the rows the tokenizer has.
        XCTAssertEqual(try r.floats("token_embd.weight"), Array(hf("embed_tokens.weight")[0 ..< n * 64]))
        // Proportional RoPE: head_dim 64, partial 0.25 -> 8 rotated pairs, 24 pinned.
        XCTAssertEqual(
            try r.floats("rope_freqs.weight"), [Float](repeating: 1, count: 8) + [Float](repeating: 1e30, count: 24))
    }

    func testOutputTypes() throws {
        let dir = try makeGGUFTempDirectory()
        let source = try GGUFTinyGemma4().write(to: dir)
        let n = GGUFTinyGemma4.vocabCount
        let want = source["model.language_model.layers.0.mlp.gate_proj.weight"]!.asType(.float32).asArray(Float.self)
        let wantEmbed = Array(
            source["model.language_model.embed_tokens.weight"]!.asType(.float32).asArray(Float.self)[0 ..< n * 64])

        for (type, fileType, weightType) in [
            (GGUFOutputType.f32, 0, GGUFTensorType.f32), (.f16, 1, .f16), (.bf16, 32, .bf16), (.q8_0, 7, .q8_0),
        ] {
            let out = dir.appendingPathComponent("\(weightType.name).gguf")
            let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: type)
            XCTAssertEqual(summary.fileType, weightType)
            let r = try GGUFReader(url: out)
            XCTAssertEqual(r["general.file_type"], .uint32(UInt32(fileType)))
            XCTAssertEqual(r.tensorInfo("blk.0.ffn_gate.weight")?.type, weightType)
            XCTAssertEqual(r.tensorInfo("token_embd.weight")?.type, weightType)
            // Always float32, whatever was asked for.
            for always in [
                "blk.0.attn_norm.weight", "blk.1.attn_k_norm.weight", "output_norm.weight",
                "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp.scale", "blk.0.ffn_down_exps.scale",
                "blk.0.layer_output_scale.weight", "rope_freqs.weight", "blk.0.post_ffw_norm_1.weight",
            ] {
                XCTAssertEqual(r.tensorInfo(always)?.type, .f32, "\(weightType.name) \(always)")
            }
            // Expert down projections have rows of 48, not a multiple of the Q8_0 block.
            XCTAssertEqual(r.tensorInfo("blk.0.ffn_down_exps.weight")?.type, weightType == .q8_0 ? .f16 : weightType)
            XCTAssertEqual(r.tensorInfo("blk.0.ffn_gate_up_exps.weight")?.type, weightType)

            let got = try r.floats("blk.0.ffn_gate.weight")
            let embed = try r.floats("token_embd.weight")
            switch weightType {
            case .f32, .f16, .bf16:
                XCTAssertEqual(got, want, weightType.name)
                XCTAssertEqual(embed, wantEmbed)
            case .q8_0:
                for (g, w) in zip(got, want) { XCTAssertEqual(g, w, accuracy: 0.02) }
                // 7 weights per layer plus the layer-0 value projection and the embedding;
                // the two expert down projections fall back to half precision.
                XCTAssertEqual(summary.tensorsByType["Q8_0"], 2 * 7 + 1 + 1)
                XCTAssertEqual(summary.tensorsByType["F16"], 2)
                XCTAssertEqual(summary.tensorsByType.values.reduce(0, +), summary.tensorCount)
            default:
                XCTFail("\(weightType.name) is not exercised here")
            }
        }
    }

    func testAutoFollowsTheCheckpointPrecision() throws {
        let dir = try makeGGUFTempDirectory()
        try GGUFTinyGemma4().write(to: dir)
        let out = dir.appendingPathComponent("auto.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out)
        XCTAssertEqual(summary.fileType, .bf16)

        // A float32 checkpoint is exported as float16, as the reference converter does.
        let dir32 = try makeGGUFTempDirectory()
        var tiny = GGUFTinyGemma4()
        tiny.mutate = { t in for (k, v) in t { t[k] = v.asType(.float32) } }
        try tiny.write(to: dir32)
        let out32 = dir32.appendingPathComponent("auto.gguf")
        let s32 = try GGUFExport.gemma4(hfDirectory: dir32, output: out32)
        XCTAssertEqual(s32.fileType, .f16)
        XCTAssertEqual(try GGUFReader(url: out32).tensorInfo("blk.0.attn_q.weight")?.type, .f16)
    }

    func testPrunedExpertCountComesFromTheTensors() throws {
        // config.json still describes the original 128-expert model with 8 active; the tensors
        // were pruned to 8 experts and the router now picks 8 of 8.
        let dir = try makeGGUFTempDirectory()
        var tiny = GGUFTinyGemma4()
        tiny.experts = 8
        tiny.configExperts = 128
        tiny.configTopK = 16
        tiny.configMoeFF = 704
        try tiny.write(to: dir)
        let out = dir.appendingPathComponent("pruned.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        let r = try GGUFReader(url: out)
        XCTAssertEqual(r["gemma4.expert_count"], .uint32(8))
        XCTAssertEqual(r["gemma4.expert_used_count"], .uint32(8), "clamped to the experts that exist")
        XCTAssertEqual(r["gemma4.expert_feed_forward_length"], .uint32(48))
        XCTAssertEqual(summary.warnings.count, 3)
        XCTAssertTrue(summary.warnings.contains { $0.contains("128") && $0.contains("8") })
        XCTAssertEqual(r.tensorInfo("blk.0.ffn_gate_inp.weight")?.shape, [8, 64])
    }

    func testFewerExpertsStillExport() throws {
        let dir = try makeGGUFTempDirectory()
        var tiny = GGUFTinyGemma4()
        tiny.experts = 3
        tiny.topK = 2
        try tiny.write(to: dir)
        let out = dir.appendingPathComponent("three.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        let r = try GGUFReader(url: out)
        XCTAssertEqual(r["gemma4.expert_count"], .uint32(3))
        XCTAssertEqual(r.tensorInfo("blk.1.ffn_gate_up_exps.weight")?.shape, [3, 96, 64])
        XCTAssertEqual(r.tensorInfo("blk.1.ffn_down_exps.scale")?.shape, [3])
    }

    func testUntiedOutputIsKeptAndTruncated() throws {
        let dir = try makeGGUFTempDirectory()
        var tiny = GGUFTinyGemma4()
        tiny.tied = false
        try tiny.write(to: dir)
        let out = dir.appendingPathComponent("untied.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        let r = try GGUFReader(url: out)
        XCTAssertEqual(r.tensorInfo("output.weight")?.shape, [GGUFTinyGemma4.vocabCount, 64])
    }

    func testQuantizedExportOfAFloat32CheckpointReadsOnlyWhatItNeeds() throws {
        // Float32 source with an 8-bit target takes the MLX path, including row truncation.
        let dir = try makeGGUFTempDirectory()
        var tiny = GGUFTinyGemma4()
        tiny.mutate = { t in for (k, v) in t { t[k] = v.asType(.float32) } }
        let source = try tiny.write(to: dir)
        let out = dir.appendingPathComponent("q.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .q8_0)
        let r = try GGUFReader(url: out)
        let n = GGUFTinyGemma4.vocabCount
        XCTAssertEqual(r.tensorInfo("token_embd.weight")?.shape, [n, 64])
        let want = Array(source["model.language_model.embed_tokens.weight"]!.asArray(Float.self)[0 ..< n * 64])
        for (g, w) in zip(try r.floats("token_embd.weight"), want) { XCTAssertEqual(g, w, accuracy: 0.02) }
    }

    func testTextOnlyLayoutExportsTheSameFile() throws {
        // A text-only save: flat config.json and tensors named model.* instead of
        // model.language_model.*, and no vision tensors.
        var nested = GGUFTinyGemma4()
        nested.vision = false
        var flat = nested
        flat.flatConfig = true
        flat.tensorPrefix = "model."
        let a = try makeGGUFTempDirectory()
        let b = try makeGGUFTempDirectory()
        MLXRandom.seed(7)
        try nested.write(to: a)
        MLXRandom.seed(7)
        try flat.write(to: b)
        let outA = a.appendingPathComponent("a.gguf")
        let outB = b.appendingPathComponent("b.gguf")
        try GGUFExport.gemma4(hfDirectory: a, output: outA, type: .bf16, name: "same")
        try GGUFExport.gemma4(hfDirectory: b, output: outB, type: .bf16, name: "same")
        XCTAssertEqual(try Data(contentsOf: outA), try Data(contentsOf: outB))
    }

    // MARK: Other Gemma 4 shapes: dense, shared KV, per-layer inputs

    static func denseVariant() -> GGUFTinyGemma4 {
        var tiny = GGUFTinyGemma4()
        tiny.moe = false
        tiny.layerTypes = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        tiny.sharedKVLayers = 2
        tiny.doubleWideMLP = true
        tiny.perLayerInput = 16
        return tiny
    }

    func testDenseModelWithSharedKVAndPerLayerInputs() throws {
        let dir = try makeGGUFTempDirectory()
        let source = try Self.denseVariant().write(to: dir)
        let out = dir.appendingPathComponent("dense.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        XCTAssertNil(summary.expertCount)
        let r = try GGUFReader(url: out)
        XCTAssertFalse(r.metadata.contains { $0.key.contains("expert") }, "no expert keys on a dense model")
        XCTAssertEqual(r["gemma4.block_count"], .uint32(4))
        XCTAssertEqual(r["gemma4.feed_forward_length"], .int32s([128, 128, 256, 256]), "double-wide on shared layers")
        XCTAssertEqual(r["gemma4.attention.head_count_kv"], .int32s([2, 1, 2, 1]))
        XCTAssertEqual(r["gemma4.attention.shared_kv_layers"], .uint32(2))
        XCTAssertEqual(r["gemma4.embedding_length_per_layer_input"], .uint32(16))
        XCTAssertEqual(r["gemma4.attention.sliding_window_pattern"], .bools([true, false, true, false]))
        XCTAssertNil(r.tensorInfo("blk.2.attn_k.weight"), "shared layers have no key projection")
        XCTAssertNil(r.tensorInfo("blk.3.attn_k_norm.weight"))
        XCTAssertNotNil(r.tensorInfo("blk.2.attn_q.weight"))
        XCTAssertEqual(r.tensorInfo("blk.3.ffn_up.weight")?.shape, [256, 64])
        XCTAssertEqual(r.tensorInfo("blk.1.ffn_up.weight")?.shape, [128, 64])
        let n = GGUFTinyGemma4.vocabCount
        XCTAssertEqual(r.tensorInfo("per_layer_token_embd.weight")?.shape, [n, 64])
        XCTAssertEqual(r.tensorInfo("per_layer_model_proj.weight")?.shape, [64, 64])
        XCTAssertEqual(r.tensorInfo("per_layer_proj_norm.weight")?.shape, [16])
        XCTAssertEqual(r.tensorInfo("per_layer_proj_norm.weight")?.type, .f32)
        XCTAssertEqual(r.tensorInfo("blk.2.inp_gate.weight")?.shape, [16, 64])
        XCTAssertEqual(r.tensorInfo("blk.2.proj.weight")?.shape, [64, 16])
        XCTAssertEqual(r.tensorInfo("blk.2.post_norm.weight")?.shape, [64])
        XCTAssertEqual(r.tensorInfo("blk.2.post_norm.weight")?.type, .f32)
        let want = source["model.language_model.embed_tokens_per_layer.weight"]!.asType(.float32).asArray(Float.self)
        XCTAssertEqual(try r.floats("per_layer_token_embd.weight"), Array(want[0 ..< n * 64]))
        XCTAssertNil(r.tensorInfo("blk.0.ffn_gate_inp.weight"))
        XCTAssertNil(r.tensorInfo("blk.0.pre_ffw_norm_2.weight"))
    }

    // MARK: Rejected checkpoints

    func testRejectsInconsistentExpertCounts() throws {
        let dir = try makeGGUFTempDirectory()
        var tiny = GGUFTinyGemma4()
        tiny.expertsPerLayer = [8, 4]
        try tiny.write(to: dir)
        XCTAssertThrowsError(
            try GGUFExport.gemma4(hfDirectory: dir, output: dir.appendingPathComponent("x.gguf"), type: .bf16)
        ) { XCTAssertTrue("\($0)".contains("earlier layers"), "\($0)") }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("x.gguf").path))
    }

    func testRejectsMissingWrongAndUnknownTensors() throws {
        let p = GGUFTinyGemma4.prefix
        func export(_ mutate: @escaping (inout [String: MLXArray]) -> Void) throws -> Error? {
            let dir = try makeGGUFTempDirectory()
            var tiny = GGUFTinyGemma4()
            tiny.mutate = mutate
            try tiny.write(to: dir)
            do {
                try GGUFExport.gemma4(hfDirectory: dir, output: dir.appendingPathComponent("x.gguf"), type: .bf16)
                return nil
            } catch { return error }
        }
        let missing = try export { $0[p + "layers.1.self_attn.k_proj.weight"] = nil }
        XCTAssertTrue("\(try XCTUnwrap(missing))".contains("missing blk.1.attn_k.weight"), "\(String(describing: missing))")

        let wrong = try export { $0[p + "layers.0.self_attn.q_proj.weight"] = MLXArray.zeros([100, 64]).asType(.bfloat16) }
        XCTAssertTrue("\(try XCTUnwrap(wrong))".contains("blk.0.attn_q.weight"), "\(String(describing: wrong))")

        let unknown = try export { $0[p + "layers.0.mystery.weight"] = MLXArray.zeros([2]).asType(.bfloat16) }
        XCTAssertTrue("\(try XCTUnwrap(unknown))".contains("mystery"), "\(String(describing: unknown))")

        let badScale = try export { $0[p + "layers.0.router.per_expert_scale"] = MLXArray.zeros([5]).asType(.bfloat16) }
        XCTAssertNotNil(badScale)

        let intDType = try export { $0[p + "layers.0.mlp.gate_proj.weight"] = MLXArray.zeros([128, 64], dtype: .int32) }
        XCTAssertNotNil(intDType)

        let shortEmbedding = try export {
            $0[p + "embed_tokens.weight"] = MLXArray.zeros([10, 64]).asType(.bfloat16)
        }
        XCTAssertTrue("\(try XCTUnwrap(shortEmbedding))".contains("rows"), "\(String(describing: shortEmbedding))")
    }

    // MARK: Tokenizer seams

    func testExportWithoutTokenizerMetadataKeepsTheModelComplete() throws {
        let dir = try makeGGUFTempDirectory()
        try GGUFTinyGemma4().write(to: dir)
        let out = dir.appendingPathComponent("novocab.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16, tokenizer: .none)
        let r = try GGUFReader(url: out)
        XCTAssertNil(summary.tokenCount)
        XCTAssertFalse(r.metadata.contains { $0.key.hasPrefix("tokenizer.") })
        XCTAssertEqual(r["gemma4.expert_count"], .uint32(8))
        XCTAssertEqual(
            r.tensorInfo("token_embd.weight")?.shape, [GGUFTinyGemma4.vocabCount + 4, 64], "rows are left alone")
    }

    func testCustomTokenizerMetadataSetsTheVocabularySize() throws {
        let dir = try makeGGUFTempDirectory()
        try GGUFTinyGemma4().write(to: dir)
        let custom = GGUFTokenizerMetadata(
            entries: [("tokenizer.ggml.model", .string("gemma4")), ("tokenizer.ggml.tokens", .strings(["a", "b"]))],
            tokenCount: 100)
        let out = dir.appendingPathComponent("custom.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16, tokenizer: .custom(custom))
        let r = try GGUFReader(url: out)
        XCTAssertEqual(r.tensorInfo("token_embd.weight")?.shape, [100, 64])
        XCTAssertEqual(r["tokenizer.ggml.tokens"], .strings(["a", "b"]))
        XCTAssertEqual(r["tokenizer.ggml.add_bos_token"], .bool(true))
        XCTAssertEqual(r["tokenizer.ggml.add_space_prefix"], .bool(false))
    }

    // MARK: Optional: checks against the real 26B-A4B layout (files not committed)

    /// With `TEST_RUNNER_GGUF_ORACLE_VOCAB` (llama.cpp's `models/ggml-vocab-gemma-4.gguf`, made by
    /// its converter from the 26B-A4B model) and `TEST_RUNNER_GGUF_ORACLE_TOKENIZER_DIR` (that
    /// model's `config.json`), every non-tokenizer key the converter wrote must match, value
    /// and type.
    func testHyperparametersMatchTheReferenceFile() throws {
        let env = ProcessInfo.processInfo.environment
        guard let vocabPath = env["GGUF_ORACLE_VOCAB"], let dir = env["GGUF_ORACLE_TOKENIZER_DIR"] else {
            throw XCTSkip("set TEST_RUNNER_GGUF_ORACLE_VOCAB and TEST_RUNNER_GGUF_ORACLE_TOKENIZER_DIR")
        }
        let oracle = try GGUFReader(url: URL(fileURLWithPath: vocabPath))
        let config = try GGUFGemma4Config.load(directory: URL(fileURLWithPath: dir))
        var shape = GGUFGemma4Shape()
        shape.expertCount = 128
        shape.expertFeedForwardLength = 704
        let mine = try GGUFGemma4Metadata.build(config: config, shape: shape, fileType: .f16, name: nil)
        let skipped: Set<String> = ["general.name", "general.finetune"]
        for (key, value) in mine {
            XCTAssertEqual(oracle[key], value, key)
        }
        let mineKeys = Set(mine.map { $0.0 })
        for (key, _) in oracle.metadata where !key.hasPrefix("tokenizer.") && !skipped.contains(key) {
            XCTAssertTrue(mineKeys.contains(key), "reference has \(key), the export does not")
        }
        XCTAssertEqual(oracle["general.file_type"], .uint32(1))
    }

    /// With `TEST_RUNNER_GGUF_REAL_HEADERS` set to a directory of safetensors headers (the JSON
    /// after the 8-byte length) from the real model's shards, plus the oracle config, runs the real
    /// tensor list through name mapping and layout validation without any tensor data.
    func testRealCheckpointLayoutPassesValidation() throws {
        let env = ProcessInfo.processInfo.environment
        guard let headers = env["GGUF_REAL_HEADERS"], let dir = env["GGUF_ORACLE_TOKENIZER_DIR"] else {
            throw XCTSkip("set TEST_RUNNER_GGUF_REAL_HEADERS and TEST_RUNNER_GGUF_ORACLE_TOKENIZER_DIR")
        }
        let config = try GGUFGemma4Config.load(directory: URL(fileURLWithPath: dir))
        var tensors: [String: GGUFSafetensorsEntry] = [:]
        var unknown: [String] = []
        var skipped = 0
        for file in try FileManager.default.contentsOfDirectory(atPath: headers).sorted() where file.hasSuffix(".json") {
            let json = try XCTUnwrap(GGUFJSON.load(URL(fileURLWithPath: headers).appendingPathComponent(file)))
            for (name, info) in json.object ?? [] where name != "__metadata__" {
                let shape = (info["shape"]?.array ?? []).compactMap { $0.int }
                let entry = GGUFSafetensorsEntry(
                    name: name, dtype: info["dtype"]?.string ?? "", shape: shape,
                    file: URL(fileURLWithPath: "/dev/null"), offset: 0, byteCount: 0)
                switch GGUFGemma4Name.map(name) {
                case .skip: skipped += 1
                case .unknown: unknown.append(name)
                case .mapped(let g): tensors[g] = entry
                }
            }
        }
        XCTAssertEqual(unknown, [])
        XCTAssertGreaterThan(skipped, 0, "vision tensors are dropped")
        var shape = GGUFGemma4Shape()
        try GGUFExport.resolveExperts(config: config, tensors: tensors, into: &shape)
        XCTAssertEqual(shape.expertCount, 128)
        XCTAssertEqual(shape.expertFeedForwardLength, 704)
        XCTAssertEqual(shape.warnings, [])
        try GGUFExport.validate(config: config, tensors: tensors, expert: shape, vocabRows: 262144)
        // 30 layers; the 5 full-attention layers have no value projection.
        XCTAssertEqual(tensors.keys.filter { $0.hasSuffix(".attn_v.weight") }.count, 25)
        XCTAssertEqual(tensors.keys.filter { $0.hasSuffix(".ffn_gate_up_exps.weight") }.count, 30)
        XCTAssertNil(tensors["output.weight"])
        // 8-bit output keeps the expert weights quantized on the real model: both widths are
        // multiples of 32 (2816 and 704).
        XCTAssertEqual(
            GGUFGemma4Precision.type(ggufName: "blk.0.ffn_down_exps.weight", shape: [128, 2816, 704], requested: .q8_0), .q8_0)
    }

    // MARK: Optional: leave exports for an external loader check

    /// With `TEST_RUNNER_GGUF_KEEP_DIR` set, writes exports of the tiny model there so a real
    /// llama.cpp build can be pointed at them.
    func testLeaveExportsForExternalChecks() throws {
        guard let keep = ProcessInfo.processInfo.environment["GGUF_KEEP_DIR"] else {
            throw XCTSkip("set TEST_RUNNER_GGUF_KEEP_DIR to keep exports")
        }
        let keepURL = URL(fileURLWithPath: keep)
        try FileManager.default.createDirectory(at: keepURL, withIntermediateDirectories: true)
        let dir = try makeGGUFTempDirectory()
        try GGUFTinyGemma4().write(to: dir)
        for (type, name) in [(GGUFOutputType.bf16, "bf16"), (.f16, "f16"), (.q8_0, "q8_0"), (.f32, "f32")] {
            try GGUFExport.gemma4(
                hfDirectory: dir, output: keepURL.appendingPathComponent("tiny-\(name).gguf"), type: type)
        }
        let dense = try makeGGUFTempDirectory()
        try Self.denseVariant().write(to: dense)
        try GGUFExport.gemma4(
            hfDirectory: dense, output: keepURL.appendingPathComponent("tiny-dense-bf16.gguf"), type: .bf16)
    }
}

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Shears

/// Builds a tiny random Gemma 4 MoE checkpoint in Hugging Face layout: two layers (one
/// sliding-window, one full-attention without a value projection), hidden size 64, eight experts,
/// a byte-fallback BPE tokenizer, safetensors split across two shards, and vision tensors that
/// a text export must drop.
struct GGUFTinyGemma4 {
    static let hidden = 64
    static let heads = 4
    static let headDim = 32
    static let globalHeadDim = 64
    static let kvHeads = 2
    static let globalKVHeads = 1
    static let ff = 128
    static let prefix = "model.language_model."

    /// Layer kinds. The default is one sliding-window layer and one full-attention layer.
    var layerTypes = ["sliding_attention", "full_attention"]
    /// Trailing layers that reuse earlier layers' KV caches and carry no key/value tensors.
    var sharedKVLayers = 0
    /// Double the dense MLP width on the shared-KV layers.
    var doubleWideMLP = false
    /// Per-layer input embedding width (0 for none).
    var perLayerInput = 0
    /// `false` builds a dense model with no router or experts.
    var moe = true
    var experts = 8
    var topK = 2
    /// Deliberately not a multiple of 32, so 8-bit output must fall back to half precision.
    var moeFF = 48
    /// What config.json claims, when it should differ from the tensors (a pruned checkpoint).
    var configExperts: Int?
    var configTopK: Int?
    var configMoeFF: Int?
    var padRows = 4
    var vision = true
    var tokenizer = true
    var tied = true
    /// Tensor name prefix: the multimodal checkpoint's `model.language_model.`, or `model.`.
    var tensorPrefix = GGUFTinyGemma4.prefix
    /// Write the text settings at the top level of config.json, as a text-only save does.
    var flatConfig = false
    /// Expert count per layer, to build an inconsistent checkpoint.
    var expertsPerLayer: [Int]?
    var mutate: (inout [String: MLXArray]) -> Void = { _ in }

    /// The tokenizer's vocabulary size (specials, a few letters, 256 byte tokens, 3 added).
    static let vocabCount = 5 + 1 + 26 + 2 + 256 + 3

    var layers: Int { layerTypes.count }

    func config() -> [String: Any] {
        var text: [String: Any] = [
            "model_type": "gemma4_text",
            "hidden_size": Self.hidden, "num_hidden_layers": layers,
            "intermediate_size": Self.ff, "num_attention_heads": Self.heads,
            "head_dim": Self.headDim, "global_head_dim": Self.globalHeadDim,
            "num_key_value_heads": Self.kvHeads, "num_global_key_value_heads": Self.globalKVHeads,
            "attention_k_eq_v": true, "rms_norm_eps": 1e-6, "max_position_embeddings": 512,
            "num_kv_shared_layers": sharedKVLayers, "hidden_size_per_layer_input": perLayerInput,
            "use_double_wide_mlp": doubleWideMLP, "sliding_window": 16, "final_logit_softcapping": 30.0,
            "enable_moe_block": moe, "layer_types": layerTypes,
            "tie_word_embeddings": tied,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention": [
                    "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional",
                ],
            ],
        ]
        if moe {
            text["num_experts"] = configExperts ?? experts
            text["top_k_experts"] = configTopK ?? topK
            text["moe_intermediate_size"] = configMoeFF ?? moeFF
        }
        text["vocab_size"] = Self.vocabCount + padRows
        if flatConfig {
            text["architectures"] = ["Gemma4ForCausalLM"]
            return text
        }
        return [
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "eos_token_id": [1, 106], "text_config": text,
        ]
    }

    /// All text tensors by Hugging Face name, bfloat16.
    func tensors() -> [String: MLXArray] {
        func rnd(_ shape: [Int], scale: Float = 0.5) -> MLXArray {
            (MLXRandom.normal(shape) * scale).asType(.bfloat16)
        }
        var t: [String: MLXArray] = [:]
        let p = tensorPrefix
        let h = Self.hidden
        t[p + "embed_tokens.weight"] = rnd([Self.vocabCount + padRows, h])
        t[p + "norm.weight"] = rnd([h])
        if !tied { t["lm_head.weight"] = rnd([Self.vocabCount + padRows, h]) }
        let rows = Self.vocabCount + padRows
        if perLayerInput > 0 {
            t[p + "embed_tokens_per_layer.weight"] = rnd([rows, layers * perLayerInput])
            t[p + "per_layer_model_projection.weight"] = rnd([layers * perLayerInput, h])
            t[p + "per_layer_projection_norm.weight"] = rnd([perLayerInput])
        }
        for i in 0 ..< layers {
            let l = "\(p)layers.\(i)."
            let full = layerTypes[i] == "full_attention"
            let shared = i >= layers - sharedKVLayers
            let hd = full ? Self.globalHeadDim : Self.headDim
            let kv = (full ? Self.globalKVHeads : Self.kvHeads) * hd
            let q = Self.heads * hd
            let e = expertsPerLayer?[i] ?? experts
            let ff = doubleWideMLP && shared ? 2 * Self.ff : Self.ff
            var norms = [
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ]
            if moe {
                norms += ["post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"]
            }
            for n in norms { t[l + n + ".weight"] = rnd([h]) }
            t[l + "self_attn.q_proj.weight"] = rnd([q, h])
            if !shared {
                t[l + "self_attn.k_proj.weight"] = rnd([kv, h])
                if !full { t[l + "self_attn.v_proj.weight"] = rnd([kv, h]) }
                t[l + "self_attn.k_norm.weight"] = rnd([hd])
            }
            t[l + "self_attn.o_proj.weight"] = rnd([h, q])
            t[l + "self_attn.q_norm.weight"] = rnd([hd])
            t[l + "mlp.gate_proj.weight"] = rnd([ff, h])
            t[l + "mlp.up_proj.weight"] = rnd([ff, h])
            t[l + "mlp.down_proj.weight"] = rnd([h, ff])
            if moe {
                t[l + "experts.gate_up_proj"] = rnd([e, 2 * moeFF, h])
                t[l + "experts.down_proj"] = rnd([e, h, moeFF])
                t[l + "router.proj.weight"] = rnd([e, h])
                t[l + "router.scale"] = rnd([h])
                t[l + "router.per_expert_scale"] = rnd([e])
            }
            t[l + "layer_scalar"] = rnd([1])
            if perLayerInput > 0 {
                t[l + "per_layer_input_gate.weight"] = rnd([perLayerInput, h])
                t[l + "per_layer_projection.weight"] = rnd([h, perLayerInput])
                t[l + "post_per_layer_input_norm.weight"] = rnd([h])
            }
        }
        if vision {
            t["model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"] = rnd([8, 8])
            t["model.embed_vision.embedding_projection.weight"] = rnd([h, 8])
        }
        var m = t
        mutate(&m)
        return m
    }

    /// Writes the checkpoint to `dir` and returns the tensors written.
    @discardableResult
    func write(to dir: URL) throws -> [String: MLXArray] {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let all = tensors()
        try JSONSerialization.data(withJSONObject: config(), options: [.sortedKeys])
            .write(to: dir.appendingPathComponent("config.json"))

        // Two shards: the embedding and layer 0 in the first, the rest in the second.
        var first: [String: MLXArray] = [:]
        var second: [String: MLXArray] = [:]
        for (k, v) in all {
            if k.contains("embed_tokens") || k.contains("layers.0.") { first[k] = v } else { second[k] = v }
        }
        try save(arrays: first, url: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try save(arrays: second, url: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        var weightMap: [String: String] = [:]
        for k in first.keys { weightMap[k] = "model-00001-of-00002.safetensors" }
        for k in second.keys { weightMap[k] = "model-00002-of-00002.safetensors" }
        try JSONSerialization.data(
            withJSONObject: ["metadata": ["total_size": 0], "weight_map": weightMap], options: [.sortedKeys]
        ).write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        if tokenizer { try Self.writeTokenizer(to: dir) }
        return all
    }

    /// A byte-fallback BPE tokenizer in the layout of the real one: special tokens that are
    /// also listed as added tokens at low ids, a base vocabulary, and added tokens above it.
    static func writeTokenizer(to dir: URL) throws {
        var vocab: [String: Int] = [:]
        let specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
        for (i, s) in specials.enumerated() { vocab[s] = i }
        vocab["▁"] = 5
        for (i, c) in "abcdefghijklmnopqrstuvwxyz".enumerated() { vocab[String(c)] = 6 + i }
        vocab["▁a"] = 32
        vocab["ab"] = 33
        for b in 0 ..< 256 { vocab[String(format: "<0x%02X>", b)] = 34 + b }
        let added = ["<|turn>", "<turn|>", "<|channel>"]
        func entry(_ id: Int, _ content: String) -> [String: Any] {
            [
                "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            ]
        }
        var addedTokens = specials.enumerated().map { entry($0.offset, $0.element) }
        for (i, s) in added.enumerated() { addedTokens.append(entry(290 + i, s)) }
        let tokenizerJSON: [String: Any] = [
            "version": "1.0", "added_tokens": addedTokens,
            "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "▁"],
            "model": [
                "type": "BPE", "vocab": vocab, "merges": [["▁", "a"], ["a", "b"]],
                "byte_fallback": true,
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerJSON).write(
            to: dir.appendingPathComponent("tokenizer.json"))
        let tokenizerConfig: [String: Any] = [
            "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>",
            "mask_token": "<mask>", "eot_token": "<turn|>", "soc_token": "<|channel>",
            "extra_special_tokens": ["<|turn>"], "add_eos_token": false,
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(
            to: dir.appendingPathComponent("tokenizer_config.json"))
        try Self.chatTemplate.write(
            to: dir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
    }

    static let chatTemplate =
        "{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}"
}

extension ShearsTestCase {
    func makeGGUFTempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gguf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}

import Foundation

/// The parts of a Hugging Face Gemma 4 `config.json` that GGUF export needs.
///
/// Reads either the nested `text_config` form (`Gemma4ForConditionalGeneration`) or a flat
/// text-only config.
public struct GGUFGemma4Config: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    /// KV heads on sliding-window layers. `nil` when the config doesn't say.
    public var kvHeadsSliding: Int?
    /// KV heads on full-attention layers. `nil` when the config doesn't say.
    public var kvHeadsFull: Int?
    public var headDimSliding: Int
    public var headDimFull: Int
    public var intermediateSize: Int
    public var expertIntermediateSize: Int?
    public var expertCount: Int?
    public var expertsPerToken: Int?
    /// `true` for each sliding-window layer.
    public var slidingLayers: [Bool]
    public var slidingWindow: Int?
    public var ropeThetaFull: Double
    public var ropeThetaSliding: Double?
    public var partialRotaryFactorFull: Double
    public var partialRotaryFactorSliding: Double
    public var rmsNormEps: Double
    public var contextLength: Int
    public var finalLogitSoftcap: Double?
    public var kvSharedLayers: Int
    public var perLayerInputSize: Int
    public var useDoubleWideMLP: Bool
    /// `architectures` from config.json.
    public var architectures: [String]

    public init(json root: [String: Any]) throws {
        let text = (root["text_config"] as? [String: Any]).map { root.merging($0) { _, new in new } } ?? root

        func int(_ k: String) -> Int? { (text[k] as? NSNumber).flatMap { Self.isBool($0) ? nil : $0.intValue } }
        func dbl(_ k: String) -> Double? { (text[k] as? NSNumber).flatMap { Self.isBool($0) ? nil : $0.doubleValue } }
        func need(_ k: String) throws -> Int {
            guard let v = int(k) else { throw GGUFError.other("config.json: missing \(k)") }
            return v
        }

        architectures = (root["architectures"] as? [String]) ?? []
        hiddenSize = try need("hidden_size")
        layerCount = try need("num_hidden_layers")
        headCount = int("num_attention_heads") ?? 8
        intermediateSize = try need("intermediate_size")
        expertIntermediateSize = int("expert_intermediate_size") ?? int("moe_intermediate_size")
        expertCount = int("num_experts") ?? int("num_local_experts")
        expertsPerToken = int("top_k_experts") ?? int("num_experts_per_tok")
        guard let types = text["layer_types"] as? [String], types.count == layerCount else {
            throw GGUFError.other("config.json: layer_types must list \(layerCount) layers")
        }
        slidingLayers = types.map { $0 == "sliding_attention" }
        slidingWindow = int("sliding_window")
        rmsNormEps = dbl("rms_norm_eps") ?? 1e-6
        contextLength = int("max_position_embeddings") ?? 131072
        finalLogitSoftcap = dbl("final_logit_softcapping")
        kvSharedLayers = int("num_kv_shared_layers") ?? 0
        perLayerInputSize = int("hidden_size_per_layer_input") ?? 0
        useDoubleWideMLP = text["use_double_wide_mlp"] as? Bool ?? false

        // Per-layer overrides are only consulted when the global value is absent, as in the
        // reference converter.
        let perLayer = text["per_layer_config"] as? [String: Any]
        func fromFullLayer(_ key: String) -> Int? {
            guard let perLayer else { return nil }
            for (idx, cfg) in perLayer.sorted(by: { (Int($0.key) ?? 0) < (Int($1.key) ?? 0) }) {
                guard let i = Int(idx), i < types.count, types[i] == "full_attention",
                    let c = cfg as? [String: Any], let v = (c[key] as? NSNumber)?.intValue
                else { continue }
                return v
            }
            return nil
        }
        guard let swaDim = int("head_dim") else { throw GGUFError.other("config.json: missing head_dim") }
        headDimSliding = swaDim
        guard let fullDim = int("global_head_dim") ?? fromFullLayer("head_dim") else {
            throw GGUFError.other("config.json: missing global_head_dim")
        }
        headDimFull = fullDim
        kvHeadsSliding = int("num_key_value_heads")
        kvHeadsFull = int("num_global_key_value_heads") ?? fromFullLayer("num_key_value_heads")

        guard let rope = text["rope_parameters"] as? [String: Any],
            let full = rope["full_attention"] as? [String: Any]
        else { throw GGUFError.other("config.json: rope_parameters.full_attention is missing") }
        guard (full["rope_type"] as? String) == "proportional" else {
            throw GGUFError.other("config.json: only proportional RoPE on full-attention layers is supported")
        }
        guard let factor = (full["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw GGUFError.other("config.json: rope_parameters.full_attention.partial_rotary_factor is missing")
        }
        partialRotaryFactorFull = factor
        ropeThetaFull = (full["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000
        ropeThetaSliding = ((rope["sliding_attention"] as? [String: Any])?["rope_theta"] as? NSNumber)?.doubleValue
        partialRotaryFactorSliding = (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 1.0
    }

    static func isBool(_ n: NSNumber) -> Bool { n.isBoolean }

    public static func load(directory: URL) throws -> GGUFGemma4Config {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw GGUFError.other("cannot read config.json in \(directory.path)") }
        return try GGUFGemma4Config(json: json)
    }

    /// First layer index that reuses an earlier layer's KV cache.
    public var firstKVSharedLayer: Int { layerCount - kvSharedLayers }

    public func hasFullAttention() -> Bool { slidingLayers.contains(false) }

    public func headDim(layer: Int) -> Int { slidingLayers[layer] ? headDimSliding : headDimFull }

    public func kvHeads(layer: Int) -> Int {
        let sliding = kvHeadsSliding ?? 4
        return slidingLayers[layer] ? sliding : (kvHeadsFull ?? sliding)
    }

    public func denseFeedForwardLength(layer: Int) -> Int {
        useDoubleWideMLP && layer >= firstKVSharedLayer ? intermediateSize * 2 : intermediateSize
    }
}

/// Facts about the checkpoint that come from its tensors rather than its config. After pruning
/// the tensors are the truth: `config.json` may still describe the original expert count.
public struct GGUFGemma4Shape: Sendable {
    public var expertCount: Int?
    public var expertFeedForwardLength: Int?
    public var warnings: [String] = []
}

/// How one Hugging Face tensor name maps into the GGUF `gemma4` architecture.
enum GGUFGemma4Name {
    enum Result: Equatable {
        /// Multimodal or bookkeeping tensor that a text-only export leaves out.
        case skip
        case mapped(String)
        case unknown
    }

    static let skipPrefixes = [
        "mlp", "vit.", "vpm.", "siglip2.", "conformer.", "connector.", "merger.", "resampler.",
        "sound_encoder.", "sound_projection.", "speech_embeddings.",
    ]
    static let skipFragments = [
        "visual.", "vision.", "audio.", "talker.", "vision_", "audio_", "token2wav.", "code2wav.",
        "projector.", "pre_mm_projector_norm", "image_newline", "view_seperator", "patch_embed",
        "patch_embedding", "patch_merger.", "patch_merge_mlp.",
    ]
    static let skipSuffixes = [".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq"]

    static let layerMap: [String: String] = [
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "ffn_norm.weight",
        "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
        "pre_feedforward_layernorm_2.weight": "pre_ffw_norm_2.weight",
        "post_feedforward_layernorm_1.weight": "post_ffw_norm_1.weight",
        "post_feedforward_layernorm_2.weight": "post_ffw_norm_2.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        // Experts are stored stacked and fused (gate rows then up rows), as llama.cpp expects.
        "experts.gate_up_proj.weight": "ffn_gate_up_exps.weight",
        "experts.down_proj.weight": "ffn_down_exps.weight",
        "router.proj.weight": "ffn_gate_inp.weight",
        "router.scale": "ffn_gate_inp.scale",
        "router.per_expert_scale": "ffn_down_exps.scale",
        "layer_scalar.weight": "layer_output_scale.weight",
        "per_layer_input_gate.weight": "inp_gate.weight",
        "per_layer_projection.weight": "proj.weight",
        "post_per_layer_input_norm.weight": "post_norm.weight",
    ]

    static let globalMap: [String: String] = [
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "model.lm_head.weight": "output.weight",
        "model.embed_tokens_per_layer.weight": "per_layer_token_embd.weight",
        "model.per_layer_model_projection.weight": "per_layer_model_proj.weight",
        "model.per_layer_projection_norm.weight": "per_layer_proj_norm.weight",
    ]

    /// Maps a Hugging Face tensor name (any of the `model.language_model.`, `model.` or bare
    /// prefixes) to its GGUF name.
    static func map(_ original: String) -> Result {
        var name = original
        let probe = name.hasPrefix("model.") ? String(name.dropFirst(6)) : name
        if skipPrefixes.contains(where: { probe.hasPrefix($0) }) { return .skip }
        if skipFragments.contains(where: { probe.contains($0) }) { return .skip }
        if skipSuffixes.contains(where: { name.hasSuffix($0) }) { return .skip }

        name = name.replacingOccurrences(of: "language_model.", with: "")
        if !name.hasPrefix("model.") && name != "lm_head.weight" { name = "model." + name }

        // Same fix-ups as the reference converter: scalar-ish tensors and fused expert
        // tensors are stored without a ".weight" suffix.
        if name.hasSuffix("per_dim_scale") || name.hasSuffix("layer_scalar") { name += ".weight" }
        if name.contains(".experts.") && !name.hasSuffix(".weight") { name += ".weight" }

        if let g = globalMap[name] { return .mapped(g) }

        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count > 3, parts[0] == "model", parts[1] == "layers", let bid = Int(parts[2]) else {
            return .unknown
        }
        let rest = parts[3...].joined(separator: ".")
        guard let mapped = layerMap[rest] else { return .unknown }
        return .mapped("blk.\(bid).\(mapped)")
    }

    /// The layer index in a GGUF per-layer tensor name, if it has one.
    static func layer(of ggufName: String) -> Int? {
        let parts = ggufName.split(separator: ".")
        guard parts.count > 2, parts[0] == "blk" else { return nil }
        return Int(parts[1])
    }
}

/// Decides storage types the way the reference converter does.
enum GGUFGemma4Precision {
    /// Type for a tensor given the requested output type.
    ///
    /// Always float32: anything with one dimension or fewer, tensors named `*_norm.weight`, the
    /// router weight, and anything whose name doesn't end in `.weight` (scales). Quantized
    /// output falls back to float16 where the row length isn't a multiple of the block size.
    static func type(ggufName: String, shape: [Int], requested: GGUFTensorType) -> GGUFTensorType {
        if shape.count <= 1 { return .f32 }
        if ggufName.hasSuffix("_norm.weight") { return .f32 }
        if !ggufName.hasSuffix(".weight") { return .f32 }
        if ggufName.hasSuffix(".ffn_gate_inp.weight") { return .f32 }
        if shape[shape.count - 1] % requested.blockSize != 0 { return .f16 }
        return requested
    }
}

/// The `gemma4` architecture's GGUF metadata, in the key order and with the value types the
/// reference converter writes.
enum GGUFGemma4Metadata {
    static func build(
        config c: GGUFGemma4Config, shape: GGUFGemma4Shape, fileType: GGUFTensorType, name: String?
    ) throws -> [(String, GGUFValue)] {
        func u32(_ v: Int) throws -> GGUFValue {
            guard v >= 0, v <= Int(UInt32.max) else { throw GGUFError.other("metadata value \(v) out of range") }
            return .uint32(UInt32(v))
        }
        let a = "gemma4"
        var kv: [(String, GGUFValue)] = []
        kv.append(("general.architecture", .string(a)))
        kv.append(("general.type", .string("model")))
        if let name { kv.append(("general.name", .string(name))) }
        kv.append(("\(a).block_count", try u32(c.layerCount)))
        kv.append(("\(a).context_length", try u32(c.contextLength)))
        kv.append(("\(a).embedding_length", try u32(c.hiddenSize)))
        if c.useDoubleWideMLP {
            kv.append(("\(a).feed_forward_length", .int32s((0 ..< c.layerCount).map { c.denseFeedForwardLength(layer: $0) })))
        } else {
            kv.append(("\(a).feed_forward_length", try u32(c.intermediateSize)))
        }
        kv.append(("\(a).attention.head_count", try u32(c.headCount)))
        if let sliding = c.kvHeadsSliding, let full = c.kvHeadsFull {
            kv.append(("\(a).attention.head_count_kv", .int32s(c.slidingLayers.map { $0 ? sliding : full })))
        } else {
            kv.append(("\(a).attention.head_count_kv", try u32(c.kvHeadsSliding ?? 4)))
        }
        kv.append(("\(a).rope.freq_base", .float32(Float(c.ropeThetaFull))))
        if let swa = c.ropeThetaSliding { kv.append(("\(a).rope.freq_base_swa", .float32(Float(swa)))) }
        kv.append(("\(a).attention.layer_norm_rms_epsilon", .float32(Float(c.rmsNormEps))))
        if let e = shape.expertCount {
            kv.append(("\(a).expert_count", try u32(e)))
            if let used = c.expertsPerToken { kv.append(("\(a).expert_used_count", try u32(min(used, e)))) }
        }
        kv.append(("\(a).attention.key_length", try u32(c.headDimFull)))
        kv.append(("\(a).attention.value_length", try u32(c.headDimFull)))
        kv.append(("general.file_type", .uint32(GGUFFileType.value(for: fileType))))
        if let cap = c.finalLogitSoftcap, cap != 0 {
            kv.append(("\(a).final_logit_softcapping", .float32(Float(cap))))
        }
        if let w = c.slidingWindow { kv.append(("\(a).attention.sliding_window", try u32(w))) }
        kv.append(("\(a).attention.shared_kv_layers", try u32(c.kvSharedLayers)))
        kv.append(("\(a).embedding_length_per_layer_input", try u32(c.perLayerInputSize)))
        kv.append(("\(a).attention.sliding_window_pattern", .bools(c.slidingLayers)))
        kv.append(("\(a).attention.key_length_swa", try u32(c.headDimSliding)))
        kv.append(("\(a).attention.value_length_swa", try u32(c.headDimSliding)))
        if let f = shape.expertFeedForwardLength { kv.append(("\(a).expert_feed_forward_length", try u32(f))) }
        kv.append(("\(a).rope.dimension_count", try u32(c.headDimFull)))
        kv.append(
            ("\(a).rope.dimension_count_swa",
                try u32(Int(Double(c.headDimSliding) * c.partialRotaryFactorSliding))))
        kv.append(("general.quantization_version", .uint32(2)))
        return kv
    }

    /// Frequency factors for the full-attention layers' proportional RoPE: the first
    /// `headDim * partial / 2` pairs rotate normally and the rest are pushed to ~infinity so
    /// they don't rotate. ggml's NeoX rotation can't skip the middle of a head, so this is how
    /// the unrotated tail is expressed.
    static func ropeFrequencyFactors(_ c: GGUFGemma4Config) -> [Float] {
        let rotated = Int(Double(c.headDimFull) * c.partialRotaryFactorFull / 2)
        let unrotated = c.headDimFull / 2 - rotated
        return [Float](repeating: 1, count: rotated) + [Float](repeating: 1e30, count: unrotated)
    }
}

// Ported from ml-explore/mlx-swift-lm, MIT License, Copyright (c) 2024 ml-explore
// (Libraries/MLXLLM/Models/Gemma4Text.swift, Libraries/MLXLLM/Models/Gemma4.swift).
// See THIRD_PARTY_NOTICES.md.

import Foundation

/// The text half of a Gemma 4 `config.json`.
///
/// Decodes either a text-only config (`model_type: gemma4_text`, fields at the top level)
/// or a multimodal one (`model_type: gemma4`, fields under `text_config`); use
/// ``load(from:)`` for a file. Where the key is absent the defaults are Hugging Face transformers': no logit softcap,
/// a 5:1 sliding:full layer pattern whose last layer is always full attention, and 0.25 partial
/// rotary factor on full-attention layers when there are no `rope_parameters` at all. Real
/// configs set all of these.
///
/// Per-layer embeddings (`hidden_size_per_layer_input > 0`) and KV sharing
/// (`num_kv_shared_layers > 0`), both used by the small dense Gemma 4 models, are not
/// implemented; ``validate()`` rejects such configs rather than building the wrong model.
public struct Gemma4Configuration: Codable, Sendable {
    public var modelType: String = "gemma4_text"
    public var hiddenSize: Int = 1536
    public var numHiddenLayers: Int = 35
    public var intermediateSize: Int = 6144
    public var numAttentionHeads: Int = 8
    /// Head size of sliding-window layers.
    public var headDim: Int = 256
    /// Head size of full-attention layers.
    public var globalHeadDim: Int = 512
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 262144
    public var numKeyValueHeads: Int = 1
    /// KV heads of full-attention layers when ``attentionKEqV`` is set.
    public var numGlobalKeyValueHeads: Int?
    public var numKvSharedLayers: Int = 20
    public var hiddenSizePerLayerInput: Int = 256
    public var slidingWindow: Int = 512
    /// Layers per repeat when `layer_types` is absent: that many minus one sliding, then one full.
    /// Hugging Face's default is 6 (5:1).
    public var slidingWindowPattern: Int = 6
    public var maxPositionEmbeddings: Int = 131072
    /// Full-attention layers have no `v_proj`; values are `v_norm(k_proj(x))`.
    public var attentionKEqV: Bool = false
    /// `tanh(logits / c) * c`; `nil` (the default when the key is absent) or `<= 0` disables it.
    public var finalLogitSoftcapping: Float?
    public var useDoubleWideMlp: Bool = true
    public var enableMoEBlock: Bool = false
    public var numExperts: Int?
    public var topKExperts: Int?
    public var moeIntermediateSize: Int?
    /// `"sliding_attention"` or `"full_attention"` per layer.
    public var layerTypes: [String] = []
    public var tieWordEmbeddings: Bool = true

    public var slidingRopeTheta: Float = 10000
    public var fullRopeTheta: Float = 1_000_000
    /// 0.25 when the config has no `rope_parameters` (Hugging Face's default for full attention);
    /// 1.0 when it has them but the full-attention entry omits `partial_rotary_factor`.
    public var fullPartialRotaryFactor: Float = 0.25
    /// `rope_parameters.full_attention.factor` for proportional RoPE (scales the frequencies).
    public var fullRopeFactor: Float = 1.0
    /// `rope_parameters.full_attention.rope_type`: `proportional` or `default` (whole-head RoPE).
    public var fullRopeType: String = "proportional"
    public var slidingRopeType: String = "default"

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKEqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    /// A number, string or bool inside `rope_parameters`.
    enum RopeValue: Codable {
        case number(Float)
        case string(String)
        case bool(Bool)

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let f = try? c.decode(Float.self) {
                self = .number(f)
            } else if let b = try? c.decode(Bool.self) {
                self = .bool(b)
            } else {
                self = .string(try c.decode(String.self))
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .number(let f): try c.encode(f)
            case .string(let s): try c.encode(s)
            case .bool(let b): try c.encode(b)
            }
        }

        var float: Float? { if case .number(let f) = self { f } else { nil } }
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func v<T: Decodable>(_ k: CodingKeys, _ d: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? d
        }
        modelType = try v(.modelType, "gemma4_text")
        hiddenSize = try v(.hiddenSize, 1536)
        numHiddenLayers = try v(.numHiddenLayers, 35)
        intermediateSize = try v(.intermediateSize, 6144)
        numAttentionHeads = try v(.numAttentionHeads, 8)
        headDim = try v(.headDim, 256)
        globalHeadDim = try v(.globalHeadDim, 512)
        rmsNormEps = try v(.rmsNormEps, 1e-6)
        vocabSize = try v(.vocabSize, 262144)
        numKeyValueHeads = try v(.numKeyValueHeads, 1)
        numGlobalKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        numKvSharedLayers = try v(.numKvSharedLayers, 20)
        hiddenSizePerLayerInput = try v(.hiddenSizePerLayerInput, 256)
        slidingWindow = try v(.slidingWindow, 512)
        slidingWindowPattern = try v(.slidingWindowPattern, 6)
        maxPositionEmbeddings = try v(.maxPositionEmbeddings, 131072)
        attentionKEqV = try v(.attentionKEqV, false)
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        useDoubleWideMlp = try v(.useDoubleWideMlp, true)
        enableMoEBlock = try v(.enableMoEBlock, false)
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        tieWordEmbeddings = try v(.tieWordEmbeddings, true)

        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            // Every `slidingWindowPattern`-th layer is full attention.
            let period = max(1, slidingWindowPattern)
            layerTypes = (0 ..< numHiddenLayers).map {
                ($0 + 1) % period == 0 ? "full_attention" : "sliding_attention"
            }
        }
        // Hugging Face forces the last layer to full attention, whatever the list says.
        if !layerTypes.isEmpty { layerTypes[layerTypes.count - 1] = "full_attention" }

        if let rope = try c.decodeIfPresent([String: [String: RopeValue]].self, forKey: .ropeParameters) {
            fullPartialRotaryFactor = 1.0
            if let s = rope["sliding_attention"] {
                slidingRopeTheta = s["rope_theta"]?.float ?? 10000
                if case .string(let t)? = s["rope_type"] { slidingRopeType = t }
            }
            if let f = rope["full_attention"] {
                fullRopeTheta = f["rope_theta"]?.float ?? 1_000_000
                fullPartialRotaryFactor = f["partial_rotary_factor"]?.float ?? 1.0
                fullRopeFactor = f["factor"]?.float ?? 1.0
                if case .string(let t)? = f["rope_type"] { fullRopeType = t }
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(globalHeadDim, forKey: .globalHeadDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encodeIfPresent(numGlobalKeyValueHeads, forKey: .numGlobalKeyValueHeads)
        try c.encode(numKvSharedLayers, forKey: .numKvSharedLayers)
        try c.encode(hiddenSizePerLayerInput, forKey: .hiddenSizePerLayerInput)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(slidingWindowPattern, forKey: .slidingWindowPattern)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(attentionKEqV, forKey: .attentionKEqV)
        try c.encode(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try c.encode(useDoubleWideMlp, forKey: .useDoubleWideMlp)
        try c.encode(enableMoEBlock, forKey: .enableMoEBlock)
        try c.encodeIfPresent(numExperts, forKey: .numExperts)
        try c.encodeIfPresent(topKExperts, forKey: .topKExperts)
        try c.encodeIfPresent(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        let rope: [String: [String: RopeValue]] = [
            "sliding_attention": [
                "rope_theta": .number(slidingRopeTheta), "rope_type": .string(slidingRopeType),
            ],
            "full_attention": [
                "rope_theta": .number(fullRopeTheta),
                "partial_rotary_factor": .number(fullPartialRotaryFactor),
                "factor": .number(fullRopeFactor),
                "rope_type": .string(fullRopeType),
            ],
        ]
        try c.encode(rope, forKey: .ropeParameters)
    }

    /// Decode `config.json`, reading `text_config` when present.
    public static func load(from url: URL) throws -> Gemma4Configuration {
        let data = try Data(contentsOf: url)
        struct Outer: Decodable {
            var textConfig: Gemma4Configuration?
            var vocabSize: Int?
            enum CodingKeys: String, CodingKey {
                case textConfig = "text_config"
                case vocabSize = "vocab_size"
            }
        }
        let outer = try JSONDecoder().decode(Outer.self, from: data)
        var config: Gemma4Configuration
        if var text = outer.textConfig {
            if let v = outer.vocabSize { text.vocabSize = v }
            config = text
        } else {
            config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        }
        try config.validate()
        return config
    }

    /// Throws for configs this port can't build faithfully.
    public func validate() throws {
        func bad(_ s: String) -> MoELoaderError { .badConfig(s) }
        guard layerTypes.count == numHiddenLayers else {
            throw bad("layer_types has \(layerTypes.count) entries for \(numHiddenLayers) layers")
        }
        for t in layerTypes where t != "sliding_attention" && t != "full_attention" {
            throw bad("unknown layer type \(t)")
        }
        guard slidingRopeType == "default" else {
            throw bad("sliding-attention rope_type '\(slidingRopeType)' is not supported (only default)")
        }
        guard fullRopeType == "proportional" || fullRopeType == "default" else {
            throw bad("full-attention rope_type '\(fullRopeType)' is not supported (proportional or default)")
        }
        guard hiddenSizePerLayerInput == 0 else {
            throw bad("per-layer embeddings (hidden_size_per_layer_input > 0) are not supported")
        }
        guard numKvSharedLayers == 0 else {
            throw bad("KV sharing (num_kv_shared_layers > 0) is not supported")
        }
        if enableMoEBlock {
            guard let e = numExperts, let k = topKExperts, moeIntermediateSize != nil,
                e > 0, k > 0, k <= e
            else {
                throw bad("enable_moe_block needs num_experts, top_k_experts, moe_intermediate_size")
            }
        }
    }

    func isSliding(_ layer: Int) -> Bool { layerTypes[layer] == "sliding_attention" }
}

// Ported from ml-explore/mlx-swift-lm, MIT License, Copyright (c) 2024 ml-explore
// (Libraries/MLXLLM/Models/Gemma4Text.swift). See THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

// MARK: - Decoder layers

/// One dense Gemma 4 decoder layer:
///
/// ```
/// h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
/// out = (h + post_feedforward_layernorm(ffn(h))) * layer_scalar
/// ```
///
/// where `ffn` is `mlp(pre_feedforward_layernorm(h))`. ``Gemma4MoEDecoderLayer`` adds a
/// sparse branch to `ffn`.
///
/// As a `UnaryLayer` (``callAsFunction(_:)``) the layer treats its input as a complete
/// sequence starting at position 0 and builds its own causal (and, on sliding layers,
/// windowed) mask from `x.dim(1)`. That makes every decoder layer usable directly as a
/// single block, at any calibration sequence length.
open class Gemma4DecoderLayer: Module, UnaryLayer {
    public let layerIndex: Int
    public let isSliding: Bool
    public let slidingWindow: Int

    @ModuleInfo(key: "self_attn") public var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") public var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFeedforwardLayernorm: RMSNorm
    @ParameterInfo(key: "layer_scalar") public var layerScalar: MLXArray

    public init(_ config: Gemma4Configuration, layer: Int) {
        layerIndex = layer
        isSliding = config.isSliding(layer)
        slidingWindow = config.slidingWindow
        let d = config.hiddenSize, eps = config.rmsNormEps
        _selfAttn.wrappedValue = Gemma4Attention(config, layer: layer)
        _mlp.wrappedValue = Gemma4MLP(dims: d, hiddenDims: config.intermediateSize)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _preFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _postFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    /// The mask this layer uses for `n` new positions after `offset` cached ones.
    public func mask(n: Int, offset: Int = 0) -> MLXArray {
        gemma4AttentionMask(n: n, offset: offset, window: isSliding ? slidingWindow : nil)
    }

    /// `x + post_attention_layernorm(self_attn(input_layernorm(x)))`: the residual stream
    /// entering the feed-forward half, which is also what the MoE router reads.
    public func attentionResidual(_ x: MLXArray, mask: MLXArray?, cache: Gemma4KVCache?) -> MLXArray {
        x + postAttentionLayernorm(selfAttn(inputLayernorm(x), mask: mask, cache: cache))
    }

    /// The feed-forward branches, before `post_feedforward_layernorm`.
    open func feedForward(_ h: MLXArray) -> MLXArray {
        mlp(preFeedforwardLayernorm(h))
    }

    open func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: Gemma4KVCache?) -> MLXArray {
        let h = attentionResidual(x, mask: mask, cache: cache)
        return (h + postFeedforwardLayernorm(feedForward(h))) * layerScalar
    }

    /// Whole-sequence forward from position 0 with no cache.
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, mask: mask(n: x.dim(1)), cache: nil)
    }
}

/// A Gemma 4 decoder layer with the MoE block (`enable_moe_block`): the dense MLP and a
/// sparse expert branch run in parallel on the post-attention residual `h`,
///
/// ```
/// dense  = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))
/// sparse = post_feedforward_layernorm_2(Σ_k w_k · expert_k(pre_feedforward_layernorm_2(h)))
/// ```
///
/// with `(indices, w) = router(h)`. The router reads `h` itself, not the normed input
/// the experts see. `router` and `experts` are direct children of the layer, matching the
/// Hugging Face key names.
open class Gemma4MoEDecoderLayer: Gemma4DecoderLayer, MixtureOfExperts {
    @ModuleInfo(key: "router") public var router: Gemma4Router
    @ModuleInfo(key: "experts") public var experts: Gemma4Experts
    @ModuleInfo(key: "post_feedforward_layernorm_1") public var postFeedforwardLayernorm1: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") public var postFeedforwardLayernorm2: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") public var preFeedforwardLayernorm2: RMSNorm

    public var stats: ExpertStats?
    /// All of this layer's (kept) experts as a set, for the merged expert layout.
    var mergedSet: ExpertSet?

    public override init(_ config: Gemma4Configuration, layer: Int) {
        precondition(config.enableMoEBlock, "config has enable_moe_block = false")
        let d = config.hiddenSize, eps = config.rmsNormEps
        let e = config.numExperts!, k = config.topKExperts!
        _router.wrappedValue = Gemma4Router(dims: d, numExperts: e, topK: k, eps: eps)
        _experts.wrappedValue = Gemma4Experts(
            dims: d, hiddenDims: config.moeIntermediateSize!, numExperts: e)
        _postFeedforwardLayernorm1.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _postFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _preFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        super.init(config, layer: layer)
    }

    public var expertRouter: any ExpertRouter { router }
    public var expertBank: SwitchGLU { experts.switchGLU }

    /// The sparse branch alone, before `post_feedforward_layernorm_2`.
    public func sparse(_ h: MLXArray) -> MLXArray {
        let (indices, scores) = router.route(h)
        // A pruned layer (kept count near top-k) runs its experts as one wide MLP: see ``mergedExpertBranch``.
        // Decoding a single token reads only the picked experts' weights in the routed path, so there the merged
        // layout is used only when it reads no more (kept == top-k).
        if stats == nil, ExpertCompute.merged, router.numExperts <= ExpertCompute.inferenceKeptCeiling {
            let kept = router.numExperts
            let tokens = indices.size / max(1, indices.dim(-1))
            if tokens >= ExpertCompute.mergedMinTokens || kept <= router.topK {
                if mergedSet?.keep.count != kept { mergedSet = ExpertSet.all(kept) }
                if let set = mergedSet, mergedExpertsApplicable(self, set: set, lora: nil) {
                    return mergedExpertBranch(
                        self, x: preFeedforwardLayernorm2(h), indices: indices, weights: scores, set: set, lora: nil, sliced: true)
                }
            }
        }
        let y = expertBank(preFeedforwardLayernorm2(h), indices)
        stats?.record(indices: indices, scores: scores, expertOutputs: y)
        return (y * expandedDimensions(scores.asType(y.dtype), axis: -1)).sum(axis: -2)
    }

    open override func feedForward(_ h: MLXArray) -> MLXArray {
        let dense = postFeedforwardLayernorm1(mlp(preFeedforwardLayernorm(h)))
        return dense + postFeedforwardLayernorm2(sparse(h))
    }
}

// MARK: - Model

/// Embeddings, decoder stack and final norm. Keys under `model.` in ``Gemma4TextModel``.
open class Gemma4Backbone: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Gemma4DecoderLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    public init(_ config: Gemma4Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4TextModel.makeLayer(config, layer: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

/// The Gemma 4 text model (dense or MoE).
///
/// Module keys follow mlx-lm's sanitised layout (`model.layers.N.…`, `lm_head.weight`
/// when embeddings are untied); ``Gemma4Checkpoint`` maps them to and from the Hugging
/// Face names.
open class Gemma4TextModel: Module {
    public let configuration: Gemma4Configuration
    @ModuleInfo(key: "model") public var model: Gemma4Backbone
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ config: Gemma4Configuration) {
        configuration = config
        _model.wrappedValue = Gemma4Backbone(config)
        _lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    /// The layer type `config` calls for at `layer`.
    public static func makeLayer(_ config: Gemma4Configuration, layer: Int) -> Gemma4DecoderLayer {
        config.enableMoEBlock
            ? Gemma4MoEDecoderLayer(config, layer: layer) : Gemma4DecoderLayer(config, layer: layer)
    }

    public var layers: [Gemma4DecoderLayer] { model.layers }

    /// When set, the model's embedding table is this small working set instead of the whole `embed_tokens`
    /// (which a model loaded for it does not hold): the input side embeds from the window's rows and the
    /// output side scores only the window's tokens, every other token's logit being `-inf`. See
    /// ``VocabularyWindow``. Not a parameter; inference only.
    public var vocabularyWindow: VocabularyWindow?

    /// `embed_tokens(tokens) * sqrt(hidden_size)`: the hidden state entering layer 0.
    public func embed(_ tokens: MLXArray) -> MLXArray {
        let scale = Float(configuration.hiddenSize).squareRoot()
        if let w = vocabularyWindow { return w.embed(tokens) * scale }
        return model.embedTokens(tokens) * scale
    }

    /// Final norm, output projection and logit softcapping applied to the last layer's
    /// output.
    public func logits(fromHidden h: MLXArray) -> MLXArray {
        let normed = model.norm(h)
        func softcap(_ x: MLXArray) -> MLXArray {
            guard let cap = configuration.finalLogitSoftcapping, cap > 0 else { return x }
            return tanh(x / cap) * cap
        }
        // Softcapping runs on the window's scores, before they are spread to the vocabulary: capping the
        // `-inf` fill would turn it into -cap.
        if let w = vocabularyWindow { return w.expand(softcap(w.project(normed))) }
        return softcap(lmHead.map { $0(normed) } ?? model.embedTokens.asLinear(normed))
    }

    /// The scores of just `candidates` for the position(s) in `h`, `[.., candidates.count]`, as ``logits(fromHidden:)``
    /// computes them (norm, projection, softcap). Reads only those rows of the table.
    public func candidateLogits(fromHidden h: MLXArray, candidates: [Int]) throws -> MLXArray {
        let normed = model.norm(h)
        var out: MLXArray
        if let w = vocabularyWindow {
            out = try w.project(normed, onlyTokens: candidates)
        } else if lmHead != nil || candidates.isEmpty {
            throw InferenceError.invalid("candidate scoring needs a tied embedding table")
        } else {
            let ids = MLXArray(candidates.map(Int32.init))
            if let q = model.embedTokens as? QuantizedEmbedding {
                let rows = dequantized(
                    q.weight[ids], scales: q.scales[ids], biases: q.biases.map { $0[ids] }, groupSize: q.groupSize,
                    bits: q.bits, mode: q.mode)
                out = matmul(normed, rows.asType(normed.dtype).T)
            } else {
                out = matmul(normed, model.embedTokens.weight[ids].asType(normed.dtype).T)
            }
        }
        if let cap = configuration.finalLogitSoftcapping, cap > 0 { out = tanh(out / cap) * cap }
        return out
    }

    /// One fresh cache per layer, for incremental decoding.
    public func newCache() -> [Gemma4KVCache] { layers.map { _ in Gemma4KVCache() } }

    /// - Parameters:
    ///   - tokens: `[B, L]` token ids
    ///   - cache: from ``newCache()``; when given, `tokens` continue the cached sequence
    /// - Returns: logits `[B, L, vocab]`
    open func callAsFunction(_ tokens: MLXArray, cache: [Gemma4KVCache]? = nil) -> MLXArray {
        var h = embed(tokens)
        let n = tokens.dim(1)
        let offset = cache?.first?.offset ?? 0
        let sliding = gemma4AttentionMask(n: n, offset: offset, window: configuration.slidingWindow)
        let full = gemma4AttentionMask(n: n, offset: offset)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? sliding : full, cache: cache?[i])
        }
        return logits(fromHidden: h)
    }
}

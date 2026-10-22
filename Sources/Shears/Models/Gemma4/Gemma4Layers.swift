// Ported from ml-explore/mlx-swift-lm, MIT License, Copyright (c) 2024 ml-explore
// (Libraries/MLXLLM/Models/Gemma4Text.swift, Libraries/MLXLMCommon/RoPEUtils.swift,
// Libraries/MLXLMCommon/KVCache.swift). See THIRD_PARTY_NOTICES.md.
//
// Differences from the source: no compiled fusions (plain ops, so everything is safe
// under `valueAndGrad`), no quantized KV cache, no KV sharing, no per-layer embeddings.

import Foundation
import MLX
import MLXNN

// MARK: - Masks and cache

/// Boolean attention mask `[n, offset + n - keyStart]`: query `offset + i` may attend key `j` (an
/// absolute position, counted from `keyStart`) when `j <= offset + i` and, with a window,
/// `offset + i - j < window`. `keyStart` is 0 unless the leading keys are known to be out of reach.
public func gemma4AttentionMask(n: Int, offset: Int = 0, window: Int? = nil, keyStart: Int = 0) -> MLXArray {
    let keys = MLXArray((keyStart ..< (offset + n)).map(Int32.init))[.newAxis]
    let queries = MLXArray((offset ..< (offset + n)).map(Int32.init))[0..., .newAxis]
    var mask = queries .>= keys
    if let window {
        mask = mask & (queries .< keys + Int32(window))
    }
    return mask
}

/// A minimal append-only KV cache for one attention layer. Sliding-window layers keep
/// every key too and rely on the mask, which is simple and fine at evaluation lengths.
public final class Gemma4KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    /// Number of positions already cached.
    public private(set) var offset = 0

    public init() {}

    /// Append `[B, H, L, D]` keys and values and return everything cached so far.
    func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, k], axis: 2)
            self.values = concatenated([values, v], axis: 2)
        } else {
            self.keys = k
            self.values = v
        }
        offset += k.dim(2)
        return (self.keys!, self.values!)
    }
}

// MARK: - RoPE

/// Rotary embedding on the first `partialRotaryFactor` of each half of the head, as
/// Gemma 4's full-attention layers use it (mlx-lm's `ProportionalRoPE`). With a factor
/// of 1 this is ordinary non-traditional RoPE.
final class Gemma4ProportionalRoPE {
    let dims: Int
    let rotatedDims: Int
    let freqs: MLXArray?

    init(dims: Int, base: Float, partialRotaryFactor: Float, factor: Float = 1) {
        self.dims = dims
        let angles = Int(partialRotaryFactor * Float(dims) / 2)
        rotatedDims = 2 * angles
        if rotatedDims > 0 {
            let exponents = MLXArray(Array(stride(from: 0, to: rotatedDims, by: 2)).map(Float.init))
                / Float(dims)
            freqs = factor * MLX.pow(MLXArray(base), exponents)
        } else {
            freqs = nil
        }
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0, let freqs else { return x }
        let half = dims / 2
        let rotatedHalf = rotatedDims / 2

        let head: MLXArray
        let tail: MLXArray?
        if x.dim(-1) > dims {
            let parts = split(x, indices: [dims], axis: -1)
            head = parts[0]
            tail = parts[1]
        } else {
            head = x
            tail = nil
        }
        let halves = split(head, indices: [half], axis: -1)
        let l = split(halves[0], indices: [rotatedHalf], axis: -1)
        let r = split(halves[1], indices: [rotatedHalf], axis: -1)
        var rotated = concatenated([l[0], r[0]], axis: -1)
        rotated = MLXFast.RoPE(
            rotated, dimensions: rotatedDims, traditional: false, base: nil, scale: 1,
            offset: offset, freqs: freqs)
        let rp = split(rotated, indices: [rotatedHalf], axis: -1)
        let updated = concatenated(
            [concatenated([rp[0], l[1]], axis: -1), concatenated([rp[1], r[1]], axis: -1)],
            axis: -1)
        if let tail { return concatenated([updated, tail], axis: -1) }
        return updated
    }
}

// MARK: - Attention

/// Gemma 4 attention: per-head `q_norm` / `k_norm`, a parameter-free `v_norm`, and an
/// attention scale of 1 (the norms take its place).
///
/// Sliding layers use `head_dim`, `num_key_value_heads` and default RoPE over the whole
/// head; full layers use `global_head_dim`, proportional RoPE over part of the head and,
/// with `attention_k_eq_v`, `num_global_key_value_heads` and no `v_proj` (values come
/// from the un-normed keys).
open class Gemma4Attention: Module {
    public let isSliding: Bool
    public let headDim: Int
    public let numHeads: Int
    public let numKVHeads: Int
    let eps: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear?
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    let slidingTheta: Float
    let rope: Gemma4ProportionalRoPE?

    public init(_ config: Gemma4Configuration, layer: Int) {
        isSliding = config.isSliding(layer)
        headDim = isSliding ? config.headDim : config.globalHeadDim
        numHeads = config.numAttentionHeads
        let kEqV = config.attentionKEqV && !isSliding
        numKVHeads = kEqV ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads)
            : config.numKeyValueHeads
        eps = config.rmsNormEps
        let d = config.hiddenSize

        _qProj.wrappedValue = Linear(d, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(d, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = kEqV ? nil : Linear(d, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, d, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)

        slidingTheta = config.slidingRopeTheta
        rope = isSliding ? nil : Gemma4ProportionalRoPE(
            dims: headDim, base: config.fullRopeTheta,
            partialRotaryFactor: config.fullRopeType == "default" ? 1.0 : config.fullPartialRotaryFactor,
            factor: config.fullRopeType == "default" ? 1.0 : config.fullRopeFactor)
        super.init()
    }

    func applyRoPE(_ x: MLXArray, offset: Int) -> MLXArray {
        if let rope { return rope(x, offset: offset) }
        return MLXFast.RoPE(
            x, dimensions: headDim, traditional: false, base: slidingTheta, scale: 1,
            offset: offset)
    }

    /// `rmsNorm` without a learned scale.
    func vNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps)
    }

    /// - Parameters:
    ///   - x: `[B, L, D]`
    ///   - mask: boolean `[L, offset + L]` (see ``gemma4AttentionMask(n:offset:window:)``)
    ///   - cache: appended to, and its offset used as the RoPE position, when given
    open func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: Gemma4KVCache?) -> MLXArray {
        oProj(attended(x, mask: mask, cache: cache))
    }

    /// Queries, keys and values after the norms and RoPE, each `[B, heads, L, headDim]`.
    func projected(_ x: MLXArray, offset: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, numHeads, headDim)).transposed(0, 2, 1, 3)
        q = applyRoPE(q, offset: offset)

        let kRaw = kProj(x).reshaped(b, l, numKVHeads, headDim)
        var k = kNorm(kRaw).transposed(0, 2, 1, 3)
        k = applyRoPE(k, offset: offset)
        let v = vNorm(vProj.map { $0(x).reshaped(b, l, numKVHeads, headDim) } ?? kRaw)
            .transposed(0, 2, 1, 3)
        return (q, k, v)
    }

    /// The attention output before `o_proj`, `[B, L, heads * headDim]`. Exposed so an adapter
    /// on the output projection can see its input.
    open func attended(_ x: MLXArray, mask: MLXArray?, cache: Gemma4KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        var (q, k, v) = projected(x, offset: offset)
        if let cache { (k, v) = cache.update(keys: k, values: v) }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1, mask: mask)
        return out.transposed(0, 2, 1, 3).reshaped(b, l, -1)
    }

    /// Longest block of queries scored against the keys at once by ``attendedWhole(_:window:)``;
    /// 0 turns chunking off.
    ///
    /// MLX's fused attention kernel is not used when a gradient is being traced (Metal's unfused
    /// path is faster there), nor for head dimension 512 (Gemma 4's global layers) or 32, so those
    /// cases score `heads x L x L` at once. Chunking the queries bounds that at
    /// `heads x chunk x L`, and a chunk is rematerialised in the backward pass instead of kept.
    nonisolated(unsafe) public static var queryChunk = 512

    /// Whole-sequence causal attention (from position 0, no cache) with an optional sliding
    /// `window`, equal to ``attended(_:mask:cache:)`` with the corresponding mask.
    ///
    /// Gemma 4 applies no attention softcap and a scale of 1 (the q/k norms take its place), so
    /// `scaledDotProductAttention` expresses it exactly: the causal and windowed mask goes in as
    /// a boolean array, grouped-query attention is handled by the kernel, and the global layers'
    /// keys-equal-values case only changes where `v` comes from. Past `queryChunk` positions, each
    /// block of queries sees only the keys it can attend to (so a sliding layer costs
    /// `L x window`, not `L x L`) and runs under gradient checkpointing.
    public func attendedWhole(_ x: MLXArray, window: Int?) -> MLXArray {
        attendedWholeKeepingKV(x, window: window).out
    }

    /// ``attendedWhole(_:window:)`` that also returns the keys and values it attended over (after the
    /// norms and RoPE, `[B, kvHeads, L, headDim]`), which a later segment of the same sequence can
    /// attend to through ``attended(after:_:window:)``.
    public func attendedWholeKeepingKV(_ x: MLXArray, window: Int?) -> (out: MLXArray, keys: MLXArray, values: MLXArray) {
        let (q, k, v) = projected(x, offset: 0)
        return (scoreChunked(q: q, k: k, v: v, queryOffset: 0, keyStart: 0, window: window), k, v)
    }

    /// Attention for `x` (`[R, S, D]`), the continuation of a sequence whose first `P` positions
    /// are the shared `prefix` (keys and values `[1, kvHeads, P, headDim]`, from
    /// ``attendedWholeKeepingKV(_:window:)`` on the prefix alone). Every row sees the same prefix
    /// keys and its own suffix keys, with positions starting at `P` and, on a sliding layer, exactly
    /// the keys it would see in the whole sequence. Equal to ``attendedWhole(_:window:)`` on
    /// `prefix + x[r]` restricted to the suffix positions, for each row `r`.
    public func attended(after prefix: (keys: MLXArray, values: MLXArray), _ x: MLXArray, window: Int?) -> MLXArray {
        let p = prefix.keys.dim(2)
        let rows = x.dim(0)
        let (q, k, v) = projected(x, offset: p)
        // Prefix keys no suffix query can reach are dropped before they are copied per row.
        let keyStart = window.map { max(0, p - $0 + 1) } ?? 0
        func shared(_ a: MLXArray) -> MLXArray {
            broadcast(a[0..., 0..., keyStart ..< p, 0...], to: [rows, a.dim(1), p - keyStart, a.dim(3)])
        }
        let keys = concatenated([shared(prefix.keys), k], axis: 2)
        let values = concatenated([shared(prefix.values), v], axis: 2)
        return scoreChunked(q: q, k: keys, v: values, queryOffset: p, keyStart: keyStart, window: window)
    }

    /// Causal (and optionally windowed) attention of `q` (`[B, heads, L, headDim]`, absolute positions
    /// `queryOffset ..< queryOffset + L`) over `k` and `v` (`[B, kvHeads, n, headDim]`, absolute
    /// positions `keyStart ..< keyStart + n`), as `[B, L, heads * headDim]`. See
    /// ``attendedWhole(_:window:)`` for the chunking.
    func scoreChunked(
        q: MLXArray, k: MLXArray, v: MLXArray, queryOffset: Int, keyStart: Int, window: Int?
    ) -> MLXArray {
        let (b, l) = (q.dim(0), q.dim(2))
        let chunk = Self.queryChunk
        guard chunk > 0, l > chunk else {
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1,
                mask: gemma4AttentionMask(n: l, offset: queryOffset, window: window, keyStart: keyStart))
            return out.transposed(0, 2, 1, 3).reshaped(b, l, -1)
        }
        var outs: [MLXArray] = []
        for s in stride(from: 0, to: l, by: chunk) {
            let e = min(l, s + chunk)
            let k0 = window.map { max(keyStart, queryOffset + s - $0 + 1) } ?? keyStart
            let queries = MLXArray((queryOffset + s ..< queryOffset + e).map(Int32.init))[0..., .newAxis]
            let keys = MLXArray((k0 ..< queryOffset + e).map(Int32.init))[.newAxis]
            var m = queries .>= keys
            if let window { m = m & (queries .< keys + Int32(window)) }
            let f = checkpoint { (a: [MLXArray]) -> [MLXArray] in
                [MLXFast.scaledDotProductAttention(
                    queries: a[0], keys: a[1], values: a[2], scale: 1, mask: m)]
            }
            let (ks, ke) = (k0 - keyStart, queryOffset + e - keyStart)
            outs.append(f([q[0..., 0..., s ..< e, 0...], k[0..., 0..., ks ..< ke, 0...], v[0..., 0..., ks ..< ke, 0...]])[0])
        }
        return concatenated(outs, axis: 2).transposed(0, 2, 1, 3).reshaped(b, l, -1)
    }
}

// MARK: - Dense MLP

/// `down(gelu_tanh(gate(x)) * up(x))`.
open class Gemma4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(dims: Int, hiddenDims: Int) {
        _gateProj.wrappedValue = Linear(dims, hiddenDims, bias: false)
        _upProj.wrappedValue = Linear(dims, hiddenDims, bias: false)
        _downProj.wrappedValue = Linear(hiddenDims, dims, bias: false)
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE router and experts

/// Gemma 4's expert router.
///
/// `scores = proj(rmsNorm(x, scale / sqrt(D)))`; the top-k are taken over the raw scores,
/// softmaxed among themselves, then multiplied by `per_expert_scale[e]`. Because the
/// softmax runs over the selected experts only, dropping experts a token didn't select
/// leaves its routing weights exactly unchanged.
///
/// Keys: `proj.weight` `[E, D]`, `scale` `[D]`, `per_expert_scale` `[E]`.
open class Gemma4Router: Module, ExpertRouter {
    @ModuleInfo(key: "proj") public var proj: Linear
    @ParameterInfo(key: "scale") public var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") public var perExpertScale: MLXArray

    public var topK: Int
    public let eps: Float

    public init(dims: Int, numExperts: Int, topK: Int, eps: Float = 1e-6) {
        _proj.wrappedValue = Linear(dims, numExperts, bias: false)
        _scale.wrappedValue = MLXArray.ones([dims])
        _perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        self.topK = topK
        self.eps = eps
        super.init()
    }

    public var numExperts: Int { proj.weight.dim(0) }

    /// The router's own normalised input: `rmsNorm(x, scale / sqrt(D))`.
    open func normalizedInput(_ x: MLXArray) -> MLXArray {
        let rootSize = pow(Float(scale.dim(0)), -0.5)
        return MLXFast.rmsNorm(x, weight: (scale * rootSize).asType(x.dtype), eps: eps)
    }

    open func route(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let logits = proj(normalizedInput(x))
        let k = topK
        let indices = stopGradient(topKIndices(logits, k))
        var weights = softmax(takeAlong(logits, indices, axis: -1), axis: -1, precise: true)
        weights = weights * perExpertScale[indices].asType(weights.dtype)
        return (indices, weights)
    }

    open func keepExperts(_ keep: MLXArray) {
        // Every row-indexed tensor of the projection (a quantized one also has scales/biases).
        proj.update(parameters: ModuleParameters.unflattened(
            proj.parameters().flattened().map { ($0.0, $0.1.take(keep, axis: 0)) }))
        update(parameters: ModuleParameters.unflattened(
            [("per_expert_scale", perExpertScale.take(keep, axis: 0))]))
        topK = min(topK, keep.dim(0))
    }
}

/// Holds the expert bank under `switch_glu`, so the decoder layer's keys are
/// `experts.switch_glu.{gate,up,down}_proj.weight` (mlx-lm's sanitised Gemma 4 layout).
open class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") public var switchGLU: SwitchGLU

    public init(dims: Int, hiddenDims: Int, numExperts: Int) {
        _switchGLU.wrappedValue = SwitchGLU(
            inputDims: dims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: geluApproximate)
        super.init()
    }
}

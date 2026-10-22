import MLX
import MLXNN
import MLXRandom

/// Softmax-then-top-k router (the Qwen3-MoE / Mixtral-style gate).
///
/// `gateWeight` is `[E, dims]` and is stored under the key `weight`, so inside a
/// ``MoEBlock`` it loads from `mlp.gate.weight`.
open class TopKRouter: Module, ExpertRouter {
    @ParameterInfo(key: "weight") public var gateWeight: MLXArray

    public var topK: Int
    public var normalizeTopK: Bool

    public init(dims: Int, numExperts: Int, topK: Int, normalizeTopK: Bool = true) {
        let scale = (1 / Float(dims)).squareRoot()
        self._gateWeight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [numExperts, dims])
        self.topK = topK
        self.normalizeTopK = normalizeTopK
        super.init()
    }

    public init(gateWeight: MLXArray, topK: Int, normalizeTopK: Bool = true) {
        self._gateWeight.wrappedValue = gateWeight
        self.topK = topK
        self.normalizeTopK = normalizeTopK
        super.init()
    }

    public var numExperts: Int { gateWeight.dim(0) }

    public func route(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) { self(x) }

    /// - Returns: `indices` `[..., k]` (no gradient) and `scores` `[..., k]` (differentiable
    ///   w.r.t. `gateWeight` and `x`).
    open func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let gates = softmax(matmul(x, gateWeight.T), axis: -1, precise: true)
        let k = topK
        let indices = stopGradient(topKIndices(gates, k))
        var scores = takeAlong(gates, indices, axis: -1)
        if normalizeTopK {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return (indices, scores)
    }

    /// Keep only the router rows in `keep` and clamp `topK` to what remains.
    open func keepExperts(_ keep: MLXArray) {
        update(parameters: ModuleParameters.unflattened(
            [("weight", gateWeight.take(keep, axis: 0))]))
        topK = min(topK, keep.dim(0))
    }
}

/// Router plus expert bank: `sum_k score_k * expert_k(x)`.
///
/// Keys match mlx-lm's Qwen3-MoE sparse block (`gate`, `switch_mlp`), so an
/// `mlp.` prefix maps straight onto a checkpoint.
open class MoEBlock: Module, UnaryLayer, MixtureOfExperts {
    @ModuleInfo(key: "gate") public var router: TopKRouter
    @ModuleInfo(key: "switch_mlp") public var experts: SwitchGLU

    /// Calibration hook. Recording calls `eval`, so leave this `nil` while training.
    public var stats: ExpertStats?

    public init(dims: Int, hiddenDims: Int, numExperts: Int, topK: Int, normalizeTopK: Bool = true) {
        self._router.wrappedValue = TopKRouter(
            dims: dims, numExperts: numExperts, topK: topK, normalizeTopK: normalizeTopK)
        self._experts.wrappedValue = SwitchGLU(
            inputDims: dims, hiddenDims: hiddenDims, numExperts: numExperts)
        super.init()
    }

    public init(router: TopKRouter, experts: SwitchGLU) {
        precondition(router.numExperts == experts.numExperts, "router/expert count mismatch")
        self._router.wrappedValue = router
        self._experts.wrappedValue = experts
        super.init()
    }

    public var numExperts: Int { router.numExperts }
    public var expertRouter: any ExpertRouter { router }
    public var expertBank: SwitchGLU { experts }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = router(x)
        let y = experts(x, indices)
        stats?.record(indices: indices, scores: scores, expertOutputs: y)
        return (y * expandedDimensions(scores, axis: -1)).sum(axis: -2)
    }
}

/// A pre-norm residual MoE layer, `x + moe(rmsNorm(x))`, for building toy models
/// and as the unit block-local healing operates on.
open class MoEResidualLayer: Module, UnaryLayer {
    @ModuleInfo(key: "post_attention_layernorm") public var norm: RMSNorm
    @ModuleInfo(key: "mlp") public var moe: MoEBlock

    public init(dims: Int, hiddenDims: Int, numExperts: Int, topK: Int) {
        self._norm.wrappedValue = RMSNorm(dimensions: dims)
        self._moe.wrappedValue = MoEBlock(
            dims: dims, hiddenDims: hiddenDims, numExperts: numExperts, topK: topK)
        super.init()
    }

    public init(norm: RMSNorm, moe: MoEBlock) {
        self._norm.wrappedValue = norm
        self._moe.wrappedValue = moe
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + moe(norm(x))
    }
}

extension Module {
    /// Copy this module's parameters (new arrays, not shared references) into `other`,
    /// which must have the same structure. Used to make a student from a teacher before
    /// pruning one of them in place.
    public func copyParameters(into other: Module) {
        let copied = parameters().flattened().map { ($0.0, $0.1 * 1) }
        other.update(parameters: ModuleParameters.unflattened(copied))
        eval(other)
    }
}

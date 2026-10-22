import MLX
import MLXNN

/// A router that picks `topK` of `numExperts` experts per token.
///
/// ``TopKRouter`` (softmax, then top-k) and ``Gemma4Router`` (top-k over raw logits, then
/// softmax, then a per-expert scale) both conform. Everything in Shears that needs to
/// know about routing (``ExpertStats``, ``ExpertPruner``, the ``MoETraining`` freezing
/// helpers) goes through this protocol, so a new router type only has to implement it.
public protocol ExpertRouter: Module {
    var numExperts: Int { get }
    var topK: Int { get }

    /// - Returns: `indices` `[..., k]` (no gradient) and `scores` `[..., k]`, the weights
    ///   applied to each selected expert's output (differentiable w.r.t. the router's
    ///   parameters and `x`).
    func route(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray)

    /// Keep only the experts in `keep` (sorted int32 ids): slice every per-expert
    /// parameter along its expert axis, and clamp `topK` to what remains.
    func keepExperts(_ keep: MLXArray)
}

/// A router plus the bank of experts it selects from.
///
/// ``MoEBlock`` conforms, as does ``Gemma4MoEDecoderLayer``, whose router and experts
/// sit directly in the decoder layer and read different inputs. ``ExpertPruner``,
/// ``ExpertQuantizer`` and ``MoETraining`` work on any conforming type.
public protocol MixtureOfExperts: AnyObject {
    var expertRouter: any ExpertRouter { get }
    var expertBank: SwitchGLU { get }
    /// Calibration hook. Recording calls `eval`, so leave this `nil` while training.
    var stats: ExpertStats? { get set }
}

extension MixtureOfExperts {
    public var numExperts: Int { expertRouter.numExperts }
}

/// The indices of the `k` largest entries along the last axis, in the form mlx-lm routes with:
/// `argpartition(x, kth=-k)[..., -k:]`. Which of several equal scores is picked (and the order of
/// the picked ones) depends on the partition's form, so routers here use this one rather than
/// partitioning the negated scores: ties then break as they do in the Python reference.
///
/// `kth` is passed as `n - k`, not negative, so nothing relies on the binding normalising it.
public func topKIndices(_ x: MLXArray, _ k: Int) -> MLXArray {
    let n = x.dim(-1)
    return argPartition(x, kth: n - k, axis: -1)[.ellipsis, (n - k)...]
}

import Foundation
import MLX
import MLXNN

/// How closely a student's routing follows the original router's, on one set of rows.
///
/// The "original routing" is the uncorrected router's softmax over the kept experts, applied to the student's own
/// router input; the "corrected routing" adds the router correction's logits first. So the numbers measure exactly
/// what the router correction changed, and are zero / one for a block whose correction is still zero.
public struct RoutingAgreement: Sendable, Equatable, Codable {
    /// Mean `KL(original routing || corrected routing)` per position over the kept experts, pin-weighted.
    public var kl: Float
    /// Mean fraction of each position's top-k experts that the correction leaves unchanged, pin-weighted.
    public var topKOverlap: Float
    /// Mean squared size of the correction on the kept experts' logits, pin-weighted.
    public var logitDeltaSquared: Float
    public init(kl: Float, topKOverlap: Float, logitDeltaSquared: Float) {
        self.kl = kl
        self.topKOverlap = topKOverlap
        self.logitDeltaSquared = logitDeltaSquared
    }
}

/// Channels of the per-position routing tensor `[..., 3]`.
enum RoutingChannel {
    static let kl = 0, l2 = 1, overlap = 2, count = 3
}

/// `[..., 3]` (float32): the KL, the squared size and the top-k overlap of the router correction at each position.
/// `scoresKept` are the uncorrected logits of the kept experts `[..., K]`; `keep` the kept ids (to gather the
/// correction's columns of `b`, which spans every expert). Nothing here is differentiated.
func routingChannels(
    scoresKept: MLXArray, routerInput: MLXArray, tweak: RouterTweak, keep: MLXArray, topK: Int
) -> MLXArray {
    let s0 = stopGradient(scoresKept.asType(.float32))
    let rin = stopGradient(routerInput.asType(.float32))
    let corr = matmul(matmul(rin, tweak.a), tweak.b.take(keep, axis: 1))
    let s1 = s0 + corr
    let kl = HealingLoss.kl(teacherLogProbs: HealingLoss.logProbs(s0), studentLogProbs: HealingLoss.logProbs(s1))
    let l2 = square(corr).mean(axis: -1)
    let k = min(topK, s0.dim(-1))
    let a = topKIndices(s0, k).expandedDimensions(axis: -1)
    let b = topKIndices(s1, k).expandedDimensions(axis: -2)
    let shared = (a .== b).any(axis: -1).asType(.float32).sum(axis: -1)
    return stacked([kl, l2, shared / Float(k)], axis: -1)
}

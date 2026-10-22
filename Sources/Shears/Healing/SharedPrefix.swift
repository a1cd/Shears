import Foundation
import MLX
import MLXNN

// Shared-prefix forward passes.
//
// Many training rows begin with the same tokens (a long common prompt) and differ in a suffix.
// Attention is causal, so every layer's hidden state at a prefix position is the same for all of
// those rows. These functions compute the prefix once and let each row's suffix attend to the
// prefix's keys and values, which is exactly what running every row whole computes, for the cost
// of the prefix once plus the suffixes.
//
// Nothing here is a special gradient: the prefix keys and values are ordinary arrays in the same
// graph, shared by every suffix, so a loss summed over the rows differentiates to the sum of the
// whole rows' gradients (the prefix's K/V receive the sum of every row's cotangent).

/// One decoder layer over a shared prefix and `R` suffixes.
///
/// For every row `r`, `concatenated([prefix, suffix[r]])` run through ``healingLayerForward(_:_:set:adapters:sliced:)``
/// equals `concatenated([prefixOut, suffixOut[r]])`, up to float rounding. The teacher is `set` and
/// `adapters` both `nil`. A layer that does not ``HealableMoELayer/supportsSharedPrefix`` runs each
/// row whole instead (correct, but with no saving).
///
/// - Parameters:
///   - prefix: `[1, P, D]`, the layer input at the shared positions `0 ..< P`.
///   - suffix: `[R, S, D]`, the layer input at positions `P ..< P + S` of each row.
/// - Returns: the layer output at the prefix, `[1, P, D]`, and at each suffix, `[R, S, D]`.
func healingLayerForwardShared(
    _ layer: any HealableMoELayer, prefix: MLXArray, suffix: MLXArray, set: ExpertSet?,
    adapters: LayerAdapters?, sliced: Bool = false
) -> (prefix: MLXArray, suffix: MLXArray) {
    precondition(prefix.ndim == 3 && prefix.dim(0) == 1, "the shared prefix is [1, P, D]")
    precondition(suffix.ndim == 3 && suffix.dim(2) == prefix.dim(2), "suffixes are [R, S, D] with the prefix's D")
    let (p, rows) = (prefix.dim(1), suffix.dim(0))
    guard layer.supportsSharedPrefix else {
        let whole = concatenated([broadcast(prefix, to: [rows, p, prefix.dim(2)]), suffix], axis: 1)
        let out = healingLayerForward(layer, whole, set: set, adapters: adapters, sliced: sliced)
        return (out[0 ..< 1, 0 ..< p, 0...], out[0..., p..., 0...])
    }
    healingLayerForwardCount += 1
    let (prefixParts, kv) = layer.prefixAttentionPart(prefix, adapters: adapters)
    let suffixParts = layer.suffixAttentionPart(suffix, after: kv, adapters: adapters)
    return (
        healingLayerRemainder(layer, prefixParts, set: set, adapters: adapters, sliced: sliced),
        healingLayerRemainder(layer, suffixParts, set: set, adapters: adapters, sliced: sliced))
}

extension HealingNetwork {
    /// ``forward(_:from:through:student:collect:checkpointed:keptLayers:detachBefore:)`` for rows that share a
    /// prefix: layers `from...through` (model layer indices) run once over `prefix` (`[1, P, D]`,
    /// from position 0) and over `suffix` (`[R, S, D]`, positions `P ...` of each row), the suffixes
    /// attending to the prefix. Row `r` of the result equals running
    /// `concatenated([prefix, suffix[r]])` through ``forward(_:from:through:student:collect:checkpointed:keptLayers:detachBefore:)``.
    ///
    /// - Returns: the final prefix and suffix states, and for each layer in `collect` its output.
    public func forwardShared(
        prefix p0: MLXArray, suffix s0: MLXArray, from: Int? = nil, through: Int? = nil, student: Bool,
        collect: Set<Int> = [], checkpointed: Bool = false, keptLayers: Int = 0, detachBefore: Int? = nil
    ) -> (prefix: MLXArray, suffix: MLXArray, collected: [Int: (prefix: MLXArray, suffix: MLXArray)]) {
        var (p, s) = (p0, s0)
        var out: [Int: (prefix: MLXArray, suffix: MLXArray)] = [:]
        let (pShape, sShape) = (p0.shape, s0.shape)
        let pCount = p0.size
        let lastRun = through ?? lastLayer
        for L in (from ?? firstLayer) ... lastRun {
            let i = L - firstLayer
            let layer = layers[i]
            if detachBefore == L { (p, s) = (stopGradient(p), stopGradient(s)) }
            if student {
                let set = sets[i]
                if let ad = adapters {
                    let la = ad.layers[i]
                    if checkpointed, L <= lastRun - keptLayers, detachBefore == nil || L >= detachBefore! {
                        let sl = sliced
                        // `checkpoint` takes one array: pack both states into it and unpack inside.
                        let packed = checkpoint(la) { (m: LayerAdapters, x: MLXArray) in
                            let pp = x[0 ..< pCount].reshaped(pShape)
                            let ss = x[pCount...].reshaped(sShape)
                            let r = healingLayerForwardShared(layer, prefix: pp, suffix: ss, set: set, adapters: m, sliced: sl)
                            return concatenated([r.prefix.reshaped(-1), r.suffix.reshaped(-1)])
                        }(concatenated([p.reshaped(-1), s.reshaped(-1)]))
                        p = packed[0 ..< pCount].reshaped(pShape)
                        s = packed[pCount...].reshaped(sShape)
                    } else {
                        (p, s) = healingLayerForwardShared(layer, prefix: p, suffix: s, set: set, adapters: la, sliced: sliced)
                    }
                } else {
                    (p, s) = healingLayerForwardShared(layer, prefix: p, suffix: s, set: set, adapters: nil, sliced: sliced)
                }
            } else {
                precondition(!sliced, "a sliced student holds only its kept experts and cannot run the teacher")
                (p, s) = healingLayerForwardShared(layer, prefix: p, suffix: s, set: nil, adapters: nil)
            }
            if collect.contains(L) { out[L] = (p, s) }
        }
        return (p, s, out)
    }
}

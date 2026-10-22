import Foundation
import MLX
import MLXNN

/// Settings for how the expert branch is computed. Each is part of a compiled step's signature
/// (see ``CompiledLayerStep/signature(kept:sliced:)``), so changing one between runs re-traces.
public enum ExpertCompute {
    /// Use the merged layout (see ``mergedExpertBranch(_:x:indices:weights:set:lora:sliced:)``) whenever it applies.
    nonisolated(unsafe) public static var merged = true
    /// The merged layout is used when the kept count is at most this multiple of the router's top-k
    /// (its work grows with the kept count, the routed path's with top-k), default 1.5. 0 turns it off,
    /// a large value forces it.
    nonisolated(unsafe) public static var mergedMaxKeptRatio: Double = 1.5
    /// How many tokens the expert math processes at once; `nil` is all of them. Results are identical (the math is
    /// per token); a smaller chunk lowers the working memory of a step (each chunk is recomputed in the backward
    /// pass instead of kept) at some cost in speed. The block pipeline sets it from the memory budget.
    nonisolated(unsafe) public static var tokenChunk: Int?
    /// Inference (``Gemma4MoEDecoderLayer/sparse(_:)``) uses the merged layout only for layers with at most this many
    /// experts, so a full unpruned model never pays for the check.
    nonisolated(unsafe) public static var inferenceKeptCeiling = 64
    /// Inference runs the merged layout from this many tokens in a call (a prompt, a batch); a single decoded token
    /// stays routed unless the kept count equals top-k.
    nonisolated(unsafe) public static var mergedMinTokens = 16
    /// Counts the expert branches built in the merged layout (for tests).
    nonisolated(unsafe) static var mergedBranchCount = 0

    static var signature: String { "\(merged ? "m" : "r")\(mergedMaxKeptRatio)c\(tokenChunk ?? 0)" }
}

/// Whether the merged layout can compute this layer's expert branch.
func mergedExpertsApplicable(
    _ layer: any HealableMoELayer, set: ExpertSet, lora: ExpertLoRA?
) -> Bool {
    guard ExpertCompute.merged else { return false }
    let k = set.keep.count
    guard Double(k) <= ExpertCompute.mergedMaxKeptRatio * Double(layer.expertRouter.topK) else { return false }
    if let lora, lora.gateUpFormat != nil || lora.downFormat != nil { return false }
    let bank = layer.expertBank
    for p in [bank.gateProj, bank.upProj, bank.downProj] {
        if p.bias != nil { return false }
        if let q = p as? QuantizedSwitchLinear {
            if q.hasLoRA { return false }
            // Down's packed rows are re-laid K-major: each expert's input width must be whole groups.
            if q.inputDims % q.groupSize != 0 { return false }
        }
    }
    return true
}

/// The kept experts' gate (or up) weights stacked into one `[K * out, in]` projection applied to `x`.
private func wideInput(_ p: SwitchLinear, _ x: MLXArray, keep: MLXArray?) -> MLXArray {
    if let q = p as? QuantizedSwitchLinear {
        var (w, s, b) = (q.weight, q.scales, q.biases)
        if let keep {
            w = w.take(keep, axis: 0)
            s = s.take(keep, axis: 0)
            b = b?.take(keep, axis: 0)
        }
        let rows = w.dim(0) * w.dim(1)
        return quantizedMM(
            x, w.reshaped(rows, -1), scales: s.reshaped(rows, -1), biases: b?.reshaped(rows, -1),
            transpose: true, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
    }
    var w = p.weight
    if let keep { w = w.take(keep, axis: 0) }
    return matmul(x, w.reshaped(w.dim(0) * w.dim(1), -1).T)
}

/// The kept experts' down weights as one `[out, K * in]` projection applied to `x` (`[..., K * in]`).
private func wideOutput(_ p: SwitchLinear, _ x: MLXArray, keep: MLXArray?) -> MLXArray {
    if let q = p as? QuantizedSwitchLinear {
        var (w, s, b) = (q.weight, q.scales, q.biases)
        if let keep {
            w = w.take(keep, axis: 0)
            s = s.take(keep, axis: 0)
            b = b?.take(keep, axis: 0)
        }
        let out = w.dim(1)
        func joined(_ a: MLXArray) -> MLXArray { a.transposed(1, 0, 2).reshaped(out, -1) }
        return quantizedMM(
            x, joined(w), scales: joined(s), biases: b.map(joined), transpose: true, groupSize: q.groupSize,
            bits: q.bits, mode: q.mode)
    }
    var w = p.weight
    if let keep { w = w.take(keep, axis: 0) }
    return matmul(x, w.transposed(1, 0, 2).reshaped(w.dim(1), -1).T)
}

/// The expert branch `sum_j w_j * expert_j(x)` as one wide MLP over all `K` kept experts.
///
/// Gate and up of the kept experts are stacked into `hidden -> K * width` projections, the activation and product
/// run on the whole wide hidden layer, each expert's chunk of it is scaled by that expert's routing weight (zero
/// for an expert the token did not pick, so it contributes exactly nothing and gets exactly zero gradient), and the
/// down projections are stacked into one `K * width -> hidden` projection: three dense matmuls with no gather,
/// sort or unsort. Equal to the routed path (``expertBranch(_:x:indices:weights:set:lora:sliced:)`` with the
/// merged layout off) up to float rounding, in forward and gradients. A per-expert LoRA runs batched over the
/// expert axis; a shared one is two plain low-rank products on the wide layers. The LoRA delta is added in
/// float32 as in the routed path.
func mergedExpertBranch(
    _ layer: any HealableMoELayer, x: MLXArray, indices: MLXArray, weights: MLXArray, set: ExpertSet,
    lora: ExpertLoRA?, sliced: Bool
) -> MLXArray {
    ExpertCompute.mergedBranchCount += 1
    let bank = layer.expertBank
    let kept = set.keep.count
    let d = x.dim(-1)
    let lead = Array(x.shape.dropLast())
    let slot = stopGradient(sliced ? indices : set.slotTable[indices])
    // Weight of each kept slot per token [..., K]: the routing weight where the token picked it, else 0.
    let picked = expandedDimensions(slot, axis: -1) .== MLXArray(Array(0 ..< Int32(kept)))
    let wSlot = (picked.asType(weights.dtype) * expandedDimensions(weights, axis: -1)).sum(axis: -2).asType(x.dtype)
    let n = lead.reduce(1, *)
    let x2 = x.reshaped(n, d)
    let w2 = wSlot.reshaped(n, kept)
    let keepIds: MLXArray? = sliced ? nil : set.keepArray
    let h = bank.gateProj.outputDims
    let scale = lora?.scale ?? 0
    let loraArrays: [MLXArray] = lora.map { l in [l.gateA, l.gateB, l.upA, l.upB, l.downA, l.downB] } ?? []
    let shared = lora?.shared ?? false

    func core(_ x: MLXArray, _ w: MLXArray, _ la: [MLXArray]) -> MLXArray {
        let m = x.dim(0)
        var g = wideInput(bank.gateProj, x, keep: keepIds)
        var u = wideInput(bank.upProj, x, keep: keepIds)
        func lowRankIn(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            if shared {
                return matmul(matmul(x, a.asType(x.dtype)).asType(.float32), b)
            }
            let r = a.dim(2)
            let aFlat = a.transposed(1, 0, 2).reshaped(d, kept * r)
            let xa = matmul(x, aFlat.asType(x.dtype)).asType(.float32).reshaped(m, kept, r).transposed(1, 0, 2)
            return matmul(xa, b).transposed(1, 0, 2).reshaped(m, kept * h)
        }
        func add(_ base: MLXArray, _ delta: MLXArray) -> MLXArray {
            (base.asType(.float32) + scale * delta).asType(base.dtype)
        }
        if !la.isEmpty {
            g = add(g, lowRankIn(la[0], la[1]))
            u = add(u, lowRankIn(la[2], la[3]))
        }
        let hidden = bank.activation(g) * u
        let hw = (hidden.reshaped(m, kept, h) * expandedDimensions(w, axis: -1)).reshaped(m, kept * h)
        var out = wideOutput(bank.downProj, hw, keep: keepIds)
        if !la.isEmpty {
            let delta: MLXArray
            if shared {
                delta = matmul(matmul(hw, la[4].asType(hw.dtype)).asType(.float32), la[5])
            } else {
                let hk = hw.reshaped(m, kept, h).transposed(1, 0, 2)
                delta = matmul(matmul(hk, la[4].asType(hw.dtype)).asType(.float32), la[5]).sum(axis: 0)
            }
            out = add(out, delta)
        }
        return out
    }

    let result: MLXArray
    if let chunk = ExpertCompute.tokenChunk, chunk > 0, n > chunk {
        let step = checkpoint { (a: [MLXArray]) -> [MLXArray] in [core(a[0], a[1], Array(a[2...]))] }
        var parts: [MLXArray] = []
        var s = 0
        while s < n {
            let e = min(n, s + chunk)
            parts.append(step([x2[s ..< e], w2[s ..< e]] + loraArrays)[0])
            s = e
        }
        result = concatenated(parts, axis: 0)
    } else {
        result = core(x2, w2, loraArrays)
    }
    return result.reshaped(lead + [d])
}

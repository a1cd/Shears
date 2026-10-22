import Foundation
import MLX

/// The loss and metrics of healing.
public enum HealingLoss {
    /// Position-weighted, per-dimension-normalised distance between hidden states.
    ///
    /// For each pinned layer, with teacher state `r` `[1, T, D]`:
    /// `scale = max(sqrt(mean over positions of r^2), 1e-3)` per dimension, then
    /// `sum_t w_t * mean_d(((h - r) / scale)^2) / max(sum_t w_t, 1)`, averaged over the pinned
    /// layers. The per-dimension scale keeps a few huge-magnitude outlier dimensions from
    /// dominating, which raw MSE would let them do. The `max(sum w, 1)` floor is the recipe's:
    /// weights summing to less than one are not renormalised.
    public static func pin(
        _ pinned: [Int: MLXArray], teacher ref: [Int: MLXArray], weights w: MLXArray
    ) -> MLXArray {
        var total = MLXArray(Float(0))
        for (layer, h) in pinned {
            total = total + pinTerm(h, ref[layer]!, w)
        }
        return total / Float(max(1, pinned.count))
    }

    /// ``pin(_:teacher:weights:)`` for states with a batch axis `[N, T, D]` against one teacher state: the
    /// mean over the batch of each element's pin term (``pin`` itself reads element 0 only). With `N == 1`
    /// it is ``pin`` exactly.
    public static func pinBatched(
        _ pinned: [Int: MLXArray], teacher ref: [Int: MLXArray], weights w: MLXArray
    ) -> MLXArray {
        var total = MLXArray(Float(0))
        for (layer, h) in pinned {
            let r = ref[layer]!
            let n = h.dim(0)
            if n == 1 { total = total + pinTerm(h, r, w); continue }
            var sum = MLXArray(Float(0))
            for i in 0 ..< n { sum = sum + pinTerm(h[i ..< i + 1], r, w) }
            total = total + sum / Float(n)
        }
        return total / Float(max(1, pinned.count))
    }

    /// ``pinBatched(_:teacher:weights:)`` for states right-padded with zeros to a longer length than the row's
    /// own (the padding a compiled step pads to, so one compiled graph serves rows of several lengths).
    /// `h`, the teacher state and `w` are `[N, B, D]`, `[1, B, D]` and `[B]`, zero beyond the row's
    /// `length` (a float scalar array: an input of the compiled graph, not a constant of it). The result
    /// equals ``pinBatched`` on the unpadded row: the padded positions carry weight zero, and the teacher's
    /// per-dimension scale averages over the row's true `length` and not over `B`.
    public static func pinBatchedMasked(
        _ pinned: [Int: MLXArray], teacher ref: [Int: MLXArray], weights w: MLXArray, length: MLXArray
    ) -> MLXArray {
        var total = MLXArray(Float(0))
        for (layer, h) in pinned {
            let r = ref[layer]!
            let n = h.dim(0)
            if n == 1 { total = total + pinTermMasked(h, r, w, length); continue }
            var sum = MLXArray(Float(0))
            for i in 0 ..< n { sum = sum + pinTermMasked(h[i ..< i + 1], r, w, length) }
            total = total + sum / Float(n)
        }
        return total / Float(max(1, pinned.count))
    }

    /// The pin loss of a step that stacks several rows, each with `copies` inputs (clean and dirty) against the
    /// row's one teacher state: `h` is `[rows * copies, T, D]`, row `i`'s inputs at `i * copies ..< (i + 1) * copies`,
    /// right-padded to `T`. Each element is scored on its row's own positions only (causal attention keeps those
    /// exact under padding) and the loss is the mean over all of them. With one row and one copy it is
    /// ``pin(_:teacher:weights:)`` exactly.
    public static func pinStacked(
        _ h: MLXArray, teachers: [MLXArray], weights: [MLXArray], copies: Int
    ) -> MLXArray {
        var total = MLXArray(Float(0))
        for (i, r) in teachers.enumerated() {
            for c in 0 ..< copies {
                let j = i * copies + c
                total = total + pinTerm(h[j ..< j + 1, 0 ..< r.dim(1)], r, weights[i])
            }
        }
        return total / Float(max(1, teachers.count * copies))
    }

    /// The same quantity, per layer and as plain numbers (for reconstruction-error tables).
    public static func pinErrors(
        _ pinned: [Int: MLXArray], teacher ref: [Int: MLXArray], weights w: MLXArray
    ) -> [Int: Float] {
        var out: [Int: Float] = [:]
        for (layer, h) in pinned { out[layer] = pinTerm(h, ref[layer]!, w).item(Float.self) }
        return out
    }

    static func pinTerm(_ h: MLXArray, _ r: MLXArray, _ w: MLXArray) -> MLXArray {
        let scale = maximum(sqrt((r * r).mean(axis: 1, keepDims: true)), 1e-3)
        let d = square((h - r) / scale)
        return (d.mean(axis: -1)[0] * w).sum() / maximum(w.sum(), 1)
    }

    /// ``pinTerm`` over a row padded to `B` positions (zeros past `length`, weights zero there).
    static func pinTermMasked(_ h: MLXArray, _ r: MLXArray, _ w: MLXArray, _ length: MLXArray) -> MLXArray {
        // The mean over the true positions: the padded ones are zero in `r`, so the sum is the row's, and the
        // division is by the row's length. Summed in float32 (a half-precision sum of squares could overflow
        // where the mean it replaces does not) and cast back to the state's dtype, as the mean is.
        let meanSquare = ((r * r).asType(.float32).sum(axis: 1, keepDims: true) / length.asType(.float32)).asType(r.dtype)
        let scale = maximum(sqrt(meanSquare), 1e-3)
        let d = square((h - r) / scale)
        return (d.mean(axis: -1)[0] * w).sum() / maximum(w.sum(), 1)
    }

    /// `KL(teacher || student)` over the last axis, from log-probabilities.
    public static func kl(teacherLogProbs t: MLXArray, studentLogProbs s: MLXArray) -> MLXArray {
        (exp(t) * (t - s)).sum(axis: -1)
    }

    /// Log-softmax in float32.
    public static func logProbs(_ logits: MLXArray) -> MLXArray {
        let l = logits.asType(.float32)
        return l - logSumExp(l, axis: -1, keepDims: true)
    }

    /// Position weights that sit at `floor` early in the sequence and ramp linearly to 1 over the
    /// `rampLength` positions before `boundary`, staying at 1 from `boundary` on. The recipe's
    /// defaults are floor 0.15 and ramp 64, with `boundary` at the end of the prompt.
    public static func rampedWeights(
        count: Int, boundary: Int, floor: Float = 0.15, rampLength: Int = 64
    ) -> MLXArray {
        let idx = MLXArray.arange(count, dtype: .float32)
        let ramp = clip(
            (idx - Float(boundary - rampLength)) / Float(max(1, rampLength)), min: 0, max: 1)
        return floor + (1 - floor) * ramp
    }
}

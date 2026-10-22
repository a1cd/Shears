import Foundation
import MLX
import MLXNN

/// Warmup then cosine decay, as `mlx.optimizers` composes them:
/// `join_schedules([linear_schedule(0, peak, warmup), cosine_decay(peak, max(1, total - warmup), floor)], [warmup])`.
///
/// Evaluated at the optimizer's update count *before* it increments, so the first update runs
/// at 0 when there is a warmup.
///
/// The schedule is independent of how many steps a run takes: `total` is the schedule's own
/// length, and `offset` is where on the schedule the run's first update sits, so a run can
/// continue an earlier run's schedule (`offset` = the updates already made), or start a new
/// schedule over the same adapters (offset 0, a new peak and length). `lengthChanges` re-length
/// the schedule part-way, as a run that edits its schedule length mid-way does: from schedule
/// position `from` on, the value is the one a schedule of the new length has at that position.
public struct LearningRateSchedule: Sendable, Equatable, Codable {
    public struct LengthChange: Sendable, Equatable, Codable {
        public var from: Int
        public var total: Int
        public init(from: Int, total: Int) {
            self.from = from
            self.total = total
        }
    }

    public var peak: Float
    public var floor: Float
    public var warmup: Int
    public var total: Int
    public var offset: Int
    public var lengthChanges: [LengthChange]

    public init(
        peak: Float, floor: Float, warmup: Int, total: Int, offset: Int = 0,
        lengthChanges: [LengthChange] = []
    ) {
        self.peak = peak
        self.floor = floor
        self.warmup = warmup
        self.total = total
        self.offset = offset
        self.lengthChanges = lengthChanges.sorted { $0.from < $1.from }
    }

    /// The learning rate of the update made when the optimizer has applied `step` updates in this
    /// run (the schedule position is `offset + step`).
    public func value(at step: Int) -> Float {
        let position = step + offset
        var length = total
        for c in lengthChanges where position >= c.from { length = c.total }
        if warmup > 0, position < warmup {
            return Float(position) * (peak / Float(warmup))
        }
        let decaySteps = max(1, length - warmup)
        let s = Float(min(position - warmup, decaySteps))
        let decay = 0.5 * (1 + cos(Float.pi / Float(decaySteps) * s))
        return floor + decay * (peak - floor)
    }
}

/// AdamW as `mlx.optimizers.AdamW` defines it, with state that can be saved and restored:
/// no bias correction, decoupled decay applied as `p * (1 - lr * wd)` before the Adam step,
/// betas (0.9, 0.999), eps 1e-8.
public final class HealingAdamW {
    public var weightDecay: Float
    public var betas: (Float, Float) = (0.9, 0.999)
    public var eps: Float = 1e-8
    /// Multiplies the learning rate of parameters under a router correction (`.router.` in their names).
    public var routerLRMultiplier: Float = 1
    /// Number of updates applied so far; the schedule is read at this count.
    public private(set) var step = 0
    private var m: [String: MLXArray] = [:]
    private var v: [String: MLXArray] = [:]

    public init(weightDecay: Float = 0.01) { self.weightDecay = weightDecay }

    /// Run the per-layer update (``update(layers:indices:grads:learningRate:)``) as one compiled, fused kernel
    /// per tensor instead of about ten dispatches. The same arithmetic in the same order (the fused kernel may
    /// contract multiply-adds, so results agree to float rounding, tested). On by default.
    nonisolated(unsafe) public static var compileStep = true

    /// `[param, m, v, grad, lr, decay]` to `[param', m', v']`; betas and epsilon are fixed when it is first used.
    private lazy var fused: ([MLXArray]) -> [MLXArray] = {
        let (b1, b2, eps) = (betas.0, betas.1, self.eps)
        return compile { (a: [MLXArray]) -> [MLXArray] in
            let (p, m0, v0, g, lr, decay) = (a[0], a[1], a[2], a[3], a[4], a[5])
            let m1 = b1 * m0 + (1 - b1) * g
            let v1 = b2 * v0 + (1 - b2) * square(g)
            return [p * decay - lr * m1 / (sqrt(v1) + eps), m1, v1]
        }
    }()

    /// Applies one update to `model` from `grads` (keys are the flattened parameter names).
    ///
    /// With `advanceStep: false` the update count is left alone, for an update made in pieces (a
    /// streamed step applies each layer window's gradients as they arrive); the caller then calls
    /// ``completeStep()`` once. `grads` may cover only part of the model.
    public func update(model: Module, grads: [String: MLXArray], learningRate lr: Float, advanceStep: Bool = true) {
        let params = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        var out: [(String, MLXArray)] = []
        let baseRate = lr
        for (k, g) in grads {
            guard let p = params[k] else { continue }
            let lr = rate(baseRate, key: k)
            let m0 = m[k] ?? MLXArray.zeros(like: g)
            let v0 = v[k] ?? MLXArray.zeros(like: g)
            let m1 = betas.0 * m0 + (1 - betas.0) * g
            let v1 = betas.1 * v0 + (1 - betas.1) * square(g)
            m[k] = m1
            v[k] = v1
            let decayed = p * (1 - lr * weightDecay)
            out.append((k, decayed - lr * m1 / (sqrt(v1) + eps)))
        }
        model.update(parameters: ModuleParameters.unflattened(out))
        if advanceStep { step += 1 }
    }

    /// The update of ``update(model:grads:learningRate:advanceStep:)`` for the layers `indices` of `adapters`
    /// alone (`grads` keyed `layers.<i>.…`), touching only those layers' parameters and moments, and
    /// returning the arrays to `eval`. The update count is not advanced (call ``completeStep()``). Same math
    /// as the whole-model call, without walking and rewriting the other layers' trees.
    public func update(layers adapters: ModelAdapters, indices: [Int], grads: [String: MLXArray], learningRate lr: Float) -> [MLXArray] {
        var touched: [MLXArray] = []
        let baseRate = lr
        for i in indices {
            let la = adapters.layers[i]
            let prefix = "layers.\(i)."
            var out: [(String, MLXArray)] = []
            for (local, p) in la.trainableParameters().flattened() {
                let k = prefix + local
                guard let g = grads[k] else { continue }
                let lr = rate(baseRate, key: k)
                let m0 = m[k] ?? MLXArray.zeros(like: g), v0 = v[k] ?? MLXArray.zeros(like: g)
                let r = Self.compileStep
                    ? fused([p, m0, v0, g, MLXArray(lr), MLXArray(1 - lr * weightDecay)])
                    : {
                        let m1 = betas.0 * m0 + (1 - betas.0) * g
                        let v1 = betas.1 * v0 + (1 - betas.1) * square(g)
                        return [p * (1 - lr * weightDecay) - lr * m1 / (sqrt(v1) + eps), m1, v1]
                    }()
                m[k] = r[1]
                v[k] = r[2]
                out.append((local, r[0]))
                touched.append(contentsOf: r)
            }
            la.update(parameters: ModuleParameters.unflattened(out))
        }
        return touched
    }

    private func rate(_ lr: Float, key: String) -> Float {
        routerLRMultiplier == 1 || !key.contains(".router.") ? lr : lr * routerLRMultiplier
    }

    /// Counts one update made with `advanceStep: false`.
    public func completeStep() { step += 1 }

    /// Arrays to `eval` after an update.
    public var stateArrays: [MLXArray] { Array(m.values) + Array(v.values) }

    public func stateTensors() -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, a) in m { out["m." + k] = a }
        for (k, a) in v { out["v." + k] = a }
        return out
    }

    public func restore(tensors: [String: MLXArray], step: Int) {
        m = [:]
        v = [:]
        for (k, a) in tensors {
            if k.hasPrefix("m.") { m[String(k.dropFirst(2))] = a }
            else if k.hasPrefix("v.") { v[String(k.dropFirst(2))] = a }
        }
        self.step = step
    }
}

/// Sum and scale gradient dictionaries (keys are flattened parameter names).
enum GradMath {
    static func add(_ a: [String: MLXArray]?, _ b: [String: MLXArray]) -> [String: MLXArray] {
        guard let a else { return b }
        var out = a
        for (k, v) in b { out[k] = (out[k] ?? MLXArray.zeros(like: v)) + v }
        return out
    }

    static func scale(_ a: [String: MLXArray], by s: Float) -> [String: MLXArray] {
        a.mapValues { $0 * s }
    }

    /// `mlx.optimizers.clip_grad_norm`: scales by `max_norm / (norm + 1e-6)` when the global
    /// norm exceeds `max_norm`.
    static func clip(_ a: [String: MLXArray], maxNorm: Float) -> [String: MLXArray] {
        var sq = MLXArray(Float(0))
        for v in a.values { sq = sq + square(v.asType(.float32)).sum() }
        let norm = sqrt(sq)
        let factor = maxNorm / (norm + 1e-6)
        return a.mapValues { MLX.where(norm .< maxNorm, $0, $0 * factor.asType($0.dtype)) }
    }

    static func flatten(_ p: ModuleParameters) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: p.flattened())
    }
}

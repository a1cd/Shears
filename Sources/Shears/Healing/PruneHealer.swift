import Foundation
import MLX
import MLXNN

/// How the student's experts are held.
public enum StudentForm: Sendable, Equatable {
    /// Every expert stays resident behind the routing mask (the teacher is the same weights
    /// with the mask and adapters off).
    case masked
    /// Cut the model's layers to their kept experts in place at init. The model stops being a
    /// teacher, so this needs a pre-computed teacher (``TeacherPrepass``).
    case sliced
    /// The layers already hold only their kept experts (for example from
    /// ``Gemma4Checkpoint/loadModel(keeping:dtype:)``); `numExperts` is the unpruned count.
    case preSliced(numExperts: Int)
}

/// Heals a model whose experts are pruned by a routing mask.
///
/// By default the model is loaded once and is both models: the teacher is that model with every
/// adapter off and all experts routable; the student restricts each layer's router to its kept
/// set and applies per-expert LoRA, dense LoRA and a low-rank router correction. Training moves
/// only the adapters. When healing is done, ``MergeAndSliceExport`` folds the adapters into the
/// weights and slices the experts to the kept set.
///
/// **With a pre-computed teacher** (`teacher:`, filled by ``TeacherPrepass``) no teacher layer
/// runs during healing: block ends and the final state come from the store, and the KL phases'
/// teacher log-probabilities are recomputed from the stored final state. Then the student can be
/// `.sliced`: the model holds only its kept experts, and the numbers are those of the masked
/// formulation. The one thing that genuinely needs a live teacher is the dirty phase with
/// `dirtyLambda > 0` (the teacher's block applied to the *student's* input): it needs
/// `.masked`, where the teacher is the same weights, and is refused on a sliced student.
public final class PruneHealer {
    public let model: any HealableMoEModel
    public let network: HealingNetwork
    public let adapters: ModelAdapters
    public let configuration: HealingConfiguration
    public let blockEnds: [Int]
    public let student: StudentForm
    /// Where teacher block ends come from when no live teacher is run.
    public let teacherStore: BoundaryStore?
    let computeDType: DType?
    private let teacherManifest: TeacherPrepass.Manifest?
    public var blocks: [(start: Int, end: Int)] { blockRanges(ends: blockEnds) }

    /// - Parameters:
    ///   - keep: kept expert ids per layer.
    ///   - blockEnds: layer indices ending each block; default is the model's global-attention
    ///     layers.
    ///   - initialAdapters: start from these adapter tensors instead of a fresh initialisation
    ///     (continue a previous run with a new optimizer).
    ///   - student: how the student's experts are held (see ``StudentForm``).
    ///   - teacher: a store filled by ``TeacherPrepass`` for exactly the rows this run will use
    ///     (and these block ends). Required for a sliced student.
    ///   - teacherFingerprint: the teacher source's ``LayerSource/teacherFingerprint()``; when given,
    ///     a store filled by a different model is refused. (A student built from a model the
    ///     teacher is not, such as a sliced one, cannot compute it for itself.)
    public init(
        model: any HealableMoEModel, keep: [[Int]],
        adapterConfiguration: AdapterConfiguration = .init(),
        configuration: HealingConfiguration = .init(), blockEnds: [Int]? = nil,
        initialAdapters: [String: MLXArray]? = nil, student: StudentForm = .masked,
        teacher: BoundaryStore? = nil, teacherFingerprint: String? = nil
    ) throws {
        let layers = model.healableLayers
        guard keep.count == layers.count else {
            throw HealingError.invalid("keep has \(keep.count) entries for \(layers.count) layers")
        }
        if student != .masked, teacher == nil {
            throw HealingError.invalid("a sliced student has no teacher of its own; pass a pre-computed teacher store")
        }
        let sets: [ExpertSet]
        if case .preSliced(let n) = student {
            sets = keep.map { ExpertSet(keep: $0, numExperts: n) }
            for (l, s) in zip(layers, sets) where l.numExperts != s.keep.count {
                throw HealingError.invalid("layers are not cut to their kept experts")
            }
        } else {
            sets = zip(layers, keep).map { ExpertSet(keep: $1, numExperts: $0.numExperts) }
        }
        // Adapters are built from the unsliced sets, so they have one format in every form.
        let ad = ModelAdapters(layers: layers, sets: sets, configuration: adapterConfiguration)
        if let initialAdapters { try ad.load(tensors: initialAdapters) }
        let computeDType = TeacherPrepass.computeDType(layers)
        let resolvedBlockEnds = blockEnds ?? model.defaultBlockEnds
        if let teacher {
            self.teacherManifest = try TeacherPrepass.verifiedManifest(
                of: teacher, blockEnds: resolvedBlockEnds, layerCount: layers.count, teacher: teacherFingerprint)
        } else {
            self.teacherManifest = nil
        }
        if student == .sliced {
            for (l, s) in zip(layers, sets) { sliceToKeptExperts(l, set: s) }
        }
        self.student = student
        self.teacherStore = teacher
        self.computeDType = computeDType
        self.model = model
        self.adapters = ad
        self.network = HealingNetwork(layers: layers, sets: sets, adapters: ad, sliced: student != .masked)
        self.configuration = configuration
        self.blockEnds = blockEnds ?? model.defaultBlockEnds
        precondition(self.blockEnds.last == layers.count - 1, "blocks must cover every layer")
    }

    // MARK: Teacher

    func embed(_ row: HealingRow) -> MLXArray { stopGradient(model.embed(row.tokenArray)) }

    /// Teacher hidden states at `layers` (forward only, evaluated): read from the pre-computed
    /// store when there is one (block ends only), otherwise run live.
    public func teacherStates(_ row: HealingRow, at layers: Set<Int>) -> [Int: MLXArray] {
        if let store = teacherStore {
            do { try TeacherPrepass.verify(row, against: teacherManifest) } catch {
                preconditionFailure("\(error)")
            }
            var out: [Int: MLXArray] = [:]
            for l in layers {
                guard let b = blockEnds.firstIndex(of: l) else {
                    preconditionFailure("a pre-computed teacher stores block ends only; layer \(l) is not one")
                }
                var s: MLXArray
                do { s = try store.get(row: row.key, boundary: b + 1) } catch {
                    preconditionFailure("the pre-computed teacher has no state for row '\(row.key)': \(error)")
                }
                if let c = computeDType, s.dtype != c { s = s.asType(c) }
                out[l] = s
            }
            eval(Array(out.values))
            return out
        }
        let (_, states) = network.forward(embed(row), student: false, collect: layers)
        let out = states.mapValues { stopGradient($0) }
        eval(Array(out.values))
        return out
    }

    // MARK: Per-phase steps (loss and gradients, nothing applied)

    typealias StepResult = (loss: Float, grads: [String: MLXArray])

    private func blockGrad(
        input: MLXArray, target: MLXArray, start: Int, end: Int, weights: MLXArray
    ) -> StepResult {
        let ckpt = configuration.checkpointBlocks
        let keptTop = configuration.keepIntermediates.count(ofLayers: end - start + 1)
        let vg = valueAndGrad(model: adapters) { [network] (_: ModelAdapters, _: [MLXArray]) in
            let (_, p) = network.forward(
                input, from: start, through: end, student: true, collect: [end], checkpointed: ckpt, keptLayers: keptTop)
            return [HealingLoss.pin([end: p[end]!], teacher: [end: target], weights: weights)]
        }
        let (l, g) = vg(adapters, [])
        return (l[0].item(Float.self), GradMath.flatten(g))
    }

    func blockStep(_ row: HealingRow) -> StepResult {
        let w = row.weights
        let ends = teacherStates(row, at: Set(blockEnds))
        var inputs: [Int: MLXArray] = [0: embed(row)]
        for (prev, next) in zip(blocks, blocks.dropFirst()) { inputs[next.start] = ends[prev.end]! }
        var total: [String: MLXArray]?
        var lossSum: Float = 0
        for b in blocks {
            let r = blockGrad(input: inputs[b.start]!, target: ends[b.end]!, start: b.start, end: b.end, weights: w)
            total = GradMath.add(total, r.grads)
            eval(Array(total!.values))
            lossSum += r.loss
        }
        let n = Float(blocks.count)
        return (lossSum / n, GradMath.scale(total!, by: 1 / n))
    }

    /// The student's own input to every block (forward only, adapters on), and, when
    /// `withTargets`, the teacher's block applied to that same input.
    func dirtyInputs(_ row: HealingRow, withTargets: Bool) -> (inputs: [Int: MLXArray], targets: [Int: MLXArray]) {
        let bs = blocks
        var inputs: [Int: MLXArray] = [bs[0].start: embed(row)]
        if bs.count > 1 {
            let collect = Set(bs.dropLast().map(\.end))
            let (_, st) = network.forward(
                inputs[bs[0].start]!, through: bs[bs.count - 2].end, student: true, collect: collect)
            for (prev, next) in zip(bs, bs.dropFirst()) { inputs[next.start] = stopGradient(st[prev.end]!) }
        }
        eval(Array(inputs.values))
        var targets: [Int: MLXArray] = [:]
        if withTargets {
            precondition(!network.sliced, "dirtyLambda > 0 needs a live teacher; use a masked student")
            for b in bs {
                let (_, t) = network.forward(
                    inputs[b.start]!, from: b.start, through: b.end, student: false, collect: [b.end])
                targets[b.end] = stopGradient(t[b.end]!)
                eval(targets[b.end]!)
            }
        }
        return (inputs, targets)
    }

    func dirtyStep(_ row: HealingRow) -> StepResult {
        let w = row.weights
        let lambda = configuration.dirtyLambda
        let ends = teacherStates(row, at: Set(blockEnds))
        let (inputs, dirtyTargets) = dirtyInputs(row, withTargets: lambda > 0)
        var total: [String: MLXArray]?
        var lossSum: Float = 0
        for b in blocks {
            var r = blockGrad(input: inputs[b.start]!, target: ends[b.end]!, start: b.start, end: b.end, weights: w)
            if lambda > 0 {
                let r2 = blockGrad(
                    input: inputs[b.start]!, target: dirtyTargets[b.end]!, start: b.start, end: b.end, weights: w)
                r = (r.loss + lambda * r2.loss, GradMath.add(r.grads, GradMath.scale(r2.grads, by: lambda)))
            }
            total = GradMath.add(total, r.grads)
            eval(Array(total!.values))
            lossSum += r.loss
        }
        let n = Float(blocks.count)
        return (lossSum / n, GradMath.scale(total!, by: 1 / n))
    }

    func freerunStep(_ row: HealingRow) -> StepResult {
        let w = row.weights
        let ends = teacherStates(row, at: Set(blockEnds))
        let x = embed(row)
        let vg = valueAndGrad(model: adapters) { [network, blockEnds] (_: ModelAdapters, _: [MLXArray]) in
            let (_, p) = network.forward(x, student: true, collect: Set(blockEnds), checkpointed: true)
            return [HealingLoss.pin(p, teacher: ends, weights: w)]
        }
        let (l, g) = vg(adapters, [])
        let grads = GradMath.flatten(g)
        eval(l + Array(grads.values))
        return (l[0].item(Float.self), grads)
    }

    /// Positions and weights the KL phases score for `row` at `step`.
    func klPositions(_ row: HealingRow, step: Int, sample: Bool) -> ([Int], [Float]) {
        var pos = row.klAnchors.map(\.position)
        var wt = row.klAnchors.map(\.weight)
        if sample, let range = row.klSampleRange, configuration.endSamplePositions > 0 {
            var rng = SplitMix64(seed: configuration.seed &* 1000 &+ UInt64(step))
            let picks = Array(range).shuffled(using: &rng).prefix(configuration.endSamplePositions).sorted()
            pos += picks
            wt += picks.map { _ in 1 }
        }
        precondition(!pos.isEmpty, "KL phases need klAnchors or a klSampleRange on every row")
        return (pos, wt)
    }

    /// End-to-end step on the final hidden state and the next-token distribution at chosen
    /// positions. The LM head is applied to those positions only: logits over the whole
    /// vocabulary at every position would be gigabytes inside the autodiff graph.
    func klStep(_ row: HealingRow, step: Int, phase: HealingPhase) -> StepResult {
        let last = network.lastLayer
        let w = row.weights
        let isEnd = phase == .end
        let (pos, kw) = klPositions(row, step: step, sample: isEnd)
        let posArr = MLXArray(pos.map(Int32.init))
        let kwArr = MLXArray(kw)
        let hT = teacherStates(row, at: [last])[last]!
        let tLP = HealingLoss.logProbs(model.logits(fromHidden: hT[0][posArr]))
        eval(tLP)
        let x = embed(row)
        let detach: Int? = isEnd ? nil : configuration.trainFromLayer
        let pinW = isEnd ? configuration.pinWeight : configuration.answerPinWeight
        let klW = configuration.klWeight
        let vg = valueAndGrad(model: adapters) { [network, model] (_: ModelAdapters, _: [MLXArray]) in
            let (h, _) = network.forward(x, student: true, checkpointed: true, detachBefore: detach)
            let pin = HealingLoss.pin([last: h], teacher: [last: hT], weights: w)
            let sLP = HealingLoss.logProbs(model.logits(fromHidden: h[0][posArr]))
            let kl = HealingLoss.kl(teacherLogProbs: tLP, studentLogProbs: sLP)
            return [pinW * pin + klW * (kl * kwArr).sum() / kwArr.sum()]
        }
        let (l, g) = vg(adapters, [])
        let grads = GradMath.flatten(g)
        eval(l + Array(grads.values))
        return (l[0].item(Float.self), grads)
    }

    func stepFor(_ phase: HealingPhase, _ row: HealingRow, step: Int) -> StepResult {
        network.refreshRounding()
        return switch phase {
        case .block: blockStep(row)
        case .dirty: dirtyStep(row)
        case .freerun: freerunStep(row)
        case .end, .answerKL: klStep(row, step: step, phase: phase)
        }
    }

    /// The loss `phase` would compute for `row` right now (gradients computed and discarded).
    public func probeLoss(_ phase: HealingPhase, row: HealingRow, step: Int = 0) -> Float {
        stepFor(phase, row, step: step).loss
    }

    // MARK: Evaluation helpers

    /// Free-running reconstruction error at the block ends, averaged over `rows`: the student
    /// (or the teacher with `student: false`, which is zero by construction) against the
    /// teacher.
    public func reconstructionError(rows: [HealingRow], student: Bool = true) -> [Int: Float] {
        var acc: [Int: Float] = [:]
        if student { network.refreshRounding() }
        for row in rows {
            let ref = teacherStates(row, at: Set(blockEnds))
            let (_, h) = network.forward(embed(row), student: student, collect: Set(blockEnds))
            let e = HealingLoss.pinErrors(h, teacher: ref, weights: row.weights)
            for (k, v) in e { acc[k, default: 0] += v / Float(rows.count) }
            Memory.clearCache()
        }
        return acc
    }

    /// Logits of the whole model on `tokens` (`[1, T]`), student or teacher.
    public func logits(tokens: MLXArray, student: Bool) -> MLXArray {
        if student { network.refreshRounding() }
        let (h, _) = network.forward(stopGradient(model.embed(tokens)), student: student)
        return model.logits(fromHidden: h)
    }

    // MARK: Training loop

    /// Runs `plan`, resuming from `checkpointDirectory` when it holds a checkpoint.
    @discardableResult
    public func run(
        rows: HealingRowSource, plan: PhasePlan, checkpointDirectory: URL? = nil,
        memory: MemoryConfiguration = .init(), callbacks: HealingCallbacks = .init()
    ) throws -> HealingSummary {
        memory.apply()
        let c = configuration
        if network.sliced, c.dirtyLambda > 0, plan.entries.contains(where: { $0.phase == .dirty }) {
            throw HealingError.invalid(
                "the dirty phase with dirtyLambda > 0 applies the teacher to the student's own input, which needs the "
                    + "unpruned weights; run it with a masked student")
        }
        if plan.entries.contains(where: { $0.phase == .answerKL }), c.trainFromLayer > 0 {
            for k in 0 ..< min(c.trainFromLayer, adapters.layers.count) { adapters.layers[k].freeze() }
        }
        let schedule = c.learningRateSchedule(runSteps: plan.total)
        let opt = HealingAdamW(weightDecay: c.weightDecay)
        opt.routerLRMultiplier = adapters.configuration.routerLRMultiplier
        var step0 = 0
        var history: [StepRecord] = []
        if let dir = checkpointDirectory, let meta = try Checkpointing.resume(from: dir, adapters: adapters, optimizer: opt) {
            step0 = meta.step
        }
        let started = Date()
        var stepsRun = 0
        func save(_ step: Int, phase: HealingPhase, final: Bool = false) throws {
            guard let dir = checkpointDirectory else { return }
            try Checkpointing.save(
                to: dir, adapters: adapters, optimizer: opt, step: step, phase: phase,
                plan: plan, final: final)
            callbacks.onCheckpoint?(step, dir)
        }
        var outcome = HealingOutcome.finished
        var i = step0
        while i < plan.total {
            let phase = plan.phase(at: i)
            callbacks.waitWhilePaused()
            if callbacks.stopRequested() {
                try save(i, phase: phase)
                callbacks.consumeStop()
                outcome = .stopped(step: i, reason: "stop requested")
                break
            }
            if let limit = memory.peakStopBytes, Memory.peakMemory > limit {
                try save(i, phase: phase)
                outcome = .stopped(step: i, reason: "peak memory \(Memory.peakMemory) over \(limit)")
                break
            }
            let t0 = Date()
            let row = rows.row(forStep: i)
            var result = stepFor(phase, row, step: i)
            if c.clipNorm > 0 { result.grads = GradMath.clip(result.grads, maxNorm: c.clipNorm) }
            let lr = schedule.value(at: opt.step)
            let ok = result.loss.isFinite && result.loss <= c.maxLoss
            if ok {
                opt.update(model: adapters, grads: result.grads, learningRate: lr)
                eval(adapters, opt.stateArrays)
            }
            result.grads = [:]
            Memory.clearCache()
            let rec = StepRecord(
                step: i + 1, phase: phase, loss: result.loss, learningRate: lr, applied: ok,
                seconds: Date().timeIntervalSince(t0))
            history.append(rec)
            callbacks.onStep?(rec)
            stepsRun += 1
            i += 1
            if i < plan.total, i % c.evaluateEvery == 0, let ev = callbacks.evaluate {
                callbacks.onEvaluation?(i, ev(i, phase))
            }
            if c.milestoneEvery > 0, i % c.milestoneEvery == 0, let dir = checkpointDirectory {
                try Checkpointing.saveMilestone(to: dir, adapters: adapters, step: i)
            }
            if i < plan.total, i % c.checkpointEvery == 0 { try save(i, phase: phase) }
        }
        if outcome == .finished {
            try save(plan.total, phase: plan.phase(at: max(0, plan.total - 1)), final: true)
            if let ev = callbacks.evaluate {
                callbacks.onEvaluation?(plan.total, ev(plan.total, plan.phase(at: max(0, plan.total - 1))))
            }
        }
        _ = started
        return HealingSummary(
            outcome: outcome, stepsRun: stepsRun, finalStep: i, history: history,
            peakMemoryBytes: Memory.peakMemory)
    }
}

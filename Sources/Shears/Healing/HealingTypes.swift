import Foundation
import MLX

/// One training sequence.
public struct HealingRow {
    /// Caller's identifier (used by the boundary store when streaming).
    public var key: String
    public var tokens: [Int32]
    /// Per-position weight of the hidden-state pin, `[T]`. `nil` means the recipe's ramp
    /// (``HealingLoss/rampedWeights(count:boundary:floor:rampLength:)``: 0.15 early in the
    /// sequence, rising linearly to 1 over the 64 positions before ``answerStart``), never
    /// uniform weights. Supply your own for anything else.
    public var pinWeights: MLXArray?
    /// Where the part of the sequence that matters most begins (the end of the prompt, in the
    /// recipe). The default pin weights reach 1 here. `nil` means the end of the sequence.
    public var answerStart: Int?
    /// Positions whose next-token distribution the KL phases always score, with weights.
    public var klAnchors: [KLAnchor]
    /// Extra positions the `end` phase samples (deterministically per step) and scores
    /// with weight 1. `nil` for none.
    public var klSampleRange: Range<Int>?

    public init(
        key: String = "", tokens: [Int32], pinWeights: MLXArray? = nil, answerStart: Int? = nil,
        klAnchors: [KLAnchor] = [], klSampleRange: Range<Int>? = nil
    ) {
        self.key = key
        self.tokens = tokens
        self.pinWeights = pinWeights
        self.answerStart = answerStart
        self.klAnchors = klAnchors
        self.klSampleRange = klSampleRange
    }

    public var length: Int { tokens.count }
    var weights: MLXArray {
        pinWeights ?? HealingLoss.rampedWeights(count: tokens.count, boundary: answerStart ?? tokens.count)
    }
    var tokenArray: MLXArray { MLXArray(tokens)[.newAxis] }
}

public struct KLAnchor: Sendable, Equatable {
    public var position: Int
    public var weight: Float
    public init(position: Int, weight: Float = 1) {
        self.position = position
        self.weight = weight
    }
}

/// Supplies the row for each step. It must be a pure function of the step so that a resumed
/// run sees the same data as an uninterrupted one.
public protocol HealingRowSource {
    func row(forStep step: Int) -> HealingRow
}

struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Walks the rows in a fresh random order each epoch (seeded by seed and epoch).
public struct ShuffledRows: HealingRowSource {
    public let rows: [HealingRow]
    public let seed: UInt64
    public init(rows: [HealingRow], seed: UInt64 = 0) {
        precondition(!rows.isEmpty)
        self.rows = rows
        self.seed = seed
    }
    public func index(forStep step: Int) -> Int {
        let (epoch, j) = (step / rows.count, step % rows.count)
        var rng = SplitMix64(seed: seed &+ UInt64(epoch))
        return Array(rows.indices).shuffled(using: &rng)[j]
    }
    public func row(forStep step: Int) -> HealingRow { rows[index(forStep: step)] }
}

/// Draws a group by probability each step, then walks that group in a shuffled order. For
/// callers that need to balance rows by class: groups and probabilities are theirs.
public struct GroupBalancedRows: HealingRowSource {
    public let groups: [[HealingRow]]
    public let probabilities: [Double]
    public let seed: UInt64
    public init(groups: [[HealingRow]], probabilities: [Double], seed: UInt64 = 0) {
        precondition(groups.count == probabilities.count && groups.allSatisfy { !$0.isEmpty })
        self.groups = groups
        self.probabilities = probabilities
        self.seed = seed
    }
    public func row(forStep step: Int) -> HealingRow {
        var rng = SplitMix64(seed: seed &* 7 &+ UInt64(step))
        let u = Double.random(in: 0 ..< probabilities.reduce(0, +), using: &rng)
        var acc = 0.0
        var g = groups.count - 1
        for (i, p) in probabilities.enumerated() {
            acc += p
            if u < acc { g = i; break }
        }
        var prng = SplitMix64(seed: seed &+ 99 &+ UInt64(g))
        let perm = Array(groups[g].indices).shuffled(using: &prng)
        return groups[g][perm[(step / 2) % groups[g].count]]
    }
}

/// Phase names of the schedule.
public enum HealingPhase: String, Sendable, CaseIterable {
    /// Block-local, teacher-forced: each block is fed the teacher's input and pinned to the
    /// teacher's output at its end. Gradients never cross a block boundary.
    case block
    /// Whole pruned model end to end (checkpointed), pinned at every block end.
    case freerun
    /// Student-input: each block is fed the live pruned model's own input from below
    /// (forward only) and pinned to the teacher's true output at its end.
    case dirty
    /// Final hidden state pin plus full-vocabulary KL at sampled positions, end to end.
    case end
    /// Answer-position KL (plus a pin anchor on the final hidden state), end to end through
    /// the layers from ``HealingConfiguration/trainFromLayer`` up.
    case answerKL = "answer_kl"
}

/// `block@0,freerun@600,end@1400`: which phase runs from which step.
public struct PhasePlan: Sendable, Equatable {
    public var entries: [(phase: HealingPhase, start: Int)]
    public var total: Int

    public init(entries: [(phase: HealingPhase, start: Int)], total: Int) {
        precondition(entries.first?.start == 0, "the plan must start at step 0")
        precondition(zip(entries, entries.dropFirst()).allSatisfy { $0.start < $1.start })
        self.entries = entries
        self.total = total
    }

    public init(parsing text: String, total: Int) throws {
        var items: [(HealingPhase, Int)] = []
        for tok in text.split(separator: ",") {
            let p = tok.split(separator: "@")
            guard p.count == 2, let phase = HealingPhase(rawValue: String(p[0])), let s = Int(p[1]) else {
                throw HealingError.invalid("bad plan entry '\(tok)'")
            }
            items.append((phase, s))
        }
        guard items.first?.1 == 0, zip(items, items.dropFirst()).allSatisfy({ $0.1 < $1.1 }) else {
            throw HealingError.invalid("plan must start at 0 and strictly increase: \(text)")
        }
        self.init(entries: items, total: total)
    }

    public static func == (a: PhasePlan, b: PhasePlan) -> Bool {
        a.total == b.total && a.entries.count == b.entries.count
            && zip(a.entries, b.entries).allSatisfy { $0.phase == $1.phase && $0.start == $1.start }
    }

    public func phase(at step: Int) -> HealingPhase {
        var p = entries[0].phase
        for e in entries where step >= e.start { p = e.phase }
        return p
    }

    public var text: String { entries.map { "\($0.phase.rawValue)@\($0.start)" }.joined(separator: ",") }
}

/// How many layers of a block keep their forward intermediates (no recompute in the backward pass) when per-layer
/// checkpointing is on. The layers kept are the top ones (their backward runs first, so what they hold is freed
/// first). Every setting computes the same numbers; only memory and time differ: each kept layer costs about one
/// layer's activations and saves about one layer's forward.
///
/// - ``fast``: every layer keeps them (no recompute; fastest, most memory).
/// - ``lowMemory``: none does (every layer is recomputed from its saved input; least memory, slowest).
/// - ``layers(_:)``: that many, clamped to the layers in the step's top window.
/// - ``auto`` (default): ``lowMemory``, unless the run has a memory budget that leaves room, in which case the most
///   layers whose estimated peak (see ``ResidencyPlanner/estimate(_:resident:tokenChunk:keptLayers:)``) still fits
///   the budget after the window size and token chunk are chosen. The budget is spent on window and chunk first, since
///   those cost no extra memory; kept layers get what is left. Without a budget auto never keeps any.
///
/// Layers outside the top window of a windowed step always recompute (their backward is a separate vjp), whatever the
/// setting; the setting is clamped to the top window.
public enum ActivationKeep: Sendable, Hashable, Codable, CustomStringConvertible {
    case auto, fast, lowMemory
    case layers(Int)

    public init(parsing text: String) throws {
        switch text.lowercased() {
        case "auto": self = .auto
        case "fast": self = .fast
        case "lowmemory", "low-memory", "low_memory": self = .lowMemory
        default:
            guard let n = Int(text), n >= 0 else {
                throw HealingError.invalid("keepIntermediates '\(text)': use auto, fast, lowMemory or a layer count")
            }
            self = .layers(n)
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let n = try? c.decode(Int.self) {
            guard n >= 0 else { throw HealingError.invalid("keepIntermediates must not be negative") }
            self = .layers(n)
        } else {
            try self.init(parsing: try c.decode(String.self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        if case .layers(let n) = self { try c.encode(n) } else { try c.encode(description) }
    }

    public var description: String {
        switch self {
        case .auto: return "auto"
        case .fast: return "fast"
        case .lowMemory: return "lowMemory"
        case .layers(let n): return "\(n)"
        }
    }

    /// The layers kept when there is no memory budget to spend (``auto`` keeps none), out of `layers`.
    public func count(ofLayers layers: Int) -> Int {
        switch self {
        case .auto, .lowMemory: return 0
        case .fast: return layers
        case .layers(let n): return min(max(0, n), layers)
        }
    }
}

/// Knobs of a run. Defaults are the recipe's.
public struct HealingConfiguration: Sendable {
    public var learningRate: Float = 1e-4
    public var learningRateFloor: Float = 1e-5
    public var warmupSteps: Int = 20
    /// Length of the warmup-and-cosine schedule. `nil` (the default) is the length of the run
    /// (the plan's total, or a streamed run's `steps`); set it to run a slice of a longer schedule.
    public var scheduleLength: Int?
    /// Where on the schedule this run's first update sits (the optimizer's count is added to it).
    /// 0 starts a schedule; the updates an earlier run already made continue it.
    public var scheduleOffset = 0
    /// Re-lengths the schedule from a schedule position on (see ``LearningRateSchedule``).
    public var scheduleLengthChanges: [LearningRateSchedule.LengthChange] = []
    public var weightDecay: Float = 0.01
    /// Global gradient-norm clip; 0 is off.
    public var clipNorm: Float = 0
    /// A step whose loss is non-finite or above this applies no update.
    public var maxLoss: Float = 1e3
    /// Recompute activations per layer in the block and dirty phases too (the end-to-end
    /// phases always do). On by default: it is what keeps a block's memory flat in sequence
    /// length; turn it off only for short sequences, where it costs a little time for no gain.
    public var checkpointBlocks = true
    /// With ``checkpointBlocks`` on, how many of a block's layers keep their forward intermediates for the backward
    /// pass instead of recomputing them. Only what is stored changes: the result is bit-identical for every setting.
    /// See ``ActivationKeep``. Off checkpointing is `fast` whatever this says.
    public var keepIntermediates = ActivationKeep.auto
    /// Streamed block phase only: how many of a block's layers are held in memory at once, from 1 to
    /// the block's length. `nil` (the default) holds the whole block, as before. Fewer means less
    /// memory and the same result: the block is walked a window of layers at a time, each window's
    /// weights loaded when needed and dropped after, the per-layer recompute the block already does
    /// being the only recompute. Costs a second load of each layer per step (the top window, kept from
    /// the forward walk into the backward walk, is loaded once).
    public var residentLayers: Int?
    /// Streamed block phase only: bytes of MLX memory the run may use. The run picks the most layers
    /// it can hold at once within it (``residentLayers`` too, if set, is checked against it), sets
    /// MLX's memory limit to it, and stops with a checkpoint if a measured peak passes it. A budget
    /// the run cannot meet is refused before any training, with the estimate that shows why.
    public var memoryBudget: Int?
    /// Weight of the dirty phase's extra target, the teacher's block applied to the student's
    /// own input. 0 (the recipe's default) leaves only the clean target.
    public var dirtyLambda: Float = 0
    public var pinWeight: Float = 1
    public var klWeight: Float = 1
    /// Positions the `end` phase samples from ``HealingRow/klSampleRange``.
    public var endSamplePositions = 192
    /// `answerKL` trains only layers at or above this; below it the model runs forward only.
    /// If a plan contains `answerKL`, layers below it are frozen for the whole run, as the
    /// recipe does.
    public var trainFromLayer = 0
    /// Weight of the final-hidden pin in the `answerKL` phase.
    public var answerPinWeight: Float = 0.3
    public var seed: UInt64 = 0
    public var checkpointEvery = 150
    public var evaluateEvery = 300
    public var milestoneEvery = 500
    public init() {}
}

extension HealingConfiguration {
    /// The learning-rate schedule of a run of `runSteps` steps.
    public func learningRateSchedule(runSteps: Int) -> LearningRateSchedule {
        LearningRateSchedule(
            peak: learningRate, floor: learningRateFloor, warmup: warmupSteps,
            total: scheduleLength ?? runSteps, offset: scheduleOffset, lengthChanges: scheduleLengthChanges)
    }
}

/// MLX memory settings. Applied once at the start of a run.
public struct MemoryConfiguration: Sendable {
    public var memoryLimitBytes: Int?
    public var cacheLimitBytes: Int?
    /// If the allocator's peak passes this between steps, the run checkpoints and stops.
    public var peakStopBytes: Int?
    public init(memoryLimitBytes: Int? = nil, cacheLimitBytes: Int? = nil, peakStopBytes: Int? = nil) {
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.peakStopBytes = peakStopBytes
    }
    func apply() {
        if let m = memoryLimitBytes { Memory.memoryLimit = m }
        if let c = cacheLimitBytes { Memory.cacheLimit = c }
    }
}

public struct StepRecord: Sendable {
    public var step: Int
    public var phase: HealingPhase
    public var loss: Float
    public var learningRate: Float
    public var applied: Bool
    public var seconds: Double
    /// Highest MLX allocator memory this block has reached so far (`Memory.peakMemory`), bytes.
    public var peakMemoryBytes: Int = 0
    /// Highest process physical footprint during this step (`phys_footprint`, what Activity Monitor
    /// shows as "Memory"), bytes. Counts anonymous memory and leaves out clean file-backed pages.
    public var peakFootprintBytes: Int = 0
    /// Resident file-backed (memory-mapped, droppable) bytes seen during this step.
    public var fileBackedBytes: Int = 0
}

/// Hooks a caller can supply. All optional.
public struct HealingCallbacks {
    public var onStep: ((StepRecord) -> Void)?
    /// Called every ``HealingConfiguration/evaluateEvery`` steps (and at the end) with the
    /// step and phase; the returned numbers are passed to `onEvaluation`. Use
    /// ``PruneHealer/reconstructionError(rows:)`` and your own metrics here.
    public var evaluate: ((Int, HealingPhase) -> [String: Double])?
    public var onEvaluation: ((Int, [String: Double]) -> Void)?
    /// Checked between steps; `true` checkpoints and stops the run.
    public var shouldStop: (() -> Bool)?
    /// Checked between steps; while `true` the run idles (and returns cached buffers).
    public var shouldPause: (() -> Bool)?
    public var onCheckpoint: ((Int, URL) -> Void)?
    /// A file whose existence means "stop": checked between steps with `shouldStop`. When the run
    /// honours it, the run checkpoints, stops and removes the file (so the next start is not
    /// stopped by a stale one).
    public var stopFile: URL?
    /// A file whose existence means "pause": while it exists the run idles between steps.
    public var pauseFile: URL?
    public init() {}

    /// Whether the caller asked for a stop (`shouldStop` or the stop file).
    func stopRequested() -> Bool {
        if shouldStop?() == true { return true }
        if let f = stopFile, FileManager.default.fileExists(atPath: f.path) { return true }
        return false
    }

    /// Removes the stop file once a stop has been honoured and saved.
    func consumeStop() {
        if let f = stopFile { try? FileManager.default.removeItem(at: f) }
    }

    func pauseRequested() -> Bool {
        if shouldPause?() == true { return true }
        if let f = pauseFile, FileManager.default.fileExists(atPath: f.path) { return true }
        return false
    }

    /// Idles (returning cached buffers first) while a pause is requested; gives up waiting when
    /// a stop is requested too, so a paused run can still be stopped.
    func waitWhilePaused() {
        guard pauseRequested() else { return }
        Memory.clearCache()
        while pauseRequested(), !stopRequested() { Thread.sleep(forTimeInterval: 0.25) }
    }
}

public enum HealingOutcome: Sendable, Equatable {
    case finished
    case stopped(step: Int, reason: String)
}

public struct HealingSummary: Sendable {
    public var outcome: HealingOutcome
    public var stepsRun: Int
    public var finalStep: Int
    public var history: [StepRecord]
    public var peakMemoryBytes: Int
}

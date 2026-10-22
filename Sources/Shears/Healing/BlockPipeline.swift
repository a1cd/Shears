import Foundation
import MLX
import MLXNN
import MLXRandom

/// Which inputs a block trains on.
public enum InputMix: String, Sendable, Codable {
    /// The teacher's block input only (the old streamed block phase).
    case clean
    /// The real output of the healed blocks below only.
    case dirty
    /// Both, every step: the clean and the dirty input of the same row are stacked as a batch of two, run
    /// through one forward and backward, and the loss is the mean of their two pin losses against the
    /// teacher's clean block end. One optimizer update per step, so the last updates always cover both.
    /// Block 0 has no dirty input of its own (it equals the clean one), so it runs one input, which is the
    /// same update at half the cost.
    case both
}

/// When the teacher's block ends are computed (see ``BlockPipelineOptions/teacherPass``).
public enum TeacherPass: String, Sendable, Codable {
    case perBlock = "block"
    case perBatch = "batch"
}

/// Rows that begin with the same tokens (a common prompt) share their prefix's work.
public enum SharedPrefixMode: Sendable, Equatable {
    case off
    /// Use the longest token prefix common to every row, if it is at least `sharedPrefixMinimum` tokens.
    case auto
    /// Exactly this many tokens (every row must begin with the same ones).
    case length(Int)
}

public struct BlockPipelineOptions {
    /// Steps (one row each, in the row source's order) in one rolling data batch: its rows are prepared
    /// (teacher walk, selection), trained on, and the next batch is prepared.
    public var batchSteps = 64
    public var mix: InputMix = .both
    /// What stored states are written as; `nil` follows the model's compute dtype.
    public var storageDType: DType?
    /// Padded-token and row limits of the batches the walks run in.
    public var maxBatchTokens = 8192
    public var maxBatchRows = 16
    /// When the teacher's block ends are computed for the training rows. ``TeacherPass/perBlock`` (default): for every
    /// row of the block in one go before its training starts, so the teacher's layers (all 128 experts of each) are
    /// read once per block. ``TeacherPass/perBatch``: for each rolling batch just before its steps, which reads the
    /// whole block again for every batch.
    public var teacherPass = TeacherPass.perBlock
    /// With ``TeacherPass/perBlock``, the most bytes of hidden states one walk of the block's layers holds (rows x tokens
    /// x hidden, at the compute dtype's width, not counting the layer being walked). Rows beyond it wait for a second
    /// walk, so a block reads its teacher layers `ceil(states / budget)` times at most.
    public var teacherStateBudgetBytes = 2 << 30
    /// Pad the teacher walk's and the next frontier's batches up to a small set of shapes: the length to the next of
    /// ``StreamedBlockHealer/sequenceBuckets`` and the row count to the next power of two (when that stays within
    /// ``maxBatchTokens``), with zero rows. Attention is causal and every row is cut back to its own length afterwards,
    /// so no value changes (to float rounding; tested), but a backend that compiles a kernel per shape (MLX's CUDA
    /// JIT) meets a handful of shapes instead of a new one in every batch.
    public var padWalks = true
    /// A ceiling on MLX memory. By default the whole block is resident; with a budget the window is the largest whose
    /// estimate fits it (see ``ResidencyPlanner/chooseAuto(_:otherStagesPeak:ceiling:)``), and a run whose floor is
    /// above the budget is refused with the amount it needs.
    public var memoryBudget: Int?
    /// Selection settings; `nil` needs ``fixedKeep``.
    public var selection: SelectionConfiguration?
    /// Kept experts per layer, chosen already: no selection is run.
    public var fixedKeep: [[Int]]?
    /// Selection items by row key. A block's keep sets are chosen from the rows of its first batch that
    /// have an item (the others are only walked).
    public var items: [String: SelectionItem] = [:]
    /// Stop after this many blocks (the sweep prepares block 0 alone).
    public var maxBlocks: Int?
    /// Prepare block 0's data (selection and every batch's teacher walk) and stop before training.
    public var prepareOnly = false
    /// Train the blocks but leave the frontier alone: no next-block pass and nothing deleted, so one prepared
    /// frontier can serve several runs (the sweep).
    public var trainOnly = false
    /// Rows in one step: their states are stacked (right-padded; attention is causal, so each row's own
    /// positions are exact) through one forward and backward, and the loss is the mean over rows (and over
    /// clean and dirty). `steps` counts updates, so a run consumes `steps * rowsPerStep` rows of the source.
    public var rowsPerStep = 1
    /// Held-out rows scored during training (not trained on); with ``plateau`` they decide when to anneal.
    public var evalRows: [HealingRow] = []
    /// Shorten the remaining schedule when the held-out loss stops improving; needs ``evalRows``.
    public var plateau: PlateauAnnealing?
    /// Share the work on a common token prefix: the teacher's prefix activations are computed once per layer for
    /// a whole batch of rows (its suffixes attend to the prefix's keys and values), and a step of several rows
    /// runs the shared prefix once, forward and backward, when the whole block is resident. Exactly equal to
    /// running each row whole. A windowed step (fewer resident layers than the block) runs its rows whole.
    public var sharedPrefix = SharedPrefixMode.auto
    public var sharedPrefixMinimum = 16
    /// Tokens the expert math processes at once in the merged layout (see ``ExpertCompute/tokenChunk``); `nil` follows
    /// the residency decision, which lowers it when the memory budget needs it.
    public var tokenChunk: Int?
    /// MLX's buffer-cache limit for the run, bytes. Freed buffers are kept for reuse up to this size and count
    /// toward the process footprint. `nil` derives it: before training (embedding, the residency measurement, the
    /// teacher walk, selection) sixteen batches' hidden states, between 64 MB and 1 GB, and never above the limit
    /// already set; for training an eighth of the residency estimate, between 64 MB and 1 GB. A
    /// `MemoryConfiguration/cacheLimitBytes` given to `run` wins.
    public var cacheLimitBytes: Int?
    /// Apply and free each layer's gradients one layer at a time in the top window's update (see
    /// ``StreamedBlockHealer/freeGradientsEarly``); `nil` leaves the process-wide setting alone. The previous value is
    /// restored when the run returns.
    public var freeGradientsEarly: Bool?
    /// Queue the backward with the forward before the loss is read (see ``StreamedBlockHealer/overlapLossCheck``);
    /// `nil` leaves the process-wide setting alone. Restored when the run returns.
    public var overlapLossCheck: Bool?
    /// Bound MLX's buffer cache in the stages before training (see ``cacheLimitBytes``). Off leaves it as the caller set it.
    public var boundPrepareCache = true
    /// Test hook: force the resident layers instead of the auto rule.
    var forcedResidentLayers: Int?
    public init() {}
}

/// When held-out loss plateaus, anneal sooner: every `evalEvery` updates the block's held-out rows are scored
/// (on the inputs it trains on); after `patience` scores in a row that fail to beat the best by the relative
/// `threshold`, the cosine schedule is re-lengthened (``LearningRateSchedule/LengthChange``) so that it ends
/// `annealSteps` updates later. That drops the learning rate to where a schedule of that length would be at this
/// position and decays it to the floor over the next `annealSteps` updates. It triggers at most once per block.
public struct PlateauAnnealing: Sendable, Codable, Equatable {
    public var evalEvery = 50
    public var patience = 3
    public var threshold: Float = 0.01
    public var annealSteps = 50
    public init(evalEvery: Int = 50, patience: Int = 3, threshold: Float = 0.01, annealSteps: Int = 50) {
        self.evalEvery = evalEvery
        self.patience = patience
        self.threshold = threshold
        self.annealSteps = annealSteps
    }

    /// The monitor's running state (kept in the manifest so a resumed block continues it).
    public struct State: Sendable, Codable, Equatable {
        public var best = Float.greatestFiniteMagnitude
        public var bad = 0
        public var triggeredAt: Int?
        public init() {}
    }

    /// Feeds one held-out score at update count `step`; returns true exactly when this score triggers annealing.
    public func observe(_ loss: Float, step: Int, state: inout State) -> Bool {
        guard state.triggeredAt == nil else { return false }
        if loss < state.best * (1 - threshold) {
            state.best = min(state.best, loss)
            state.bad = 0
            return false
        }
        state.bad += 1
        if state.bad >= patience {
            state.triggeredAt = step
            return true
        }
        return false
    }
}

struct StopWhileWritingFrontier: Error {}

/// Block-sequential healing: one block at a time, end to end.
///
/// For block `b`, in rolling data batches of `batchSteps` rows:
/// 1. **Prepare** a batch: the teacher walks the block's layers one at a time over the batch's rows with
///    the activations in RAM; in the same pass (first batch only) the experts of each layer are selected
///    from the pruned stream's input and the all-expert target, and the pruned stream advances with the
///    layer's new keep set. The teacher's block ends are written to the frontier.
/// 2. **Train** on that batch: each step takes the row's clean input (the teacher's block input) and, from
///    block 1 on, its dirty input (the real output of the healed blocks below), against the teacher's clean
///    block end (see ``InputMix``).
/// When every batch is done the healed block runs over all rows to write the next frontier (dirty states,
/// plus any teacher states not written yet) and the previous frontier is deleted. Only the current
/// frontier is on disk, and there is no full-model pre-pass.
///
/// The resident layers of training are chosen by ``ResidencyPlanner/chooseAuto(_:otherStagesPeak:ceiling:)``
/// per block. Everything is resumable: per layer (selection records), per batch and per block.
public final class BlockPipeline {
    /// Counts of steps and teacher walks that took the shared-prefix path (for tests and reports).
    nonisolated(unsafe) public static var sharedPrefixStepCount = 0
    nonisolated(unsafe) public static var sharedPrefixWalkCount = 0
    public let source: LayerSource
    public let blockEnds: [Int]
    public let adapterConfiguration: AdapterConfiguration
    public let configuration: HealingConfiguration
    public let options: BlockPipelineOptions
    public let directory: URL
    public let store: DiskBoundaryStore
    /// Names a run that shares this directory's frontier but has its own manifest, block adapters and
    /// checkpoints (empty for the main run).
    public let runName: String

    public init(
        source: LayerSource, blockEnds: [Int], adapterConfiguration: AdapterConfiguration = .init(),
        configuration: HealingConfiguration = .init(), options: BlockPipelineOptions = .init(), directory: URL,
        runName: String = ""
    ) throws {
        self.runName = runName
        precondition(configuration.clipNorm == 0, "streamed blocks can't clip by global norm")
        self.source = source
        self.blockEnds = blockEnds
        self.adapterConfiguration = adapterConfiguration
        self.configuration = configuration
        self.options = options
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        store = try DiskBoundaryStore(directory: directory.appendingPathComponent("frontier"))
    }

    // MARK: state

    struct BlockState: Codable {
        var selections: [LayerSelection] = []
        var selectionDone = false
        var prepared: [Int] = []
        var trainingDone = false
        var frontierDone = false
        var decision: ResidencyDecision?
        var losses: [Float] = []
        var skipped = 0
        var plateau = PlateauAnnealing.State()
        var lengthChanges: [LearningRateSchedule.LengthChange] = []
        var heldOut: [Float] = []
        var finalHeldOut: Float?
        var finalRouting: RoutingAgreement?
    }

    struct Manifest: Codable {
        var version = 1
        var teacher: String
        var blockEnds: [Int]
        var dtype: String
        var rowsDigest: String
        var steps: Int
        var batchSteps: Int
        var mix: InputMix
        var rowsPerStep: Int? = 1
        var blocks: [String: BlockState] = [:]
        /// Highest MLX peak of any stage that is not training so far (the floor's measured part).
        var otherStagesPeak = 0
        /// Block 0's inputs (the embeddings) are written.
        var embedded = false
    }

    public struct BlockInfo {
        public var block: Int
        public var residentLayers: Int
        public var decision: ResidencyDecision?
        public var losses: [Float]
        public var skippedUpdates: Int
        public var trainingPeakBytes: Int
        public var secondsPerStep: Double
        /// Held-out scores at each evaluation (with ``BlockPipelineOptions/evalRows``).
        public var heldOut: [Float] = []
        /// The update at which plateau annealing triggered, if it did.
        public var annealedAt: Int?
        /// Held-out score of the finished block (with ``BlockPipelineOptions/evalRows``).
        public var finalHeldOut: Float?
        /// How closely the finished block's router correction follows the original router on the held-out rows
        /// (`nil` without held-out rows or without a router correction).
        public var finalRouting: RoutingAgreement?
    }

    public struct Report {
        public var adapters: [String: MLXArray]
        public var keep: [[Int]]
        public var selections: [LayerSelection]
        public var blocks: [BlockInfo]
        public var otherStagesPeakBytes: Int
        public var trainingPeakBytes: Int
        public var frontierBytes: Int
        public var outcome: HealingOutcome = .finished
    }

    private var manifest: Manifest!
    private let fm = FileManager.default
    var tag: String { runName.isEmpty ? "" : "-" + runName }
    var manifestFile: URL { directory.appendingPathComponent("pipeline\(tag).json") }

    func saveManifest() throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        try enc.encode(manifest).write(to: manifestFile, options: .atomic)
    }

    // MARK: frontier

    static func key(_ kind: String, _ row: String) -> String { kind + "\u{1f}" + row }
    var storage: DType = .float32
    func put(_ kind: String, _ row: String, _ boundary: Int, _ x: MLXArray) throws {
        try store.put(x.dtype == storage ? x : x.asType(storage), row: Self.key(kind, row), boundary: boundary)
    }
    func get(_ kind: String, _ row: String, _ boundary: Int) throws -> MLXArray {
        try store.get(row: Self.key(kind, row), boundary: boundary)
    }
    func has(_ kind: String, _ row: String, _ boundary: Int) -> Bool { store.has(row: Self.key(kind, row), boundary: boundary) }

    /// Padded `[B, T, D]` batch of states (right-padded with zeros) and each row's length. With `bucketed` (and
    /// ``BlockPipelineOptions/padWalks``) the length is rounded up to its bucket and the rows to a power of two.
    func stack(_ states: [MLXArray], compute: DType?, bucketed: Bool = false) -> MLXArray {
        let bucket = bucketed && options.padWalks
        var longest = states.map { $0.dim(1) }.max() ?? 0
        if bucket { longest = StreamedBlockHealer.sequenceBucket(for: longest) }
        var out = concatenated(states.map { x in
            var x = x
            if let compute, x.dtype != compute { x = x.asType(compute) }
            let pad = longest - x.dim(1)
            return pad == 0 ? x : concatenated([x, MLXArray.zeros([1, pad, x.dim(2)], dtype: x.dtype)], axis: 1)
        }, axis: 0)
        if bucket {
            var rows = 1
            while rows < states.count { rows *= 2 }
            if rows > states.count, rows * longest <= options.maxBatchTokens {
                out = concatenated([out, MLXArray.zeros([rows - states.count, out.dim(1), out.dim(2)], dtype: out.dtype)], axis: 0)
            }
        }
        return out
    }

    // MARK: run

    /// Rows by step: batch `k` is steps `k * batchSteps ..< min(steps, (k + 1) * batchSteps)`.
    func batchRows(_ rows: HealingRowSource, _ k: Int, steps: Int) -> [(step: Int, row: HealingRow)] {
        let lo = k * options.batchSteps, hi = min(steps, lo + options.batchSteps)
        return (lo ..< hi).flatMap { s in stepRows(rows, s).map { (s, $0) } }
    }

    /// The rows of update `step`.
    func stepRows(_ rows: HealingRowSource, _ step: Int) -> [HealingRow] {
        (0 ..< options.rowsPerStep).map { rows.row(forStep: step * options.rowsPerStep + $0) }
    }

    func unique(_ rows: [HealingRow]) -> [HealingRow] {
        var seen = Set<String>()
        return rows.filter { seen.insert($0.key).inserted }
    }

    /// The buffer-cache limit used when none is given: an eighth of the residency estimate, between 64 MB and 1 GB.
    static func automaticCacheLimit(for estimate: Int) -> Int { min(max(estimate / 8, 64 << 20), 1 << 30) }
    /// The buffer-cache limit for the stages before training: sixteen times one batch's hidden states
    /// (`maxBatchTokens x hidden` at two bytes), between 64 MB and 1 GB.
    static func prepareCacheLimit(maxBatchTokens: Int, hidden: Int) -> Int {
        min(max(16 * maxBatchTokens * hidden * 2, 64 << 20), 1 << 30)
    }

    @discardableResult
    public func run(
        rows: HealingRowSource, steps: Int, memory: MemoryConfiguration = .init(),
        callbacks: HealingCallbacks = .init(), log: ((String) -> Void)? = nil,
        onStep: ((Int, StepRecord) -> Void)? = nil, onBlockFinished: ((Int, [String: MLXArray]) -> Void)? = nil
    ) throws -> Report {
        precondition(steps > 0 && options.batchSteps > 0)
        try adapterConfiguration.validate()
        let blocks = blockRanges(ends: blockEnds)
        guard blockEnds.last == source.layerCount - 1 else { throw HealingError.invalid("blocks must cover every layer") }
        if options.fixedKeep == nil && options.selection == nil {
            throw HealingError.invalid("the pipeline needs selection settings or fixed keep sets")
        }
        if let k = options.fixedKeep, k.count != source.layerCount {
            throw HealingError.invalid("keep has \(k.count) entries for \(source.layerCount) layers")
        }
        let batchCount = (steps + options.batchSteps - 1) / options.batchSteps
        // Every distinct row the steps use: the frontier covers exactly these.
        var used: [HealingRow] = []
        var seen = Set<String>()
        for s in 0 ..< steps {
            for r in stepRows(rows, s) where seen.insert(r.key).inserted { used.append(r) }
        }
        for r in options.evalRows where seen.insert(r.key).inserted { used.append(r) }
        let longest = used.map(\.length).max() ?? 1
        let sharedP = try sharedPrefixLength(used)
        var fp = Fingerprint()
        for r in used { fp.add(r.key); fp.add(TeacherPrepass.fingerprint(r)) }
        let probe = try source.loadLayers(0 ... 0, keeping: nil)
        hiddenSize = probe.first?.hiddenSize ?? 1
        storage = options.storageDType ?? TeacherPrepass.computeDType(probe) ?? .bfloat16
        let teacher = try source.teacherFingerprint()
        var fresh = Manifest(
            teacher: teacher, blockEnds: blockEnds, dtype: "\(storage)", rowsDigest: fp.hex, steps: steps,
            batchSteps: options.batchSteps, mix: options.mix)
        fresh.rowsPerStep = options.rowsPerStep
        if fm.fileExists(atPath: manifestFile.path) {
            let old = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestFile))
            guard old.teacher == fresh.teacher, old.blockEnds == fresh.blockEnds, old.dtype == fresh.dtype,
                old.rowsDigest == fresh.rowsDigest, old.steps == steps, old.batchSteps == options.batchSteps, old.mix == options.mix,
                (old.rowsPerStep ?? 1) == options.rowsPerStep
            else {
                throw HealingError.invalid(
                    "this pipeline directory was written for a different model, rows, steps, batch size, mix or storage dtype; use a fresh directory")
            }
            manifest = old
        } else {
            manifest = fresh
            try saveManifest()
        }
        // The stages before training (embedding, the residency measurement, the teacher walk, selection) hold one
        // teacher layer and a batch of rows. MLX keeps freed buffers for reuse up to its cache limit and they count
        // toward the process footprint; left at its default (the whole memory limit) the cache grows with every
        // distinct buffer size the walk allocates, many times the bytes in use. Bound it from the batch size for
        // these stages; training sets its own below, and a `MemoryConfiguration/cacheLimitBytes` wins over both.
        var priorCacheLimit: Int?
        defer { if let p = priorCacheLimit { Memory.cacheLimit = p } }
        priorCacheLimit = Memory.cacheLimit
        if options.boundPrepareCache {
            Memory.cacheLimit = options.cacheLimitBytes
                ?? min(Memory.cacheLimit, Self.prepareCacheLimit(maxBatchTokens: options.maxBatchTokens, hidden: probe.first?.hiddenSize ?? 0))
        }
        let priorFree = StreamedBlockHealer.freeGradientsEarly, priorOverlap = StreamedBlockHealer.overlapLossCheck
        defer {
            StreamedBlockHealer.freeGradientsEarly = priorFree
            StreamedBlockHealer.overlapLossCheck = priorOverlap
        }
        if let f = options.freeGradientsEarly { StreamedBlockHealer.freeGradientsEarly = f }
        if let o = options.overlapLossCheck { StreamedBlockHealer.overlapLossCheck = o }
        // A budget below what training itself needs (one layer resident) is refused before any work; one below the
        // measured peak of the other stages can only be found once those stages have run.
        if let budget = options.memoryBudget, let k = options.fixedKeep?.first?.count ?? options.selection?.keep {
            var inputs = try ResidencyPlanner.inputs(
                source: source, keep: Array(repeating: Array(0 ..< k), count: source.layerCount), blockEnds: blockEnds,
                adapterConfiguration: adapterConfiguration, sliced: source.canSliceLayers, sequence: longest, block: 0)
            inputs.hidden *= options.rowsPerStep
            inputs.tokenCount *= options.rowsPerStep
            inputs.workspaceBytesPerToken *= options.rowsPerStep
            inputs.attentionBytes *= options.rowsPerStep
            _ = try ResidencyPlanner.chooseAuto(inputs, otherStagesPeak: 0, ceiling: budget)
        }
        memory.apply()
        var memory = memory
        if let budget = options.memoryBudget {
            memory.memoryLimitBytes = min(memory.memoryLimitBytes ?? budget, budget)
            memory.apply()
        }
        let footprint = FootprintSampler()
        // The cache limits this run derives are put back when it returns: a library call leaves no global changed.
        var keep: [[Int]] = options.fixedKeep ?? Array(repeating: [], count: source.layerCount)
        var allSelections: [LayerSelection] = []
        var adaptersAll: [String: MLXArray] = [:]
        var infos: [BlockInfo] = []
        var trainingPeak = 0
        var outcome = HealingOutcome.finished
        let blockLimit = min(blocks.count, options.maxBlocks ?? blocks.count)
        let sequences = used
        if !manifest.embedded {
            Memory.peakMemory = Memory.activeMemory
            try embed(used)
            manifest.otherStagesPeak = max(manifest.otherStagesPeak, Memory.peakMemory)
            manifest.embedded = true
            try saveManifest()
        }

        blockLoop: for b in 0 ..< blockLimit {
            let range = blocks[b]
            var st = manifest.blocks["\(b)"] ?? BlockState()
            func commit() throws { manifest.blocks["\(b)"] = st; try saveManifest() }
            let blockFile = directory.appendingPathComponent("blocks\(tag)/block-\(b).safetensors")
            // Keep sets are known from an earlier run's selection records.
            if options.fixedKeep == nil, st.selectionDone {
                for r in st.selections { keep[r.layer] = r.chosen }
            }

            if st.frontierDone || (st.trainingDone && b == blocks.count - 1) {
                allSelections.append(contentsOf: st.selections)
                if fm.fileExists(atPath: blockFile.path) { adaptersAll.merge(try loadArrays(url: blockFile)) { $1 } }
                continue
            }

            // 1. prepare the first batch (with selection), measuring its peak
            let needSelection = options.fixedKeep == nil && !st.selectionDone
            let batch0 = unique(batchRows(rows, 0, steps: steps).map(\.row) + options.evalRows)
            // Per block (default): every batch's rows go through the teacher in this one stage, so the block's teacher
            // layers are read once (per walk of the state budget), not once per rolling batch.
            let wholeBlock = options.teacherPass == .perBlock
            let needAll = wholeBlock && !st.trainingDone && (0 ..< batchCount).contains { !st.prepared.contains($0) }
            if !st.prepared.contains(0) || needSelection || needAll {
                Memory.peakMemory = Memory.activeMemory
                let label = wholeBlock
                    ? "block \(b + 1)/\(blocks.count), all \(batchCount) batch\(batchCount == 1 ? "" : "es")"
                    : "block \(b + 1)/\(blocks.count), batch 1/\(batchCount)"
                try prepare(
                    block: b, rows: wholeBlock ? sequences : batch0, selecting: needSelection, selectionRows: batch0, state: &st,
                    keep: &keep, prefix: sharedP, label: label, log: log,
                    persist: { s in
                        self.manifest.blocks["\(b)"] = s
                        try? self.saveManifest()
                        if let r = s.selections.last, needSelection {
                            log?("block \(b + 1)/\(blocks.count): selected layer \(r.layer + 1)  error \(r.chosenError)")
                        }
                    })
                if needSelection { st.selectionDone = true }
                for k in 0 ..< batchCount where (k == 0 || wholeBlock) && !st.prepared.contains(k) { st.prepared.append(k) }
                manifest.otherStagesPeak = max(manifest.otherStagesPeak, Memory.peakMemory)
                try commit()
            }
            allSelections.append(contentsOf: st.selections)
            if options.prepareOnly {
                for k in 1 ..< batchCount where !st.prepared.contains(k) {
                    try prepare(
                        block: b, rows: unique(batchRows(rows, k, steps: steps).map(\.row)), selecting: false, state: &st,
                        keep: &keep, prefix: sharedP, label: "block \(b + 1)/\(blocks.count), batch \(k + 1)/\(batchCount)", log: log,
                        persist: { _ in })
                    st.prepared.append(k)
                }
                try commit()
                infos.append(BlockInfo(block: b, residentLayers: 0, decision: nil, losses: [], skippedUpdates: 0, trainingPeakBytes: 0, secondsPerStep: 0))
                continue
            }

            // 2. residency, from what the run's other stages measured
            let kept = Array(keep[range.start ... range.end])
            let sets = kept.map { ExpertSet(keep: $0, numExperts: source.numExperts) }
            let copies = options.mix == .both && b > 0 ? 2 : 1
            let stackFactor = copies * options.rowsPerStep
            var decision = st.decision
            // Per-layer checkpointing off is `fast`: nothing is recomputed.
            let keepSetting: ActivationKeep = configuration.checkpointBlocks ? configuration.keepIntermediates : .fast
            if decision?.keepSetting != keepSetting.description { decision = nil }
            if decision == nil || options.forcedResidentLayers != nil {
                var inputs = try ResidencyPlanner.inputs(
                    source: source, keep: keep, blockEnds: blockEnds, adapterConfiguration: adapterConfiguration,
                    sliced: source.canSliceLayers, sequence: longest, block: b)
                inputs.hidden *= stackFactor
                inputs.tokenCount *= stackFactor
                inputs.workspaceBytesPerToken *= stackFactor
                inputs.attentionBytes *= stackFactor
                var d = try ResidencyPlanner.chooseAuto(
                    inputs, otherStagesPeak: manifest.otherStagesPeak, ceiling: options.memoryBudget, keep: keepSetting)
                if let f = options.forcedResidentLayers {
                    d.residentLayers = min(max(f, 1), inputs.blockLayers)
                    d.reason = "forced to \(d.residentLayers) (test)"
                }
                decision = d
                st.decision = d
                try commit()
            }
            let resident = decision!.residentLayers
            // The expert math's token chunk: the caller's, else what the memory budget asked for.
            let priorChunk = ExpertCompute.tokenChunk
            ExpertCompute.tokenChunk = options.tokenChunk ?? decision!.tokenChunk
            defer { ExpertCompute.tokenChunk = priorChunk }
            if memory.cacheLimitBytes == nil {
                Memory.cacheLimit = options.cacheLimitBytes ?? Self.automaticCacheLimit(for: decision!.estimate.total)
            }
            log?("block \(b + 1)/\(blocks.count): \(resident) of \(range.end - range.start + 1) layers resident: \(decision!.reason)")

            // 3. train, batch by batch
            var healer = StreamedBlockHealer(
                source: source, keep: keep, blockEnds: blockEnds, adapterConfiguration: adapterConfiguration,
                configuration: configuration)
            healer.slicedStudent = source.canSliceLayers
            let sliced = healer.slicedStudent
            let blockLength = range.end - range.start + 1
            let windowed = resident < blockLength
            let windows = StreamedBlockHealer.windows(count: blockLength, resident: resident)
            // Layers (of the top window) that keep their intermediates; of the whole block when it is resident.
            let keptTop = min(decision!.keptLayers ?? 0, windows.last!.count)
            let c = configuration
            var schedule = c.learningRateSchedule(runSteps: steps)
            schedule.lengthChanges = (schedule.lengthChanges + st.lengthChanges).sorted { $0.from < $1.from }
            var adapters: ModelAdapters
            let wholeLayers: [any HealableMoELayer]
            // Every layer is loaded before the generator is seeded, so initialisation does not depend on how
            // the layers were loaded.
            if windowed {
                var dims: [LayerDimensions] = []
                for i in 0 ..< blockLength {
                    dims.append(LayerDimensions(try source.loadLayers((range.start + i) ... (range.start + i), keeping: sliced ? [kept[i]] : nil)[0]))
                }
                MLXRandom.seed(c.seed &+ UInt64(range.start))
                adapters = ModelAdapters(dimensions: dims, firstLayer: range.start, sets: sets, configuration: adapterConfiguration)
                wholeLayers = []
            } else {
                wholeLayers = try source.loadLayers(range.start ... range.end, keeping: sliced ? kept : nil)
                MLXRandom.seed(c.seed &+ UInt64(range.start))
                adapters = ModelAdapters(layers: wholeLayers, firstLayer: range.start, sets: sets, configuration: adapterConfiguration)
            }
            let opt = HealingAdamW(weightDecay: c.weightDecay)
            opt.routerLRMultiplier = adapterConfiguration.routerLRMultiplier
            var losses: [Float] = st.losses
            var skips = st.skipped
            var step = 0
            let ckpt = directory.appendingPathComponent("checkpoints\(tag)/block-\(b)")
            if !st.trainingDone, let meta = try Checkpointing.resumeBlock(from: ckpt, block: b, adapters: adapters, optimizer: opt) {
                step = meta.step
                losses = meta.losses
                skips = meta.skipped
            }
            if st.trainingDone {
                try adapters.load(tensors: try loadArrays(url: blockFile))
            }
            let net: HealingNetwork? = windowed ? nil : HealingNetwork(
                layers: wholeLayers, firstLayer: range.start, sets: sets, adapters: adapters, sliced: sliced)
            let carry = WindowCarry()
            func saveCheckpoint(_ at: Int) throws {
                try Checkpointing.saveBlock(
                    to: ckpt, adapters: adapters, optimizer: opt,
                    meta: .init(
                        block: b, step: at, optimizerStep: opt.step, losses: losses, skipped: skips,
                        saved: ISO8601DateFormatter().string(from: Date())))
                callbacks.onCheckpoint?(at, ckpt)
            }
            var seconds: [Double] = []
            Memory.peakMemory = Memory.activeMemory
            if !st.trainingDone {
                var k = step / options.batchSteps
                while k < batchCount {
                    if !st.prepared.contains(k) {
                        let before = Memory.peakMemory
                        Memory.peakMemory = Memory.activeMemory
                        try prepare(
                            block: b, rows: unique(batchRows(rows, k, steps: steps).map(\.row)), selecting: false, state: &st,
                            keep: &keep, prefix: sharedP, label: "block \(b + 1)/\(blocks.count), batch \(k + 1)/\(batchCount)", log: log,
                            persist: { _ in })
                        manifest.otherStagesPeak = max(manifest.otherStagesPeak, Memory.peakMemory)
                        Memory.peakMemory = max(before, Memory.activeMemory)
                        st.prepared.append(k)
                        try commit()
                    }
                    let end = min(steps, (k + 1) * options.batchSteps)
                    while step < end {
                        callbacks.waitWhilePaused()
                        var reason: String?
                        if callbacks.stopRequested() { reason = "stop requested" }
                        else if let limit = memory.peakStopBytes, Memory.peakMemory > limit { reason = "peak memory \(Memory.peakMemory) over \(limit)" }
                        if let reason {
                            try saveCheckpoint(step)
                            callbacks.consumeStop()
                            outcome = .stopped(step: step, reason: reason)
                            st.losses = losses
                            st.skipped = skips
                            try commit()
                            break blockLoop
                        }
                        let t0 = Date()
                        let rs = stepRows(rows, step)
                        // Stored states are read from disk one step at a time; the next step's are fetched meanwhile.
                        if step + 1 < steps {
                            for r in stepRows(rows, step + 1) {
                                for kind in ["c", "d"] { store.prefetch(row: Self.key(kind, r.key), boundary: b) }
                                store.prefetch(row: Self.key("c", r.key), boundary: b + 1)
                            }
                        }
                        var inputs: [MLXArray] = [], targets: [MLXArray] = [], ws: [MLXArray] = []
                        for row in rs {
                            let clean = try get("c", row.key, b)
                            let dirty: MLXArray? = b > 0 && options.mix != .clean ? try get("d", row.key, b) : nil
                            switch (options.mix, dirty) {
                            case (.both, let d?): inputs.append(contentsOf: [clean, d])
                            case (.dirty, let d?): inputs.append(d)
                            default: inputs.append(clean)
                            }
                            targets.append(try get("c", row.key, b + 1))
                            ws.append(row.weights)
                        }
                        let input = stack(inputs, compute: nil)
                        let lossOf = { (h: MLXArray) in HealingLoss.pinStacked(h, teachers: targets, weights: ws, copies: copies) }
                        let nb = Float(blocks.count)
                        var loss: Float, ok: Bool, lr: Float
                        if !windowed, sharedP > 0, rs.count > 1 {
                            // The rows share their first `sharedP` tokens: that part runs once per layer, forward and
                            // backward, for each input kind (clean, dirty), and the suffixes attend to its K/V.
                            BlockPipeline.sharedPrefixStepCount += 1
                            let kinds = copies == 2 ? 2 : 1
                            var perKind: [[MLXArray]] = Array(repeating: [], count: kinds)
                            for (n, x) in inputs.enumerated() { perKind[n % kinds].append(x) }
                            let pres = perKind.map { $0[0][0..., 0 ..< sharedP] }
                            let sufs = perKind.map { states in stack(states.map { $0[0..., sharedP...] }, compute: nil) }
                            let net = net!
                            net.refreshRounding()
                            let ckptOn = c.checkpointBlocks
                            let vg = valueAndGrad(model: adapters) { (_: ModelAdapters, _: [MLXArray]) in
                                var total = MLXArray(Float(0))
                                for k in 0 ..< kinds {
                                    let r = net.forwardShared(
                                        prefix: pres[k], suffix: sufs[k], student: true, collect: [range.end], checkpointed: ckptOn, keptLayers: keptTop)
                                    let (hp, hs) = r.collected[range.end]!
                                    let full = concatenated([broadcast(hp, to: [hs.dim(0), hp.dim(1), hp.dim(2)]), hs], axis: 1)
                                    total = total + HealingLoss.pinStacked(full, teachers: targets, weights: ws, copies: 1)
                                }
                                return [total / Float(kinds)]
                            }
                            let (l, g) = vg(adapters, [])
                            loss = l[0].item(Float.self)
                            lr = schedule.value(at: opt.step)
                            ok = loss.isFinite && loss <= c.maxLoss
                            if ok {
                                opt.update(model: adapters, grads: GradMath.scale(GradMath.flatten(g), by: 1 / nb), learningRate: lr)
                                eval(adapters, opt.stateArrays)
                            }
                        } else if windowed {
                            (loss, ok, lr) = try healer.windowedStep(
                                block: range, windows: windows, adapters: adapters, optimizer: opt, sets: sets, kept: kept,
                                input: input, target: targets[0], weights: ws[0], learningRate: { schedule.value(at: $0) },
                                blockCount: nb, checkpointed: c.checkpointBlocks, keptLayers: keptTop, carry: carry, loss: rs.count > 1 ? lossOf : nil)
                        } else if rs.count == 1, StreamedBlockHealer.compileLayerSteps, !adapters.needsRefresh {
                            // One row: the whole block as one window through the compiled step (its layers are resident).
                            (loss, ok, lr) = try healer.windowedStep(
                                block: range, windows: windows, adapters: adapters, optimizer: opt, sets: sets, kept: kept,
                                input: input, target: targets[0], weights: ws[0], learningRate: { schedule.value(at: $0) },
                                blockCount: nb, checkpointed: c.checkpointBlocks, keptLayers: keptTop, carry: carry, preloaded: wholeLayers)
                        } else {
                            let net = net!
                            net.refreshRounding()
                            let ckptOn = c.checkpointBlocks
                            let vg = valueAndGrad(model: adapters) { (_: ModelAdapters, _: [MLXArray]) in
                                let (_, p) = net.forward(input, student: true, collect: [range.end], checkpointed: ckptOn, keptLayers: keptTop)
                                return [lossOf(p[range.end]!)]
                            }
                            let (l, g) = vg(adapters, [])
                            loss = l[0].item(Float.self)
                            lr = schedule.value(at: opt.step)
                            ok = loss.isFinite && loss <= c.maxLoss
                            if ok {
                                opt.update(model: adapters, grads: GradMath.scale(GradMath.flatten(g), by: 1 / nb), learningRate: lr)
                                eval(adapters, opt.stateArrays)
                            }
                        }
                        if !ok { skips += 1 }
                        losses.append(loss)
                        let seen = footprint.takeStepPeak()
                        let secs = Date().timeIntervalSince(t0)
                        seconds.append(secs)
                        onStep?(b, StepRecord(
                            step: step + 1, phase: .block, loss: loss, learningRate: lr, applied: ok, seconds: secs,
                            peakMemoryBytes: Memory.peakMemory, peakFootprintBytes: seen.footprint, fileBackedBytes: seen.fileBacked))
                        Memory.clearCache()
                        step += 1
                        if let pl = options.plateau, !options.evalRows.isEmpty, st.plateau.triggeredAt == nil,
                            step % max(1, pl.evalEvery) == 0
                        {
                            let score = try heldOutScore(
                                block: b, range: range, windows: windows, adapters: adapters, kept: kept, sets: sets,
                                sliced: sliced, wholeLayers: wholeLayers, copies: copies, routing: false).loss
                            st.heldOut.append(score)
                            if pl.observe(score, step: step, state: &st.plateau) {
                                let position = schedule.offset + opt.step
                                schedule.lengthChanges.append(.init(from: position, total: position + max(1, pl.annealSteps)))
                                st.lengthChanges = schedule.lengthChanges
                                log?("block \(b + 1)/\(blocks.count): held-out loss plateaued at update \(step) (\(String(format: "%.5f", score))); annealing over the next \(pl.annealSteps) updates")
                            }
                            st.losses = losses
                            try commit()
                        }
                        if step < end, step % max(1, c.checkpointEvery) == 0 { try saveCheckpoint(step) }
                    }
                    try saveCheckpoint(step)
                    k += 1
                }
                trainingPeak = max(trainingPeak, Memory.peakMemory)
                if !options.evalRows.isEmpty {
                    let final = try heldOutScore(
                        block: b, range: range, windows: windows, adapters: adapters, kept: kept, sets: sets,
                        sliced: sliced, wholeLayers: wholeLayers, copies: copies)
                    st.finalHeldOut = final.loss
                    st.finalRouting = final.routing
                }
                st.losses = losses
                st.skipped = skips
                let t = adapters.tensors()
                try fm.createDirectory(at: blockFile.deletingLastPathComponent(), withIntermediateDirectories: true)
                let tmp = blockFile.deletingLastPathComponent().appendingPathComponent("block-\(b).partial.safetensors")
                try MLX.save(arrays: t, url: tmp)
                if fm.fileExists(atPath: blockFile.path) { try fm.removeItem(at: blockFile) }
                try fm.moveItem(at: tmp, to: blockFile)
                st.trainingDone = true
                try commit()
                try? fm.removeItem(at: ckpt)
            }
            let finished = adapters.tensors()
            // The report's adapters are read back from the block's file (lazily, as `loadArrays` does), so a finished
            // block's adapters are not held in memory through the rest of the run; they are read when the caller saves them.
            adaptersAll.merge(fm.fileExists(atPath: blockFile.path) ? try loadArrays(url: blockFile) : finished) { $1 }
            infos.append(BlockInfo(
                block: b, residentLayers: resident, decision: st.decision, losses: losses, skippedUpdates: skips,
                trainingPeakBytes: trainingPeak, secondsPerStep: seconds.isEmpty ? 0 : seconds.dropFirst().reduce(0, +) / Double(max(1, seconds.count - 1)),
                heldOut: st.heldOut, annealedAt: st.plateau.triggeredAt, finalHeldOut: st.finalHeldOut,
                finalRouting: st.finalRouting))
            onBlockFinished?(b, finished)

            // 4. the next frontier: healed outputs of every row, then delete this one
            if b + 1 < blocks.count, !options.trainOnly {
                Memory.peakMemory = Memory.activeMemory
                do {
                    try frontier(block: b, rows: sequences, adapters: adapters, kept: kept, sets: sets, callbacks: callbacks)
                } catch is StopWhileWritingFrontier {
                    callbacks.consumeStop()
                    outcome = .stopped(step: steps, reason: "stop requested while writing the frontier after block \(b + 1); run again to resume")
                    break blockLoop
                }
                manifest.otherStagesPeak = max(manifest.otherStagesPeak, Memory.peakMemory)
                for r in sequences {
                    for kind in ["c", "d", "p"] { store.remove(row: Self.key(kind, r.key), boundary: b) }
                }
            }
            st.frontierDone = true
            try commit()
        }
        return Report(
            adapters: adaptersAll, keep: keep, selections: allSelections, blocks: infos,
            otherStagesPeakBytes: manifest.otherStagesPeak, trainingPeakBytes: trainingPeak,
            frontierBytes: store.diskBytes(), outcome: outcome)
    }

    /// Block 0's inputs: every row's embedding, written once with the table loaded once.
    func embed(_ rows: [HealingRow]) throws {
        let todo = rows.filter { !has("c", $0.key, 0) }
        guard !todo.isEmpty else { return }
        let embedder = try source.makeEmbedder()
        for g in TeacherPrepass.batches(lengths: todo.map(\.length), maxTokens: options.maxBatchTokens, maxRows: options.maxBatchRows) {
            let batch = g.map { todo[$0] }
            let longest = batch.map(\.length).max()!
            let tokens = concatenated(batch.map { r -> MLXArray in
                let t = r.tokenArray
                return r.length == longest ? t : concatenated([t, MLXArray.zeros([1, longest - r.length], dtype: t.dtype)], axis: 1)
            }, axis: 0)
            let h = try embedder(tokens)
            eval(h)
            for (j, r) in batch.enumerated() { try put("c", r.key, 0, h[j ..< j + 1, 0 ..< r.length]) }
            Memory.clearCache()
        }
    }

    // MARK: prepare: the teacher pass (and selection)

    /// Where a teacher walk's time went: loading the layers (and the bytes of weights that is), the first evaluation
    /// of each kernel shape this process has not run before (which on a GPU includes its JIT compile), and the
    /// rest of the evaluations.
    struct TeacherTiming {
        var passes = 0
        var rows = 0
        var layerLoads = 0
        var loadSeconds = 0.0
        var loadBytes = 0
        var firstCallSeconds = 0.0
        var firstCalls = 0
        var computeSeconds = 0.0
        var selectionSeconds = 0.0
        var fileBackedGrowth = 0
        var wallSeconds = 0.0

        static func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824) }
        func line(_ label: String) -> String {
            var s = String(format: "%@: teacher outputs %.1f s = read %.1f s (%@)", label, wallSeconds, loadSeconds, Self.gb(loadBytes))
            s += String(format: " + first-call/compile %.1f s (%d new kernel shapes) + compute %.1f s", firstCallSeconds, firstCalls, computeSeconds)
            if selectionSeconds > 0 { s += String(format: " + selection %.1f s", selectionSeconds) }
            s += "; \(rows) rows in \(passes) walk\(passes == 1 ? "" : "s"), \(layerLoads) layer loads"
            if fileBackedGrowth > 0 { s += ", file-backed pages +\(Self.gb(fileBackedGrowth)) (pages of mapped weights are read on first use, inside the compute figure)" }
            return s
        }
    }

    /// Kernel shapes (rows x tokens x hidden, dtype, attention shape) a teacher walk has evaluated in this process: the
    /// first evaluation of one is timed apart as first-call/compile time.
    var seenWalkShapes = Set<String>()
    /// The model's hidden size, read once from the probe layer at the start of a run.
    var hiddenSize = 1

    /// Walks `rows` through block `b`'s teacher layers one layer at a time with every activation in RAM,
    /// and writes the teacher's block ends. With `selecting`, each layer's experts are chosen in the same
    /// pass from the rows in `selectionRows` (default: those of `rows` that have a selection item); their pruned
    /// stream advances with the chosen set.
    ///
    /// The other rows are walked in groups whose hidden states fit ``BlockPipelineOptions/teacherStateBudgetBytes``;
    /// each group is one walk, which loads every layer of the block once. Selection happens in the first.
    func prepare(
        block b: Int, rows: [HealingRow], selecting: Bool, selectionRows: [HealingRow]? = nil, state st: inout BlockState,
        keep: inout [[Int]], prefix: Int = 0, label: String = "", log: ((String) -> Void)? = nil,
        persist: (BlockState) -> Void
    ) throws {
        let range = blockRanges(ends: blockEnds)[b]
        let candidates = selectionRows ?? rows
        let selRows = selecting ? candidates.filter { options.items[$0.key] != nil } : []
        if selecting && selRows.isEmpty {
            throw HealingError.invalid("block \(b + 1): none of its first batch's rows has a selection item")
        }
        let selKeys = Set(selRows.map(\.key))
        let plain = rows.filter { !selKeys.contains($0.key) && !has("c", $0.key, b + 1) }
        if plain.isEmpty && selRows.isEmpty { return }
        // Groups of rows of similar length, then walks of as many groups as fit the state budget. A walk costs one
        // read of every layer of the block, so the fewer walks the better; the budget only bounds the hidden states.
        let groups = TeacherPrepass.batches(lengths: plain.map(\.length), maxTokens: options.maxBatchTokens, maxRows: options.maxBatchRows)
        let perToken = max(1, hiddenSize * 4)  // the compute dtype's widest; storage may be narrower
        var walks: [[[Int]]] = []
        var current: [[Int]] = []
        var currentBytes = selRows.reduce(0) { $0 + 2 * $1.length * perToken }
        for g in groups {
            let longest = g.map { plain[$0].length }.max() ?? 0
            let bytes = g.count * longest * perToken
            if !current.isEmpty && currentBytes + bytes > options.teacherStateBudgetBytes {
                walks.append(current)
                current = []
                currentBytes = 0
            }
            current.append(g)
            currentBytes += bytes
        }
        if !current.isEmpty || walks.isEmpty { walks.append(current) }
        var timing = TeacherTiming()
        let started = Date()
        let fileBackedBefore = ProcessMemory.snapshot().fileBacked
        for (n, walk) in walks.enumerated() {
            try prepareWalk(
                block: b, plain: plain, groups: walk, selRows: n == 0 ? selRows : [], selecting: selecting && n == 0,
                state: &st, keep: &keep, prefix: prefix, timing: &timing, persist: persist)
            timing.passes += 1
        }
        timing.rows = plain.count + selRows.count
        timing.wallSeconds = Date().timeIntervalSince(started)
        timing.fileBackedGrowth = max(0, ProcessMemory.snapshot().fileBacked - fileBackedBefore)
        log?(timing.line(label))
    }

    /// One walk of block `b`'s teacher layers over `groups` of the `plain` rows (and the selection rows).
    private func prepareWalk(
        block b: Int, plain: [HealingRow], groups: [[Int]], selRows: [HealingRow], selecting: Bool,
        state st: inout BlockState, keep: inout [[Int]], prefix: Int, timing: inout TeacherTiming,
        persist: (BlockState) -> Void
    ) throws {
        let range = blockRanges(ends: blockEnds)[b]
        var compute: DType?
        var h: [MLXArray] = try groups.map { g in
            stack(try g.map { try get("c", plain[$0].key, b) }, compute: nil, bucketed: true)
        }
        var pre: [MLXArray] = [], suf: [MLXArray] = []
        var hf: [MLXArray] = try selRows.map { try get("c", $0.key, b) }
        var hp: [MLXArray] = try selRows.map { r in b == 0 ? try get("c", r.key, b) : try get("p", r.key, b) }
        let items = selRows.map { options.items[$0.key]! }
        let selector = options.selection.map { ExpertSelector(configuration: $0) }
        var existing = Dictionary(uniqueKeysWithValues: st.selections.map { ($0.layer, $0) })
        // Runs `body` (which evaluates) and files its time under first-call/compile when this kernel shape is new
        // to the process, else under compute.
        func timed(_ shape: String, _ body: () -> Void) {
            let t0 = Date()
            body()
            let dt = Date().timeIntervalSince(t0)
            if seenWalkShapes.insert(shape).inserted {
                timing.firstCallSeconds += dt
                timing.firstCalls += 1
            } else {
                timing.computeSeconds += dt
            }
        }
        for l in range.start ... range.end {
            let t0 = Date()
            let layer = try source.loadLayers(l ... l, keeping: nil)[0]
            timing.loadSeconds += Date().timeIntervalSince(t0)
            timing.layerLoads += 1
            if let m = layer as? Module { timing.loadBytes += m.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } }
            guard layer.numExperts == source.numExperts else {
                throw HealingError.invalid("the teacher source's layers no longer have all \(source.numExperts) experts")
            }
            if compute == nil { compute = TeacherPrepass.computeDType([layer]) }
            var kind = "layer"
            if let m = layer as? Gemma4MoEDecoderLayer { kind = "\(m.selfAttn.numHeads)x\(m.selfAttn.headDim)" }
            if prefix > 0 {
                BlockPipeline.sharedPrefixWalkCount += 1
                for g in h.indices {
                    if pre.count <= g {
                        pre.append(h[g][0 ..< 1, 0 ..< prefix])
                        suf.append(h[g][0..., prefix...])
                    }
                    let x = compute.map { pre[g].dtype == $0 ? (pre[g], suf[g]) : (pre[g].asType($0), suf[g].asType($0)) } ?? (pre[g], suf[g])
                    timed("shared \(x.0.shape) \(x.1.shape) \(x.1.dtype) \(kind)") {
                        (pre[g], suf[g]) = healingLayerForwardShared(layer, prefix: x.0, suffix: x.1, set: nil, adapters: nil)
                        eval(pre[g], suf[g])
                    }
                }
            } else {
                for i in h.indices {
                    let x = compute.map { h[i].dtype == $0 ? h[i] : h[i].asType($0) } ?? h[i]
                    timed("\(x.shape) \(x.dtype) \(kind)") {
                        h[i] = healingLayerForward(layer, x, set: nil, adapters: nil)
                        eval(h[i])
                    }
                }
            }
            if selecting, let selector {
                let s0 = Date()
                let rec = selector.step(layer: layer, index: l, items: items, hf: &hf, hp: &hp, existing: existing[l])
                timing.selectionSeconds += Date().timeIntervalSince(s0)
                existing[l] = rec
                keep[l] = rec.chosen
                st.selections.removeAll { $0.layer == l }
                st.selections.append(rec)
                persist(st)
            }
            Memory.clearCache()
        }
        for (g, group) in groups.enumerated() {
            for (j, i) in group.enumerated() {
                if prefix > 0 {
                    try put("c", plain[i].key, b + 1, concatenated([pre[g], suf[g][j ..< j + 1, 0 ..< (plain[i].length - prefix)]], axis: 1))
                } else {
                    try put("c", plain[i].key, b + 1, h[g][j ..< j + 1, 0 ..< plain[i].length])
                }
            }
        }
        for (i, r) in selRows.enumerated() {
            try put("c", r.key, b + 1, hf[i])
            try put("p", r.key, b + 1, hp[i])
        }
    }

    /// The shared prefix to use for `rows` (0 for none).
    func sharedPrefixLength(_ rows: [HealingRow]) throws -> Int {
        guard options.sharedPrefix != .off, rows.count > 1, let first = rows.first else { return 0 }
        var common = first.length
        for r in rows.dropFirst() {
            var n = 0
            let m = min(common, r.length)
            while n < m, r.tokens[n] == first.tokens[n] { n += 1 }
            common = n
        }
        let usable = min(common, (rows.map(\.length).min() ?? 1) - 1)
        switch options.sharedPrefix {
        case .off: return 0
        case .auto: return usable >= options.sharedPrefixMinimum ? usable : 0
        case .length(let n):
            guard n <= usable else { throw HealingError.invalid("sharedPrefix length \(n) but the rows share only \(usable) leading tokens") }
            return n
        }
    }

    // MARK: held-out score

    /// Mean pin loss of the block's student on ``BlockPipelineOptions/evalRows`` (on the inputs it trains on),
    /// forward only, one window of layers at a time.
    func heldOutScore(
        block b: Int, range: (start: Int, end: Int), windows: [ClosedRange<Int>], adapters: ModelAdapters, kept: [[Int]],
        sets: [ExpertSet], sliced: Bool, wholeLayers: [any HealableMoELayer], copies: Int, routing: Bool = true
    ) throws -> (loss: Float, routing: RoutingAgreement?) {
        let rows = options.evalRows
        // Sums over layers of each row's pin-weighted mean (KL, size, overlap) of the router correction.
        let measure = routing && adapters.layers.contains { $0.router != nil }
        var routingSum: [MLXArray] = rows.map { _ in MLXArray.zeros([RoutingChannel.count]) }
        var routingLayers = 0
        var states: [MLXArray] = try rows.map { r in
            let clean = try get("c", r.key, b)
            guard b > 0, options.mix != .clean else { return clean }
            let dirty = try get("d", r.key, b)
            return options.mix == .both ? concatenated([clean, dirty], axis: 0) : dirty
        }
        let walks: [ClosedRange<Int>] = wholeLayers.isEmpty ? windows : [0 ... (range.end - range.start)]
        for win in walks {
            let layers = wholeLayers.isEmpty
                ? try source.loadLayers((range.start + win.lowerBound) ... (range.start + win.upperBound), keeping: sliced ? Array(kept[win]) : nil)
                : Array(wholeLayers[win])
            for (j, layer) in layers.enumerated() {
                let i = win.lowerBound + j
                adapters.refreshRounding(layerAt: i, layer: layer, set: sets[i], sliced: sliced)
                if measure { routingLayers += 1 }
                for n in states.indices {
                    if measure {
                        let (y, aux) = healingLayerForwardAux(layer, states[n], set: sets[i], adapters: adapters.layers[i], sliced: sliced)
                        states[n] = y
                        let w = rows[n].weights
                        routingSum[n] = routingSum[n] + ((aux * w.reshaped(1, -1, 1)).sum(axis: 1) / maximum(w.sum(), 1)).mean(axis: 0)
                        eval(states[n], routingSum[n])
                    } else {
                        states[n] = healingLayerForward(layer, states[n], set: sets[i], adapters: adapters.layers[i], sliced: sliced)
                        eval(states[n])
                    }
                }
            }
        }
        var total: Float = 0
        for (n, r) in rows.enumerated() {
            let t = try get("c", r.key, b + 1)
            total += HealingLoss.pinStacked(states[n], teachers: [t], weights: [r.weights], copies: copies).item(Float.self)
        }
        Memory.clearCache()
        var agreement: RoutingAgreement?
        if measure, routingLayers > 0 {
            var sum = MLXArray.zeros([RoutingChannel.count])
            for r in routingSum { sum = sum + r }
            let m = (sum / Float(routingLayers * max(1, rows.count))).asArray(Float.self)
            agreement = RoutingAgreement(kl: m[RoutingChannel.kl], topKOverlap: m[RoutingChannel.overlap], logitDeltaSquared: m[RoutingChannel.l2])
        }
        return (total / Float(max(1, rows.count)), agreement)
    }

    // MARK: the next frontier

    /// Runs the healed block over every row's dirty input and writes the dirty states entering the next
    /// block, plus the teacher's block end for any row the batches did not cover.
    func frontier(
        block b: Int, rows: [HealingRow], adapters: ModelAdapters, kept: [[Int]], sets: [ExpertSet],
        callbacks: HealingCallbacks
    ) throws {
        let range = blockRanges(ends: blockEnds)[b]
        let sliced = source.canSliceLayers
        let todo = rows.filter { !has("d", $0.key, b + 1) || !has("c", $0.key, b + 1) }
        let groups = TeacherPrepass.batches(lengths: todo.map(\.length), maxTokens: options.maxBatchTokens, maxRows: options.maxBatchRows)
        for g in groups {
            if callbacks.stopRequested() { throw StopWhileWritingFrontier() }
            let batch = g.map { todo[$0] }
            let needClean = batch.filter { !has("c", $0.key, b + 1) }
            var compute: DType?
            // Student walk on the dirty input (block 0's is the clean one).
            var d = stack(try batch.map { try get(b == 0 ? "c" : "d", $0.key, b) }, compute: nil, bucketed: true)
            for l in range.start ... range.end {
                let i = l - range.start
                let layer = try source.loadLayers(l ... l, keeping: sliced ? [kept[i]] : nil)[0]
                if compute == nil { compute = TeacherPrepass.computeDType([layer]) }
                adapters.refreshRounding(layerAt: i, layer: layer, set: sets[i], sliced: sliced)
                d = healingLayerForward(layer, compute.map { d.dtype == $0 ? d : d.asType($0) } ?? d, set: sets[i], adapters: adapters.layers[i], sliced: sliced)
                eval(d)
            }
            for (j, r) in batch.enumerated() { try put("d", r.key, b + 1, stopGradient(d[j ..< j + 1, 0 ..< r.length])) }
            if !needClean.isEmpty {
                var t = stack(try needClean.map { try get("c", $0.key, b) }, compute: nil, bucketed: true)
                for l in range.start ... range.end {
                    let layer = try source.loadLayers(l ... l, keeping: nil)[0]
                    t = healingLayerForward(layer, compute.map { t.dtype == $0 ? t : t.asType($0) } ?? t, set: nil, adapters: nil)
                    eval(t)
                }
                for (j, r) in needClean.enumerated() { try put("c", r.key, b + 1, t[j ..< j + 1, 0 ..< r.length]) }
            }
            Memory.clearCache()
        }
    }
}

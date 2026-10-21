import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Layer source

/// Where a streamed run gets its layers. Only the layers of the block being worked on are
/// resident, so a source loads them on demand and drops them when the caller does.
public protocol LayerSource {
    var layerCount: Int { get }
    /// The unpruned expert count of every layer.
    var numExperts: Int { get }
    /// Hidden states entering layer 0 for each token batch (`[1, T]`). The embedding table is
    /// loaded for this call and released after it.
    func embed(_ tokens: [MLXArray]) throws -> [MLXArray]
    /// A function from token ids `[B, T]` to the hidden states entering layer 0. It keeps the
    /// embedding table resident until it is dropped, so a caller embedding many batches loads
    /// the table once.
    func makeEmbedder() throws -> (MLXArray) throws -> MLXArray
    /// The layers in `range`. With `keeping` (one kept-expert list per layer in the range) each
    /// layer comes back already cut to its kept experts, so a layer's unkept experts are never
    /// resident together with the rest of the block.
    func loadLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) throws -> [any HealableMoELayer]
    /// Whether `loadLayers(_:keeping:)` is allowed to cut layers to their kept experts. A student
    /// is trained sliced exactly when its source can.
    var canSliceLayers: Bool { get }
    /// Whether cutting layers leaves the source able to serve full layers afterwards. An in-memory
    /// model that is cut in place cannot: afterwards it is no teacher.
    var slicesWithoutConsumingSource: Bool { get }
    /// A string identifying the model the layers come from (config, tensor layout, quantization,
    /// load dtype), stored in the teacher pre-pass's manifest so a store is only reused with the
    /// model that filled it.
    func teacherFingerprint() throws -> String
    /// Hints that the layers in `range` are about to be loaded, so a source reading from disk can
    /// start fetching them while the caller computes. Must return at once and may do nothing.
    func prefetchLayers(_ range: ClosedRange<Int>)
    /// As ``prefetchLayers(_:)`` for layers that will be loaded cut to `keeping`: only the kept experts' rows.
    func prefetchLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?)
}

extension LayerSource {
    public func prefetchLayers(_ range: ClosedRange<Int>) {}
    public func prefetchLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) { prefetchLayers(range) }
    public var canSliceLayers: Bool { true }
    public var slicesWithoutConsumingSource: Bool { true }
    public func loadLayers(_ range: ClosedRange<Int>) throws -> [any HealableMoELayer] {
        try loadLayers(range, keeping: nil)
    }
    public func makeEmbedder() throws -> (MLXArray) throws -> MLXArray {
        { tokens in
            let rows = (0 ..< tokens.dim(0)).map { tokens[$0 ..< $0 + 1] }
            return concatenated(try self.embed(rows), axis: 0)
        }
    }
}

/// Wall-clock seconds and count of layer loads by ``Gemma4LayerSource``, for measuring how much of a step is I/O.
public enum LayerLoadStats {
    nonisolated(unsafe) public private(set) static var seconds = 0.0
    nonisolated(unsafe) public private(set) static var count = 0
    static func record(_ s: Double) { seconds += s; count += 1 }
    public static func reset() { seconds = 0; count = 0 }
}

/// Layers read one block at a time from a Hugging Face Gemma 4 checkpoint directory.
public struct Gemma4LayerSource: LayerSource {
    public let checkpoint: Gemma4Checkpoint
    public let dtype: DType?
    /// Load each layer's tensors as read-only views of the checkpoint files' pages instead of
    /// reading them into memory (see ``MappedSafetensors``). Falls back to copying where the system
    /// does not share the pages. Default on.
    public let mapWeights: Bool
    public init(checkpoint: Gemma4Checkpoint, dtype: DType? = nil, mapWeights: Bool = true) {
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.mapWeights = mapWeights
    }
    public func prefetchLayers(_ range: ClosedRange<Int>) { checkpoint.prefetchLayers(range) }
    public func prefetchLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) {
        if let keeping { checkpoint.prefetchLayers(range, keeping: keeping) } else { checkpoint.prefetchLayers(range) }
    }
    public var layerCount: Int { checkpoint.configuration.numHiddenLayers }
    public var numExperts: Int { checkpoint.configuration.numExperts ?? 0 }
    public func teacherFingerprint() throws -> String { try checkpoint.fingerprint(loadDType: dtype) }
    public func embed(_ tokens: [MLXArray]) throws -> [MLXArray] {
        let table = try checkpoint.loadEmbedding(dtype: dtype)
        let scale = Float(checkpoint.configuration.hiddenSize).squareRoot()
        let out = tokens.map { table($0) * scale }
        eval(out)
        return out
    }
    public func makeEmbedder() throws -> (MLXArray) throws -> MLXArray {
        let table = try checkpoint.loadEmbedding(dtype: dtype)
        let scale = Float(checkpoint.configuration.hiddenSize).squareRoot()
        return { table($0) * scale }
    }
    /// Layers are loaded and cut one at a time, so the transient peak is one layer with all its
    /// experts.
    public func loadLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) throws -> [any HealableMoELayer] {
        if let keeping {
            precondition(keeping.count == range.count, "one kept set per layer in the range")
        }
        let t0 = Date()
        defer { LayerLoadStats.record(Date().timeIntervalSince(t0)) }
        return try range.enumerated().map { j, i in
            let l = try checkpoint.loadDecoderLayer(i, dtype: dtype, mapped: mapWeights, keeping: keeping?[j])
            guard let m = l as? Gemma4MoEDecoderLayer else {
                throw HealingError.invalid("layer \(i) has no MoE block")
            }
            m.train(false)
            return m
        }
    }
}

/// Layers of a model that is already in memory (tests, small models).
///
/// Cutting these layers to their kept experts would rewrite the model's own parameters and end
/// its life as a teacher, so `loadLayers(_:keeping:)` refuses unless the source was created with
/// `cutsModelInPlace: true` (run the teacher pre-pass first, or give it a model you can discard).
/// A ``StreamedBlockHealer`` over a plain `ModelLayerSource` therefore trains behind the mask.
public struct ModelLayerSource: LayerSource {
    public let model: any HealableMoEModel
    public let numExperts: Int
    public let cutsModelInPlace: Bool
    public init(model: any HealableMoEModel, cutsModelInPlace: Bool = false) {
        self.model = model
        self.numExperts = model.healableLayers.first?.numExperts ?? 0
        self.cutsModelInPlace = cutsModelInPlace
    }
    public var canSliceLayers: Bool { cutsModelInPlace }
    /// Cutting these layers rewrites the model itself, so afterwards it is no teacher.
    public var slicesWithoutConsumingSource: Bool { !cutsModelInPlace }
    /// Hashes the layer count and every parameter's name, shape and dtype. It cannot tell apart two
    /// models of the same architecture with different weights, and a model already cut to its kept
    /// experts has a different fingerprint from the uncut one it came from.
    public func teacherFingerprint() throws -> String {
        var fp = Fingerprint()
        fp.add("in-memory layers=\(model.healableLayers.count) experts=\(numExperts)")
        if let m = model as? Module {
            for (k, v) in m.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
                fp.add("\(k) \(v.shape) \(v.dtype)")
            }
        }
        return fp.hex
    }
    public var layerCount: Int { model.healableLayers.count }
    public func embed(_ tokens: [MLXArray]) throws -> [MLXArray] { tokens.map { model.embed($0) } }
    public func makeEmbedder() throws -> (MLXArray) throws -> MLXArray { { [model] in model.embed($0) } }
    public func loadLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) throws -> [any HealableMoELayer] {
        let layers = Array(model.healableLayers[range])
        if let keeping {
            guard cutsModelInPlace else {
                throw HealingError.invalid(
                    "cutting an in-memory model to its kept experts would destroy it as a teacher; "
                        + "create the ModelLayerSource with cutsModelInPlace: true, or train behind the mask")
            }
            precondition(keeping.count == range.count, "one kept set per layer in the range")
            for (l, k) in zip(layers, keeping) where l.numExperts == numExperts {
                // A layer already cut (an earlier load of the same range) is left as it is.
                sliceToKeptExperts(l, set: ExpertSet(keep: k, numExperts: numExperts))
            }
        }
        return layers
    }
}

// MARK: - Boundary store

/// Holds the teacher's hidden state at every block boundary for every row: boundary `b` is
/// the state entering block `b` (boundary 0 is the embedding, boundary `b + 1` the end of
/// block `b`). States are stored unpadded, `[1, T, D]`, in the storage dtype the pre-pass chose.
public protocol BoundaryStore {
    func put(_ state: MLXArray, row: String, boundary: Int) throws
    func get(row: String, boundary: Int) throws -> MLXArray
    /// Whether a complete state is stored. A state is only ever visible once fully written.
    func has(row: String, boundary: Int) -> Bool
    /// Small named blobs next to the states (the pre-pass keeps its manifest here).
    func metadata(named name: String) throws -> Data?
    func putMetadata(_ data: Data, named name: String) throws
    /// Deletes a stored state (no error if it is absent), for a run that only keeps the data it still needs.
    func remove(row: String, boundary: Int)
    /// Hints that a state is about to be read, so a store on disk can start fetching it while the caller
    /// computes. Must return at once and may do nothing.
    func prefetch(row: String, boundary: Int)
}

extension BoundaryStore {
    public func prefetch(row: String, boundary: Int) {}
}

public final class InMemoryBoundaryStore: BoundaryStore {
    private var states: [String: MLXArray] = [:]
    private var blobs: [String: Data] = [:]
    public init() {}
    public func put(_ state: MLXArray, row: String, boundary: Int) throws {
        eval(state)
        states["\(row)#\(boundary)"] = state
    }
    public func get(row: String, boundary: Int) throws -> MLXArray {
        guard let s = states["\(row)#\(boundary)"] else {
            throw HealingError.invalid("no boundary \(boundary) stored for row \(row)")
        }
        return s
    }
    public func has(row: String, boundary: Int) -> Bool { states["\(row)#\(boundary)"] != nil }
    public func remove(row: String, boundary: Int) { states["\(row)#\(boundary)"] = nil }
    public func metadata(named name: String) throws -> Data? { blobs[name] }
    public func putMetadata(_ data: Data, named name: String) throws { blobs[name] = data }
}

/// One safetensors file per (row, boundary) under a directory. Files are written to a
/// temporary name and renamed into place, so a file that exists is complete.
public final class DiskBoundaryStore: BoundaryStore {
    public let directory: URL
    /// Bumped whenever file naming changes; a directory without this marker, or with another
    /// number, was written by a different layout and is refused rather than half-reused.
    static let layoutVersion = 2
    static let layoutFile = "store-layout.json"

    public init(directory: URL) throws {
        self.directory = directory
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let marker = directory.appendingPathComponent(Self.layoutFile)
        if fm.fileExists(atPath: marker.path) {
            let found = (try? JSONSerialization.jsonObject(with: Data(contentsOf: marker)) as? [String: Int])?["layout"]
            guard found == Self.layoutVersion else {
                throw HealingError.invalid("the boundary store at \(directory.lastPathComponent) uses file layout \(found.map(String.init) ?? "?"), not \(Self.layoutVersion); use a fresh directory")
            }
        } else {
            let existing = try fm.contentsOfDirectory(atPath: directory.path)
            guard existing.isEmpty else {
                throw HealingError.invalid("the boundary store directory \(directory.lastPathComponent) holds files from an older layout (or something else); use a fresh directory")
            }
            try JSONSerialization.data(withJSONObject: ["layout": Self.layoutVersion]).write(to: marker, options: .atomic)
        }
    }
    /// A readable prefix, then a hash of the key. The hash is what makes the name unique: a
    /// case-insensitive volume maps `Ab` and `ab` to one file, and a name derived from the key's
    /// characters alone can collide there (or run past the filename length limit).
    private func name(_ row: String, _ boundary: Int) -> String {
        var fp = Fingerprint()
        fp.add(row)
        let prefix = String(row.unicodeScalars.lazy.filter { $0.isASCII && CharacterSet.alphanumerics.contains($0) }
            .prefix(24).map(Character.init)).lowercased()
        return "\(prefix)-\(fp.hex.prefix(32)).\(boundary)"
    }
    private func url(_ row: String, _ boundary: Int) -> URL {
        directory.appendingPathComponent(name(row, boundary) + ".safetensors")
    }
    public func put(_ state: MLXArray, row: String, boundary: Int) throws {
        let tmp = directory.appendingPathComponent(name(row, boundary) + ".partial.safetensors")
        try MLX.save(arrays: ["h": state], url: tmp)
        let dest = url(row, boundary)
        let fm = FileManager.default
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: tmp, to: dest)
    }
    public func get(row: String, boundary: Int) throws -> MLXArray {
        try loadArrays(url: url(row, boundary))["h"]!
    }
    public func has(row: String, boundary: Int) -> Bool {
        FileManager.default.fileExists(atPath: url(row, boundary).path)
    }
    public func remove(row: String, boundary: Int) { try? FileManager.default.removeItem(at: url(row, boundary)) }
    /// Asks the kernel to read the state's file into the page cache from a background queue (no MLX array is
    /// built there), so the `get` of the next step finds it in memory.
    public func prefetch(row: String, boundary: Int) {
        let path = url(row, boundary).path
        Self.prefetchQueue.async {
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { return }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size > 0 else { return }
            ReadAhead.advise(fd: fd, offset: 0, count: Int(st.st_size))
        }
    }
    private static let prefetchQueue = DispatchQueue(label: "shears.state-prefetch", qos: .utility)
    /// Bytes of every file under the directory (what the store costs on disk).
    public func diskBytes() -> Int {
        let fm = FileManager.default
        guard let files = fm.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var n = 0
        for case let u as URL in files { n += (try? u.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        return n
    }
    public func metadata(named name: String) throws -> Data? {
        let u = directory.appendingPathComponent("\(name).json")
        return FileManager.default.fileExists(atPath: u.path) ? try Data(contentsOf: u) : nil
    }
    public func putMetadata(_ data: Data, named name: String) throws {
        try data.write(to: directory.appendingPathComponent("\(name).json"), options: .atomic)
    }
}

// MARK: - Streamed block phase

/// The block phase with one block resident at a time.
///
/// **This covers the block phase only.** The free-running, dirty and KL phases backpropagate
/// across, or read the output of, every block; with the teacher pre-computed they run on a
/// ``PruneHealer`` holding only the sliced student (see ``PruneHealer/init(model:keep:adapterConfiguration:configuration:blockEnds:initialAdapters:student:teacher:)``).
/// Selection needs the whole unpruned model resident.
///
/// It works because block-phase adapters are disjoint and each block's loss reads only the
/// teacher's states at its two boundaries. A first pass advances the teacher through the
/// blocks one at a time and stores each row's boundary states. Then each block is loaded once
/// and trained through all the steps, with its own optimizer. With the same row sequence, the
/// same schedule, gradients scaled by `1 / blockCount` (as the in-memory step does), no
/// clipping and no skipped updates, the result equals the in-memory block phase up to float
/// noise.
///
/// Memory is one block's layers and adapters (plus optimizer state for the adapters) and one
/// sequence's activations, whatever the model's depth. By default (`slicedStudent`) the block's
/// layers are cut to their kept experts as they load, so the unkept experts of a block are never
/// resident beside the adapters; the teacher is never run here at all. The price is the store:
/// `T x D` per row per boundary (see ``TeacherPrepass``).
public struct StreamedBlockHealer {
    public let source: LayerSource
    public let keep: [[Int]]
    public let blockEnds: [Int]
    public let adapterConfiguration: AdapterConfiguration
    public let configuration: HealingConfiguration
    public let initialAdapters: [String: MLXArray]?
    /// Train on layers cut to their kept experts (the default whenever the source can do that
    /// without being consumed), or on full layers behind the routing mask, which gives the same
    /// numbers with every expert resident.
    public var slicedStudent: Bool

    public struct Report {
        public var adapters: [String: MLXArray]
        public var blockLosses: [[Float]]
        /// Updates skipped per block because the loss was non-finite or above `maxLoss`.
        public var skippedUpdates: [Int]
        public var peakMemoryBytes: Int
        /// Highest process physical footprint (`phys_footprint`) sampled during training, bytes.
        public var peakFootprintBytes: Int = 0
        /// The layers held at once and the memory model's estimate for them (`nil` without a budget).
        public var residentLayers: Int = 0
        public var estimate: ResidencyEstimate?
        /// The highest MLX peak that went over the limit a run was asked to stay under (`memoryBudget` or
        /// `MemoryConfiguration.peakStopBytes`); `nil` when none did. MLX's own memory limit does not stop an
        /// allocation, so the run only stops at the next check; this says it happened.
        public var peakOverLimit: Int?
        /// `.stopped` when a stop was requested (or the peak-memory limit passed) before every block
        /// finished; `adapters` then holds the finished blocks only, and the block that was in
        /// progress is in its checkpoint.
        public var outcome: HealingOutcome = .finished
    }

    public init(
        source: LayerSource, keep: [[Int]], blockEnds: [Int],
        adapterConfiguration: AdapterConfiguration = .init(),
        configuration: HealingConfiguration = .init(), initialAdapters: [String: MLXArray]? = nil
    ) {
        precondition(configuration.clipNorm == 0, "streamed blocks can't clip by global norm")
        self.source = source
        self.slicedStudent = source.canSliceLayers
        self.keep = keep
        self.blockEnds = blockEnds
        self.adapterConfiguration = adapterConfiguration
        self.configuration = configuration
        self.initialAdapters = initialAdapters
    }

    var blocks: [(start: Int, end: Int)] { blockRanges(ends: blockEnds) }

    private func prepare(_ range: (start: Int, end: Int), sliced: Bool) throws
        -> (layers: [any HealableMoELayer], sets: [ExpertSet])
    {
        let kept = Array(keep[range.start ... range.end])
        let layers = try source.loadLayers(range.start ... range.end, keeping: sliced ? kept : nil)
        let sets = kept.map { ExpertSet(keep: $0, numExperts: source.numExperts) }
        return (layers, sets)
    }

    /// Runs the teacher pre-pass (``TeacherPrepass``) with this healer's source and blocks.
    @discardableResult
    public func precomputeBoundaries(
        rows: [HealingRow], store: BoundaryStore, options: TeacherPrepass.Options = .init(),
        shouldStop: (() -> Bool)? = nil
    ) throws -> TeacherPrepass.Report {
        try TeacherPrepass(source: source, blockEnds: blockEnds, options: options)
            .run(rows: rows, store: store, shouldStop: shouldStop)
    }

    /// Trains every block through `steps` steps. `rows` must be the rows given to
    /// ``precomputeBoundaries(rows:store:options:)``; `rows.row(forStep:)` picks the row for each step.
    ///
    /// **Resume.** With `outputDirectory`, each finished block's adapters are written (atomically)
    /// to `block-<n>.safetensors`, and blocks already written are loaded instead of retrained. With
    /// `checkpointDirectory`, the block in progress also saves its adapters, optimizer moments,
    /// update count and step to `block-<n>/ckpt` every ``HealingConfiguration/checkpointEvery`` steps
    /// (and when a stop is honoured), so an interrupted run resumes mid-block and finishes equal to
    /// an uninterrupted one. The checkpoint is removed when its block finishes. Initial adapters
    /// apply to a fresh start only, never over a checkpoint.
    ///
    /// **Initialisation.** A block's adapters start from `initialAdapters`, or from random numbers
    /// drawn after MLX's generator is seeded with `configuration.seed` plus the block's first layer,
    /// so a resumed run initialises later blocks as an uninterrupted one does.
    ///
    /// **Schedule.** The learning rate follows ``HealingConfiguration/learningRateSchedule(runSteps:)``
    /// at the block's own update count; every block sees the same schedule.
    ///
    /// - Parameters:
    ///   - onStep: called after every step of every block with the block index and the step's
    ///     record (loss, learning rate, whether the update was applied), so curves exist during
    ///     the run. A step whose loss is non-finite or above `maxLoss` applies no update, as in
    ///     the main path.
    ///   - onBlockFinished: called with the block index and its adapter tensors.
    ///   - memory: MLX limits, and the peak above which the run checkpoints and stops.
    ///   - callbacks: `shouldStop`, `shouldPause`, `stopFile`, `pauseFile` are honoured between
    ///     steps; a stop saves the block's checkpoint first. `onCheckpoint` gets the step and directory.
    public func train(
        rows: HealingRowSource, steps: Int, store: BoundaryStore, outputDirectory: URL? = nil,
        checkpointDirectory: URL? = nil, memory: MemoryConfiguration = .init(),
        callbacks: HealingCallbacks = .init(),
        onStep: ((Int, StepRecord) -> Void)? = nil,
        onBlockFinished: ((Int, [String: MLXArray]) -> Void)? = nil
    ) throws -> Report {
        try adapterConfiguration.validate()
        var memory = memory
        let c = configuration
        let longest = blocks.map { $0.end - $0.start + 1 }.max() ?? 1
        var resident = min(max(c.residentLayers ?? longest, 1), longest)
        var estimate: ResidencyEstimate?
        if let budget = c.memoryBudget {
            var sequence = 0
            for s in 0 ..< max(1, min(steps, 1024)) { sequence = max(sequence, rows.row(forStep: s).length) }
            let inputs = try ResidencyPlanner.inputs(
                source: source, keep: keep, blockEnds: blockEnds, adapterConfiguration: adapterConfiguration,
                sliced: slicedStudent, sequence: sequence)
            if let asked = c.residentLayers {
                let e = ResidencyPlanner.estimate(inputs, resident: asked)
                guard e.total <= budget else {
                    throw HealingError.budget(
                        "residentLayers \(asked) needs about \(e.total) bytes, over the memory budget of \(budget): \(e)")
                }
                estimate = e
            } else {
                estimate = try ResidencyPlanner.plan(budget: budget, inputs)
                resident = estimate!.residentLayers
            }
            memory.memoryLimitBytes = min(memory.memoryLimitBytes ?? budget, budget)
            memory.peakStopBytes = min(memory.peakStopBytes ?? budget, budget)
            memory.cacheLimitBytes = min(memory.cacheLimitBytes ?? budget / 8, budget / 8)
        }
        memory.apply()
        let footprint = FootprintSampler()
        let manifest = try TeacherPrepass.verifiedManifest(
            of: store, blockEnds: blockEnds, layerCount: source.layerCount, teacher: try source.teacherFingerprint())
        let schedule = c.learningRateSchedule(runSteps: steps)
        var all: [String: MLXArray] = [:]
        var losses: [[Float]] = []
        var skipped: [Int] = []
        var peak = 0
        var outcome = HealingOutcome.finished
        var exceeded: Int?
        let nb = Float(blocks.count)
        let fm = FileManager.default
        blockLoop: for (b, range) in blocks.enumerated() {
            let file = outputDirectory?.appendingPathComponent("block-\(b).safetensors")
            if let file, fm.fileExists(atPath: file.path) {
                all.merge(try loadArrays(url: file)) { $1 }
                losses.append([])
                skipped.append(0)
                continue
            }
            let blockCheckpoint = checkpointDirectory?.appendingPathComponent("block-\(b)")
            Memory.peakMemory = 0
            let blockLength = range.end - range.start + 1
            let windowed = resident < blockLength
            let kept = Array(keep[range.start ... range.end])
            let sets = kept.map { ExpertSet(keep: $0, numExperts: source.numExperts) }
            let layers: [any HealableMoELayer] = windowed ? [] : try prepare(range, sliced: slicedStudent).layers
            // Loading a layer draws random numbers (its parameters start random), so every layer is
            // loaded before the generator is seeded; the adapters' draws then match the whole-block path.
            var dimensions: [LayerDimensions] = []
            if windowed {
                for i in 0 ..< blockLength {
                    dimensions.append(LayerDimensions(
                        try source.loadLayers(
                            (range.start + i) ... (range.start + i), keeping: slicedStudent ? [kept[i]] : nil)[0]))
                }
            }
            MLXRandom.seed(c.seed &+ UInt64(range.start))
            let adapters: ModelAdapters
            if windowed {
                adapters = ModelAdapters(
                    dimensions: dimensions, firstLayer: range.start, sets: sets, configuration: adapterConfiguration)
            } else {
                adapters = ModelAdapters(
                    layers: layers, firstLayer: range.start, sets: sets, configuration: adapterConfiguration)
            }
            let windows = Self.windows(count: blockLength, resident: resident)
            let keptTop = c.keepIntermediates.count(ofLayers: windows.last!.count)   // no memory budget here: `auto` keeps none
            let opt = HealingAdamW(weightDecay: c.weightDecay)
            opt.routerLRMultiplier = adapterConfiguration.routerLRMultiplier
            var blockLosses: [Float] = []
            var skips = 0
            var first = 0
            if let blockCheckpoint,
                let meta = try Checkpointing.resumeBlock(from: blockCheckpoint, block: b, adapters: adapters, optimizer: opt)
            {
                first = meta.step
                blockLosses = meta.losses
                skips = meta.skipped
            } else if let initialAdapters {
                try adapters.load(subsetOf: initialAdapters)
            }
            let net: HealingNetwork? = windowed ? nil : HealingNetwork(
                layers: layers, firstLayer: range.start, sets: sets, adapters: adapters, sliced: slicedStudent)
            func saveCheckpoint(_ step: Int) throws {
                guard let blockCheckpoint else { return }
                try Checkpointing.saveBlock(
                    to: blockCheckpoint, adapters: adapters, optimizer: opt,
                    meta: .init(
                        block: b, step: step, optimizerStep: opt.step, losses: blockLosses, skipped: skips,
                        saved: ISO8601DateFormatter().string(from: Date())))
                callbacks.onCheckpoint?(step, blockCheckpoint)
            }
            let carry = WindowCarry()
            var step = first
            while step < steps {
                callbacks.waitWhilePaused()
                var reason: String?
                if callbacks.stopRequested() { reason = "stop requested" }
                else if let limit = memory.peakStopBytes, Memory.peakMemory > limit {
                    reason = "peak memory \(Memory.peakMemory) over \(limit)"
                }
                if let reason {
                    if memory.peakStopBytes.map({ Memory.peakMemory > $0 }) == true { exceeded = max(exceeded ?? 0, Memory.peakMemory) }
                    try saveCheckpoint(step)
                    callbacks.consumeStop()
                    outcome = .stopped(step: step, reason: reason)
                    peak = max(peak, Memory.peakMemory)
                    losses.append(blockLosses)
                    skipped.append(skips)
                    break blockLoop
                }
                let t0 = Date()
                let row = rows.row(forStep: step)
                try TeacherPrepass.verify(row, against: manifest)
                let input = try store.get(row: row.key, boundary: b)
                let target = try store.get(row: row.key, boundary: b + 1)
                let w = row.weights
                var loss: Float
                var ok: Bool
                var lr: Float
                if windowed {
                    (loss, ok, lr) = try windowedStep(
                        block: range, windows: windows, adapters: adapters, optimizer: opt, sets: sets, kept: kept,
                        input: input, target: target, weights: w, learningRate: { schedule.value(at: $0) },
                        blockCount: nb, checkpointed: c.checkpointBlocks, keptLayers: keptTop, carry: carry)
                    if !ok { skips += 1 }
                    blockLosses.append(loss)
                } else if StreamedBlockHealer.compileLayerSteps, !adapters.needsRefresh {
                    // The whole block as one window through the compiled step (its layers are already resident).
                    (loss, ok, lr) = try windowedStep(
                        block: range, windows: windows, adapters: adapters, optimizer: opt, sets: sets, kept: kept,
                        input: input, target: target, weights: w, learningRate: { schedule.value(at: $0) },
                        blockCount: nb, checkpointed: c.checkpointBlocks, keptLayers: keptTop, carry: carry, preloaded: layers)
                    if !ok { skips += 1 }
                    blockLosses.append(loss)
                } else {
                    let net = net!
                    net.refreshRounding()
                    let ckpt = c.checkpointBlocks
                    let vg = valueAndGrad(model: adapters) { (_: ModelAdapters, _: [MLXArray]) in
                        let (_, p) = net.forward(
                            input, student: true, collect: [range.end], checkpointed: ckpt, keptLayers: keptTop)
                        return [HealingLoss.pin([range.end: p[range.end]!], teacher: [range.end: target], weights: w)]
                    }
                    let (l, g) = vg(adapters, [])
                    loss = l[0].item(Float.self)
                    let grads = GradMath.scale(GradMath.flatten(g), by: 1 / nb)
                    blockLosses.append(loss)
                    lr = schedule.value(at: opt.step)
                    ok = loss.isFinite && loss <= c.maxLoss
                    if ok {
                        opt.update(model: adapters, grads: grads, learningRate: lr)
                        eval(adapters, opt.stateArrays)
                    } else { skips += 1 }
                }
                let seen = footprint.takeStepPeak()
                onStep?(b, StepRecord(
                    step: step + 1, phase: .block, loss: loss, learningRate: lr, applied: ok,
                    seconds: Date().timeIntervalSince(t0), peakMemoryBytes: Memory.peakMemory,
                    peakFootprintBytes: seen.footprint, fileBackedBytes: seen.fileBacked))
                Memory.clearCache()
                step += 1
                if step < steps, step % max(1, c.checkpointEvery) == 0 { try saveCheckpoint(step) }
            }
            let blockPeak = Memory.peakMemory
            peak = max(peak, blockPeak)
            let t = adapters.tensors()
            if let file {
                try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                // Written under a temporary name: a file that exists under its final name is complete.
                let tmp = file.deletingLastPathComponent().appendingPathComponent("block-\(b).partial.safetensors")
                try MLX.save(arrays: t, url: tmp)
                if fm.fileExists(atPath: file.path) { try fm.removeItem(at: file) }
                try fm.moveItem(at: tmp, to: file)
            }
            if let blockCheckpoint { try? fm.removeItem(at: blockCheckpoint) }
            all.merge(t) { $1 }
            losses.append(blockLosses)
            skipped.append(skips)
            onBlockFinished?(b, t)
            // The peak is checked before each step, so the last step of a block is checked here. The
            // block is finished and saved; a run with blocks left stops rather than start the next one.
            if let limit = memory.peakStopBytes, blockPeak > limit {
                exceeded = max(exceeded ?? 0, blockPeak)
                if b + 1 < blocks.count {
                    outcome = .stopped(step: steps, reason: "peak memory \(blockPeak) over \(limit) (block \(b + 1) is finished and saved)")
                    break blockLoop
                }
            }
        }
        return Report(
            adapters: all, blockLosses: losses, skippedUpdates: skipped, peakMemoryBytes: peak,
            peakFootprintBytes: footprint.peak, residentLayers: resident, estimate: estimate,
            peakOverLimit: exceeded, outcome: outcome)
    }
}

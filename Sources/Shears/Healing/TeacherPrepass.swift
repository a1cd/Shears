import Foundation
import MLX
import MLXNN
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif

/// Computes the teacher once, block by block, and writes everything training needs to a
/// ``BoundaryStore``; nothing later runs a teacher layer.
///
/// **Block-major.** For each block the teacher's weights (every expert, as stored) are loaded
/// once, every row is pushed through them in batches, the block's end states are written, and
/// the block is released. Block `b` reads its inputs from the store, which holds block `b - 1`'s
/// ends (boundary 0 is the embedding), so only one block's weights and one batch of activations
/// are ever resident.
///
/// **Batching is exact.** Rows of different lengths are sorted by length, packed into batches
/// under a padded-token budget, and right-padded with zeros. Attention is causal (sliding
/// windows included) so a query never reads a key after itself, every other operation is per
/// token, and the padded tail is cut off before storing: each stored row is the unpadded
/// `[1, T, D]` a row-by-row teacher would produce (up to the float noise of a different matmul
/// shape). Left padding would shift the rotary positions and is never used.
///
/// **What is stored.** The embedding and the end of every block, per row: `blocks + 1` states of
/// `T x D`. The recipe's KL phases score the teacher's *full-vocabulary* log-probabilities at
/// anchor and sampled positions, and the `end` phase samples different positions at every step.
/// Those log-probabilities are a function of the last stored state alone (final norm, output
/// head, softcap), and the head is resident during the KL phases anyway, so they are recomputed
/// exactly from the stored final state each step instead of being stored (a vocabulary-sized
/// vector per position would dwarf the states).
///
/// **Disk cost.** `sum over rows of T_r x D x bytes(storageDType) x (blocks + 1)`; see
/// ``storageBytes(rowLengths:hiddenSize:blockCount:dtype:)``. The default follows the model's
/// compute dtype, so a bf16 model (the real one) stores bf16, which is lossless. Storing bf16
/// for an fp32 model rounds the teacher's targets and each next block's input.
///
/// **Resumable.** Files appear atomically, a stage is marked done in the store's manifest only
/// after its last row, and a rerun skips finished stages and rows. The manifest fingerprints
/// the block ends, storage dtype and every row (key, length, token hash): a store written for
/// anything else is rejected instead of reused.
public struct TeacherPrepass {
    public struct Options {
        /// What the states are written as. `nil` (the default) follows the model: bf16 for a bf16
        /// model, which is lossless, and fp32 for an fp32 model, whose targets and next-block
        /// inputs bf16 would round. Pass `.bfloat16` to halve an fp32 model's store on purpose.
        public var storageDType: DType?
        /// Most padded tokens (`rows x longest row`) in one batch. A batch always holds at least
        /// one row, however long.
        public var maxBatchTokens = 8192
        public var maxBatchRows = 16
        public init(storageDType: DType? = nil, maxBatchTokens: Int = 8192, maxBatchRows: Int = 16) {
            self.storageDType = storageDType
            self.maxBatchTokens = maxBatchTokens
            self.maxBatchRows = maxBatchRows
        }
    }

    public struct Report {
        /// Allocator peak per stage run, in order (stage 0 is the embedding, stage `b + 1` block `b`).
        public var stagePeakBytes: [Int: Int]
        /// Stages that were already complete in the store.
        public var skippedStages: [Int]
        public var batchesRun: Int
        /// True when `shouldStop` ended the run early. Everything stored so far is kept; a rerun resumes.
        public var stopped = false
    }

    public let source: LayerSource
    public let blockEnds: [Int]
    public let options: Options

    public init(source: LayerSource, blockEnds: [Int], options: Options = .init()) {
        self.source = source
        self.blockEnds = blockEnds
        self.options = options
    }

    /// Bytes the store needs: `T x D x element size` for each of `blockCount + 1` boundaries,
    /// summed over rows (plus a small safetensors header per file).
    public static func storageBytes(rowLengths: [Int], hiddenSize: Int, blockCount: Int, dtype: DType) -> Int {
        let per = dtype.size
        return rowLengths.reduce(0) { $0 + $1 * hiddenSize * per } * (blockCount + 1)
    }

    struct Manifest: Codable, Equatable {
        var version = 2
        /// ``LayerSource/teacherFingerprint()`` of the model that filled the store.
        var teacher: String
        var layerCount: Int
        var blockEnds: [Int]
        var dtype: String
        var rows: [String: String]
        var completed: [Int] = []
        func sameInputs(as o: Manifest) -> Bool {
            version == o.version && teacher == o.teacher && layerCount == o.layerCount && blockEnds == o.blockEnds
                && dtype == o.dtype && rows == o.rows
        }
    }

    /// The store's manifest if it has one, after checking it was written for these block ends and
    /// layer count and that every stage finished. `nil` for a store that carries no manifest
    /// (one filled by other means). Training reads this so a store from a different blocking or
    /// an interrupted pre-pass is refused instead of giving silently wrong targets.
    static func verifiedManifest(
        of store: BoundaryStore, blockEnds: [Int], layerCount: Int, teacher: String? = nil
    ) throws -> Manifest? {
        guard let data = try store.metadata(named: "prepass") else { return nil }
        let m: Manifest
        do { m = try JSONDecoder().decode(Manifest.self, from: data) } catch {
            throw HealingError.invalid(
                "the teacher store's manifest is from an older version and does not say which model "
                    + "filled it; run the pre-pass again into a fresh store")
        }
        if let teacher, m.teacher != teacher {
            throw HealingError.invalid("the teacher store was filled by a different model (or load dtype) than this run's")
        }
        guard m.blockEnds == blockEnds, m.layerCount == layerCount else {
            throw HealingError.invalid(
                "the teacher store was written for block ends \(m.blockEnds) over \(m.layerCount) layers, "
                    + "not \(blockEnds) over \(layerCount)")
        }
        let missing = (0 ... blockEnds.count).filter { !m.completed.contains($0) }
        guard missing.isEmpty else {
            throw HealingError.invalid("the teacher pre-pass did not finish (stages \(missing) missing); rerun it")
        }
        return m
    }

    /// Throws if `row` is not the row the store was filled with under its key.
    static func verify(_ row: HealingRow, against manifest: Manifest?) throws {
        guard let manifest else { return }
        guard let f = manifest.rows[row.key] else {
            throw HealingError.invalid("row '\(row.key)' is not in the teacher store")
        }
        guard f == fingerprint(row) else {
            throw HealingError.invalid("row '\(row.key)' differs from the one the teacher store was filled with")
        }
    }

    static func fingerprint(_ row: HealingRow) -> String {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for t in row.tokens {
            h = (h ^ UInt64(UInt32(bitPattern: t))) &* 0x0000_0100_0000_01b3
        }
        return "\(row.length):\(String(h, radix: 16))"
    }

    /// Row indices grouped into batches: longest first, under the token and row budgets.
    static func batches(lengths: [Int], maxTokens: Int, maxRows: Int) -> [[Int]] {
        let order = lengths.indices.sorted { lengths[$0] != lengths[$1] ? lengths[$0] > lengths[$1] : $0 < $1 }
        var out: [[Int]] = []
        var cur: [Int] = []
        for i in order {
            if let first = cur.first, cur.count >= maxRows || (cur.count + 1) * lengths[first] > maxTokens {
                out.append(cur)
                cur = []
            }
            cur.append(i)
        }
        if !cur.isEmpty { out.append(cur) }
        return out
    }

    /// Runs the pre-pass.
    ///
    /// - Parameters:
    ///   - shouldStop: checked before every batch; when it returns `true` the run returns with
    ///     `Report.stopped` set. Rows already stored are kept, so a rerun picks up at the next batch.
    ///   - onStageFinished: called after a stage's last row with its boundary index
    ///   (0 for the embedding, then `b + 1`); throwing from it stops the run, which a later call
    ///   resumes.
    @discardableResult
    public func run(
        rows: [HealingRow], store: BoundaryStore, shouldStop: (() -> Bool)? = nil,
        onStageFinished: ((Int) throws -> Void)? = nil
    ) throws -> Report {
        guard Set(rows.map(\.key)).count == rows.count else {
            throw HealingError.invalid("row keys must be unique")
        }
        guard blockEnds.last == source.layerCount - 1 else {
            throw HealingError.invalid("blocks must cover every layer")
        }
        // One teacher block and one batch of rows are all this holds; bound MLX's buffer cache (it counts toward the
        // process footprint) instead of letting it grow to the whole memory limit. See ``BlockPipeline``.
        let priorCacheLimit = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCacheLimit }
        // Before any layer is seen the bound is the ceiling (1 GB); the first block loaded tightens it to the batch size.
        Memory.cacheLimit = min(priorCacheLimit, BlockPipeline.prepareCacheLimit(maxBatchTokens: options.maxBatchTokens, hidden: 1 << 30))
        let storage: DType
        if let d = options.storageDType {
            storage = d
        } else {
            let probe = try source.loadLayers(0 ... 0, keeping: nil)
            storage = Self.computeDType(probe) ?? .bfloat16
        }
        var manifest = Manifest(
            teacher: try source.teacherFingerprint(), layerCount: source.layerCount, blockEnds: blockEnds, dtype: "\(storage)",
            rows: Dictionary(uniqueKeysWithValues: rows.map { ($0.key, Self.fingerprint($0)) }))
        let decoder = JSONDecoder(), encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try store.metadata(named: "prepass") {
            guard let old = try? decoder.decode(Manifest.self, from: data) else {
                throw HealingError.invalid(
                    "this store's manifest is from an older version and does not say which model filled it; use a fresh store")
            }
            guard old.teacher == manifest.teacher else {
                throw HealingError.invalid(
                    "this store was filled by a different model (or load dtype); use a fresh store")
            }
            guard old.sameInputs(as: manifest) else {
                throw HealingError.invalid(
                    "this store was written for different rows, blocks or storage dtype; use a fresh store")
            }
            manifest = old
        } else {
            try store.putMetadata(try encoder.encode(manifest), named: "prepass")
        }
        func finish(_ stage: Int) throws {
            manifest.completed.append(stage)
            try store.putMetadata(try encoder.encode(manifest), named: "prepass")
            try onStageFinished?(stage)
        }

        let lengths = rows.map(\.length)
        let groups = Self.batches(lengths: lengths, maxTokens: options.maxBatchTokens, maxRows: options.maxBatchRows)
        var report = Report(stagePeakBytes: [:], skippedStages: [], batchesRun: 0)

        // Stage 0: embeddings, with the table loaded once.
        if manifest.completed.contains(0) {
            report.skippedStages.append(0)
        } else {
            Memory.peakMemory = 0
            let embed = try source.makeEmbedder()
            for g in groups where !g.allSatisfy({ store.has(row: rows[$0].key, boundary: 0) }) {
                if shouldStop?() == true { report.stopped = true; return report }
                let longest = lengths[g[0]]
                let tokens = concatenated(
                    g.map { i -> MLXArray in
                        let t = rows[i].tokenArray
                        return rows[i].length == longest
                            ? t : concatenated([t, MLXArray.zeros([1, longest - rows[i].length], dtype: t.dtype)], axis: 1)
                    }, axis: 0)
                let h = try embed(tokens)
                try write(h, group: g, rows: rows, boundary: 0, store: store, dtype: storage)
                report.batchesRun += 1
                Memory.clearCache()
            }
            report.stagePeakBytes[0] = Memory.peakMemory
            try finish(0)
        }

        for (b, range) in blockRanges(ends: blockEnds).enumerated() {
            let stage = b + 1
            if manifest.completed.contains(stage) {
                report.skippedStages.append(stage)
                continue
            }
            let pending = groups.filter { g in !g.allSatisfy { store.has(row: rows[$0].key, boundary: stage) } }
            Memory.peakMemory = 0
            if !pending.isEmpty {
                let layers = try source.loadLayers(range.start ... range.end, keeping: nil)
                // A teacher has every expert. A source that was cut in place for a sliced student
                // would otherwise silently serve the student's layers as the teacher.
                guard layers.allSatisfy({ $0.numExperts == source.numExperts }) else {
                    throw HealingError.invalid(
                        "the teacher source's layers no longer have all \(source.numExperts) experts (it was cut for a sliced student); run the pre-pass before slicing, or reload the model")
                }
                if let hidden = layers.first?.hiddenSize {
                    Memory.cacheLimit = min(
                        priorCacheLimit, BlockPipeline.prepareCacheLimit(maxBatchTokens: options.maxBatchTokens, hidden: hidden))
                }
                let sets = layers.map { ExpertSet.all($0.numExperts) }
                let net = HealingNetwork(layers: layers, firstLayer: range.start, sets: sets, adapters: nil)
                let compute = Self.computeDType(layers)
                for g in pending {
                    if shouldStop?() == true { report.stopped = true; return report }
                    let longest = lengths[g[0]]
                    let inputs = try g.map { i -> MLXArray in
                        var x = try store.get(row: rows[i].key, boundary: b)
                        if let compute, x.dtype != compute { x = x.asType(compute) }
                        let pad = longest - x.dim(1)
                        return pad == 0
                            ? x : concatenated([x, MLXArray.zeros([1, pad, x.dim(2)], dtype: x.dtype)], axis: 1)
                    }
                    // Layer by layer with an eval between: there is no backward pass to feed, so
                    // each layer's intermediates are freed before the next layer starts.
                    var h = concatenated(inputs, axis: 0)
                    for l in range.start ... range.end {
                        h = net.forward(h, from: l, through: l, student: false).0
                        eval(h)
                    }
                    try write(stopGradient(h), group: g, rows: rows, boundary: stage, store: store, dtype: storage)
                    report.batchesRun += 1
                    Memory.clearCache()
                }
            }
            report.stagePeakBytes[stage] = Memory.peakMemory
            try finish(stage)
        }
        return report
    }

    /// Stores each row of a padded batch result unpadded.
    private func write(
        _ h: MLXArray, group: [Int], rows: [HealingRow], boundary: Int, store: BoundaryStore, dtype: DType
    ) throws {
        for (j, i) in group.enumerated() {
            var s = h[j ..< j + 1, 0 ..< rows[i].length]
            if s.dtype != dtype { s = s.asType(dtype) }
            if !store.has(row: rows[i].key, boundary: boundary) {
                try store.put(s, row: rows[i].key, boundary: boundary)
            }
        }
    }

    /// The dtype the layers compute in: the first floating-point parameter's (a quantized
    /// layer's norms carry it). `nil` when it can't be told.
    static func computeDType(_ layers: [any HealableMoELayer]) -> DType? {
        guard let m = layers.first as? Module else { return nil }
        let floats = m.parameters().flattened().filter { $0.1.dtype.isFloatingPoint }
        return (floats.first { $0.0.contains("norm") } ?? floats.first)?.1.dtype
    }
}

/// A SHA-256 over a sequence of strings and byte blobs, for ``LayerSource/teacherFingerprint()``.
struct Fingerprint {
    private var hasher = SHA256()
    mutating func add(_ s: String) {
        hasher.update(data: Data(s.utf8))
        hasher.update(data: Data([0]))
    }
    mutating func add(_ d: Data) {
        var n = UInt64(d.count)
        hasher.update(data: Data(bytes: &n, count: 8))
        hasher.update(data: d)
    }
    var hex: String { hasher.finalize().map { String(format: "%02x", $0) }.joined() }
}

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Where the rows of the embedding table live on disk, read a row at a time.
///
/// The table is `[vocab, hidden]` (full precision) or, in an mlx-community 4-bit checkpoint, the
/// packed `weight`, `scales` and (affine) `biases`, each `[vocab, ...]`. Row `i` of each tensor is
/// stored contiguously, so a set of rows is a handful of `pread`s: nothing else of the table is
/// mapped, read or counted.
public final class EmbeddingRowSource {
    struct Tensor {
        var entry: MappedSafetensors.Entry
        var url: URL
    }

    public struct Quantization: Equatable {
        public var groupSize: Int
        public var bits: Int
        public var mode: QuantizationMode
    }

    let weight: Tensor
    let scales: Tensor?
    let biases: Tensor?
    public let vocabularySize: Int
    public let hiddenSize: Int
    /// nil for a full-precision table.
    public let quantization: Quantization?
    /// Cast floating-point rows to this when they are read (like `Gemma4Checkpoint.loadModel(dtype:)`).
    public let dtype: DType?

    init(
        weight: Tensor, scales: Tensor?, biases: Tensor?, vocabularySize: Int, hiddenSize: Int,
        quantization: Quantization?, dtype: DType?
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.quantization = quantization
        self.dtype = dtype
    }

    /// Bytes of one row, across the tensors that make it up.
    public var bytesPerRow: Int {
        func per(_ t: Tensor?) -> Int {
            guard let t, vocabularySize > 0 else { return 0 }
            return (t.entry.end - t.entry.begin) / vocabularySize
        }
        return per(weight) + per(scales) + per(biases)
    }

    /// The rows `ids` (any order, duplicates allowed), in that order.
    public func read(_ ids: [Int]) throws -> VocabularyRows {
        for id in ids where id < 0 || id >= vocabularySize {
            throw InferenceError.invalid("token id \(id) is outside the \(vocabularySize)-row embedding table")
        }
        // Ascending reads make runs of neighbouring rows one read; the result is put back in request order.
        let order = ids.indices.sorted { ids[$0] < ids[$1] }
        let sorted = order.map { ids[$0] }
        func gather(_ t: Tensor?) throws -> MLXArray? {
            guard let t else { return nil }
            let fd = open(t.url.path, O_RDONLY)
            guard fd >= 0 else { throw MoELoaderError.missingFile(t.url) }
            defer { close(fd) }
            var a = try MappedSafetensors.gatherRows(t.entry, rows: sorted, fd: fd, in: t.url)
            if order != Array(ids.indices) {
                var inverse = [Int32](repeating: 0, count: order.count)
                for (rank, original) in order.enumerated() { inverse[original] = Int32(rank) }
                a = a[MLXArray(inverse)]
            }
            return a
        }
        guard var w = try gather(weight) else { throw InferenceError.invalid("no embedding weight") }
        var s = try gather(scales)
        var b = try gather(biases)
        if let dtype {
            if w.dtype.isFloatingPoint { w = w.asType(dtype) }
            s = s.map { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 }
            b = b.map { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 }
        }
        return VocabularyRows(count: ids.count, weight: w, scales: s, biases: b, quantization: quantization)
    }
}

/// Some rows of the embedding table, in memory.
public struct VocabularyRows {
    public var count: Int
    var weight: MLXArray
    var scales: MLXArray?
    var biases: MLXArray?
    var quantization: EmbeddingRowSource.Quantization?

    /// Bytes held.
    public var nbytes: Int { weight.nbytes + (scales?.nbytes ?? 0) + (biases?.nbytes ?? 0) }

    /// The rows at `indices` (any shape, `Int32`) as full-precision vectors `[..., hidden]`.
    func gather(_ indices: MLXArray, dtype: DType) -> MLXArray {
        let shape = indices.shape
        let flat = indices.flattened()
        let rows: MLXArray
        if let q = quantization, let scales {
            rows = dequantized(
                weight[flat], scales: scales[flat], biases: biases.map { $0[flat] },
                groupSize: q.groupSize, bits: q.bits, mode: q.mode)
        } else {
            rows = weight[flat]
        }
        return rows.asType(dtype).reshaped(shape + [-1])
    }

    /// `x @ rows^T`: `[..., hidden]` to `[..., count]`.
    func project(_ x: MLXArray) -> MLXArray {
        if let q = quantization, let scales {
            return quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: true, groupSize: q.groupSize, bits: q.bits,
                mode: q.mode)
        }
        return matmul(x, weight.T)
    }
}

/// The embedding table as a small working set instead of 262,144 rows.
///
/// The full table stays on disk. A window holds
/// - the rows of a **static base** (a fixed list of token ids, loaded once), and
/// - the rows of the tokens of the **current context** that the base lacks (read from the file for each
///   request by ``prepare(contextTokens:)``).
///
/// Both ends of the model use it. The **input** side embeds the prompt from those rows (every token of the
/// prompt is in one of the two, by construction). The **output** side computes logits over those rows only
/// and returns them in a full-width `[.., vocab]` array whose other entries are `-inf`, so sampling,
/// constraints and log probabilities work unchanged; a token outside the window cannot be chosen. Tokens
/// the model generates come from the window, so they need no further reads.
///
/// Not thread-safe: one request at a time, like the generator.
public final class VocabularyWindow {
    public let source: EmbeddingRowSource
    /// Ids of the static base, ascending.
    public let staticIDs: [Int]
    private let staticRows: VocabularyRows
    /// Row index in `[static..., extra...]` for each vocabulary id, -1 when the window lacks it.
    private let staticMap: [Int32]
    private var map: MLXArray
    public private(set) var extraIDs: [Int] = []
    private var extraRows: VocabularyRows?
    /// Dtype of the vectors this window produces.
    public let dtype: DType

    public var vocabularySize: Int { source.vocabularySize }
    /// Rows in the window now.
    public var rowCount: Int { staticIDs.count + extraIDs.count }
    /// Bytes the window's rows occupy now.
    public var residentBytes: Int { staticRows.nbytes + (extraRows?.nbytes ?? 0) + staticMap.count * 4 }

    /// - Parameters:
    ///   - source: where the rows are read from.
    ///   - staticIDs: the static base (duplicates are removed).
    ///   - dtype: the type of the vectors it produces (the model's compute type).
    public init(source: EmbeddingRowSource, staticIDs: [Int], dtype: DType) throws {
        self.source = source
        self.dtype = dtype
        let ids = Array(Set(staticIDs)).sorted()
        guard let last = ids.last, ids.first! >= 0, last < source.vocabularySize else {
            throw InferenceError.invalid("the static vocabulary has ids outside 0..<\(source.vocabularySize)")
        }
        self.staticIDs = ids
        staticRows = try source.read(ids)
        var m = [Int32](repeating: -1, count: source.vocabularySize)
        for (row, id) in ids.enumerated() { m[id] = Int32(row) }
        staticMap = m
        map = MLXArray(m)
        eval(staticRows.weight, map)
        if let s = staticRows.scales { eval(s) }
        if let b = staticRows.biases { eval(b) }
    }

    /// Makes the window cover `contextTokens`: loads the rows of those not in the static base (and drops
    /// the previous request's extra rows). Call before every request.
    public func prepare(contextTokens: [Int]) throws {
        var missing = Set<Int>()
        for t in contextTokens {
            guard t >= 0, t < vocabularySize else {
                throw InferenceError.invalid("token id \(t) is outside the \(vocabularySize)-token vocabulary")
            }
            if staticMap[t] < 0 { missing.insert(t) }
        }
        let extra = missing.sorted()
        if extra == extraIDs, extraRows != nil || extra.isEmpty { return }
        extraIDs = extra
        extraRows = extra.isEmpty ? nil : try source.read(extra)
        var m = staticMap
        for (i, id) in extra.enumerated() { m[id] = Int32(staticIDs.count + i) }
        map = MLXArray(m)
        eval(map)
        if let e = extraRows {
            eval(e.weight)
            if let s = e.scales { eval(s) }
            if let b = e.biases { eval(b) }
        }
    }

    /// Whether the window can currently produce `id`.
    public func contains(_ id: Int) -> Bool {
        guard id >= 0, id < vocabularySize else { return false }
        if staticMap[id] >= 0 { return true }
        return extraIDs.contains(id)
    }

    /// The embedding vectors of `tokens` (`[B, L]` ids) `[B, L, hidden]`, before the `sqrt(hidden)` scale.
    /// Every id must be in the window (``prepare(contextTokens:)`` guarantees it for the prompt, and
    /// generation only produces tokens the window holds).
    public func embed(_ tokens: MLXArray) -> MLXArray {
        let idx = map[tokens.asType(.int32)]
        let s = staticIDs.count
        let fromStatic = staticRows.gather(clip(idx, min: 0, max: s - 1), dtype: dtype)
        guard let extra = extraRows else { return fromStatic }
        let fromExtra = extra.gather(clip(idx - Int32(s), min: 0, max: extra.count - 1), dtype: dtype)
        return MLX.where(expandedDimensions(idx .< Int32(s), axis: -1), fromStatic, fromExtra)
    }

    /// `x @ window^T`: scores of the window's rows, `[.., rowCount]` in the order static then extra.
    public func project(_ x: MLXArray) -> MLXArray {
        let a = staticRows.project(x)
        guard let extra = extraRows else { return a }
        return concatenated([a, extra.project(x)], axis: -1)
    }

    /// Spreads window scores `[.., rowCount]` back to `[.., vocab]`, with `-inf` for every token the
    /// window lacks.
    public func expand(_ scores: MLXArray) -> MLXArray {
        let n = rowCount
        let gathered = take(scores, clip(map, min: 0, max: n - 1), axis: -1)
        return MLX.where(map .>= 0, gathered, MLXArray(-Float.infinity).asType(scores.dtype))
    }

    /// Scores of just the rows of `ids` (each of which must be in the window), `[.., ids.count]`.
    public func project(_ x: MLXArray, onlyTokens ids: [Int]) throws -> MLXArray {
        let rows = ids.map { Int(staticMap[$0] >= 0 ? staticMap[$0] : -1) }
        guard !rows.contains(-1) else { throw InferenceError.invalid("a candidate token is not in the static vocabulary") }
        let vectors = staticRows.gather(MLXArray(rows.map(Int32.init)), dtype: dtype)
        return matmul(x, vectors.T)
    }
}

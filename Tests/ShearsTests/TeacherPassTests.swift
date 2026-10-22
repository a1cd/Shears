import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Counts the loads of full (unsliced) layers: the teacher's reads.
final class TeacherCountingSource: LayerSource {
    let inner: Gemma4LayerSource
    var teacherLoads = 0
    init(_ inner: Gemma4LayerSource) { self.inner = inner }
    var layerCount: Int { inner.layerCount }
    var numExperts: Int { inner.numExperts }
    func embed(_ tokens: [MLXArray]) throws -> [MLXArray] { try inner.embed(tokens) }
    func makeEmbedder() throws -> (MLXArray) throws -> MLXArray { try inner.makeEmbedder() }
    func loadLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) throws -> [any HealableMoELayer] {
        if keeping == nil { teacherLoads += range.count }
        return try inner.loadLayers(range, keeping: keeping)
    }
    func teacherFingerprint() throws -> String { try inner.teacherFingerprint() }
}

/// The teacher's block ends are computed once per block (default) or once per rolling batch, with the same numbers
/// either way, and the walks run on a few padded shapes.
final class TeacherPassTests: ShearsTestCase {
    func tempDir(_ name: String = "tp") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    struct Fixture {
        var checkpoint: URL
        var keep: [[Int]]
        var rows: [HealingRow]
    }

    func fixture(lengths: [Int]) throws -> Fixture {
        let m = try HealingFixture.model()
        let dir = try tempDir("tpckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: 4), adapters: nil, to: dir, options: o)
        // Distinct leading tokens, so no shared prefix and every row is its own.
        let rows = lengths.enumerated().map { HealingFixture.row(length: $1, seed: UInt64(30 + $0), key: "row\($0)") }
        return Fixture(checkpoint: dir, keep: HealingFixture.keepSets(count: 5), rows: rows)
    }

    struct Result {
        var pipeline: BlockPipeline
        var source: TeacherCountingSource
        var log: [String]
        var report: BlockPipeline.Report?
    }

    func run(
        _ f: Fixture, steps: Int, batchSteps: Int, prepareOnly: Bool = false, configure: (inout BlockPipelineOptions) -> Void
    ) throws -> Result {
        let source = TeacherCountingSource(Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: f.checkpoint), dtype: .float32))
        var o = BlockPipelineOptions()
        o.batchSteps = batchSteps
        o.mix = .both
        o.fixedKeep = f.keep
        o.storageDType = .float32
        o.maxBatchRows = 2
        o.sharedPrefix = .off
        o.prepareOnly = prepareOnly
        if prepareOnly { o.maxBlocks = 1 }
        configure(&o)
        let pipeline = try BlockPipeline(
            source: source, blockEnds: [1, 3], adapterConfiguration: HealingFixture.smallAdapters,
            configuration: HealingFixture.tinyConfiguration(), options: o, directory: try tempDir("tprun"))
        var lines: [String] = []
        let report = try pipeline.run(rows: ShuffledRows(rows: f.rows, seed: 3), steps: steps, log: { lines.append($0) })
        return Result(pipeline: pipeline, source: source, log: lines, report: report)
    }

    func targets(_ r: Result, _ f: Fixture) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for row in f.rows { out[row.key] = try r.pipeline.get("c", row.key, 1) }
        return out
    }

    static let lengths = [20, 23, 27, 31, 22, 26]

    /// One read of the block's teacher layers per block instead of per batch, same targets, same training.
    func testPerBlockReadsTheTeacherOncePerBlockAndEqualsPerBatch() throws {
        let f = try fixture(lengths: Self.lengths)
        // 6 steps of one row each in batches of 2: three batches.
        let whole = try run(f, steps: 6, batchSteps: 2) { $0.teacherPass = .perBlock }
        let each = try run(f, steps: 6, batchSteps: 2) { $0.teacherPass = .perBatch }
        // Block 0 and block 1 have two layers each; the training itself loads only sliced layers (keeping != nil).
        XCTAssertEqual(whole.source.teacherLoads, 4 + 1, "each of the 4 layers is read once (plus the run's one probe layer)")
        XCTAssertEqual(each.source.teacherLoads, 3 * 4 + 1, "each batch reads the whole block again")
        let a = whole.report!, b = each.report!
        XCTAssertEqual(Set(a.adapters.keys), Set(b.adapters.keys))
        for (k, v) in b.adapters { assertClose(a.adapters[k]!, v, atol: 1e-5, rtol: 1e-5, k) }
        for (x, y) in zip(a.blocks, b.blocks) { XCTAssertEqual(x.losses.count, y.losses.count) }
        // The timing line says what it is and splits the pause.
        let line = try XCTUnwrap(whole.log.first { $0.contains("teacher outputs") })
        XCTAssertTrue(line.contains("block 1/2, all 3 batches"), line)
        for part in ["read ", "first-call/compile ", "compute ", "layer loads"] { XCTAssertTrue(line.contains(part), "\(part) in \(line)") }
        XCTAssertEqual(each.log.filter { $0.contains("teacher outputs") && $0.contains("block 1/2") }.count, 3)
    }

    /// A state budget below one batch's hidden states splits the block into several walks, with the same numbers.
    func testStateBudgetSplitsTheBlockIntoWalks() throws {
        let f = try fixture(lengths: Self.lengths)
        let whole = try run(f, steps: 6, batchSteps: 2, prepareOnly: true) { $0.teacherPass = .perBlock }
        let small = try run(f, steps: 6, batchSteps: 2, prepareOnly: true) { $0.teacherPass = .perBlock; $0.teacherStateBudgetBytes = 1 }
        let walks = try XCTUnwrap(small.log.first { $0.contains("teacher outputs") })
        XCTAssertTrue(walks.contains("\(Self.lengths.count) rows in 3 walks") || walks.contains("rows in 3 walks"), walks)
        XCTAssertEqual(small.source.teacherLoads, 3 * 2 + 1, "two layers per walk, three walks")
        XCTAssertEqual(whole.source.teacherLoads, 2 + 1)
        let a = try targets(whole, f), b = try targets(small, f)
        for (k, v) in a { assertClose(b[k]!, v, atol: 1e-5, rtol: 1e-5, k) }
    }

    /// Rows of many lengths walk on a few padded shapes, and padding changes no target.
    func testPaddedWalksUseFewerShapesAndChangeNothing() throws {
        let f = try fixture(lengths: [20, 23, 27, 31, 22, 26, 29, 21])
        let padded = try run(f, steps: 8, batchSteps: 4, prepareOnly: true) { $0.padWalks = true; $0.maxBatchRows = 3 }
        let exact = try run(f, steps: 8, batchSteps: 4, prepareOnly: true) { $0.padWalks = false; $0.maxBatchRows = 3 }
        let nPadded = padded.pipeline.seenWalkShapes.count, nExact = exact.pipeline.seenWalkShapes.count
        print("TEACHERPASS distinct walk shapes: exact \(nExact), padded \(nPadded)")
        XCTAssertLessThan(nPadded, nExact)
        let a = try targets(padded, f), b = try targets(exact, f)
        for (k, v) in b {
            XCTAssertEqual(a[k]!.shape, v.shape, k)
            assertClose(a[k]!, v, atol: 1e-5, rtol: 1e-5, k)
        }
    }

    /// One fixed length for every walk and step: one shape each, the same training, and a clear refusal when a row
    /// does not fit.
    func testFixedSequenceLengthMeetsOneShapeAndChangesNothing() throws {
        let f = try fixture(lengths: [20, 23, 27, 31, 22, 40, 29, 21])
        let buckets = try run(f, steps: 8, batchSteps: 4) { $0.maxBatchRows = 3 }
        let fixed = try run(f, steps: 8, batchSteps: 4) { $0.maxBatchRows = 3; $0.fixedSequenceLength = 64 }
        XCTAssertEqual(StreamedBlockHealer.fixedSequenceLength, nil, "the process-wide setting is put back")
        let (a, b) = (buckets.report!, fixed.report!)
        print("TEACHERPASS buckets: \(a.compiledTraces) traces, \(a.walkShapes) walk shapes; fixed 64: \(b.compiledTraces) traces, \(b.walkShapes) walk shapes")
        XCTAssertLessThan(b.walkShapes, a.walkShapes)
        XCTAssertLessThan(b.compiledTraces, a.compiledTraces)
        XCTAssertEqual(Set(a.adapters.keys), Set(b.adapters.keys))
        for (k, v) in a.adapters { assertClose(b.adapters[k]!, v, atol: 2e-4, rtol: 2e-3, k) }
        for (x, y) in zip(a.blocks, b.blocks) {
            XCTAssertEqual(x.losses.count, y.losses.count)
            for (l1, l2) in zip(x.losses, y.losses) { XCTAssertEqual(l1, l2, accuracy: 2e-4 + 2e-3 * abs(l1)) }
        }
        XCTAssertThrowsError(try run(f, steps: 8, batchSteps: 4) { $0.fixedSequenceLength = 32 }) {
            XCTAssertTrue("\($0)".contains("shorter than the longest row"), "\($0)")
        }
        XCTAssertEqual(StreamedBlockHealer.fixedSequenceLength, nil)
    }
}

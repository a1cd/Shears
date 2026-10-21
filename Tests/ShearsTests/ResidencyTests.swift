import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Layer-window residency of the streamed block phase: the result must equal the whole-block path
/// whatever the window, weights are read as views of the checkpoint files, and a memory budget is
/// either respected or refused up front.
final class ResidencyTests: ShearsTestCase {
    func tempDir(_ name: String = "res") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    /// Rewrites each safetensors file's header padded with spaces to a multiple of 8 bytes, as Hugging Face's
    /// writer does, so every tensor of a 4- or 8-byte type starts aligned (MLX's own writer does not pad).
    func padHeaders(_ dir: URL) throws {
        for f in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) where f.pathExtension == "safetensors" {
            let d = try Data(contentsOf: f)
            let n = d.prefix(8).withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
            var header = d.subdata(in: 8 ..< 8 + n)
            while (8 + header.count) % 8 != 0 { header.append(0x20) }
            var len = UInt64(header.count).littleEndian
            var out = Data(bytes: &len, count: 8)
            out.append(header)
            out.append(d.subdata(in: 8 + n ..< d.count))
            try out.write(to: f, options: .atomic)
        }
    }

    struct Setup {
        var keep: [[Int]]
        var rows: [HealingRow]
        var rowSource: ShuffledRows
        var initial: [String: MLXArray]
        var checkpoint: URL
        var ends: [Int]
    }

    /// A `layers`-layer model written as a checkpoint, one block holding every layer.
    func setup(layers: Int = 6, keepCount: Int = 5, padded: Bool = true) throws -> Setup {
        let m = try HealingFixture.model(layers: layers)
        let keep = HealingFixture.keepSets(layers: layers, count: keepCount)
        let seed = try HealingFixture.healer(model: m, keep: keep)
        HealingFixture.randomizeAdapters(seed.adapters, scale: 0.02)
        let initial = HealingFixture.copyTensors(seed.adapters.tensors())
        let rows = [HealingFixture.row(seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        let dir = try tempDir("ckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: layers), adapters: nil, to: dir, options: o)
        if padded { try padHeaders(dir) }
        return Setup(keep: keep, rows: rows, rowSource: ShuffledRows(rows: rows, seed: 9), initial: initial, checkpoint: dir, ends: [layers - 1])
    }

    func train(
        _ s: Setup, resident: Int?, steps: Int = 5, map: Bool = true, budget: Int? = nil,
        configure: (inout HealingConfiguration) -> Void = { _ in }, sliced: Bool = true,
        ends: [Int]? = nil, memory: MemoryConfiguration = .init(),
        onStep: ((Int, StepRecord) -> Void)? = nil
    ) throws -> StreamedBlockHealer.Report {
        var c = HealingFixture.tinyConfiguration()
        c.residentLayers = resident
        c.memoryBudget = budget
        configure(&c)
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32, mapWeights: map)
        var h = StreamedBlockHealer(
            source: src, keep: s.keep, blockEnds: ends ?? s.ends, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, initialAdapters: s.initial)
        h.slicedStudent = sliced
        let store = InMemoryBoundaryStore()
        try h.precomputeBoundaries(rows: s.rows, store: store, options: .init(storageDType: .float32))
        return try h.train(rows: s.rowSource, steps: steps, store: store, memory: memory, onStep: onStep)
    }

    func assertSame(_ a: StreamedBlockHealer.Report, _ b: StreamedBlockHealer.Report, _ what: String, atol: Float = 1e-4) {
        XCTAssertEqual(Set(a.adapters.keys), Set(b.adapters.keys), what)
        for (k, v) in a.adapters { assertClose(b.adapters[k]!, v, atol: atol, rtol: 1e-3, "\(what) \(k)") }
        XCTAssertEqual(a.blockLosses.count, b.blockLosses.count)
        for (x, y) in zip(a.blockLosses.joined(), b.blockLosses.joined()) {
            XCTAssertEqual(x, y, accuracy: 1e-4 * max(1, abs(x)), "\(what) loss")
        }
    }

    // MARK: equivalence

    func testEveryWindowSizeEqualsTheWholeBlock() throws {
        let s = try setup()
        let whole = try train(s, resident: nil)
        var moved: Float = 0
        for (k, v) in whole.adapters { moved = max(moved, abs(v - s.initial[k]!).max().item(Float.self)) }
        XCTAssertGreaterThan(moved, 1e-3, "the adapters must really move for the comparison to mean anything")
        XCTAssertEqual(whole.residentLayers, 6)
        // 6 layers: windows of 1, 2, 4 (does not divide 6), 5 and the whole block.
        for r in [1, 2, 4, 5, 6] {
            let w = try train(s, resident: r)
            XCTAssertEqual(w.residentLayers, r)
            assertSame(whole, w, "residentLayers \(r)")
        }
    }

    func testWindowsEqualTheWholeBlockWithoutCheckpointingAndOnFullLayers() throws {
        let s = try setup()
        for (ckpt, sliced) in [(false, true), (true, false)] {
            let whole = try train(s, resident: nil, configure: { $0.checkpointBlocks = ckpt }, sliced: sliced)
            let w = try train(s, resident: 2, configure: { $0.checkpointBlocks = ckpt }, sliced: sliced)
            assertSame(whole, w, "checkpoint \(ckpt) sliced \(sliced)")
        }
    }

    func testASkippedStepAppliesNoUpdateInAnyWindow() throws {
        let s = try setup()
        var applied: [Bool] = []
        let r = try train(s, resident: 2, steps: 3, configure: { $0.maxLoss = 0 }, onStep: { _, rec in applied.append(rec.applied) })
        XCTAssertEqual(applied, [false, false, false])
        for (k, v) in r.adapters { assertClose(v, s.initial[k]!, atol: 0, rtol: 0, k) }
        XCTAssertEqual(r.skippedUpdates, [3])
    }

    func testMappedAndCopiedWeightsTrainIdentically() throws {
        let s = try setup()
        let a = try train(s, resident: 2, map: true)
        let b = try train(s, resident: 2, map: false)
        assertSame(a, b, "mapped vs copied", atol: 1e-5)
    }

    func testUnalignedTensorsAreCopiedAndTrainIdentically() throws {
        let s = try setup(padded: false)
        let a = try train(s, resident: 2, map: true)
        let b = try train(s, resident: 2, map: false)
        assertSame(a, b, "unpadded file, mapped vs copied", atol: 1e-5)
    }

    func testWindowedRunResumesFromACheckpointLikeAnUninterruptedOne() throws {
        let s = try setup()
        let straight = try train(s, resident: 2, steps: 6, configure: { $0.learningRate = 1e-3; $0.learningRateFloor = 1e-4 })
        // The checkpoint format is the whole-block path's: a windowed run's checkpoint resumes in
        // either path.
        let ckpt = try tempDir("resume")
        var c = HealingFixture.tinyConfiguration()
        c.learningRate = 1e-3
        c.learningRateFloor = 1e-4
        c.residentLayers = 2
        c.checkpointEvery = 1
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32)
        let h = StreamedBlockHealer(
            source: src, keep: s.keep, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, initialAdapters: s.initial)
        let store = InMemoryBoundaryStore()
        try h.precomputeBoundaries(rows: s.rows, store: store, options: .init(storageDType: .float32))
        var calls = 0
        var cb = HealingCallbacks()
        cb.shouldStop = { calls += 1; return calls == 4 }
        let first = try h.train(rows: s.rowSource, steps: 6, store: store, checkpointDirectory: ckpt, callbacks: cb)
        guard case .stopped = first.outcome else { return XCTFail("expected a stop") }
        var c2 = c
        c2.residentLayers = nil  // resume in the whole-block path
        let h2 = StreamedBlockHealer(
            source: src, keep: s.keep, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c2, initialAdapters: s.initial)
        let resumed = try h2.train(rows: s.rowSource, steps: 6, store: store, checkpointDirectory: ckpt)
        for (k, v) in straight.adapters { assertClose(resumed.adapters[k]!, v, atol: 2e-4, rtol: 1e-3, k) }
    }

    /// The same compute, not just the same math: layer forwards built per step (a recompute builds one
    /// again) must equal the whole-block path's for every window.
    func testWindowsDoNoMoreLayerForwardsThanTheWholeBlock() throws {
        // Compiled steps count their forwards by hand, so this guarantee is checked on the uncompiled path.
        let compiled = StreamedBlockHealer.compileLayerSteps
        StreamedBlockHealer.compileLayerSteps = false
        defer { StreamedBlockHealer.compileLayerSteps = compiled }
        let s = try setup()
        func forwardsPerStep(_ r: Int?) throws -> Int {
            var counts: [Int] = []
            _ = try train(s, resident: r, steps: 4, onStep: { _, _ in counts.append(healingLayerForwardCount) })
            let deltas = zip(counts.dropFirst(), counts).map { $0 - $1 }  // each step's own forwards
            XCTAssertEqual(Set(deltas).count, 1, "every step does the same work: \(deltas)")
            return deltas[0]
        }
        let whole = try forwardsPerStep(nil)
        XCTAssertEqual(whole, 12, "two per layer: the forward and the recompute")
        for r in [1, 2, 4, 5] {
            XCTAssertEqual(try forwardsPerStep(r), whole, "residentLayers \(r) does a different amount of compute")
        }
    }

    // MARK: memory-mapped weights

    func testMappedLayerSharesTheFilesPages() throws {
        #if !canImport(Darwin)
        try XCTSkipUnless(
            MappedSafetensors.zeroCopySupported,
            "only MLX's Metal allocator wraps external memory; the CPU and CUDA allocators copy it, so nothing is mapped")
        #endif
        let s = try setup(layers: 2, keepCount: 8)
        let cp = try Gemma4Checkpoint(directory: s.checkpoint)
        XCTAssertTrue(MappedSafetensors.zeroCopySupported, "this system copies mapped pages")
        let mapped = try cp.loadDecoderLayer(0, dtype: .float32, mapped: true)
        let copied = try cp.loadDecoderLayer(0, dtype: .float32, mapped: false)
        let a = Dictionary(uniqueKeysWithValues: mapped.parameters().flattened())
        let b = Dictionary(uniqueKeysWithValues: copied.parameters().flattened())
        for (k, v) in a { assertClose(v, b[k]!, atol: 0, rtol: 0, k) }
        // Overwrite a norm weight in the file in place: a view of the file's pages sees it, a copy does not.
        let files = try FileManager.default.contentsOfDirectory(at: s.checkpoint, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        // A tensor whose bytes start on its element size (those are mapped; others are read into memory).
        var found: (URL, MappedSafetensors.Entry, String)?
        for f in files {
            for (n, e) in try MappedSafetensors.index(of: f).sorted(by: { $0.key < $1.key }) where found == nil {
                guard let r = n.range(of: "layers.0.") else { continue }
                let k = String(n[r.upperBound...])
                if a[k] != nil, e.begin % 4 == 0, e.end - e.begin >= 4, e.dtype == .float32 { found = (f, e, k) }
            }
        }
        let (file, e, key) = found!
        let before = a[key]!.reshaped(-1)[0].item(Float.self)
        let fh = try FileHandle(forUpdating: file)
        try fh.seek(toOffset: UInt64(e.begin))
        var poison = Float(12345)
        try fh.write(contentsOf: Data(bytes: &poison, count: 4))
        try fh.close()
        XCTAssertEqual(a[key]!.reshaped(-1)[0].item(Float.self), 12345, "the mapped tensor should show the file's bytes")
        XCTAssertEqual(b[key]!.reshaped(-1)[0].item(Float.self), before, "a copy must not")
    }

    func testMappedPagesAreFileBackedAndOutsideTheFootprint() throws {
        #if !canImport(Darwin)
        try XCTSkipUnless(
            MappedSafetensors.zeroCopySupported,
            "only MLX's Metal allocator wraps external memory; the CPU and CUDA allocators copy it, so nothing is mapped")
        #endif
        let big = 96 * 1024 * 1024
        let url = try tempDir("map").appendingPathComponent("w.safetensors")
        var header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[\(big / 4)],\"data_offsets\":[0,\(big)]}}"
        while header.utf8.count % 8 != 0 { header += " " }
        var d = Data()
        var n = UInt64(header.utf8.count)
        d.append(Data(bytes: &n, count: 8))
        d.append(Data(header.utf8))
        FileManager.default.createFile(atPath: url.path, contents: d)
        let h = try FileHandle(forWritingTo: url)
        try h.seekToEnd()
        let chunk = Data((0 ..< 1 << 20).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        for _ in 0 ..< big / chunk.count { try h.write(contentsOf: chunk) }
        try h.close()
        let entry = try MappedSafetensors.index(of: url)["w"]!
        let fd = open(url.path, O_RDONLY)
        defer { close(fd) }
        let before = ProcessMemory.snapshot()
        let w = try MappedSafetensors.tensor(entry, fd: fd, in: url)
        // Read every page on the CPU through a copy-free reduction path: touch the mapping directly.
        eval(w.reshaped(1024, -1).sum())
        let wrapped = ProcessMemory.snapshot()
        XCTAssertLessThan(wrapped.footprint - before.footprint, big / 4, "mapped weights must not be charged to the footprint")
        XCTAssertGreaterThanOrEqual(Memory.activeMemory, big, "MLX counts the wrapped bytes")
    }

    // MARK: memory budget

    func testABudgetIsRespectedAndPicksTheLargestFittingWindow() throws {
        let s = try setup()
        // Unconstrained peak, then budgets between the whole-block estimate and the one-layer one.
        let free = try train(s, resident: nil, steps: 2)
        let one = try train(s, resident: 1, steps: 2)
        XCTAssertLessThan(one.peakMemoryBytes, free.peakMemoryBytes, "one resident layer should peak lower than the whole block")
        let inputs = try ResidencyPlanner.inputs(
            source: Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32),
            keep: s.keep, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, sliced: true, sequence: 24)
        // Without the slack the model must still cover what was measured, and not by an absurd margin.
        var bare = inputs
        bare.slackBytes = 0
        for (r, measured) in [(1, one.peakMemoryBytes), (6, free.peakMemoryBytes)] {
            let e = ResidencyPlanner.estimate(bare, resident: r).total
            XCTAssertGreaterThanOrEqual(Double(e), 0.95 * Double(measured), "R=\(r): estimate \(e) under what was measured (\(measured))")
            XCTAssertLessThanOrEqual(Double(e), 2.5 * Double(measured), "R=\(r): estimate \(e) far above what was measured (\(measured))")
        }
        let est = (1 ... 6).map { ResidencyPlanner.estimate(inputs, resident: $0).total }
        XCTAssertEqual(est, est.sorted(), "more resident layers cannot need less")
        let budget = (est[2] + est[3]) / 2
        let r = try train(s, resident: nil, steps: 2, budget: budget)
        XCTAssertTrue((2 ... 4).contains(r.residentLayers), "a window between the two the budget separates (\(r.residentLayers))")
        XCTAssertLessThanOrEqual(r.peakMemoryBytes, budget, "measured MLX peak must stay within the budget")
        XCTAssertNotNil(r.estimate)
        XCTAssertGreaterThanOrEqual(r.estimate!.total, r.peakMemoryBytes, "the model must not under-estimate")
    }

    func testAnOvershootOnABlocksLastStepIsReportedAndStopsTheRun() throws {
        let s = try setup()
        // One step per block: the check before a step never sees its peak, only the one after the block does.
        var m = MemoryConfiguration()
        m.peakStopBytes = 1000
        let r = try train(s, resident: 2, steps: 1, ends: [2, 5], memory: m)
        guard case .stopped(_, let why) = r.outcome else { return XCTFail("expected a stop, got \(r.outcome)") }
        XCTAssertTrue(why.contains("over 1000"), why)
        XCTAssertEqual(r.blockLosses.count, 1, "the first block finished and saved, the second never started")
        XCTAssertGreaterThan(r.peakOverLimit ?? 0, 1000)
        let calm = try train(s, resident: 2, steps: 1, ends: [2, 5])
        XCTAssertNil(calm.peakOverLimit)
        XCTAssertEqual(calm.outcome, .finished)
    }

    func testABudgetThatCannotBeMetIsRefusedBeforeTraining() throws {
        let s = try setup()
        var stepped = false
        XCTAssertThrowsError(try train(s, resident: nil, budget: 1_000_000, onStep: { _, _ in stepped = true })) { e in
            XCTAssertTrue("\(e)".contains("cannot be met"), "\(e)")
        }
        XCTAssertFalse(stepped)
        // A window asked for explicitly is checked against the budget too.
        let est = ResidencyPlanner.estimate(
            try ResidencyPlanner.inputs(
                source: Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32),
                keep: s.keep, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, sliced: true, sequence: 24),
            resident: 6)
        XCTAssertThrowsError(try train(s, resident: 6, budget: est.total / 2))
    }

    func testTheMemoryModelPicksOneLayerAtTheTightestBudget() throws {
        let i = ResidencyInputs(
            blockLayers: 6, layerWeightBytes: 1 << 30, adapterBytesPerLayer: 1 << 26, sequence: 100, hidden: 100,
            workspaceBytesPerToken: 1000, attentionBytes: 1000)
        let one = ResidencyPlanner.estimate(i, resident: 1).total
        XCTAssertEqual(try ResidencyPlanner.plan(budget: one, i).residentLayers, 1)
        XCTAssertThrowsError(try ResidencyPlanner.plan(budget: one - 1, i))
        XCTAssertEqual(try ResidencyPlanner.plan(budget: .max, i).residentLayers, 6)
    }
}

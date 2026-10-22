import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A source that counts layer loads and can stand in for any other.
final class CountingSource: LayerSource {
    let inner: LayerSource
    var loads: [ClosedRange<Int>] = []
    init(_ inner: LayerSource) { self.inner = inner }
    var layerCount: Int { inner.layerCount }
    var numExperts: Int { inner.numExperts }
    var slicesWithoutConsumingSource: Bool { inner.slicesWithoutConsumingSource }
    var canSliceLayers: Bool { inner.canSliceLayers }
    func teacherFingerprint() throws -> String { try inner.teacherFingerprint() }
    func embed(_ tokens: [MLXArray]) throws -> [MLXArray] { try inner.embed(tokens) }
    func makeEmbedder() throws -> (MLXArray) throws -> MLXArray { try inner.makeEmbedder() }
    func loadLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) throws -> [any HealableMoELayer] {
        loads.append(range)
        return try inner.loadLayers(range, keeping: keeping)
    }
}

/// The pre-computed teacher and the sliced student: both must be exactly the live, masked
/// formulation, to float noise.
final class StoredTeacherTests: ShearsTestCase {
    func tempDir(_ name: String = "stored") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func plainRow(_ length: Int, seed: UInt64, key: String) -> HealingRow {
        var rng = SplitMix64(seed: seed)
        return HealingRow(
            key: key, tokens: (0 ..< length).map { _ in Int32.random(in: 0 ..< Int32(HealingFixture.vocab), using: &rng) })
    }

    /// Teacher state at every block end for one row, computed on its own.
    func rowByRow(_ m: Gemma4TextModel, _ row: HealingRow) -> [Int: MLXArray] {
        let h = try! HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4))
        return h.teacherStates(row, at: Set(h.blockEnds))
    }

    /// An independent model with the same weights (fresh arrays, so slicing it leaves `m` alone).
    func copy(of m: Gemma4TextModel) throws -> Gemma4TextModel {
        let c = Gemma4TextModel(m.configuration)
        c.update(parameters: ModuleParameters.unflattened(m.parameters().flattened().map { ($0.0, $0.1 * 1) }))
        c.train(false)
        eval(c)
        return c
    }

    // MARK: pre-pass

    func checkPrepass(dtype: DType, storage: DType, tolerance: Float, file: StaticString = #filePath, line: UInt = #line) throws {
        let m = try HealingFixture.model(dtype: dtype)
        // Lengths straddle the sliding window (8); the budget forces padding in every batch.
        let rows = [
            plainRow(5, seed: 1, key: "short"), plainRow(24, seed: 2, key: "a"), plainRow(11, seed: 3, key: "b"),
            plainRow(24, seed: 4, key: "c"), plainRow(17, seed: 5, key: "d"), plainRow(8, seed: 6, key: "e"),
        ]
        let store = try DiskBoundaryStore(directory: try tempDir("bounds"))
        var o = TeacherPrepass.Options(storageDType: storage, maxBatchTokens: 60, maxBatchRows: 3)
        o.maxBatchTokens = 60
        let report = try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds, options: o)
            .run(rows: rows, store: store)
        XCTAssertGreaterThan(report.batchesRun, rows.count / 3)
        for row in rows {
            let want = rowByRow(m, row)
            let embedded = m.embed(row.tokenArray)
            let b0 = try store.get(row: row.key, boundary: 0)
            XCTAssertEqual(b0.shape, [1, row.length, 64], file: file, line: line)
            XCTAssertEqual(b0.dtype, storage, file: file, line: line)
            assertClose(b0.asType(.float32), embedded.asType(.float32), atol: tolerance, rtol: tolerance, "embedding \(row.key)", file: file, line: line)
            for (b, end) in m.defaultBlockEnds.enumerated() {
                let got = try store.get(row: row.key, boundary: b + 1)
                XCTAssertEqual(got.shape, [1, row.length, 64], file: file, line: line)
                assertClose(got.asType(.float32), want[end]!.asType(.float32), atol: tolerance, rtol: tolerance, "block \(b) row \(row.key)", file: file, line: line)
            }
        }
        // The documented disk cost.
        let files = try FileManager.default.contentsOfDirectory(at: store.directory, includingPropertiesForKeys: [.fileSizeKey])
            .filter { $0.pathExtension == "safetensors" }
        let onDisk = try files.reduce(0) { $0 + (try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
        let formula = TeacherPrepass.storageBytes(
            rowLengths: rows.map(\.length), hiddenSize: 64, blockCount: m.defaultBlockEnds.count, dtype: storage)
        XCTAssertGreaterThanOrEqual(onDisk, formula, file: file, line: line)
        XCTAssertLessThan(onDisk, formula + files.count * 256, "header overhead is small", file: file, line: line)
        XCTAssertEqual(files.count, rows.count * (m.defaultBlockEnds.count + 1), file: file, line: line)
    }

    func testBatchedPrepassEqualsRowByRowFloat32() throws {
        try checkPrepass(dtype: .float32, storage: .float32, tolerance: 1e-4)
    }

    func testBatchedPrepassEqualsRowByRowBFloat16() throws {
        // bf16 model, bf16 storage: lossless storage; only batch-shape rounding remains.
        let m = try HealingFixture.model(dtype: .bfloat16)
        let rows = [plainRow(24, seed: 2, key: "a"), plainRow(11, seed: 3, key: "b"), plainRow(5, seed: 1, key: "s")]
        let store = InMemoryBoundaryStore()
        try TeacherPrepass(
            source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds,
            options: .init(storageDType: .bfloat16, maxBatchTokens: 100, maxBatchRows: 3)
        ).run(rows: rows, store: store)
        for row in rows {
            let want = rowByRow(m, row)
            for (b, end) in m.defaultBlockEnds.enumerated() {
                let got = try store.get(row: row.key, boundary: b + 1).asType(.float32)
                let ref = want[end]!.asType(.float32)
                let rel = (abs(got - ref).mean() / abs(ref).mean()).item(Float.self)
                XCTAssertLessThan(rel, 0.03, "block \(b) row \(row.key)")
            }
        }
    }

    func testPrepassPacksBatchesByLengthUnderBudget() {
        let g = TeacherPrepass.batches(lengths: [5, 24, 11, 24, 17], maxTokens: 60, maxRows: 3)
        XCTAssertEqual(g.flatMap { $0 }.sorted(), [0, 1, 2, 3, 4])
        XCTAssertEqual(g.first, [1, 3])  // 3 x 24 = 72 > 60
        for b in g { XCTAssertLessThanOrEqual(b.count, 3) }
        // A row longer than the budget still gets its own batch.
        XCTAssertEqual(TeacherPrepass.batches(lengths: [100, 3], maxTokens: 10, maxRows: 4), [[0], [1]])
    }

    func testPrepassResumesBlockByBlockAndRejectsAStaleStore() throws {
        let m = try HealingFixture.model()
        let rows = [plainRow(24, seed: 2, key: "a"), plainRow(11, seed: 3, key: "b")]
        let dir = try tempDir("bounds")
        struct Stop: Error {}
        let first = CountingSource(ModelLayerSource(model: m))
        XCTAssertThrowsError(
            try TeacherPrepass(source: first, blockEnds: m.defaultBlockEnds, options: .init(storageDType: .float32))
                .run(rows: rows, store: try DiskBoundaryStore(directory: dir), onStageFinished: { if $0 == 1 { throw Stop() } }))
        XCTAssertEqual(first.loads, [0 ... 1])

        let second = CountingSource(ModelLayerSource(model: m))
        let store = try DiskBoundaryStore(directory: dir)
        let report = try TeacherPrepass(source: second, blockEnds: m.defaultBlockEnds, options: .init(storageDType: .float32))
            .run(rows: rows, store: store)
        XCTAssertEqual(report.skippedStages, [0, 1])
        XCTAssertEqual(second.loads, [2 ... 3], "finished blocks are not loaded again")
        for row in rows {
            let want = rowByRow(m, row)
            for (b, end) in m.defaultBlockEnds.enumerated() {
                assertClose(try store.get(row: row.key, boundary: b + 1), want[end]!, atol: 1e-4, rtol: 1e-4)
            }
        }
        // Different blocking, dtype or data must never reuse the store.
        XCTAssertThrowsError(try TeacherPrepass(source: second, blockEnds: [3], options: .init(storageDType: .float32)).run(rows: rows, store: store))
        XCTAssertThrowsError(try TeacherPrepass(source: second, blockEnds: m.defaultBlockEnds, options: .init(storageDType: .bfloat16)).run(rows: rows, store: store))
        var changed = rows
        changed[0].tokens[3] = (changed[0].tokens[3] + 1) % Int32(HealingFixture.vocab)
        XCTAssertThrowsError(try TeacherPrepass(source: second, blockEnds: m.defaultBlockEnds, options: .init(storageDType: .float32)).run(rows: changed, store: store))
    }

    // MARK: sliced student

    /// Every phase's loss and every adapter gradient, from three students: live masked (the
    /// reference), masked with stored targets, and sliced with stored targets.
    func checkStudents(
        reference: Gemma4TextModel, sliced makeSliced: (_ keep: [[Int]]) throws -> (Gemma4TextModel, StudentForm),
        keep: [[Int]], sortThreshold: Int? = nil, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        if let t = sortThreshold { HealingFixture.forceSortedGather(reference, threshold: t) }
        let rows = [HealingFixture.row(length: 24, seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        let store = InMemoryBoundaryStore()
        try TeacherPrepass(
            source: ModelLayerSource(model: reference), blockEnds: reference.defaultBlockEnds,
            options: .init(storageDType: .float32, maxBatchTokens: 64)
        ).run(rows: rows, store: store)
        for phase in HealingPhase.allCases {
            var c = HealingFixture.tinyConfiguration()
            if phase == .answerKL { c.trainFromLayer = 1 }
            let live = try HealingFixture.healer(model: reference, keep: keep, configuration: c)
            HealingFixture.randomizeAdapters(live.adapters)
            let initial = HealingFixture.copyTensors(live.adapters.tensors())
            let stored = try PruneHealer(
                model: reference, keep: keep, adapterConfiguration: HealingFixture.smallAdapters, configuration: c,
                initialAdapters: initial, teacher: store)
            let (sm, form) = try makeSliced(keep)
            if let t = sortThreshold { HealingFixture.forceSortedGather(sm, threshold: t) }
            let slim = try PruneHealer(
                model: sm, keep: keep, adapterConfiguration: HealingFixture.smallAdapters, configuration: c,
                initialAdapters: initial, student: form, teacher: store)
            for row in rows {
                let want = live.stepFor(phase, row, step: 3)
                XCTAssertGreaterThan(want.loss, 0, "\(phase)", file: file, line: line)
                for (name, h) in [("stored-masked", stored), ("sliced", slim)] {
                    let got = h.stepFor(phase, row, step: 3)
                    XCTAssertEqual(got.loss, want.loss, accuracy: 1e-4 * max(1, abs(want.loss)), "\(name) \(phase) loss", file: file, line: line)
                    XCTAssertEqual(Set(got.grads.keys), Set(want.grads.keys), file: file, line: line)
                    var moved: Float = 0
                    for (k, g) in want.grads {
                        moved = max(moved, abs(g).max().item(Float.self))
                        assertClose(got.grads[k]!, g, atol: 2e-5, rtol: 2e-3, "\(name) \(phase) grad \(k)", file: file, line: line)
                    }
                    XCTAssertGreaterThan(moved, 0, "gradients are not all zero", file: file, line: line)
                }
            }
        }
    }

    func testSlicedStudentEqualsMaskedStudentInEveryPhase() throws {
        let m = try HealingFixture.model()
        var keep = HealingFixture.keepSets(count: 4)
        keep[2] = [5]  // fewer kept experts than top-k
        try checkStudents(reference: m, sliced: { _ in (try self.copy(of: m), .sliced) }, keep: keep)
    }

    /// The same with the sorted gather path, which real runs (64 or more token-expert rows) use
    /// and the small sequences above do not reach.
    func testSlicedStudentEqualsMaskedStudentInEveryPhaseWithSortedGather() throws {
        let m = try HealingFixture.model()
        var keep = HealingFixture.keepSets(count: 4)
        keep[2] = [5]
        try checkStudents(reference: m, sliced: { _ in (try self.copy(of: m), .sliced) }, keep: keep, sortThreshold: 1)
    }

    func testSlicedQuantizedStudentEqualsMaskedStudentInEveryPhase() throws {
        try checkSlicedQuantized(sortThreshold: nil)
    }

    func testSlicedQuantizedStudentEqualsMaskedStudentInEveryPhaseWithSortedGather() throws {
        try checkSlicedQuantized(sortThreshold: 1)
    }

    func checkSlicedQuantized(sortThreshold: Int?) throws {
        let dir = try tempDir("q")
        try QuantizedFixture.write(try HealingFixture.model(), to: dir)
        let cp = try Gemma4Checkpoint(directory: dir)
        let q = try cp.loadModel()
        q.train(false)
        var keep = HealingFixture.keepSets(count: 4, seed: 9)
        keep[0] = [2]
        try checkStudents(reference: q, sliced: { keep in
            // Loaded already cut, layer by layer: the unpruned experts are never all resident.
            let s = try Gemma4Checkpoint(directory: dir).loadModel(keeping: keep)
            s.train(false)
            return (s, .preSliced(numExperts: HealingFixture.experts))
        }, keep: keep, sortThreshold: sortThreshold)
    }

    func testLoadingKeptExpertsEqualsLoadingThenSlicing() throws {
        let dir = try tempDir("q")
        try QuantizedFixture.write(try HealingFixture.model(), to: dir)
        let keep = HealingFixture.keepSets(count: 3, seed: 2)
        let direct = try Gemma4Checkpoint(directory: dir).loadModel(keeping: keep)
        let full = try Gemma4Checkpoint(directory: dir).loadModel()
        for (l, k) in zip(full.healableLayers, keep) { sliceToKeptExperts(l, set: ExpertSet(keep: k, numExperts: 8)) }
        let a = Dictionary(uniqueKeysWithValues: direct.parameters().flattened())
        let b = Dictionary(uniqueKeysWithValues: full.parameters().flattened())
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        for (k, v) in a { assertClose(v.asType(.float32), b[k]!.asType(.float32), atol: 0, rtol: 0, k) }
        XCTAssertEqual(direct.healableLayers[0].numExperts, 3)
        // Loading again from the same checkpoint still gives a full teacher.
        let again = try Gemma4Checkpoint(directory: dir).loadModel()
        XCTAssertEqual(again.healableLayers[0].numExperts, 8)
    }

    func testSlicedStudentHoldsOnlyItsKeptExpertsAndRefusesTheTeacher() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 3)
        let store = InMemoryBoundaryStore()
        let rows = [HealingFixture.row()]
        try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds, options: .init(storageDType: .float32))
            .run(rows: rows, store: store)
        let s = try copy(of: m)
        _ = try PruneHealer(
            model: s, keep: keep, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: HealingFixture.tinyConfiguration(), student: .sliced, teacher: store)
        for l in s.healableLayers {
            XCTAssertEqual(l.numExperts, 3)
            XCTAssertEqual(l.expertBank.gateProj.numExperts, 3)
        }
        // The teacher model is a different object and is untouched.
        XCTAssertEqual(m.healableLayers[0].numExperts, 8)
        // No store, no sliced student.
        XCTAssertThrowsError(try PruneHealer(model: try copy(of: m), keep: keep, student: .sliced))
    }

    func testDirtyLambdaNeedsALiveTeacher() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let row = HealingFixture.row()
        let store = InMemoryBoundaryStore()
        try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds, options: .init(storageDType: .float32))
            .run(rows: [row], store: store)
        var c = HealingFixture.tinyConfiguration()
        c.dirtyLambda = 0.5
        let plan = PhasePlan(entries: [(.dirty, 0)], total: 2)
        // Sliced: refused up front, not silently trained on the clean target.
        let sliced = try PruneHealer(
            model: try copy(of: m), keep: keep, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, student: .sliced, teacher: store)
        XCTAssertThrowsError(try sliced.run(rows: ShuffledRows(rows: [row]), plan: plan))
        // Masked with a stored clean target: the live teacher is still there for the extra term.
        let live = try HealingFixture.healer(model: m, keep: keep, configuration: c)
        HealingFixture.randomizeAdapters(live.adapters)
        let initial = HealingFixture.copyTensors(live.adapters.tensors())
        let stored = try PruneHealer(
            model: m, keep: keep, adapterConfiguration: HealingFixture.smallAdapters, configuration: c,
            initialAdapters: initial, teacher: store)
        let a = live.stepFor(.dirty, row, step: 0), b = stored.stepFor(.dirty, row, step: 0)
        XCTAssertEqual(a.loss, b.loss, accuracy: 1e-4)
        let clean = HealingFixture.tinyConfiguration()
        let noExtra = try PruneHealer(
            model: m, keep: keep, adapterConfiguration: HealingFixture.smallAdapters, configuration: clean,
            initialAdapters: initial, teacher: store)
        XCTAssertNotEqual(noExtra.stepFor(.dirty, row, step: 0).loss, b.loss, "lambda changes the loss")
    }

    func testInMemoryModelsAreNotCutUnlessAskedAndStorageFollowsTheModel() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 3)
        let plain = ModelLayerSource(model: m)
        XCTAssertThrowsError(try plain.loadLayers(0 ... 1, keeping: Array(keep[0 ... 1])))
        XCTAssertEqual(m.healableLayers[0].numExperts, 8, "the refusal left the model alone")
        let streamer = StreamedBlockHealer(source: plain, keep: keep + [keep[0]], blockEnds: m.defaultBlockEnds)
        XCTAssertFalse(streamer.slicedStudent, "an uncuttable source trains behind the mask")
        let cutting = ModelLayerSource(model: m, cutsModelInPlace: true)
        XCTAssertTrue(cutting.canSliceLayers)
        XCTAssertFalse(cutting.slicesWithoutConsumingSource, "cutting in place consumes the model")
        XCTAssertFalse(plain.canSliceLayers)
        XCTAssertTrue(plain.slicesWithoutConsumingSource)

        // Default storage: an fp32 model is stored as fp32, a bf16 model as bf16.
        let store = InMemoryBoundaryStore()
        try TeacherPrepass(source: plain, blockEnds: m.defaultBlockEnds).run(rows: [plainRow(10, seed: 1, key: "a")], store: store)
        XCTAssertEqual(try store.get(row: "a", boundary: 2).dtype, .float32)
        let b = try HealingFixture.model(dtype: .bfloat16)
        let store2 = InMemoryBoundaryStore()
        try TeacherPrepass(source: ModelLayerSource(model: b), blockEnds: b.defaultBlockEnds).run(rows: [plainRow(10, seed: 1, key: "a")], store: store2)
        XCTAssertEqual(try store2.get(row: "a", boundary: 2).dtype, .bfloat16)
    }

    func testTrainingRefusesAStoreFromAnotherBlockingOrOtherRows() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let row = HealingFixture.row()
        let opts = TeacherPrepass.Options(storageDType: .float32)
        let other = InMemoryBoundaryStore()
        try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: [3], options: opts).run(rows: [row], store: other)
        XCTAssertThrowsError(try PruneHealer(model: m, keep: keep, teacher: other), "different block ends")
        let good = InMemoryBoundaryStore()
        try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds, options: opts).run(rows: [row], store: good)
        _ = try PruneHealer(model: m, keep: keep, adapterConfiguration: HealingFixture.smallAdapters, teacher: good)
        var changed = row
        changed.tokens[2] = (changed.tokens[2] + 1) % Int32(HealingFixture.vocab)
        let streamer = StreamedBlockHealer(source: ModelLayerSource(model: m), keep: keep, blockEnds: m.defaultBlockEnds)
        XCTAssertThrowsError(try streamer.train(rows: ShuffledRows(rows: [changed]), steps: 1, store: good), "changed tokens")
        XCTAssertThrowsError(try streamer.train(rows: ShuffledRows(rows: [changed]), steps: 1, store: other), "other blocking")
        // A pre-pass that stopped halfway is refused too.
        struct Stop: Error {}
        let partial = InMemoryBoundaryStore()
        _ = try? TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds, options: opts)
            .run(rows: [row], store: partial, onStageFinished: { if $0 == 1 { throw Stop() } })
        XCTAssertThrowsError(try PruneHealer(model: m, keep: keep, teacher: partial), "unfinished pre-pass")
    }

    // MARK: streamed training

    func testStreamedSlicedStudentEqualsMaskedStudent() throws {
        let m = try HealingFixture.model()
        var keep = HealingFixture.keepSets(count: 4)
        keep[3] = [1]
        let seed = try HealingFixture.healer(model: m, keep: keep)
        HealingFixture.randomizeAdapters(seed.adapters, scale: 0.02)
        let initial = HealingFixture.copyTensors(seed.adapters.tensors())
        let rows = [HealingFixture.row(seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        let source = ShuffledRows(rows: rows, seed: 9)
        let ckpt = try tempDir("ckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: Array(repeating: Array(0 ..< 8), count: 4), adapters: nil, to: ckpt, options: o)
        let layers = CountingSource(Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .float32))
        var reports: [StreamedBlockHealer.Report] = []
        let store = try DiskBoundaryStore(directory: try tempDir("bounds"))
        for sliced in [false, true] {
            var s = StreamedBlockHealer(
                source: layers, keep: keep, blockEnds: m.defaultBlockEnds,
                adapterConfiguration: HealingFixture.smallAdapters, configuration: HealingFixture.tinyConfiguration(),
                initialAdapters: initial)
            s.slicedStudent = sliced
            if !sliced { try s.precomputeBoundaries(rows: rows, store: store, options: .init(storageDType: .float32)) }
            reports.append(try s.train(rows: source, steps: 5, store: store))
        }
        var moved: Float = 0
        for (k, v) in reports[0].adapters {
            moved = max(moved, abs(v - initial[k]!).max().item(Float.self))
            assertClose(reports[1].adapters[k]!, v, atol: 5e-5, rtol: 1e-3, k)
        }
        XCTAssertGreaterThan(moved, 1e-3)
    }

    // MARK: the teacher's identity, and a cut source is no teacher

    func testAPrepassOverASlicedSourceIsRefused() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 3)
        let row = HealingFixture.row()
        let victim = try copy(of: m)
        let source = ModelLayerSource(model: victim, cutsModelInPlace: true)
        let store = InMemoryBoundaryStore()
        try TeacherPrepass(source: source, blockEnds: victim.defaultBlockEnds, options: .init(storageDType: .float32))
            .run(rows: [row], store: store)
        // Train a sliced student from the stored teacher: this cuts the model in place.
        let streamer = StreamedBlockHealer(source: source, keep: keep, blockEnds: victim.defaultBlockEnds)
        XCTAssertTrue(streamer.slicedStudent)
        _ = try streamer.train(rows: ShuffledRows(rows: [row]), steps: 1, store: store)
        XCTAssertEqual(victim.healableLayers[0].numExperts, 3, "the source was cut")
        // A second pre-pass on the same source would use the sliced layers as the teacher.
        XCTAssertThrowsError(
            try TeacherPrepass(source: source, blockEnds: victim.defaultBlockEnds, options: .init(storageDType: .float32))
                .run(rows: [row], store: InMemoryBoundaryStore())
        ) { XCTAssertTrue("\($0)".contains("no longer have all"), "\($0)") }
    }

    func exportedCheckpoint(_ m: Gemma4TextModel, eps: Double? = nil) throws -> URL {
        let dir = try tempDir("ckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: Array(repeating: Array(0 ..< 8), count: 4), adapters: nil, to: dir, options: o)
        if let eps {
            let url = dir.appendingPathComponent("config.json")
            var json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            json["rms_norm_eps"] = eps
            try JSONSerialization.data(withJSONObject: json).write(to: url)
        }
        return dir
    }

    func testAStoreFromAnotherModelIsRefused() throws {
        let m = try HealingFixture.model()
        let rows = [plainRow(10, seed: 1, key: "a")]
        let a = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: try exportedCheckpoint(m)), dtype: .float32)
        let b = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: try exportedCheckpoint(m, eps: 1e-3)), dtype: .float32)
        let a16 = Gemma4LayerSource(checkpoint: a.checkpoint, dtype: .bfloat16)
        XCTAssertNotEqual(try a.teacherFingerprint(), try b.teacherFingerprint(), "config differs")
        XCTAssertNotEqual(try a.teacherFingerprint(), try a16.teacherFingerprint(), "load dtype differs")
        XCTAssertEqual(try a.teacherFingerprint(), try a.teacherFingerprint())

        let store = try DiskBoundaryStore(directory: try tempDir("bounds"))
        let opts = TeacherPrepass.Options(storageDType: .float32)
        try TeacherPrepass(source: a, blockEnds: m.defaultBlockEnds, options: opts).run(rows: rows, store: store)
        // Resuming with the same model is fine; with another it is refused.
        XCTAssertNoThrow(try TeacherPrepass(source: a, blockEnds: m.defaultBlockEnds, options: opts).run(rows: rows, store: store))
        XCTAssertThrowsError(try TeacherPrepass(source: b, blockEnds: m.defaultBlockEnds, options: opts).run(rows: rows, store: store)) {
            XCTAssertTrue("\($0)".contains("different model"), "\($0)")
        }
        // Training refuses it as well, from either side.
        let rowSource = ShuffledRows(rows: [HealingRow(key: "a", tokens: rows[0].tokens)])
        let keep = HealingFixture.keepSets(count: 4)
        XCTAssertThrowsError(try StreamedBlockHealer(source: b, keep: keep, blockEnds: m.defaultBlockEnds).train(rows: rowSource, steps: 1, store: store))
        XCTAssertThrowsError(try PruneHealer(model: m, keep: keep, teacher: store, teacherFingerprint: try b.teacherFingerprint()))
        XCTAssertNoThrow(try PruneHealer(model: m, keep: keep, teacher: store, teacherFingerprint: try a.teacherFingerprint()))
        _ = try StreamedBlockHealer(source: a, keep: keep, blockEnds: m.defaultBlockEnds).train(rows: rowSource, steps: 1, store: store)
    }

    func testAManifestWithoutAFingerprintIsRefused() throws {
        let m = try HealingFixture.model()
        let store = InMemoryBoundaryStore()
        let old = #"{"version":1,"layerCount":4,"blockEnds":[1,3],"dtype":"float32","rows":{},"completed":[0,1,2]}"#
        try store.putMetadata(Data(old.utf8), named: "prepass")
        XCTAssertThrowsError(
            try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds).run(rows: [], store: store)
        ) { XCTAssertTrue("\($0)".contains("older version"), "\($0)") }
        XCTAssertThrowsError(try PruneHealer(model: m, keep: HealingFixture.keepSets(count: 4), teacher: store))
    }

    // MARK: disk store naming

    func testKeysThatDifferOnlyInCaseGetTheirOwnFiles() throws {
        let dir = try tempDir("bounds")
        let store = try DiskBoundaryStore(directory: dir)
        let one = MLXArray.ones([1, 3, 4]), two = MLXArray.zeros([1, 3, 4])
        try store.put(one, row: "Ab", boundary: 1)
        // On a case-insensitive volume a name built from the key's characters is the same file, so
        // this would look already written (and a row would train on another row's targets).
        XCTAssertFalse(store.has(row: "ab", boundary: 1))
        try store.put(two, row: "ab", boundary: 1)
        XCTAssertEqual(try store.get(row: "Ab", boundary: 1).sum().item(Float.self), 12)
        XCTAssertEqual(try store.get(row: "ab", boundary: 1).sum().item(Float.self), 0)
        // Awkward keys: separators, non-ASCII, and a key past the filename length limit.
        for key in ["a/b", "../x", "naïve", String(repeating: "k", count: 400), ""] {
            try store.put(one, row: key, boundary: 0)
            XCTAssertTrue(store.has(row: key, boundary: 0), key)
        }
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).map { $0.lowercased() }
        XCTAssertEqual(Set(files).count, files.count, "no two files share a name in any case")
    }

    func testAStoreOfAnotherLayoutIsRefused() throws {
        let dir = try tempDir("old")
        try Data().write(to: dir.appendingPathComponent("rowa.1.safetensors"))  // an old-named file
        XCTAssertThrowsError(try DiskBoundaryStore(directory: dir))
        let fresh = try tempDir("fresh")
        _ = try DiskBoundaryStore(directory: fresh)
        XCTAssertNoThrow(try DiskBoundaryStore(directory: fresh), "reopening a store of the current layout")
        try JSONSerialization.data(withJSONObject: ["layout": 1]).write(to: fresh.appendingPathComponent(DiskBoundaryStore.layoutFile))
        XCTAssertThrowsError(try DiskBoundaryStore(directory: fresh))
    }
}

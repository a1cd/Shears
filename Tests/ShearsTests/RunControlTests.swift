import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The learning-rate schedule's reference values, mid-block checkpoints in the streamed path, and
/// the stop / pause hooks of every long-running path.
final class RunControlTests: ShearsTestCase {
    func tempDir(_ name: String = "run") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    // MARK: schedule against mlx.optimizers

    // Values from `join_schedules([linear_schedule(0, peak, warmup), cosine_decay(peak,
    // max(1, total - warmup), floor)], [warmup])` in mlx 0.30.6 (`mlx.optimizers`), evaluated at the
    // given step. The reference computes in float32, so compare at a relative 1e-5.
    typealias Vec = [(Int, Float)]
    static let refA: Vec = [
        (0, 0), (1, 0.000142857141), (3, 0.000428571424), (6, 0.000857142848), (7, 0.000999999931),
        (8, 0.000999209587), (20, 0.000872878241), (33, 0.000563334965), (40, 0.00038087758),
        (59, 0.000100790334), (60, 9.99999975e-05), (61, 9.99999975e-05), (90, 9.99999975e-05),
        (200, 9.99999975e-05),
    ]
    static let refB: Vec = [
        (0, 0), (2, 0.000199999995), (5, 0.000499999966), (6, 0.000499094196), (15, 0.000415285205),
        (25, 0.000224932781), (39, 5.09057791e-05), (40, 4.99999987e-05), (100, 4.99999987e-05),
    ]
    /// Peak 1e-3, floor 1e-4, warmup 7: length 60 before position 25, length 90 from there on (the
    /// schedule that was in force at each position).
    static let refChange: Vec = [
        (0, 0), (10, 0.000992903719), (24, 0.000790220511), (25, 0.000899537001),
        (26, 0.000888561714), (40, 0.000692293455), (60, 0.000360250677), (89, 0.000100322315),
        (90, 9.99999975e-05), (120, 9.99999975e-05),
    ]
    /// Peak 5e-3, floor 5e-4, warmup 2, length 40, at positions 5...8.
    static let refC: Vec = [(5, 0.00493115047), (6, 0.00487808883), (7, 0.00481049018), (8, 0.00472881552)]
    /// Peak 5e-3, floor 5e-4, warmup 2, length 6.
    static let refD: Vec = [
        (0, 0), (1, 0.00249999994), (2, 0.00499999989), (3, 0.00434098998), (4, 0.00274999975),
        (5, 0.00115900976),
    ]

    func assertVec(_ s: LearningRateSchedule, _ v: Vec, shift: Int = 0, line: UInt = #line) {
        for (pos, want) in v {
            XCTAssertEqual(s.value(at: pos - shift), want, accuracy: max(1e-9, abs(want) * 1e-5), "position \(pos)", line: line)
        }
    }

    func testScheduleMatchesMLXReference() {
        assertVec(LearningRateSchedule(peak: 1e-3, floor: 1e-4, warmup: 7, total: 60), Self.refA)
        assertVec(LearningRateSchedule(peak: 5e-4, floor: 5e-5, warmup: 5, total: 40), Self.refB)
        assertVec(LearningRateSchedule(peak: 5e-3, floor: 5e-4, warmup: 2, total: 6), Self.refD)
    }

    func testScheduleOffsetContinuesTheSameSchedule() {
        // A run that starts at schedule position 5 sees positions 5, 6, ... of the length-40 schedule.
        let s = LearningRateSchedule(peak: 5e-3, floor: 5e-4, warmup: 2, total: 40, offset: 5)
        assertVec(s, Self.refC, shift: 5)
        // Offset 20 on schedule A: step k of the run is position 20 + k.
        let a = LearningRateSchedule(peak: 1e-3, floor: 1e-4, warmup: 7, total: 60, offset: 20)
        for (pos, want) in Self.refA where pos >= 20 {
            XCTAssertEqual(a.value(at: pos - 20), want, accuracy: max(1e-9, abs(want) * 1e-5))
        }
    }

    func testScheduleLengthChangeMatchesTheScheduleInForceAtEachPosition() {
        let s = LearningRateSchedule(
            peak: 1e-3, floor: 1e-4, warmup: 7, total: 60, lengthChanges: [.init(from: 25, total: 90)])
        assertVec(s, Self.refChange)
        // The same change seen from a run that started 10 positions in.
        let shifted = LearningRateSchedule(
            peak: 1e-3, floor: 1e-4, warmup: 7, total: 60, offset: 10, lengthChanges: [.init(from: 25, total: 90)])
        for (pos, want) in Self.refChange where pos >= 10 {
            XCTAssertEqual(shifted.value(at: pos - 10), want, accuracy: max(1e-9, abs(want) * 1e-5))
        }
    }

    func testConfigurationScheduleIsIndependentOfRunSteps() {
        var c = HealingConfiguration()
        c.learningRate = 5e-3
        c.learningRateFloor = 5e-4
        c.warmupSteps = 2
        // By default the schedule is as long as the run.
        XCTAssertEqual(c.learningRateSchedule(runSteps: 6).total, 6)
        assertVec(c.learningRateSchedule(runSteps: 6), Self.refD)
        c.scheduleLength = 40
        c.scheduleOffset = 5
        let s = c.learningRateSchedule(runSteps: 3)
        XCTAssertEqual(s.total, 40)
        assertVec(s, Self.refC, shift: 5)
    }

    /// The learning rates a real run applies follow the offset schedule, in both paths.
    func testRunsApplyTheOffsetSchedule() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        var c = HealingFixture.tinyConfiguration()
        c.scheduleLength = 40
        c.scheduleOffset = 5
        let want = Self.refC.map(\.1)
        let rows = [HealingFixture.row(seed: 3, key: "a")]
        let source = ShuffledRows(rows: rows, seed: 9)

        let h = try HealingFixture.healer(model: m, keep: keep, configuration: c)
        let summary = try h.run(rows: source, plan: PhasePlan(entries: [(.block, 0)], total: 4))
        XCTAssertEqual(summary.history.map(\.learningRate), want)

        let store = InMemoryBoundaryStore()
        let streamer = StreamedBlockHealer(
            source: ModelLayerSource(model: m), keep: keep, blockEnds: m.defaultBlockEnds,
            adapterConfiguration: HealingFixture.smallAdapters, configuration: c)
        try streamer.precomputeBoundaries(rows: rows, store: store)
        var lrs: [Int: [Float]] = [:]
        _ = try streamer.train(rows: source, steps: 4, store: store, onStep: { b, r in lrs[b, default: []].append(r.learningRate) })
        XCTAssertEqual(lrs[0], want)
        XCTAssertEqual(lrs[1], want)
    }

    // MARK: mid-block checkpoints

    struct StreamSetup {
        var model: Gemma4TextModel
        var keep: [[Int]]
        var rows: [HealingRow]
        var source: ShuffledRows
        var configuration: HealingConfiguration
        var store: InMemoryBoundaryStore
        func healer() -> StreamedBlockHealer {
            StreamedBlockHealer(
                source: ModelLayerSource(model: model), keep: keep, blockEnds: model.defaultBlockEnds,
                adapterConfiguration: HealingFixture.smallAdapters, configuration: configuration)
        }
    }

    func streamSetup() throws -> StreamSetup {
        let m = try HealingFixture.model()
        let rows = [HealingFixture.row(seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        var c = HealingFixture.tinyConfiguration()
        c.checkpointEvery = 3
        let s = StreamSetup(
            model: m, keep: HealingFixture.keepSets(count: 4), rows: rows, source: ShuffledRows(rows: rows, seed: 9),
            configuration: c, store: InMemoryBoundaryStore())
        try s.healer().precomputeBoundaries(rows: rows, store: s.store)
        return s
    }

    static let steps = 8

    func worstDifference(_ a: [String: MLXArray], _ b: [String: MLXArray]) -> Float {
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        var worst: Float = 0
        for (k, v) in a { if let w = b[k] { worst = max(worst, abs(v - w).max().item(Float.self)) } }
        return worst
    }

    /// Adapters start from MLX's generator (no `initialAdapters` here), so this also proves a later
    /// block's initialisation does not depend on whether the run was interrupted.
    func uninterrupted(_ s: StreamSetup) throws -> [String: MLXArray] {
        let r = try s.healer().train(rows: s.source, steps: Self.steps, store: s.store)
        return HealingFixture.copyTensors(r.adapters)
    }

    /// Tolerance of the existing resume test: gather-matmul gradients differ between identical runs
    /// by about 1e-5, and a run moves the adapters by far more than this.
    static let tolerance: Float = 2e-3

    func testStopInsideABlockThenResumeEqualsUninterrupted() throws {
        let s = try streamSetup()
        let want = try uninterrupted(s)
        for stopAfter in [3, Self.steps + 3] {  // inside block 0, then inside block 1
            let out = try tempDir("blocks"), ck = try tempDir("ckpt")
            var seen = 0
            var cb = HealingCallbacks()
            cb.shouldStop = { seen >= stopAfter }
            let first = try s.healer().train(
                rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck,
                callbacks: cb, onStep: { _, _ in seen += 1 })
            XCTAssertEqual(first.outcome, .stopped(step: stopAfter % Self.steps, reason: "stop requested"))
            let second = try s.healer().train(
                rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck)
            XCTAssertEqual(second.outcome, .finished)
            let worst = worstDifference(want, second.adapters)
            XCTAssertLessThan(worst, Self.tolerance, "resume after stop at \(stopAfter) diverged")
            // A finished block leaves no checkpoint behind.
            XCTAssertFalse(FileManager.default.fileExists(atPath: ck.appendingPathComponent("block-0/ckpt").path))
        }
    }

    func testCrashInsideABlockThenResumeEqualsUninterrupted() throws {
        let s = try streamSetup()
        let want = try uninterrupted(s)
        let out = try tempDir("blocks"), ck = try tempDir("ckpt"), crashed = try tempDir("crashed")
        var cb = HealingCallbacks()
        // Take what is on disk the moment block 1's step-6 checkpoint lands, as if the process died
        // right after it; steps 6 and 7 of that run are then lost.
        cb.onCheckpoint = { step, dir in
            guard dir.lastPathComponent == "block-1", step == 6 else { return }
            try? FileManager.default.removeItem(at: crashed)
            try? FileManager.default.copyItem(at: ck, to: crashed)
        }
        _ = try s.healer().train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck,
            callbacks: cb)
        // The crashed run had finished block 0 (its file) and not block 1.
        let out2 = try tempDir("blocks2")
        try FileManager.default.copyItem(
            at: out.appendingPathComponent("block-0.safetensors"), to: out2.appendingPathComponent("block-0.safetensors"))
        var restarted: [Int] = []
        let resumed = try s.healer().train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out2, checkpointDirectory: crashed,
            onStep: { b, r in if b == 1 { restarted.append(r.step) } })
        XCTAssertEqual(restarted, [7, 8], "block 1 must resume at step 6, not restart")
        XCTAssertLessThan(worstDifference(want, resumed.adapters), Self.tolerance)
        var moved: Float = 0
        for (_, v) in want { moved = max(moved, abs(v).max().item(Float.self)) }
        XCTAssertGreaterThan(moved, 0.05)
    }

    func testResumeFallsBackToTheOldCheckpointWhenASaveWasInterrupted() throws {
        let s = try streamSetup()
        let want = try uninterrupted(s)
        let out = try tempDir("blocks"), ck = try tempDir("ckpt")
        var seen = 0
        var cb = HealingCallbacks()
        cb.shouldStop = { seen >= Self.steps + 4 }
        _ = try s.healer().train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck,
            callbacks: cb, onStep: { _, _ in seen += 1 })
        // Between a save's two renames only ckpt.old exists.
        let dir = ck.appendingPathComponent("block-1")
        try FileManager.default.moveItem(at: dir.appendingPathComponent("ckpt"), to: dir.appendingPathComponent("ckpt.old"))
        var firstResumedStep: Int?
        let resumed = try s.healer().train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck,
            onStep: { b, r in if b == 1, firstResumedStep == nil { firstResumedStep = r.step } })
        XCTAssertEqual(firstResumedStep, 5)
        XCTAssertLessThan(worstDifference(want, resumed.adapters), Self.tolerance)
    }

    func testInitialAdaptersNeverOverrideACheckpoint() throws {
        let s = try streamSetup()
        let seedHealer = try HealingFixture.healer(model: s.model, keep: s.keep)
        let initial = HealingFixture.copyTensors(seedHealer.adapters.tensors())
        let withInitial = StreamedBlockHealer(
            source: ModelLayerSource(model: s.model), keep: s.keep, blockEnds: s.model.defaultBlockEnds,
            adapterConfiguration: HealingFixture.smallAdapters, configuration: s.configuration, initialAdapters: initial)
        let want = HealingFixture.copyTensors(try withInitial.train(rows: s.source, steps: Self.steps, store: s.store).adapters)
        let out = try tempDir("blocks"), ck = try tempDir("ckpt")
        var seen = 0
        var cb = HealingCallbacks()
        cb.shouldStop = { seen >= 4 }
        _ = try withInitial.train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck,
            callbacks: cb, onStep: { _, _ in seen += 1 })
        let resumed = try withInitial.train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck)
        XCTAssertLessThan(worstDifference(want, resumed.adapters), Self.tolerance)
    }

    // MARK: stop and pause

    func testStreamedStopFileSavesACheckpointAndIsConsumed() throws {
        let s = try streamSetup()
        let out = try tempDir("blocks"), ck = try tempDir("ckpt")
        let stopFile = ck.appendingPathComponent("STOP")
        var cb = HealingCallbacks()
        cb.stopFile = stopFile
        let report = try s.healer().train(
            rows: s.source, steps: Self.steps, store: s.store, outputDirectory: out, checkpointDirectory: ck,
            callbacks: cb,
            onStep: { b, r in if b == 0, r.step == 2 { FileManager.default.createFile(atPath: stopFile.path, contents: Data()) } })
        XCTAssertEqual(report.outcome, .stopped(step: 2, reason: "stop requested"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: ck.appendingPathComponent("block-0/ckpt/meta.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: stopFile.path), "a stop is consumed once honoured")
        XCTAssertTrue(report.adapters.isEmpty, "no block finished")
    }

    func testStreamedPeakMemoryStopSavesACheckpoint() throws {
        let s = try streamSetup()
        let ck = try tempDir("ckpt")
        var mem = MemoryConfiguration()
        mem.peakStopBytes = 1  // anything the allocator has touched is over this
        let report = try s.healer().train(
            rows: s.source, steps: Self.steps, store: s.store, checkpointDirectory: ck, memory: mem)
        guard case .stopped(let step, let reason) = report.outcome else { return XCTFail("did not stop") }
        XCTAssertEqual(step, 0)
        XCTAssertTrue(reason.contains("peak memory"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: ck.appendingPathComponent("block-0/ckpt/meta.json").path))
    }

    func testMainPathStopFileAndPause() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let source = ShuffledRows(rows: [HealingFixture.row(seed: 3, key: "a")], seed: 9)
        let plan = try PhasePlan(entries: [(.block, 0)], total: 6)
        let dir = try tempDir("ckpt")
        let stopFile = dir.appendingPathComponent("STOP")
        var cb = HealingCallbacks()
        cb.stopFile = stopFile
        cb.onStep = { r in if r.step == 2 { FileManager.default.createFile(atPath: stopFile.path, contents: Data()) } }
        let h = try HealingFixture.healer(model: m, keep: keep)
        let summary = try h.run(rows: source, plan: plan, checkpointDirectory: dir, callbacks: cb)
        XCTAssertEqual(summary.outcome, .stopped(step: 2, reason: "stop requested"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: stopFile.path))
        let resumed = try HealingFixture.healer(model: m, keep: keep).run(rows: source, plan: plan, checkpointDirectory: dir)
        XCTAssertEqual(resumed.stepsRun, 4)

        // A pause file holds the run between steps until it is removed.
        let pauseFile = dir.appendingPathComponent("PAUSE")
        FileManager.default.createFile(atPath: pauseFile.path, contents: Data())
        var pcb = HealingCallbacks()
        pcb.pauseFile = pauseFile
        let started = Date()
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.8) { try? FileManager.default.removeItem(at: pauseFile) }
        let other = try tempDir("ckpt2")
        let paused = try HealingFixture.healer(model: m, keep: keep).run(
            rows: source, plan: plan, checkpointDirectory: other, callbacks: pcb)
        XCTAssertEqual(paused.outcome, .finished)
        XCTAssertGreaterThan(Date().timeIntervalSince(started), 0.7)
    }

    func testPrepassStopsBetweenBatchesAndResumes() throws {
        let s = try streamSetup()
        let rows = s.rows
        let store = InMemoryBoundaryStore()
        var options = TeacherPrepass.Options()
        options.maxBatchRows = 1
        var batches = 0
        let first = try s.healer().precomputeBoundaries(
            rows: rows, store: store, options: options, shouldStop: { batches += 1; return batches > 2 })
        XCTAssertTrue(first.stopped)
        XCTAssertEqual(first.batchesRun, 2)
        let second = try s.healer().precomputeBoundaries(rows: rows, store: store, options: options)
        XCTAssertFalse(second.stopped)
        let r = try s.healer().train(rows: s.source, steps: Self.steps, store: store)
        let want = try uninterrupted(s)
        // The pre-pass is deterministic for the same rows, so the stored targets are the same.
        XCTAssertLessThan(worstDifference(want, r.adapters), Self.tolerance)
    }
}

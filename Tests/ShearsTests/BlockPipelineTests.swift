import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

private func XCTAssertEqual(_ a: [Float], _ b: [Float], accuracy: Float, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, file: file, line: line)
    for (x, y) in zip(a, b) { XCTAssertEqual(x, y, accuracy: accuracy, file: file, line: line) }
}


/// Forwards to a source and records MLX's buffer-cache limit at each layer load and embedder request.
private final class RecordingSource: LayerSource {
    let inner: Gemma4LayerSource
    var limits: [Int] = []
    var embedLimits: [Int] = []
    init(_ inner: Gemma4LayerSource) { self.inner = inner }
    var layerCount: Int { inner.layerCount }
    var numExperts: Int { inner.numExperts }
    func embed(_ tokens: [MLXArray]) throws -> [MLXArray] { embedLimits.append(Memory.cacheLimit); return try inner.embed(tokens) }
    func makeEmbedder() throws -> (MLXArray) throws -> MLXArray { embedLimits.append(Memory.cacheLimit); return try inner.makeEmbedder() }
    func loadLayers(_ range: ClosedRange<Int>, keeping: [[Int]]?) throws -> [any HealableMoELayer] {
        limits.append(Memory.cacheLimit)
        return try inner.loadLayers(range, keeping: keeping)
    }
    func teacherFingerprint() throws -> String { try inner.teacherFingerprint() }
}

/// The block-sequential pipeline against a hand-written sequence of the same steps.
final class BlockPipelineTests: ShearsTestCase {
    func tempDir(_ name: String = "bp") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    struct Setup {
        var model: Gemma4TextModel
        var checkpoint: URL
        var keep: [[Int]]
        var rows: [HealingRow]
        var ends: [Int]
        var source: Gemma4LayerSource { get throws { Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: checkpoint), dtype: .float32) } }
    }

    func setup(layers: Int = 4, ends: [Int]? = nil, keepCount: Int = 5, rowCount: Int = 3, length: Int = 20) throws -> Setup {
        let m = try HealingFixture.model(layers: layers)
        let dir = try tempDir("bpckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: layers), adapters: nil, to: dir, options: o)
        let rows = (0 ..< rowCount).map { HealingFixture.row(length: length, seed: UInt64(3 + $0), key: "row\($0)") }
        return Setup(
            model: m, checkpoint: dir, keep: HealingFixture.keepSets(layers: layers, count: keepCount), rows: rows,
            ends: ends ?? m.defaultBlockEnds)
    }

    func config(lr: Float = 5e-3) -> HealingConfiguration {
        var c = HealingFixture.tinyConfiguration()
        c.learningRate = lr
        return c
    }

    func pipeline(
        _ s: Setup, dir: URL, mix: InputMix, batchSteps: Int, resident: Int? = nil, configure: (inout BlockPipelineOptions) -> Void = { _ in }
    ) throws -> BlockPipeline {
        var o = BlockPipelineOptions()
        o.batchSteps = batchSteps
        o.mix = mix
        o.fixedKeep = s.keep
        o.storageDType = .float32
        o.forcedResidentLayers = resident
        configure(&o)
        return try BlockPipeline(
            source: try s.source, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: config(), options: o, directory: dir)
    }

    // MARK: the manual sequence

    /// Teacher states entering every block and at the end, from the in-memory model.
    func teacherStates(_ s: Setup, _ row: HealingRow) -> [MLXArray] {
        let layers = s.model.healableLayers
        let net = HealingNetwork(layers: layers, sets: layers.map { ExpertSet.all($0.numExperts) }, adapters: nil)
        var h = s.model.embed(row.tokenArray)
        var out = [h]
        var start = 0
        for e in s.ends {
            h = net.forward(h, from: start, through: e, student: false).0
            eval(h)
            out.append(h)
            start = e + 1
        }
        return out
    }

    /// Trains block `b` by hand (whole block resident, no windows) and returns its adapters and the healed
    /// block's output for each input.
    func manualBlock(
        _ s: Setup, b: Int, steps: Int, order: [HealingRow], inputs: (HealingRow) -> [MLXArray], targets: [String: MLXArray],
        outputsFor: [(key: String, input: MLXArray)]
    ) throws -> (adapters: ModelAdapters, outputs: [String: MLXArray]) {
        let blocks = blockRanges(ends: s.ends)
        let range = blocks[b]
        let kept = Array(s.keep[range.start ... range.end])
        let layers = try (try s.source).loadLayers(range.start ... range.end, keeping: kept)
        let sets = kept.map { ExpertSet(keep: $0, numExperts: HealingFixture.experts) }
        MLXRandom.seed(config().seed &+ UInt64(range.start))
        let adapters = ModelAdapters(layers: layers, firstLayer: range.start, sets: sets, configuration: HealingFixture.smallAdapters)
        let net = HealingNetwork(layers: layers, firstLayer: range.start, sets: sets, adapters: adapters, sliced: true)
        let opt = HealingAdamW(weightDecay: config().weightDecay)
        let schedule = config().learningRateSchedule(runSteps: steps)
        for step in 0 ..< steps {
            let row = order[step]
            let xs = inputs(row)
            let target = targets[row.key]!
            let w = row.weights
            let vg = valueAndGrad(model: adapters) { (_: ModelAdapters, _: [MLXArray]) in
                var total = MLXArray(Float(0))
                for x in xs {
                    let (_, p) = net.forward(x, student: true, collect: [range.end], checkpointed: true)
                    total = total + HealingLoss.pin([range.end: p[range.end]!], teacher: [range.end: target], weights: w)
                }
                return [total / Float(xs.count)]
            }
            let (_, g) = vg(adapters, [])
            opt.update(model: adapters, grads: GradMath.scale(GradMath.flatten(g), by: 1 / Float(blocks.count)), learningRate: schedule.value(at: opt.step))
            eval(adapters, opt.stateArrays)
        }
        var outs: [String: MLXArray] = [:]
        for (k, x) in outputsFor {
            let o = net.forward(x, student: true).0
            eval(o)
            outs[k] = o
        }
        return (adapters, outs)
    }

    func assertAdapters(_ a: [String: MLXArray], _ b: ModelAdapters, _ what: String) {
        let t = b.tensors()
        XCTAssertEqual(Set(a.keys), Set(t.keys), what)
        for (k, v) in t { if let x = a[k] { assertClose(x, v, atol: 2e-4, rtol: 1e-3, "\(what) \(k)") } }
    }

    // MARK: tests

    func testPipelineEqualsTheManualSequenceWithCleanAndDirtyInputs() throws {
        let s = try setup()
        let steps = 4
        let order = (0 ..< steps).map { s.rows[$0 % s.rows.count] }
        let src = ShuffledRows(rows: s.rows, seed: 9)
        let ordered = (0 ..< steps).map { src.row(forStep: $0) }
        _ = order
        let teacher = Dictionary(uniqueKeysWithValues: s.rows.map { ($0.key, teacherStates(s, $0)) })

        // Manual: block 0 on clean inputs, then block 1 on both (stacked, loss the mean of the two pins).
        let b0 = try manualBlock(
            s, b: 0, steps: steps, order: ordered, inputs: { [teacher[$0.key]![0]] }, targets: teacher.mapValues { $0[1] },
            outputsFor: s.rows.map { ($0.key, teacher[$0.key]![0]) })
        let b1 = try manualBlock(
            s, b: 1, steps: steps, order: ordered, inputs: { [teacher[$0.key]![1], b0.outputs[$0.key]!] },
            targets: teacher.mapValues { $0[2] }, outputsFor: [])
        var expected = b0.adapters.tensors()
        expected.merge(b1.adapters.tensors()) { $1 }

        for resident in [nil, 1, 2] as [Int?] {
            for batchSteps in [steps, 2] {
                let p = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: batchSteps, resident: resident)
                let r = try p.run(rows: src, steps: steps)
                XCTAssertEqual(Set(r.adapters.keys), Set(expected.keys))
                for (k, v) in expected { assertClose(r.adapters[k]!, v, atol: 1.5e-3, rtol: 1e-2, "resident \(String(describing: resident)) batch \(batchSteps) \(k)") }
                // The dirty frontier entering block 1 is the independently computed healed block-0 output.
                for row in s.rows where ordered.contains(where: { $0.key == row.key }) {
                    assertClose(try p.get("d", row.key, 1), b0.outputs[row.key]!, atol: 2e-4, rtol: 1e-3, "dirty \(row.key)")
                    assertClose(try p.get("c", row.key, 2), teacher[row.key]![2], atol: 1e-4, rtol: 1e-3, "target \(row.key)")
                }
                XCTAssertEqual(r.blocks.map(\.residentLayers).first.map { $0 >= 1 }, true)
            }
        }
        // The adapters really moved, or the comparison means nothing.
        var moved: Float = 0
        for (_, v) in expected { moved = max(moved, abs(v).max().item(Float.self)) }
        XCTAssertGreaterThan(moved, 1e-3)
    }

    func testDirtyInputsDifferFromCleanOnceBlockZeroHasMoved() throws {
        let s = try setup()
        let src = ShuffledRows(rows: s.rows, seed: 9)
        let p = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 3)
        _ = try p.run(rows: src, steps: 3)
        for r in s.rows {
            let d = try p.get("d", r.key, 1), c = try p.get("c", r.key, 1)
            XCTAssertGreaterThan(abs(d - c).max().item(Float.self), 1e-4, "block 0 trained, so its dirty output must differ from the teacher's")
        }
    }

    func testSelectionInThePassEqualsSeparateSelection() throws {
        let s = try setup(keepCount: 4)
        var sc = SelectionConfiguration(keep: 4)
        sc.importanceTokens = 16
        sc.poolSize = 6
        sc.poolByMass = 3
        sc.poolByImportance = 3
        sc.swapRounds = 1
        sc.timeLimitPerLayer = nil
        // The search is order-sensitive (it subsamples the concatenated positions), so the separate run gets the
        // items in the order the pipeline's first batch has them.
        let order = ShuffledRows(rows: s.rows, seed: 1)
        let items = (0 ..< s.rows.count).map { order.row(forStep: $0) }.map { r in SelectionItem(key: r.key, tokens: r.tokens, searchPositions: Array(8 ..< r.length)) }
        let separate = ExpertSelector(model: s.model, configuration: sc).select(items: items)
        let p = try pipeline(s, dir: try tempDir(), mix: .clean, batchSteps: s.rows.count) { o in
            o.fixedKeep = nil
            o.selection = sc
            o.items = Dictionary(uniqueKeysWithValues: items.map { ($0.key, $0) })
            o.prepareOnly = true
        }
        let r = try p.run(rows: order, steps: s.rows.count)
        XCTAssertEqual(r.keep, KeepSets.keep(from: separate))
        XCTAssertEqual(r.selections.count, separate.count)
        for (a, b) in zip(r.selections.sorted { $0.layer < $1.layer }, separate) {
            XCTAssertEqual(a.chosenError, b.chosenError, accuracy: 1e-4 * max(1, abs(b.chosenError)))
        }
    }

    func testPrunedImportanceSearchPicksTheSameSetsAsTheExhaustiveOne() throws {
        let s = try setup(keepCount: 4, rowCount: 4, length: 24)
        // Skewed routing, as a trained router's is: two experts per layer get a zero logit, so they are rarely
        // picked and rank last by routing mass (a toy's random router is near uniform, where no candidate can be
        // dropped safely).
        let mask = MLXArray([1, 0, 1, 1, 0, 1, 1, 1] as [Float])[0..., .newAxis]
        for l in s.model.healableLayers {
            let m = l as! Gemma4MoEDecoderLayer
            m.router.proj.update(parameters: ModuleParameters.unflattened([("weight", m.router.proj.weight * mask)]))
        }
        eval(s.model)
        let items = s.rows.map { r in SelectionItem(key: r.key, tokens: r.tokens, searchPositions: Array(6 ..< r.length)) }
        func sets(_ candidates: Int?) -> [[Int]] {
            var sc = SelectionConfiguration(keep: 4)
            sc.importanceTokens = 40
            sc.poolSize = 4
            sc.poolByMass = 2
            sc.poolByImportance = 2
            sc.swapRounds = 2
            sc.timeLimitPerLayer = nil
            sc.importanceCandidates = candidates
            return KeepSets.keep(from: ExpertSelector(model: s.model, configuration: sc).select(items: items))
        }
        let full = sets(nil)
        XCTAssertEqual(sets(HealingFixture.experts), full, "every expert a candidate is the exhaustive search")
        XCTAssertEqual(sets(6), full, "skipping the two lowest-mass experts changes nothing when routing is skewed")
    }

    func testCompiledOptimizerStepEqualsTheUncompiledOne() throws {
        let s = try setup()
        let layers = try (try s.source).loadLayers(0 ... 1, keeping: Array(s.keep[0 ... 1]))
        let sets = s.keep[0 ... 1].map { ExpertSet(keep: $0, numExperts: HealingFixture.experts) }
        func run(compiled: Bool) -> ([String: MLXArray], [String: MLXArray]) {
            MLXRandom.seed(21)
            let a = ModelAdapters(layers: layers, firstLayer: 0, sets: sets, configuration: HealingFixture.smallAdapters)
            HealingFixture.randomizeAdapters(a, scale: 0.1)
            let opt = HealingAdamW(weightDecay: 0.01)
            HealingAdamW.compileStep = compiled
            defer { HealingAdamW.compileStep = true }
            for step in 0 ..< 3 {
                MLXRandom.seed(UInt64(100 + step))
                var grads: [String: MLXArray] = [:]
                for (k, v) in a.parameters().flattened() { grads[k] = MLXRandom.normal(v.shape) * 0.01 }
                eval(opt.update(layers: a, indices: [0, 1], grads: grads, learningRate: 1e-3 * Float(step + 1)))
            }
            return (a.tensors(), opt.stateTensors())
        }
        let (pc, sc) = run(compiled: true)
        let (pu, su) = run(compiled: false)
        for (k, v) in pu { assertClose(pc[k]!, v, atol: 1e-7, rtol: 1e-5, k) }
        for (k, v) in su { assertClose(sc[k]!, v, atol: 1e-9, rtol: 1e-5, k) }
        var moved: Float = 0
        for (k, v) in pu { moved = max(moved, abs(pc[k]! - v).max().item(Float.self)) }
        XCTAssertLessThan(moved, 1e-6)
    }

    func testAnInterruptedRunResumesLikeAnUninterruptedOne() throws {
        let s = try setup(layers: 6, ends: [1, 3, 5])
        let src = ShuffledRows(rows: s.rows, seed: 4)
        let whole = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 2).run(rows: src, steps: 4)
        let dir = try tempDir()
        var stops = 0
        var cb = HealingCallbacks()
        var seen = 0
        cb.shouldStop = { seen += 1; return seen == 7 }
        var pieces = 0
        var result: BlockPipeline.Report?
        for _ in 0 ..< 6 {
            let r = try pipeline(s, dir: dir, mix: .both, batchSteps: 2).run(rows: src, steps: 4, callbacks: cb)
            pieces += 1
            if case .stopped = r.outcome { stops += 1; continue }
            result = r
            break
        }
        XCTAssertEqual(stops, 1, "exactly one stop was requested")
        let resumed = try XCTUnwrap(result)
        XCTAssertEqual(Set(whole.adapters.keys), Set(resumed.adapters.keys))
        for (k, v) in whole.adapters { assertClose(resumed.adapters[k]!, v, atol: 2e-4, rtol: 1e-3, k) }
    }

    func testOnlyTheCurrentFrontierStaysOnDisk() throws {
        let s = try setup(layers: 6, ends: [1, 3, 5])
        let src = ShuffledRows(rows: s.rows, seed: 4)
        let p = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 3)
        _ = try p.run(rows: src, steps: 3)
        for r in s.rows {
            for kind in ["c", "d", "p"] {
                XCTAssertFalse(p.has(kind, r.key, 0), "boundary 0 is deleted once block 1 has its frontier")
                XCTAssertFalse(p.has(kind, r.key, 1), "boundary 1 is deleted once block 2 has its frontier")
            }
            XCTAssertTrue(p.has("c", r.key, 2) && p.has("d", r.key, 2))
        }
    }

    // MARK: residency

    func testAutoResidencyKeepsTheWholeBlockUnlessABudgetForcesLess() throws {
        var i = ResidencyInputs(
            blockLayers: 6, layerWeightBytes: 10 << 20, adapterBytesPerLayer: 4 << 20, sequence: 512, hidden: 256,
            workspaceBytesPerToken: 8192, attentionBytes: 2 << 20)
        i.slackBytes = 0
        let e = (1 ... 6).map { ResidencyPlanner.estimate(i, resident: $0).total }
        XCTAssertEqual(e, e.sorted())
        XCTAssertLessThan(e[0], e[5])
        // No budget: the whole block, whatever the other stages peaked at (even a tiny peak).
        for peak in [1, e[0], e[2] + 1, e[5] * 4] {
            let d = try ResidencyPlanner.chooseAuto(i, otherStagesPeak: peak, ceiling: nil)
            XCTAssertEqual(d.residentLayers, 6, "peak \(peak)")
            XCTAssertEqual(d.floorBytes, max(peak, e[0]))
        }
        // A roomy budget changes nothing.
        XCTAssertEqual(try ResidencyPlanner.chooseAuto(i, otherStagesPeak: e[2] + 1, ceiling: e[5] * 10).residentLayers, 6)
        // A budget that cannot hold the block gets the largest window that fits it.
        for r in 1 ... 5 {
            let d = try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 1, ceiling: e[r - 1] + 1)
            XCTAssertEqual(d.residentLayers, r, "budget just above estimate(\(r))")
        }
        // Below one layer's estimate (the floor) it is refused.
        XCTAssertThrowsError(try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 1, ceiling: e[0] - 1)) { error in
            guard case HealingError.budget(let m) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(m.contains("GB"), m)
        }
    }

    func testThePipelineReportsItsResidencyDecisionAndRefusesATooSmallBudget() throws {
        let s = try setup()
        let src = ShuffledRows(rows: s.rows, seed: 9)
        let r = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 3).run(rows: src, steps: 3)
        for b in r.blocks {
            let d = try XCTUnwrap(b.decision)
            XCTAssertGreaterThanOrEqual(d.residentLayers, 1)
            XCTAssertGreaterThanOrEqual(d.floorBytes, d.trainingMinimum)
            XCTAssertFalse(d.reason.isEmpty)
        }
        XCTAssertThrowsError(try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 3) { $0.memoryBudget = 1 << 20 }.run(rows: src, steps: 3))
    }

    // MARK: several rows per step, shared prefix, plateau annealing

    func sharedRows(count: Int = 4, prefix: Int = 20) -> [HealingRow] {
        var rng = SplitMix64(seed: 77)
        let head = (0 ..< prefix).map { _ in Int32.random(in: 0 ..< Int32(HealingFixture.vocab), using: &rng) }
        return (0 ..< count).map { i in
            let tail = (0 ..< (6 + 2 * (i % 3))).map { _ in Int32.random(in: 0 ..< Int32(HealingFixture.vocab), using: &rng) }
            return HealingRow(key: "sh\(i)", tokens: head + tail, answerStart: prefix)
        }
    }

    func testSeveralRowsPerStepTrainOnTheMeanOfTheirLossesAndEqualAcrossWindows() throws {
        var s = try setup()
        s.rows = (0 ..< 4).map { HealingFixture.row(length: 18 + 3 * ($0 % 3), seed: UInt64(30 + $0), key: "mr\($0)") }
        let src = SequentialRowSource(rows: s.rows)
        var firstLoss: Float?
        let whole = try pipeline(s, dir: try tempDir(), mix: .clean, batchSteps: 2) { $0.rowsPerStep = 2; $0.sharedPrefix = .off }
            .run(rows: src, steps: 2, onStep: { b, r in if b == 0 && r.step == 1 { firstLoss = r.loss } })
        // Step 0's loss is the mean of the two rows' own losses, each on its own length (padding is invisible).
        let teacher = Dictionary(uniqueKeysWithValues: s.rows.map { ($0.key, teacherStates(s, $0)) })
        let range = blockRanges(ends: s.ends)[0]
        let kept = Array(s.keep[range.start ... range.end])
        let layers = try (try s.source).loadLayers(range.start ... range.end, keeping: kept)
        let sets = kept.map { ExpertSet(keep: $0, numExperts: HealingFixture.experts) }
        MLXRandom.seed(config().seed &+ UInt64(range.start))
        let ad = ModelAdapters(layers: layers, firstLayer: 0, sets: sets, configuration: HealingFixture.smallAdapters)
        let net = HealingNetwork(layers: layers, sets: sets, adapters: ad, sliced: true)
        var want: Float = 0
        for r in s.rows[0 ..< 2] {
            let h = net.forward(teacher[r.key]![0], student: true, collect: [range.end]).1[range.end]!
            want += HealingLoss.pin([range.end: h], teacher: [range.end: teacher[r.key]![1]], weights: r.weights).item(Float.self) / 2
        }
        XCTAssertEqual(try XCTUnwrap(firstLoss), want, accuracy: 1e-4 * max(1, abs(want)))
        for resident in [1, 2] {
            let w = try pipeline(s, dir: try tempDir(), mix: .clean, batchSteps: 2, resident: resident) { $0.rowsPerStep = 2; $0.sharedPrefix = .off }
                .run(rows: src, steps: 2)
            for (k, v) in whole.adapters { assertClose(w.adapters[k]!, v, atol: 1.5e-3, rtol: 1e-2, "resident \(resident) \(k)") }
        }
    }

    func testASharedPrefixGivesExactlyTheSameFrontierAndAdapters() throws {
        var s = try setup()
        s.rows = sharedRows()
        let src = SequentialRowSource(rows: s.rows)
        BlockPipeline.sharedPrefixStepCount = 0
        BlockPipeline.sharedPrefixWalkCount = 0
        let plain = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 2, resident: 2) { $0.rowsPerStep = 2; $0.sharedPrefix = .off }
        let rp = try plain.run(rows: src, steps: 2)
        XCTAssertEqual(BlockPipeline.sharedPrefixStepCount + BlockPipeline.sharedPrefixWalkCount, 0)
        let shared = try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 2, resident: 2) { $0.rowsPerStep = 2; $0.sharedPrefix = .auto }
        let rs = try shared.run(rows: src, steps: 2)
        XCTAssertGreaterThan(BlockPipeline.sharedPrefixStepCount, 0, "training took the shared path")
        XCTAssertGreaterThan(BlockPipeline.sharedPrefixWalkCount, 0, "the teacher pass took the shared path")
        for r in s.rows {
            assertClose(try shared.get("c", r.key, 2), try plain.get("c", r.key, 2), atol: 2e-4, rtol: 1e-3, "teacher end \(r.key)")
            assertClose(try shared.get("d", r.key, 1), try plain.get("d", r.key, 1), atol: 2e-4, rtol: 1e-3, "dirty \(r.key)")
        }
        XCTAssertEqual(Set(rs.adapters.keys), Set(rp.adapters.keys))
        for (k, v) in rp.adapters { assertClose(rs.adapters[k]!, v, atol: 1.5e-3, rtol: 1e-2, k) }
        // A forced length that the rows do not share is refused.
        XCTAssertThrowsError(try pipeline(s, dir: try tempDir(), mix: .both, batchSteps: 2, resident: 2) { $0.rowsPerStep = 2; $0.sharedPrefix = .length(25) }.run(rows: src, steps: 2))
    }

    func testPlateauMonitorTriggersOnceAfterPatienceScoresWithoutImprovement() {
        let p = PlateauAnnealing(evalEvery: 5, patience: 2, threshold: 0.1, annealSteps: 10)
        var st = PlateauAnnealing.State()
        XCTAssertFalse(p.observe(1.0, step: 5, state: &st))
        XCTAssertFalse(p.observe(0.8, step: 10, state: &st), "20% better: improvement")
        XCTAssertFalse(p.observe(0.78, step: 15, state: &st), "2% better is below the threshold: first bad score")
        XCTAssertTrue(p.observe(0.79, step: 20, state: &st), "second bad score triggers")
        XCTAssertEqual(st.triggeredAt, 20)
        XCTAssertFalse(p.observe(0.79, step: 25, state: &st), "at most once")
    }

    func testPlateauStateAndShortenedScheduleSurviveAnInterruption() throws {
        var s = try setup()
        s.rows = (0 ..< 4).map { HealingFixture.row(length: 20, seed: UInt64(50 + $0), key: "p\($0)") }
        let held = HealingFixture.row(length: 20, seed: 99, key: "held")
        let src = SequentialRowSource(rows: s.rows)
        let plateau = PlateauAnnealing(evalEvery: 2, patience: 1, threshold: 0.99, annealSteps: 3)
        func make(_ dir: URL) throws -> BlockPipeline {
            try pipeline(s, dir: dir, mix: .both, batchSteps: 4) { $0.evalRows = [held]; $0.plateau = plateau }
        }
        var whole: [Float] = []
        let reference = try make(try tempDir()).run(rows: src, steps: 12, onStep: { b, r in if b == 0 { whole.append(r.learningRate) } })
        XCTAssertEqual(reference.blocks[0].annealedAt, 4)
        // Interrupt after the trigger (update 4), rerun, and the learning rates, scores and trigger match.
        let dir = try tempDir()
        var polls = 0
        var cb = HealingCallbacks()
        cb.shouldStop = { polls += 1; return polls == 7 }
        var resumed: [Float] = []
        var stopped = 0
        var last: BlockPipeline.Report?
        for _ in 0 ..< 3 {
            let r = try make(dir).run(rows: src, steps: 12, callbacks: cb, onStep: { b, rec in if b == 0 { resumed.append(rec.learningRate) } })
            if case .stopped = r.outcome { stopped += 1; continue }
            last = r
            break
        }
        XCTAssertEqual(stopped, 1)
        let r = try XCTUnwrap(last)
        XCTAssertEqual(resumed, whole, "the shortened schedule is the same after the resume")
        XCTAssertEqual(r.blocks[0].annealedAt, 4)
        XCTAssertEqual(r.blocks[0].heldOut, reference.blocks[0].heldOut, accuracy: 1e-4)
    }

    func testPlateauAnnealingShortensTheSchedule() throws {
        var s = try setup()
        s.rows = (0 ..< 4).map { HealingFixture.row(length: 20, seed: UInt64(50 + $0), key: "p\($0)") }
        let held = HealingFixture.row(length: 20, seed: 99, key: "held")
        let src = SequentialRowSource(rows: s.rows)
        func lrs(_ plateau: PlateauAnnealing?, dir: URL) throws -> ([Float], BlockPipeline.Report) {
            var out: [Float] = []
            let p = try pipeline(s, dir: dir, mix: .both, batchSteps: 4) { o in
                o.evalRows = [held]
                o.plateau = plateau
            }
            let r = try p.run(rows: src, steps: 12, onStep: { b, rec in if b == 0 { out.append(rec.learningRate) } })
            return (out, r)
        }
        let (base, _) = try lrs(nil, dir: try tempDir())
        // An impossible improvement bar: the second score already counts as a plateau.
        let plateau = PlateauAnnealing(evalEvery: 2, patience: 1, threshold: 0.99, annealSteps: 3)
        let (cut, report) = try lrs(plateau, dir: try tempDir())
        XCTAssertEqual(report.blocks[0].annealedAt, 4)
        XCTAssertEqual(report.blocks[0].heldOut.count, 2, "no scores after annealing triggered")
        XCTAssertEqual(Array(cut[0 ..< 4]), Array(base[0 ..< 4]), "nothing changes before the trigger")
        XCTAssertLessThan(cut[5], base[5] * 0.9, "the learning rate has dropped")
        XCTAssertEqual(cut.last!, 5e-4, accuracy: 1e-6, "and ends at the floor")
        XCTAssertGreaterThan(base.last!, cut.last! * 1.2)
    }

    /// Off by default: TEST_RUNNER_SHEARS_BENCH=1. Seconds per step of the pipeline on a toy block (hidden 512, 6 layers,
    /// 4 rows of 1024 tokens sharing a 896-token prefix), with and without the shared prefix, whole block resident.
    func testSharedPrefixSpeedBenchmark() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        let layers = 6
        let types = (0 ..< layers).map { $0 % 2 == 1 ? "full_attention" : "sliding_attention" }
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": 512, "num_hidden_layers": layers, "intermediate_size": 1024,
            "num_attention_heads": 4, "head_dim": 64, "global_head_dim": 128, "num_key_value_heads": 2,
            "num_global_key_value_heads": 1, "attention_k_eq_v": true, "vocab_size": 1024, "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0, "use_double_wide_mlp": false, "sliding_window": 128,
            "final_logit_softcapping": 30.0, "enable_moe_block": true, "num_experts": 16, "top_k_experts": 2,
            "moe_intermediate_size": 512, "layer_types": types, "tie_word_embeddings": true,
        ]
        let cfg = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        let m = Gemma4TextModel(cfg)
        MLXRandom.seed(5)
        m.update(parameters: ModuleParameters.unflattened(m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }))
        m.train(false)
        eval(m)
        let dir = try tempDir("spbench")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: Array(repeating: Array(0 ..< 16), count: layers), adapters: nil, to: dir, options: o)
        var rng = SplitMix64(seed: 3)
        let head = (0 ..< 896).map { _ in Int32.random(in: 0 ..< 1024, using: &rng) }
        let rows = (0 ..< 8).map { i in
            HealingRow(key: "b\(i)", tokens: head + (0 ..< 128).map { _ in Int32.random(in: 0 ..< 1024, using: &rng) }, answerStart: 896)
        }
        let keep = Array(repeating: Array(0 ..< 8), count: layers)
        for mode in [SharedPrefixMode.off, .auto] {
            var secs: [Double] = []
            var opts = BlockPipelineOptions()
            opts.fixedKeep = keep
            opts.storageDType = .float32
            opts.rowsPerStep = 4
            opts.mix = .clean
            opts.sharedPrefix = mode
            opts.forcedResidentLayers = layers
            opts.batchSteps = 4
            var c = HealingConfiguration()
            c.warmupSteps = 0
            let p = try BlockPipeline(
                source: Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: dir), dtype: .float32), blockEnds: [layers - 1],
                adapterConfiguration: AdapterConfiguration(expertRank: 16, denseRank: 16, routerRank: 8), configuration: c, options: opts,
                directory: try tempDir("spbenchrun"))
            let t0 = Date()
            _ = try p.run(rows: SequentialRowSource(rows: rows), steps: 4, onStep: { _, r in secs.append(r.seconds) })
            print("BENCH sharedPrefix=\(mode) median_s_per_step=\(secs.dropFirst().sorted()[secs.count / 2]) total_s=\(Date().timeIntervalSince(t0))")
        }
    }

    // MARK: gradients freed layer by layer

    /// Updating and freeing each layer's gradients one at a time gives exactly the adapters the combined update gives.
    func testEarlyGradientFreeingAndOverlappedLossCheckAreBitIdentical() throws {
        let s = try setup(layers: 4, ends: [3], keepCount: 5, rowCount: 3, length: 20)
        let priorFree = StreamedBlockHealer.freeGradientsEarly, priorOverlap = StreamedBlockHealer.overlapLossCheck
        defer { StreamedBlockHealer.freeGradientsEarly = priorFree; StreamedBlockHealer.overlapLossCheck = priorOverlap }
        var results: [BlockPipeline.Report] = []
        for (early, overlap) in [(false, false), (true, false), (false, true), (true, true)] {
            StreamedBlockHealer.freeGradientsEarly = early
            StreamedBlockHealer.overlapLossCheck = overlap
            let freed0 = StreamedBlockHealer.earlyFreedLayers, over0 = StreamedBlockHealer.overlappedSteps
            let p = try pipeline(s, dir: try tempDir(), mix: .clean, batchSteps: 6) { $0.maxBlocks = 1 }
            results.append(try p.run(rows: ShuffledRows(rows: s.rows, seed: 1), steps: 6))
            XCTAssertEqual(StreamedBlockHealer.earlyFreedLayers - freed0, early ? 4 * 6 : 0)
            XCTAssertEqual(StreamedBlockHealer.overlappedSteps - over0, overlap ? 6 : 0)
        }
        let a = results[0]
        XCTAssertFalse(a.adapters.isEmpty)
        for (n, b) in results.enumerated().dropFirst() {
            XCTAssertEqual(Set(a.adapters.keys), Set(b.adapters.keys))
            for (k, v) in a.adapters {
                XCTAssertTrue((v .== b.adapters[k]!).all().item(Bool.self), "variant \(n): \(k) differs bit for bit")
            }
            XCTAssertEqual(a.blocks[0].losses, b.blocks[0].losses, "variant \(n)")
        }
    }

    // MARK: kept intermediates in the pipeline

    func residencyInputsForKeep() -> ResidencyInputs {
        var i = ResidencyInputs(
            blockLayers: 6, layerWeightBytes: 10 << 20, adapterBytesPerLayer: 4 << 20, sequence: 512, hidden: 256,
            workspaceBytesPerToken: 8192, attentionBytes: 2 << 20)
        i.slackBytes = 0
        return i
    }

    /// The setting's rule: explicit values are kept as asked; `auto` keeps none without a budget, and with one the most
    /// layers whose estimate fits it, after the window and token chunk are chosen.
    func testTheKeepSettingResolvesAgainstTheBudget() throws {
        let i = residencyInputsForKeep()
        let est = { (k: Int) in ResidencyPlanner.estimate(i, resident: 6, keptLayers: k).total }
        XCTAssertEqual((0 ... 6).map(est), (0 ... 6).map(est).sorted())
        XCTAssertLessThan(est(0), est(6))
        func kept(_ keep: ActivationKeep, ceiling: Int?) throws -> (Int, Int) {
            let d = try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 1, ceiling: ceiling, keep: keep)
            return (d.keptLayers!, d.residentLayers)
        }
        // No budget: auto keeps none; explicit settings are honoured.
        for (keep, want) in [(ActivationKeep.auto, 0), (.lowMemory, 0), (.fast, 6), (.layers(3), 3), (.layers(99), 6)] {
            let (k, r) = try kept(keep, ceiling: nil)
            XCTAssertEqual(k, want, "\(keep) without a budget")
            XCTAssertEqual(r, 6)
        }
        // A budget that only just holds the block with nothing kept: auto keeps none; each step up buys exactly one more.
        for want in 0 ... 6 {
            let (k, r) = try kept(.auto, ceiling: est(want))
            XCTAssertEqual(k, want, "auto under a budget that fits \(want)")
            XCTAssertEqual(r, 6)
        }
        // More room than anything needs: auto does not go past the block.
        XCTAssertEqual(try kept(.auto, ceiling: est(6) * 10).0, 6)
        // A tight budget: `fast` gives up kept layers before it unloads any layer.
        let (k, r) = try kept(.fast, ceiling: est(2))
        XCTAssertEqual(r, 6)
        XCTAssertEqual(k, 2)
        // The decision names the setting it was made under (a resumed run under another one decides again).
        XCTAssertEqual(try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 1, ceiling: nil, keep: .layers(2)).keepSetting, "2")
    }

    func testTheKeepSettingParsesFromConfigValues() throws {
        for (text, want) in [("auto", ActivationKeep.auto), ("fast", .fast), ("lowMemory", .lowMemory), ("3", .layers(3)), ("0", .layers(0))] {
            XCTAssertEqual(try ActivationKeep(parsing: text), want)
        }
        XCTAssertThrowsError(try ActivationKeep(parsing: "some"))
        XCTAssertThrowsError(try ActivationKeep(parsing: "-1"))
        let decoded = try JSONDecoder().decode([ActivationKeep].self, from: Data(#"["fast", 2, "lowMemory", "auto"]"#.utf8))
        XCTAssertEqual(decoded, [.fast, .layers(2), .lowMemory, .auto])
    }

    /// Through the whole pipeline (the one-row compiled whole-block step): `fast`, `lowMemory`, an integer, and
    /// checkpointing off give the same adapters and losses to the bit, and the decision records what was kept.
    func testEveryKeepSettingTrainsBitIdentically() throws {
        let s = try setup(layers: 4, ends: [3], keepCount: 5, rowCount: 3, length: 20)
        var results: [(String, BlockPipeline.Report)] = []
        for (name, keep) in [("lowMemory", ActivationKeep.lowMemory), ("fast", .fast), ("2", .layers(2)), ("lowMemory again", .lowMemory)] {
            var c = config()
            c.keepIntermediates = keep
            var o = BlockPipelineOptions()
            o.batchSteps = 6; o.mix = .clean; o.fixedKeep = s.keep; o.storageDType = .float32; o.maxBlocks = 1
            let p = try BlockPipeline(
                source: try s.source, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters,
                configuration: c, options: o, directory: try tempDir())
            let r = try p.run(rows: ShuffledRows(rows: s.rows, seed: 1), steps: 6)
            let want = [ActivationKeep.lowMemory: 0, .fast: 4, .layers(2): 2][keep]!
            XCTAssertEqual(r.blocks[0].decision?.keptLayers, want, name)
            results.append((name, r))
        }
        let a = results[0].1
        for (name, b) in results.dropFirst() {
            for (k, v) in a.adapters { XCTAssertTrue((v .== b.adapters[k]!).all().item(Bool.self), "\(name): \(k) differs bit for bit") }
            XCTAssertEqual(a.blocks[0].losses, b.blocks[0].losses, name)
        }
        // checkpointing off is `fast`
        var c = config()
        c.checkpointBlocks = false
        var o = BlockPipelineOptions()
        o.batchSteps = 6; o.mix = .clean; o.fixedKeep = s.keep; o.storageDType = .float32; o.maxBlocks = 1
        let off = try BlockPipeline(
            source: try s.source, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, options: o, directory: try tempDir()
        ).run(rows: ShuffledRows(rows: s.rows, seed: 1), steps: 6)
        XCTAssertEqual(off.blocks[0].decision?.keptLayers, 4)
    }

    /// The options set the two statics for the run and put them back.
    func testPipelineOptionsSetAndRestoreTheStepSwitches() throws {
        let s = try setup(layers: 4, ends: [3], keepCount: 5, rowCount: 3, length: 20)
        let priorFree = StreamedBlockHealer.freeGradientsEarly, priorOverlap = StreamedBlockHealer.overlapLossCheck
        defer { StreamedBlockHealer.freeGradientsEarly = priorFree; StreamedBlockHealer.overlapLossCheck = priorOverlap }
        StreamedBlockHealer.freeGradientsEarly = true
        StreamedBlockHealer.overlapLossCheck = true
        let freed0 = StreamedBlockHealer.earlyFreedLayers, over0 = StreamedBlockHealer.overlappedSteps
        let p = try pipeline(s, dir: try tempDir(), mix: .clean, batchSteps: 4) {
            $0.maxBlocks = 1
            $0.freeGradientsEarly = false
            $0.overlapLossCheck = false
        }
        _ = try p.run(rows: ShuffledRows(rows: s.rows, seed: 1), steps: 4)
        XCTAssertEqual(StreamedBlockHealer.earlyFreedLayers, freed0)
        XCTAssertEqual(StreamedBlockHealer.overlappedSteps, over0)
        XCTAssertTrue(StreamedBlockHealer.freeGradientsEarly)
        XCTAssertTrue(StreamedBlockHealer.overlapLossCheck)
    }

    // MARK: the buffer cache is bounded before training

    /// Embedding, the residency measurement and the teacher walk run with MLX's buffer cache bounded from the batch size,
    /// not at the whole memory limit (freed buffers count toward the process footprint), and the global limit is put back.
    func testPreTrainingStagesBoundTheBufferCache() throws {
        let s = try setup(layers: 4, ends: [1, 3], keepCount: 5, rowCount: 3, length: 20)
        let big = 4 << 30
        Memory.cacheLimit = big
        defer { Memory.cacheLimit = 256 << 20 }
        let rec = RecordingSource(try s.source)
        var o = BlockPipelineOptions()
        o.batchSteps = 3
        o.mix = .clean
        o.fixedKeep = s.keep
        o.storageDType = .float32
        o.prepareOnly = true
        let p = try BlockPipeline(
            source: rec, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, configuration: config(), options: o,
            directory: try tempDir())
        _ = try p.run(rows: ShuffledRows(rows: s.rows, seed: 1), steps: 3)
        XCTAssertFalse(rec.limits.isEmpty)
        let bound = BlockPipeline.prepareCacheLimit(maxBatchTokens: o.maxBatchTokens, hidden: try HealingFixture.config().hiddenSize)
        XCTAssertLessThan(bound, big)
        for l in rec.limits.dropFirst() {  // the first load is the probe that reads the hidden size
             XCTAssertLessThanOrEqual(l, bound, "a load ran with the cache limit at \(l)") }
        XCTAssertFalse(rec.embedLimits.isEmpty)
        for l in rec.embedLimits { XCTAssertLessThanOrEqual(l, bound, "the embedding ran with the cache limit at \(l)") }
        XCTAssertEqual(Memory.cacheLimit, big, "the caller's limit is put back")

        // The same for the stand-alone teacher pre-pass.
        let rec2 = RecordingSource(try s.source)
        let store = InMemoryBoundaryStore()
        _ = try TeacherPrepass(source: rec2, blockEnds: s.ends, options: .init(storageDType: .float32)).run(rows: s.rows, store: store)
        XCTAssertFalse(rec2.limits.isEmpty)
        for l in rec2.limits.dropFirst() {  // the first block is loaded before its layer size is known
             XCTAssertLessThanOrEqual(l, bound, "a pre-pass load ran with the cache limit at \(l)") }
        XCTAssertEqual(Memory.cacheLimit, big)

        // A limit asked for wins.
        let rec3 = RecordingSource(try s.source)
        o.cacheLimitBytes = 100 << 20
        let p3 = try BlockPipeline(
            source: rec3, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, configuration: config(), options: o,
            directory: try tempDir())
        _ = try p3.run(rows: ShuffledRows(rows: s.rows, seed: 1), steps: 3)
        XCTAssertTrue(rec3.limits.dropFirst().allSatisfy { $0 == 100 << 20 })
    }
}

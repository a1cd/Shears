import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class HealingExportTests: ShearsTestCase {
    func tempDir(_ name: String = "heal") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    /// The recipe's exactness property: the model with routing restricted to the kept set and
    /// the adapters on equals the sliced model with the adapters merged.
    func assertExportMatchesMaskedModel(keepCount: Int, line: UInt = #line) throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: keepCount)
        let h = try HealingFixture.healer(model: m, keep: keep)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: h.adapters, to: dir, options: o)

        let reloaded = try Gemma4Checkpoint(directory: dir).loadModel(dtype: .float32)
        reloaded.train(false)
        XCTAssertEqual(reloaded.configuration.numExperts, keepCount, line: line)
        XCTAssertEqual(reloaded.configuration.topKExperts, min(HealingFixture.topK, keepCount), line: line)
        XCTAssertEqual((reloaded.layers[0] as! Gemma4MoEDecoderLayer).experts.switchGLU.gateProj.weight.dim(0), keepCount)

        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        let masked = h.logits(tokens: t, student: true)
        let sliced = reloaded(t)
        let teacher = h.logits(tokens: t, student: false)
        assertClose(masked, sliced, atol: 2e-3, rtol: 2e-3, "export vs masked+adapters", line: line)
        // Not vacuous: pruning and the adapters both move the output a lot more than that.
        XCTAssertGreaterThan(abs(masked - teacher).max().item(Float.self), 20 * abs(masked - sliced).max().item(Float.self), line: line)
        // And the adapters (not just the mask) matter: mask alone differs from mask + adapters.
        let zero = try HealingFixture.healer(model: m, keep: keep)
        let maskOnly = zero.logits(tokens: t, student: true)
        XCTAssertGreaterThan(abs(maskOnly - masked).max().item(Float.self), 20 * abs(masked - sliced).max().item(Float.self), line: line)
    }

    func testExportEqualsMaskedModelWhenKeptEqualsTopK() throws {
        try assertExportMatchesMaskedModel(keepCount: HealingFixture.topK)
    }

    func testExportEqualsMaskedModelWhenKeptExceedsTopK() throws {
        try assertExportMatchesMaskedModel(keepCount: 4)
    }

    func testMergedExportGoesThroughGGUFExport() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let h = try HealingFixture.healer(model: m, keep: keep)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        let src = try tempDir("tok")
        try GGUFTinyGemma4.writeTokenizer(to: src)
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .bfloat16
        o.sourceDirectory = src
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: h.adapters, to: dir, options: o)
        let out = dir.appendingPathComponent("pruned.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        XCTAssertEqual(summary.expertCount, 4)
        XCTAssertEqual(summary.expertsUsed, 2)
        XCTAssertEqual(summary.expertFeedForwardLength, 32)
        XCTAssertEqual(summary.warnings, [])
        let r = try GGUFReader(url: out)
        XCTAssertNotNil(try? r.floats("blk.0.attn_norm.weight"))
    }

    func testExportRefusesUnevenKeepCounts() throws {
        let m = try HealingFixture.model()
        let dir = try tempDir()
        var keep = HealingFixture.keepSets(count: 4)
        keep[1] = [0, 1, 2]
        XCTAssertThrowsError(try MergeAndSliceExport.write(model: m, keep: keep, adapters: nil, to: dir))
    }

    // MARK: streamed block phase

    func testStreamedBlockPhaseEqualsInMemoryBlockPhase() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let seed = try HealingFixture.healer(model: m, keep: keep)
        let initial = HealingFixture.copyTensors(seed.adapters.tensors())
        let rows = [HealingFixture.row(seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        let source = ShuffledRows(rows: rows, seed: 9)
        let steps = 6
        var c = HealingFixture.tinyConfiguration()
        c.learningRate = 5e-3

        let main = try HealingFixture.healer(model: m, keep: keep, configuration: c, initial: initial)
        try main.run(rows: source, plan: PhasePlan(entries: [(.block, 0)], total: steps))
        let want = main.adapters.tensors()

        // Streamed from layers in memory, with an in-memory store.
        let streamer = StreamedBlockHealer(
            source: ModelLayerSource(model: m), keep: keep, blockEnds: m.defaultBlockEnds,
            adapterConfiguration: HealingFixture.smallAdapters, configuration: c, initialAdapters: initial)
        let store = InMemoryBoundaryStore()
        try streamer.precomputeBoundaries(rows: rows, store: store)
        let report = try streamer.train(rows: source, steps: steps, store: store)

        // Streamed from a checkpoint directory (layers read one block at a time), with a disk store.
        let ckptDir = try tempDir("ckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: 4), adapters: nil, to: ckptDir, options: o)
        let cp = try Gemma4Checkpoint(directory: ckptDir)
        let streamer2 = StreamedBlockHealer(
            source: Gemma4LayerSource(checkpoint: cp, dtype: .float32), keep: keep,
            blockEnds: m.defaultBlockEnds, adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, initialAdapters: initial)
        let disk = try DiskBoundaryStore(directory: try tempDir("bounds"))
        try streamer2.precomputeBoundaries(rows: rows, store: disk)
        let report2 = try streamer2.train(rows: source, steps: steps, store: disk, outputDirectory: try tempDir("blocks"))

        var moved: Float = 0
        for (k, v) in want { moved = max(moved, abs(v - initial[k]!).max().item(Float.self)) }
        XCTAssertGreaterThan(moved, 0.01)
        for r in [report, report2] {
            XCTAssertEqual(Set(r.adapters.keys), Set(want.keys))
            var worst: Float = 0
            for (k, v) in want { worst = max(worst, abs(v - r.adapters[k]!).max().item(Float.self)) }
            XCTAssertLessThan(worst, 2e-3, "streamed block phase differs from the in-memory one (moved \(moved))")
        }
    }

    // MARK: selection

    func testSelectionKeepingEverythingIsExact() throws {
        let m = try HealingFixture.model()
        let items = (0 ..< 3).map { i in
            let r = HealingFixture.row(seed: UInt64(20 + i))
            return SelectionItem(key: "s\(i)", tokens: r.tokens, searchPositions: Array(0 ..< r.length))
        }
        var c = SelectionConfiguration(keep: HealingFixture.experts)
        c.timeLimitPerLayer = nil
        let out = ExpertSelector(model: m, configuration: c).select(items: items)
        XCTAssertEqual(out.count, 4)
        for r in out {
            XCTAssertEqual(r.chosen, Array(0 ..< HealingFixture.experts))
            XCTAssertLessThan(r.chosenError, 1e-8)
            XCTAssertLessThan(r.layerOutputRelError["selection"]!, 1e-6)
        }
    }

    func testSelectionFindsAnOptimalSetAndReportsHonestErrors() throws {
        let m = try HealingFixture.model()
        var items = (0 ..< 3).map { i -> SelectionItem in
            let r = HealingFixture.row(seed: UInt64(30 + i))
            return SelectionItem(key: "s\(i)", tokens: r.tokens, searchPositions: Array(0 ..< r.length))
        }
        let v = HealingFixture.row(seed: 40)
        items.append(SelectionItem(key: "v", tokens: v.tokens, searchPositions: [], isValidation: true))
        var c = SelectionConfiguration(keep: 4)
        c.timeLimitPerLayer = nil
        c.poolByMass = 8
        c.poolByImportance = 8
        c.poolSize = 8
        let out = ExpertSelector(model: m, configuration: c).select(items: items, maxLayers: 1)
        let rec = out[0]
        XCTAssertEqual(rec.chosen.count, 4)
        XCTAssertLessThanOrEqual(rec.chosenError, rec.popularitySwapError + 1e-9)
        XCTAssertLessThanOrEqual(rec.chosenError, rec.greedyError + 1e-9)
        XCTAssertLessThanOrEqual(rec.chosenError, rec.popularityError + 1e-9)

        // Independent brute force over all C(8,4) sets, from the layer's own parts.
        let layer = m.healableLayers[0]
        var xs: [MLXArray] = [], ls: [MLXArray] = []
        for it in items where !it.isValidation {
            let parts = layer.attentionPart(m.embed(MLXArray(it.tokens)[.newAxis]), adapters: nil)
            xs.append(parts.expertInput[0]); ls.append(parts.routerLogits[0])
        }
        let x = concatenated(xs), logits = concatenated(ls)
        let (ai, aw) = routeNative(layer, logits)
        let target = layer.normalizeExpertBranch(expertBranch(layer, x: x, indices: ai, weights: aw, set: nil, lora: nil)).asType(.float32)
        func error(_ keep: [Int]) -> Float {
            let (i, w) = route(layer, logits: logits, set: ExpertSet(keep: keep, numExperts: 8), tweak: nil, routerInput: nil)
            let y = layer.normalizeExpertBranch(expertBranch(layer, x: x, indices: i, weights: w, set: nil, lora: nil)).asType(.float32)
            return (square(y - target).sum(axis: -1) / square(target).sum(axis: -1)).mean().item(Float.self)
        }
        var all: [(Float, [Int])] = []
        for a in 0 ..< 8 { for b in a + 1 ..< 8 { for c in b + 1 ..< 8 { for d in c + 1 ..< 8 { all.append((error([a, b, c, d]), [a, b, c, d])) } } } }
        all.sort { $0.0 < $1.0 }
        let mine = error(rec.chosen)
        XCTAssertEqual(mine, rec.chosenError, accuracy: 1e-4 + 1e-3 * mine, "reported error is not the set's true error")
        let rank = all.firstIndex { $0.1 == rec.chosen }!
        XCTAssertLessThan(rank, 3, "search landed at rank \(rank) of 70 (best \(all[0].0), got \(mine))")
        XCTAssertGreaterThan(all.last!.0, 1.5 * all[0].0, "toy is not discriminating")
        XCTAssertNotNil(rec.validationBranchError["chosen"])
        // The greedy start alone is already a good set (the swap search then only polishes it).
        XCTAssertLessThan(rec.greedyError, all[35].0, "greedy start is worse than the median set")
        XCTAssertLessThan(rec.popularityError, all.last!.0 + 1e-9)
    }
}

final class HealingDefaultsTests: ShearsTestCase {
    func testDefaultPinWeightsAreTheRecipesRampNotUniform() {
        let row = HealingRow(tokens: [Int32](repeating: 1, count: 100), answerStart: 90)
        let w = row.weights.asArray(Float.self)
        XCTAssertEqual(w[0], 0.15, accuracy: 1e-6)
        XCTAssertEqual(w[89], 0.15 + 0.85 * (63.0 / 64.0), accuracy: 1e-5)
        XCTAssertEqual(w[90], 1, accuracy: 1e-6)
        XCTAssertEqual(w[99], 1, accuracy: 1e-6)
        // Without an answer start the ramp ends at the last position, still not all ones.
        XCTAssertLessThan(HealingRow(tokens: [Int32](repeating: 1, count: 100)).weights.asArray(Float.self)[10], 0.2)
        // Explicit weights are used as given.
        XCTAssertEqual(HealingRow(tokens: [1, 2], pinWeights: MLXArray([2, 3] as [Float])).weights.asArray(Float.self), [2, 3])
    }

    func testSelectionDefaultsMatchTheRecipe() {
        let c = SelectionConfiguration(keep: 8)
        XCTAssertEqual(c.candidateBatch, 4)
        XCTAssertEqual(c.timeLimitPerLayer, 180)
        XCTAssertEqual(c.swapRounds, 3)
        XCTAssertEqual([c.poolByMass, c.poolByImportance, c.poolSize, c.importanceTokens], [40, 24, 64, 400])
    }

    func testPrecomputedKeepSetsLoadInEveryShape() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("keep-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        func write(_ name: String, _ obj: Any) throws -> URL {
            let u = dir.appendingPathComponent(name)
            try JSONSerialization.data(withJSONObject: obj).write(to: u)
            return u
        }
        // The recipe's selection record, with extra per-layer fields.
        let record: [String: Any] = ["K": 3, "layers": [
            "0": ["layer": 0, "chosen": [5, 1, 3], "chosen_err": 0.1],
            "1": ["layer": 1, "chosen": [0, 2, 7]],
        ]]
        XCTAssertEqual(try KeepSets.load(from: write("rec.json", record), layerCount: 2), [[1, 3, 5], [0, 2, 7]])
        XCTAssertEqual(try KeepSets.load(from: write("plain.json", [[2, 1], [4, 3]])), [[1, 2], [3, 4]])
        XCTAssertEqual(try KeepSets.load(from: write("keep.json", ["keep": [[0, 1]]])), [[0, 1]])
        XCTAssertThrowsError(try KeepSets.load(from: write("short.json", [[0]]), layerCount: 2))
        let out = dir.appendingPathComponent("saved.json")
        try KeepSets.save([[3, 1]], to: out)
        XCTAssertEqual(try KeepSets.load(from: out), [[1, 3]])
    }

    func testRoutingProbeAndNormalizedSum() throws {
        let m = try HealingFixture.model()
        let toks = HealingFixture.row().tokens
        let mass = RoutingProbe.mass(model: m, sequences: [(toks, nil), (toks, [3, 4])])
        XCTAssertEqual(mass.count, 4)
        XCTAssertEqual(mass[0].count, 8)
        // Each position's routing weights are positive and per-expert-scaled, so mass is positive
        // somewhere; and the full-sequence pass dominates the two-position one.
        let only = RoutingProbe.mass(model: m, sequences: [(toks, [3, 4])])
        XCTAssertGreaterThan(mass[0].reduce(0, +), only[0].reduce(0, +))
        let comb = RoutingProbe.normalizedSum([[[1, 3], [2, 2]], [[5, 5], [0, 4]]])
        XCTAssertEqual(comb[0], [0.25 + 0.5, 0.75 + 0.5])
        XCTAssertEqual(comb[1], [0.5, 0.5 + 1])
    }

    func testStreamedModeReportsStepsAndSkipsBadLosses() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let rows = [HealingFixture.row(key: "a")]
        var c = HealingFixture.tinyConfiguration()
        c.maxLoss = 1e-9  // every loss is "bad"
        let s = StreamedBlockHealer(
            source: ModelLayerSource(model: m), keep: keep, blockEnds: m.defaultBlockEnds,
            adapterConfiguration: HealingFixture.smallAdapters, configuration: c)
        let store = InMemoryBoundaryStore()
        try s.precomputeBoundaries(rows: rows, store: store)
        var records: [(Int, StepRecord)] = []
        var finished: [Int] = []
        let report = try s.train(rows: ShuffledRows(rows: rows), steps: 3, store: store,
            onStep: { records.append(($0, $1)) }, onBlockFinished: { b, _ in finished.append(b) })
        XCTAssertEqual(records.count, 2 * 3)
        XCTAssertTrue(records.allSatisfy { !$0.1.applied && $0.1.loss.isFinite })
        XCTAssertEqual(report.skippedUpdates, [3, 3])
        XCTAssertEqual(finished, [0, 1])
        XCTAssertEqual(records.map(\.0), [0, 0, 0, 1, 1, 1])
        // Nothing moved: the b halves are still zero.
        for (k, v) in report.adapters where k.hasSuffix(".b") || k.hasSuffix("_b") {
            XCTAssertEqual(abs(v).max().item(Float.self), 0, k)
        }
    }
}

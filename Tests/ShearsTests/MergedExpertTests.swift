import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The merged expert layout (one wide MLP over the kept experts) against the routed gather path, and the shared
/// expert LoRA against the per-expert LoRA it equals.
final class MergedExpertTests: ShearsTestCase {
    override func tearDown() {
        ExpertCompute.merged = true
        ExpertCompute.mergedMaxKeptRatio = 1.5
        ExpertCompute.tokenChunk = nil
        super.tearDown()
    }

    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("merged-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    /// A fresh layer 0, optionally cut to `keep`, its expert set and randomized adapters.
    func fixture(
        keepCount: Int, sliced: Bool, shared: Bool = false, quantized: Bool = false, dtype: DType = .float32,
        router: Bool = true, reusing: (layer: Gemma4MoEDecoderLayer, set: ExpertSet)? = nil
    ) throws -> (layer: Gemma4MoEDecoderLayer, set: ExpertSet, adapters: LayerAdapters) {
        let layer: Gemma4MoEDecoderLayer, set: ExpertSet
        if let reusing {
            (layer, set) = reusing
        } else {
            let m = try HealingFixture.model(layers: 2, dtype: dtype)
            layer = m.layers[0] as! Gemma4MoEDecoderLayer
            set = ExpertSet(keep: HealingFixture.keepSets(layers: 1, count: keepCount)[0], numExperts: HealingFixture.experts)
            if quantized { ExpertQuantizer.quantizeExperts(layer, groupSize: 32, bits: 4) }
            if sliced { sliceToKeptExperts(layer, set: set) }
        }
        var c = AdapterConfiguration(expertRank: 6, denseRank: 4, routerRank: router ? 4 : 0)
        c.sharedExperts = shared
        MLXRandom.seed(21)
        let la = LayerAdapters(layer: layer, keptExperts: keepCount, numExperts: HealingFixture.experts, configuration: c)
        let p = la.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
        la.update(parameters: ModuleParameters.unflattened(p))
        eval(la)
        return (layer, set, la)
    }

    /// Output, adapter gradients and input gradient of one layer forward, for a fixed cotangent.
    func run(
        _ f: (layer: Gemma4MoEDecoderLayer, set: ExpertSet, adapters: LayerAdapters), sliced: Bool, merged: Bool,
        x: MLXArray, cot: MLXArray
    ) -> (out: MLXArray, grads: [String: MLXArray], gx: MLXArray) {
        ExpertCompute.merged = merged
        let out = healingLayerForward(f.layer, x, set: f.set, adapters: f.adapters, sliced: sliced)
        let vg = valueAndGrad(model: f.adapters) { (m: LayerAdapters, _: [MLXArray]) in
            [(healingLayerForward(f.layer, x, set: f.set, adapters: m, sliced: sliced).asType(.float32) * cot).sum()]
        }
        let (_, g) = vg(f.adapters, [])
        let gx = grad { (x: MLXArray) in
            (healingLayerForward(f.layer, x, set: f.set, adapters: f.adapters, sliced: sliced).asType(.float32) * cot).sum()
        }(x)
        eval(out, gx)
        return (out, Dictionary(uniqueKeysWithValues: g.flattened()), gx)
    }

    func assertEquivalent(
        keepCount: Int, sliced: Bool, shared: Bool = false, quantized: Bool = false, dtype: DType = .float32,
        atol: Float = 2e-4, rtol: Float = 2e-4, line: UInt = #line
    ) throws {
        let f = try fixture(keepCount: keepCount, sliced: sliced, shared: shared, quantized: quantized, dtype: dtype)
        ExpertCompute.mergedMaxKeptRatio = 100
        MLXRandom.seed(5)
        let x = MLXRandom.normal([2, 9, 64]).asType(dtype)
        let cot = MLXRandom.normal([2, 9, 64])
        let before = ExpertCompute.mergedBranchCount
        let routed = run(f, sliced: sliced, merged: false, x: x, cot: cot)
        XCTAssertEqual(ExpertCompute.mergedBranchCount, before, "the routed run must not take the merged layout", line: line)
        let merged = run(f, sliced: sliced, merged: true, x: x, cot: cot)
        XCTAssertGreaterThan(ExpertCompute.mergedBranchCount, before, "the merged layout did not run", line: line)
        assertClose(merged.out, routed.out, atol: atol, rtol: rtol, "forward", file: #filePath, line: line)
        assertClose(merged.gx, routed.gx, atol: atol, rtol: rtol, "input grad", file: #filePath, line: line)
        XCTAssertEqual(Set(merged.grads.keys), Set(routed.grads.keys), line: line)
        var experts = 0
        for (k, v) in routed.grads {
            // The scale of this gradient sets how close "close" has to be.
            let scale = max(1, abs(v).max().item(Float.self))
            assertClose(merged.grads[k]!, v, atol: atol * scale, rtol: rtol, "grad \(k)", file: #filePath, line: line)
            if k.hasPrefix("experts.") { experts += 1; XCTAssertGreaterThan(abs(v).max().item(Float.self), 0, k, line: line) }
        }
        XCTAssertEqual(experts, 6, line: line)
    }

    // MARK: merged == routed

    func testMergedEqualsRoutedWhenKeptEqualsTopK() throws {
        try assertEquivalent(keepCount: HealingFixture.topK, sliced: true)
    }

    func testMergedEqualsRoutedWhenKeptExceedsTopKIncludingUnpickedExperts() throws {
        // 4 kept, top-2: every token leaves two kept experts out, which must contribute exactly nothing.
        try assertEquivalent(keepCount: 4, sliced: true)
        try assertEquivalent(keepCount: 8, sliced: true)
    }

    func testMergedEqualsRoutedBehindTheMask() throws {
        try assertEquivalent(keepCount: 2, sliced: false)
        try assertEquivalent(keepCount: 5, sliced: false)
    }

    func testMergedEqualsRoutedForQuantizedBanks() throws {
        try assertEquivalent(keepCount: 2, sliced: true, quantized: true)
        try assertEquivalent(keepCount: 4, sliced: true, quantized: true)
    }

    func testMergedEqualsRoutedInBfloat16Loosely() throws {
        try assertEquivalent(keepCount: 2, sliced: true, dtype: .bfloat16, atol: 0.15, rtol: 0.1)
        try assertEquivalent(keepCount: 4, sliced: true, dtype: .bfloat16, atol: 0.15, rtol: 0.1)
    }

    func testTokenChunkingChangesNothing() throws {
        let f = try fixture(keepCount: 4, sliced: true)
        ExpertCompute.mergedMaxKeptRatio = 100
        MLXRandom.seed(6)
        let x = MLXRandom.normal([2, 9, 64])
        let cot = MLXRandom.normal([2, 9, 64])
        let whole = run(f, sliced: true, merged: true, x: x, cot: cot)
        for chunk in [1, 4, 7] {
            ExpertCompute.tokenChunk = chunk
            let c = run(f, sliced: true, merged: true, x: x, cot: cot)
            assertClose(c.out, whole.out, atol: 1e-5, rtol: 1e-5, "chunk \(chunk) forward")
            assertClose(c.gx, whole.gx, atol: 1e-4, rtol: 1e-4, "chunk \(chunk) input grad")
            for (k, v) in whole.grads { assertClose(c.grads[k]!, v, atol: 1e-4, rtol: 1e-4, "chunk \(chunk) \(k)") }
        }
    }

    func testRatioDecidesTheLayout() throws {
        let f = try fixture(keepCount: 4, sliced: true)
        let x = MLXRandom.normal([1, 5, 64])
        func built(_ ratio: Double) -> Bool {
            ExpertCompute.mergedMaxKeptRatio = ratio
            let before = ExpertCompute.mergedBranchCount
            eval(healingLayerForward(f.layer, x, set: f.set, adapters: f.adapters, sliced: true))
            return ExpertCompute.mergedBranchCount > before
        }
        XCTAssertFalse(built(1.5), "4 kept over top-2 is 2x")
        XCTAssertTrue(built(2))
        XCTAssertFalse(built(0))
    }

    // MARK: per-expert LoRA in the merged layout equals the existing adapter format

    func testPerExpertLoRARunsInTheMergedLayoutWithTheExistingFormat() throws {
        // Shapes are the existing format's: [n, in, r] / [n, r, out].
        let f = try fixture(keepCount: 2, sliced: true)
        let e = f.adapters.experts!
        XCTAssertEqual(e.gateA.shape, [2, 64, 6])
        XCTAssertEqual(e.gateB.shape, [2, 6, 32])
        XCTAssertEqual(e.downA.shape, [2, 32, 6])
        XCTAssertEqual(e.downB.shape, [2, 6, 64])
    }

    // MARK: shared expert LoRA

    func testSharedLoRAMergedEqualsItsPerExpertExpansionRouted() throws {
        for keepCount in [2, 4] {
            let f = try fixture(keepCount: keepCount, sliced: true, shared: true)
            let e = f.adapters.experts!
            XCTAssertEqual(e.gateA.shape, [64, 6])
            XCTAssertEqual(e.gateB.shape, [6, keepCount * 32])
            XCTAssertEqual(e.downA.shape, [keepCount * 32, 6])
            ExpertCompute.mergedMaxKeptRatio = 100
            MLXRandom.seed(8)
            let x = MLXRandom.normal([2, 9, 64])
            let cot = MLXRandom.normal([2, 9, 64])
            let merged = run(f, sliced: true, merged: true, x: x, cot: cot)
            // The same layer with the expansion as a plain per-expert LoRA, routed.
            let pe = e.perExpertTensors
            let f2 = try fixture(keepCount: keepCount, sliced: true, reusing: (f.layer, f.set))
            var flat = f.adapters.parameters().flattened().filter { !$0.0.hasPrefix("experts.") }
            flat += [
                ("experts.gate_a", pe.gateA), ("experts.gate_b", pe.gateB), ("experts.up_a", pe.upA),
                ("experts.up_b", pe.upB), ("experts.down_a", pe.downA), ("experts.down_b", pe.downB),
            ]
            f2.adapters.update(parameters: ModuleParameters.unflattened(flat))
            eval(f2.adapters)
            let routed = run(f2, sliced: true, merged: false, x: x, cot: cot)
            assertClose(merged.out, routed.out, atol: 2e-4, rtol: 2e-4, "shared vs expansion, K=\(keepCount)")
        }
    }

    func testSharedLoRAGradientsEqualThroughTheExpansion() throws {
        let f = try fixture(keepCount: 4, sliced: true, shared: true)
        ExpertCompute.mergedMaxKeptRatio = 100
        MLXRandom.seed(9)
        let x = MLXRandom.normal([2, 9, 64])
        let cot = MLXRandom.normal([2, 9, 64])
        // Merged layout vs the routed path (which expands the shared factors): gradients of the shared tensors.
        let m = run(f, sliced: true, merged: true, x: x, cot: cot)
        let r = run(f, sliced: true, merged: false, x: x, cot: cot)
        for (k, v) in r.grads where k.hasPrefix("experts.") {
            let scale = max(1, abs(v).max().item(Float.self))
            assertClose(m.grads[k]!, v, atol: 2e-4 * scale, rtol: 2e-4, k)
        }
    }

    func assertSharedExport(keepCount: Int, merged: Bool, line: UInt = #line) throws {
        ExpertCompute.merged = merged
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: keepCount)
        var c = AdapterConfiguration(expertRank: 8, denseRank: 8, routerRank: 4)
        c.sharedExperts = true
        let h = try HealingFixture.healer(model: m, keep: keep, adapters: c)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: h.adapters, to: dir, options: o)
        let reloaded = try Gemma4Checkpoint(directory: dir).loadModel(dtype: .float32)
        reloaded.train(false)
        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        let trained = h.logits(tokens: t, student: true)
        let exported = reloaded(t)
        let err = abs(trained - exported).max().item(Float.self)
        assertClose(trained, exported, atol: 2e-3, rtol: 2e-3, "exported vs trained (shared)", file: #filePath, line: line)
        // Not vacuous: the shared LoRA moves the output far more than the export error.
        let zero = try HealingFixture.healer(model: m, keep: keep, adapters: c)
        let without = zero.logits(tokens: t, student: true)
        XCTAssertGreaterThan(abs(without - trained).max().item(Float.self), 20 * err, line: line)
    }

    func testSharedLoRAExportEqualsTrainedModel() throws {
        try assertSharedExport(keepCount: 2, merged: true)
        try assertSharedExport(keepCount: 4, merged: true)
        try assertSharedExport(keepCount: 4, merged: false)
    }

    func testSharedLoRARefusesQuantizationAwareTraining() {
        var c = AdapterConfiguration(sharedExperts: true)
        c.qat = QATConfiguration(.q8_0)
        XCTAssertThrowsError(try c.validate())
    }

    // MARK: inference and budget wiring

    func testPrunedModelInferenceMatchesRoutedInPrefillAndDecode() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 3)
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: nil, to: dir, options: o)
        let pruned = try Gemma4Checkpoint(directory: dir).loadModel(dtype: .float32)
        pruned.train(false)
        let t = MLXArray(HealingFixture.row(length: 24).tokens)[.newAxis]
        ExpertCompute.merged = false
        let routed = pruned(t)
        eval(routed)
        ExpertCompute.merged = true
        let before = ExpertCompute.mergedBranchCount
        let merged = pruned(t)
        XCTAssertGreaterThan(ExpertCompute.mergedBranchCount, before, "prefill must take the merged layout (3 kept over top-2)")
        assertClose(merged, routed, atol: 2e-4, rtol: 2e-4, "prefill logits")
        // One decoded token (kept 3 > top-k 2): stays routed.
        let one = t[0..., 0 ..< 1]
        let b2 = ExpertCompute.mergedBranchCount
        eval(pruned(one))
        XCTAssertEqual(ExpertCompute.mergedBranchCount, b2)
    }

    func testBudgetLowersTheTokenChunkBeforeDroppingResidentLayers() throws {
        var i = ResidencyInputs(
            blockLayers: 6, layerWeightBytes: 4 << 20, adapterBytesPerLayer: 1 << 20, sequence: 2048, hidden: 256,
            workspaceBytesPerToken: 4096, attentionBytes: 1 << 20)
        i.slackBytes = 0
        i.expertWorkspaceBytesPerToken = 64 << 10
        i.expertChunkable = true
        let whole = ResidencyPlanner.estimate(i, resident: 6).total
        let c512 = ResidencyPlanner.estimate(i, resident: 6, tokenChunk: 512).total
        XCTAssertLessThan(c512, whole)
        let free = try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 0, ceiling: nil)
        XCTAssertNil(free.tokenChunk)
        XCTAssertEqual(free.residentLayers, 6)
        let tight = try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 0, ceiling: c512)
        XCTAssertEqual(tight.residentLayers, 6, "all layers stay resident")
        XCTAssertLessThanOrEqual(tight.tokenChunk ?? Int.max, 512)
        XCTAssertNotNil(tight.tokenChunk)
        XCTAssertLessThanOrEqual(tight.estimate.total, c512)
        // A routed layout cannot chunk: the budget then takes layers.
        i.expertChunkable = false
        let routed = try ResidencyPlanner.chooseAuto(i, otherStagesPeak: 0, ceiling: ResidencyPlanner.estimate(i, resident: 3).total)
        XCTAssertNil(routed.tokenChunk)
        XCTAssertEqual(routed.residentLayers, 3)
    }

    // MARK: the real code path: block pipeline, compiled steps, checkpointing, optimizer

    func pipelineRun(
        _ t: BlockPipelineTests, _ s: BlockPipelineTests.Setup, merged: Bool, chunk: Int?, shared: Bool = false,
        steps: Int = 4
    ) throws -> (losses: [Float], adapters: [String: MLXArray], built: Int) {
        ExpertCompute.merged = merged
        ExpertCompute.mergedMaxKeptRatio = 100
        defer { ExpertCompute.merged = true; ExpertCompute.mergedMaxKeptRatio = 1.5; ExpertCompute.tokenChunk = nil }
        var o = BlockPipelineOptions()
        o.batchSteps = steps
        o.mix = .clean
        o.fixedKeep = s.keep
        o.storageDType = .float32
        o.tokenChunk = chunk
        var ac = HealingFixture.smallAdapters
        ac.sharedExperts = shared
        var c = t.config()
        c.checkpointBlocks = true
        let dir = try tempDir()
        let before = ExpertCompute.mergedBranchCount
        let p = try BlockPipeline(source: try s.source, blockEnds: s.ends, adapterConfiguration: ac, configuration: c, options: o, directory: dir)
        let r = try p.run(rows: ShuffledRows(rows: s.rows, seed: 4), steps: steps)
        XCTAssertTrue(StreamedBlockHealer.compileLayerSteps)
        return (r.blocks[0].losses, r.adapters, ExpertCompute.mergedBranchCount - before)
    }

    func testPipelineTrainingIsTheSameInMergedAndRoutedLayouts() throws {
        // Four optimizer steps amplify float32 rounding differences between the layouts. On the CPU backend the
        // worst adapter difference is 1.9e-4 on one CPU (AMD EPYC 7763, run alone) and 7.4e-4 in a full run on
        // another (the same tensor each time, layers.0.mlp_up.a, keep 2), against 5e-4 on Metal. The step-1 and
        // step-2 differences are 0 and 3.5e-6, so this is growth through training, not a layout error.
        let cpu = Device.defaultDevice().deviceType == .cpu
        let atol: Float = cpu ? 2e-3 : 5e-4
        for (keep, chunk, shared) in [(2, nil, false), (4, nil, false), (4, 7, false), (4, nil, true)] as [(Int, Int?, Bool)] {
            let t = BlockPipelineTests()
            let setup = try t.setup(layers: 4, ends: [3], keepCount: keep, rowCount: 3, length: 20)
            let routed = try pipelineRun(t, setup, merged: false, chunk: chunk, shared: shared)
            XCTAssertEqual(routed.built, 0)
            let merged = try pipelineRun(t, setup, merged: true, chunk: chunk, shared: shared)
            XCTAssertGreaterThan(merged.built, 0, "the merged layout ran in the compiled step (keep \(keep) chunk \(String(describing: chunk)))")
            XCTAssertEqual(merged.losses.count, routed.losses.count)
            for (a, b) in zip(merged.losses, routed.losses) { XCTAssertEqual(a, b, accuracy: 2e-4 * max(1, abs(b))) }
            XCTAssertEqual(Set(merged.adapters.keys), Set(routed.adapters.keys))
            // After two steps the layouts agree to float32 rounding (0 to 3.5e-6 measured on a CPU, keep 2), so a
            // layout bug cannot hide behind the looser four-step bound below.
            let routed2 = try pipelineRun(t, setup, merged: false, chunk: chunk, shared: shared, steps: 2)
            let merged2 = try pipelineRun(t, setup, merged: true, chunk: chunk, shared: shared, steps: 2)
            for (k, v) in routed2.adapters {
                assertClose(merged2.adapters[k]!, v, atol: cpu ? 2e-4 : 5e-4, rtol: 5e-3, "two steps, keep \(keep) chunk \(String(describing: chunk)) \(k)")
            }
            var moved = 0
            for (k, v) in routed.adapters {
                assertClose(merged.adapters[k]!, v, atol: atol, rtol: 5e-3, "keep \(keep) chunk \(String(describing: chunk)) \(k)")
                if k.contains(".experts.") && k.hasSuffix("_b") && abs(v).max().item(Float.self) > 0 { moved += 1 }
            }
            XCTAssertGreaterThan(moved, 0, "expert adapters actually trained")
        }
    }

    func testCacheLimitIsDerivedOverriddenAndRestored() throws {
        XCTAssertEqual(BlockPipeline.automaticCacheLimit(for: 100 << 20), 64 << 20, "floor 64 MB")
        XCTAssertEqual(BlockPipeline.automaticCacheLimit(for: 4 << 30), 512 << 20, "an eighth")
        XCTAssertEqual(BlockPipeline.automaticCacheLimit(for: 64 << 30), 1 << 30, "ceiling 1 GB")
        let prior = 300 << 20
        Memory.cacheLimit = prior
        var seen: [Int] = []
        func run(_ configure: (inout BlockPipelineOptions) -> Void, memory: MemoryConfiguration = .init()) throws {
            let t = BlockPipelineTests()
            let s = try t.setup(layers: 4, ends: [3], keepCount: 2, rowCount: 2, length: 20)
            var o = BlockPipelineOptions()
            o.batchSteps = 2
            o.mix = .clean
            o.fixedKeep = s.keep
            o.storageDType = .float32
            configure(&o)
            let p = try BlockPipeline(source: try s.source, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, configuration: t.config(), options: o, directory: try tempDir())
            _ = try p.run(rows: ShuffledRows(rows: s.rows, seed: 4), steps: 2, memory: memory, onStep: { _, _ in seen.append(Memory.cacheLimit) })
        }
        try run({ $0.cacheLimitBytes = 123 << 20 })
        XCTAssertEqual(Set(seen), [123 << 20], "the option sets it during the run")
        XCTAssertEqual(Memory.cacheLimit, prior, "and it is put back")
        seen = []
        try run({ _ in })
        XCTAssertEqual(Set(seen), [64 << 20], "derived from the estimate (tiny toy: the floor)")
        XCTAssertEqual(Memory.cacheLimit, prior)
        seen = []
        try run({ $0.cacheLimitBytes = 123 << 20 }, memory: .init(cacheLimitBytes: 77 << 20))
        XCTAssertEqual(Set(seen), [77 << 20], "a caller's MemoryConfiguration wins")
    }
}

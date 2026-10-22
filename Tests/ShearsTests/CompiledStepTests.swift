import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The compiled layer steps of the streamed block trainer (``CompiledLayerStep``) must give the
/// uncompiled step's result: the same losses, the same adapters after several steps, for rows of
/// different lengths, whatever the window, and compiled graphs must be shared by the rows of one bucket.
final class CompiledStepTests: ShearsTestCase {
    static let layers = 6

    override func setUp() {
        super.setUp()
        let compile = StreamedBlockHealer.compileLayerSteps
        let buckets = StreamedBlockHealer.sequenceBuckets
        addTeardownBlock {
            StreamedBlockHealer.compileLayerSteps = compile
            StreamedBlockHealer.sequenceBuckets = buckets
        }
    }

    func tempDir(_ name: String = "cst") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    struct Setup {
        var keep: [[Int]]
        var rows: [HealingRow]
        var initial: [String: MLXArray]
        var checkpoint: URL
    }

    /// A six-layer model written as a checkpoint, one block holding every layer; rows of the given lengths.
    func setup(lengths: [Int], keepCount: Int = 5) throws -> Setup {
        let m = try HealingFixture.model(layers: Self.layers)
        let keep = HealingFixture.keepSets(layers: Self.layers, count: keepCount)
        let seed = try HealingFixture.healer(model: m, keep: keep)
        HealingFixture.randomizeAdapters(seed.adapters, scale: 0.02)
        let initial = HealingFixture.copyTensors(seed.adapters.tensors())
        let rows = lengths.enumerated().map { HealingFixture.row(length: $1, seed: UInt64(3 + $0), key: "r\($0)") }
        let dir = try tempDir("ckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: Self.layers),
            adapters: nil, to: dir, options: o)
        return Setup(keep: keep, rows: rows, initial: initial, checkpoint: dir)
    }

    func healer(_ s: Setup, resident: Int?, checkpointed: Bool = true) throws -> StreamedBlockHealer {
        var c = HealingFixture.tinyConfiguration()
        c.residentLayers = resident
        c.checkpointBlocks = checkpointed
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32, mapWeights: true)
        return StreamedBlockHealer(
            source: src, keep: s.keep, blockEnds: [Self.layers - 1], adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, initialAdapters: s.initial)
    }

    /// The trainer's own entry point (windows of `resident` layers, or the whole block when `nil`).
    func train(
        _ s: Setup, resident: Int?, compile: Bool, steps: Int = 6, checkpointed: Bool = true
    ) throws -> StreamedBlockHealer.Report {
        StreamedBlockHealer.compileLayerSteps = compile
        let h = try healer(s, resident: resident, checkpointed: checkpointed)
        let store = InMemoryBoundaryStore()
        try h.precomputeBoundaries(rows: s.rows, store: store, options: .init(storageDType: .float32))
        return try h.train(rows: ShuffledRows(rows: s.rows, seed: 9), steps: steps, store: store)
    }

    struct Direct {
        var losses: [Float]
        var adapters: [String: MLXArray]
        var carry: WindowCarry
        /// The optimizer's moments (`m.` and `v.` per adapter tensor): after one step `m` is a tenth of the gradient.
        var moments: [String: MLXArray]
        /// MLX's peak memory over the steps alone (reset after the boundaries are computed and the layers loaded).
        var peak = 0
    }

    /// `steps` calls of ``StreamedBlockHealer/windowedStep`` driven by hand over one window size (a window
    /// as long as the block is the whole block, trained through the same step), with the trainer's schedule,
    /// optimizer and adapters.
    func direct(
        _ s: Setup, window: Int, compile: Bool, steps: Int = 6, checkpointed: Bool = true, keptLayers: Int = 0
    ) throws -> Direct {
        StreamedBlockHealer.compileLayerSteps = compile
        var h = try healer(s, resident: window, checkpointed: checkpointed)
        h.slicedStudent = true
        let store = InMemoryBoundaryStore()
        try h.precomputeBoundaries(rows: s.rows, store: store, options: .init(storageDType: .float32))
        let c = h.configuration
        let n = Self.layers
        let kept = s.keep
        let sets = kept.map { ExpertSet(keep: $0, numExperts: HealingFixture.experts) }
        var dims: [LayerDimensions] = []
        for i in 0 ..< n { dims.append(LayerDimensions(try h.source.loadLayers(i ... i, keeping: [kept[i]])[0])) }
        let adapters = ModelAdapters(dimensions: dims, firstLayer: 0, sets: sets, configuration: HealingFixture.smallAdapters)
        try adapters.load(subsetOf: s.initial)
        let opt = HealingAdamW(weightDecay: c.weightDecay)
        let schedule = c.learningRateSchedule(runSteps: steps)
        let windows = StreamedBlockHealer.windows(count: n, resident: window)
        let carry = WindowCarry()
        let rows = ShuffledRows(rows: s.rows, seed: 9)
        var losses: [Float] = []
        Memory.clearCache()
        Memory.peakMemory = Memory.activeMemory
        for step in 0 ..< steps {
            let row = rows.row(forStep: step)
            let (loss, ok, _) = try h.windowedStep(
                block: (0, n - 1), windows: windows, adapters: adapters, optimizer: opt, sets: sets, kept: kept,
                input: try store.get(row: row.key, boundary: 0), target: try store.get(row: row.key, boundary: 1),
                weights: row.weights, learningRate: { schedule.value(at: $0) }, blockCount: 1,
                checkpointed: c.checkpointBlocks, keptLayers: keptLayers, carry: carry)
            XCTAssertTrue(ok)
            losses.append(loss)
        }
        return Direct(losses: losses, adapters: adapters.tensors(), carry: carry, moments: opt.stateTensors(), peak: Memory.peakMemory)
    }

    func assertSame(
        _ a: [String: MLXArray], _ b: [String: MLXArray], losses la: [Float], _ lb: [Float], _ what: String,
        atol: Float = 1e-3
    ) {
        XCTAssertEqual(Set(a.keys), Set(b.keys), what)
        for (k, v) in a { assertClose(b[k]!, v, atol: atol, rtol: 1e-3, "\(what) \(k)") }
        XCTAssertEqual(la.count, lb.count, what)
        for (x, y) in zip(la, lb) { XCTAssertEqual(x, y, accuracy: 1e-4 * max(1, abs(x)), "\(what) loss") }
    }

    func movement(_ t: [String: MLXArray], from initial: [String: MLXArray]) -> Float {
        var moved: Float = 0
        for (k, v) in t { moved = max(moved, abs(v - initial[k]!).max().item(Float.self)) }
        return moved
    }

    // MARK: the masked loss

    /// A row padded to a longer length scores exactly what the row does: value, and the gradient at its
    /// own positions (zero in the padding).
    func testMaskedLossEqualsTheUnpaddedLossAndGradient() {
        let (t, pad, d) = (20, 44, 16)
        MLXRandom.seed(4)
        let h = MLXRandom.normal([1, t, d])
        let r = 1.7 * MLXRandom.normal([1, t, d]) + 0.3
        let w = HealingLoss.rampedWeights(count: t, boundary: t - 6, floor: 0.15, rampLength: 8)
        eval(h, r, w)
        let plain = valueAndGrad({ (a: [MLXArray]) in [HealingLoss.pinBatched([0: a[0]], teacher: [0: r], weights: w)] })
        let (lp, gp) = plain([h])
        // The student's state in the padding is whatever the layers made of the zero input: not zero.
        let hj = MLX.where(MLXArray.arange(pad).reshaped(1, pad, 1) .< t, zeroPadded(h, axis: 1, to: pad), MLXArray(Float(0.5)))
        let rp = zeroPadded(r, axis: 1, to: pad), wp = zeroPadded(w, axis: 0, to: pad)
        let masked = valueAndGrad({ (a: [MLXArray]) in
            [HealingLoss.pinBatchedMasked([0: a[0]], teacher: [0: rp], weights: wp, length: MLXArray(Float(t)))]
        })
        let (lm, gm) = masked([hj])
        eval(lp, gp, lm, gm)
        assertClose(lm[0], lp[0], atol: 1e-6, rtol: 1e-5, "loss")
        assertClose(gm[0][0..., 0 ..< t, 0...], gp[0], atol: 1e-6, rtol: 1e-4, "gradient")
        XCTAssertEqual(abs(gm[0][0..., t..., 0...]).max().item(Float.self), 0, "no gradient in the padding")
        // The length is a real input: a wrong one changes the loss (the scale is read from it).
        let wrong = HealingLoss.pinBatchedMasked([0: hj], teacher: [0: rp], weights: wp, length: MLXArray(Float(t + 9)))
        XCTAssertGreaterThan(abs(wrong.item(Float.self) - lp[0].item(Float.self)), 1e-3 * abs(lp[0].item(Float.self)))
    }

    func testBucketsRoundUpAndPastTheLastBucketToItsMultiples() {
        StreamedBlockHealer.sequenceBuckets = [16, 32, 64]
        XCTAssertEqual([1, 16, 17, 32, 33, 64].map(StreamedBlockHealer.sequenceBucket(for:)), [16, 16, 32, 32, 64, 64])
        XCTAssertEqual([65, 128, 129].map(StreamedBlockHealer.sequenceBucket(for:)), [128, 128, 192])
    }


    // MARK: kept intermediates (selective checkpointing)

    /// The two runs are the same to the bit: every loss, adapter tensor and optimizer moment.
    func assertBitIdentical(_ a: Direct, _ b: Direct, _ what: String) {
        XCTAssertEqual(a.losses, b.losses, "\(what): losses")
        for (name, x, y) in [("adapters", a.adapters, b.adapters), ("moments", a.moments, b.moments)] {
            XCTAssertEqual(Set(x.keys), Set(y.keys), "\(what) \(name)")
            for (k, v) in x {
                guard let w = y[k] else { continue }
                XCTAssertTrue(arrayEqual(v, w).item(Bool.self), "\(what) \(name) \(k) differs")
            }
        }
    }

    /// How many layers keep their intermediates changes nothing in the arithmetic: the same step to the bit, the
    /// whole block as one window, compiled. (Run twice first: the comparison means something only if one setting
    /// repeats itself exactly.)
    func testKeptLayersGiveBitIdenticalSteps() throws {
        StreamedBlockHealer.sequenceBuckets = [32]
        let s = try setup(lengths: [20, 28, 12])
        let base = try direct(s, window: Self.layers, compile: true, keptLayers: 0)
        XCTAssertGreaterThan(movement(base.adapters, from: s.initial), 1e-3, "the adapters must really move")
        assertBitIdentical(base, try direct(s, window: Self.layers, compile: true, keptLayers: 0), "setting repeats itself")
        for k in [1, 3, Self.layers] {
            assertBitIdentical(base, try direct(s, window: Self.layers, compile: true, keptLayers: k), "kept \(k) of \(Self.layers)")
        }
    }

    /// The same, uncompiled (to a tight tolerance: see below); and the forwards built show the layers really are kept or recomputed: one forward per
    /// layer, plus a second for each layer that is not kept.
    func testKeptLayersGiveBitIdenticalUncompiledStepsAndRecomputeTheRest() throws {
        let s = try setup(lengths: [20, 28])
        // The forwards built by the steps alone: the teacher's pre-pass runs inside `direct` too, so count from there.
        var base: Direct?
        var counts: [Int: Int] = [:]
        for k in [0, 0, 2, Self.layers] {
            let before = healingLayerForwardCount
            let d = try direct(s, window: Self.layers, compile: false, steps: 2, keptLayers: k)
            counts[k] = healingLayerForwardCount - before
            // The eager (uncompiled) step does not repeat itself to the bit even with one setting (the first two runs below
            // are both 0 kept), so this is held to a tight tolerance; the compiled step, which ships, is held to equality above.
            if let base {
                assertSame(base.adapters, d.adapters, losses: base.losses, d.losses, "uncompiled, kept \(k)", atol: 1e-5)
            } else { base = d }
        }
        // Each kept layer saves one recomputed forward per step.
        XCTAssertEqual(counts[0]! - counts[2]!, 2 * 2, "forwards saved by keeping 2 layers over 2 steps")
        XCTAssertEqual(counts[0]! - counts[Self.layers]!, 2 * Self.layers)
    }

    /// A windowed step keeps only within its top window; lower windows recompute whatever is asked, and nothing changes.
    func testKeptLayersGiveBitIdenticalWindowedSteps() throws {
        StreamedBlockHealer.sequenceBuckets = [32]
        let s = try setup(lengths: [20, 28])
        let base = try direct(s, window: 2, compile: true, keptLayers: 0)
        for k in [1, 2, 5] {
            assertBitIdentical(base, try direct(s, window: 2, compile: true, keptLayers: k), "window 2, kept \(k)")
        }
    }

    /// One compiled-step cache serves every setting: a different kept count is a different graph, not the cached one.
    func testKeptLayerCountIsPartOfTheCompiledGraphKey() throws {
        StreamedBlockHealer.sequenceBuckets = [32]
        let s = try setup(lengths: [20])
        let compiled = CompiledLayerStep()
        // Drive the compiled top directly over two layers with 0, then 1, then 0 layers kept.
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32, mapWeights: true)
        let layers = try src.loadLayers(0 ... 1, keeping: s.keep[0 ... 1].map { $0 })
        let sets = s.keep[0 ... 1].map { ExpertSet(keep: $0, numExperts: HealingFixture.experts) }
        let dims = layers.map { LayerDimensions($0) }
        let ad = ModelAdapters(dimensions: dims, firstLayer: 0, sets: sets, configuration: HealingFixture.smallAdapters)
        let row = s.rows[0]
        let x = MLXRandom.normal([1, 32, layers[0].hiddenSize])
        let target = MLXRandom.normal([1, 32, layers[0].hiddenSize])
        let w = zeroPadded(row.weights, axis: 0, to: 32)
        func run(recomputed: Int) -> CompiledLayerStep.TopResult {
            compiled.top(
                first: 0, layers: layers, sets: sets, adapters: Array(ad.layers), sliced: true, recomputed: recomputed, signature: "k",
                x: x, wantInputGrad: false, target: target, weights: w, length: MLXArray(Float(20)), gradScale: MLXArray(Float(1)))
        }
        _ = run(recomputed: 2); let t1 = compiled.traces
        _ = run(recomputed: 2); XCTAssertEqual(compiled.traces, t1, "the same setting reuses its graph")
        _ = run(recomputed: 1); XCTAssertEqual(compiled.traces, t1 + 1, "another kept count traces its own graph")
        _ = run(recomputed: 1); XCTAssertEqual(compiled.traces, t1 + 1)
    }

    /// Kept layers hold their intermediates until the backward: the MLX peak of a step rises with the count.
    func testKeptLayersRaiseThePeak() throws {
        StreamedBlockHealer.sequenceBuckets = [256]
        let s = try setup(lengths: [250])
        var peaks: [Int] = []
        for k in [0, Self.layers] { peaks.append(try direct(s, window: Self.layers, compile: true, steps: 3, keptLayers: k).peak) }
        XCTAssertGreaterThan(peaks[1], peaks[0] + peaks[0] / 20, "keeping every layer's intermediates must cost memory: \(peaks)")
    }

    // MARK: windowed steps

    /// Windows of one and two layers, rows of three lengths landing in two buckets (12 -> 16, 20 and
    /// 28 -> 32): the trainer's result is the same compiled or not.
    func testCompiledWindowsEqualUncompiledAcrossRowLengthsAndBuckets() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 20, 28])
        for w in [1, 2] {
            let off = try train(s, resident: w, compile: false)
            XCTAssertGreaterThan(movement(off.adapters, from: s.initial), 1e-3, "the adapters must really move")
            let on = try train(s, resident: w, compile: true)
            assertSame(off.adapters, on.adapters, losses: Array(off.blockLosses.joined()), Array(on.blockLosses.joined()), "window \(w)")
        }
    }

    /// Rows of different lengths inside one bucket.
    func testRowsOfOneBucketAgree() throws {
        StreamedBlockHealer.sequenceBuckets = [64]
        let s = try setup(lengths: [20, 24, 37])
        let off = try train(s, resident: 2, compile: false)
        let on = try train(s, resident: 2, compile: true)
        assertSame(off.adapters, on.adapters, losses: Array(off.blockLosses.joined()), Array(on.blockLosses.joined()), "one bucket")
    }

    /// Without checkpointing the top window keeps its activations; the compiled step must match that too.
    func testCompiledEqualsUncompiledWithoutCheckpointing() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 28])
        let off = try train(s, resident: 2, compile: false, checkpointed: false)
        let on = try train(s, resident: 2, compile: true, checkpointed: false)
        assertSame(off.adapters, on.adapters, losses: Array(off.blockLosses.joined()), Array(on.blockLosses.joined()), "no checkpointing")
    }

    /// Attention that scores its queries in chunks (each chunk under its own checkpoint, inside the layer's
    /// checkpoint) compiles too, and keeps the padded positions out of the valid ones.
    func testCompiledEqualsUncompiledWithChunkedAttention() throws {
        let chunk = Gemma4Attention.queryChunk
        addTeardownBlock { Gemma4Attention.queryChunk = chunk }
        Gemma4Attention.queryChunk = 8
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 28])
        let off = try train(s, resident: 2, compile: false)
        let on = try train(s, resident: 2, compile: true)
        assertSame(off.adapters, on.adapters, losses: Array(off.blockLosses.joined()), Array(on.blockLosses.joined()), "chunked attention")
    }

    /// The whole block as one window through ``windowedStep``, compiled, equals the trainer's own whole-block path.
    func testCompiledWholeBlockEqualsTheWholeBlockPath() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 20, 28])
        let whole = try train(s, resident: nil, compile: false)
        let off = try direct(s, window: Self.layers, compile: false)
        let on = try direct(s, window: Self.layers, compile: true)
        assertSame(whole.adapters, off.adapters, losses: Array(whole.blockLosses.joined()), off.losses, "uncompiled one window vs whole block", atol: 1e-4)
        assertSame(off.adapters, on.adapters, losses: off.losses, on.losses, "compiled whole block")
        XCTAssertGreaterThan(movement(on.adapters, from: s.initial), 1e-3)
    }

    /// The trainer's own whole-block path (`residentLayers` unset or the block's length) with compilation on equals
    /// it with compilation off. (The whole-block path routes through the compiled single-window step; where it
    /// does not, both sides are the uncompiled path and this holds trivially.)
    func testTrainerWholeBlockPathEqualsUncompiled() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 20, 28])
        let off = try train(s, resident: nil, compile: false)
        let on = try train(s, resident: nil, compile: true)
        assertSame(off.adapters, on.adapters, losses: Array(off.blockLosses.joined()), Array(on.blockLosses.joined()), "whole block")
        XCTAssertGreaterThan(movement(on.adapters, from: s.initial), 1e-3)
    }

    /// Compiled windows of 1 and 2 layers, driven by hand, equal the uncompiled ones step for step.
    func testDirectStepsEqualAcrossWindows() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 20, 28])
        for w in [1, 2, 4] {
            let off = try direct(s, window: w, compile: false)
            let on = try direct(s, window: w, compile: true)
            assertSame(off.adapters, on.adapters, losses: off.losses, on.losses, "window \(w)")
        }
    }

    /// The gradients themselves, not only where Adam's normalisation takes them: after one step the optimizer's
    /// first moment is a tenth of the (scaled) gradient, so comparing it compares the gradients of every
    /// adapter tensor, relative to that tensor's largest entry. Rows of both buckets, three window sizes.
    func testCompiledGradientsEqualUncompiled() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 20, 28])
        for w in [1, 2, Self.layers] {
            for steps in [1, 2] {
                let off = try direct(s, window: w, compile: false, steps: steps)
                let on = try direct(s, window: w, compile: true, steps: steps)
                let tolerance: Float = steps == 1 ? 2e-4 : 1e-3
                for k in off.moments.keys.sorted() where k.hasPrefix("m.") {
                    let a = off.moments[k]!, b = on.moments[k]!
                    let scale = abs(a).max().item(Float.self)
                    XCTAssertGreaterThan(scale, 0, "\(k) has a gradient")
                    assertClose(b, a, atol: tolerance * scale, rtol: 0, "window \(w) steps \(steps) \(k)")
                }
                XCTAssertEqual(off.losses.count, steps)
                for (x, y) in zip(off.losses, on.losses) { XCTAssertEqual(x, y, accuracy: 1e-5 * max(1, abs(x))) }
            }
        }
    }

    // MARK: reuse

    /// Rows of three lengths in two buckets compile each function twice, not three times, and later
    /// steps compile nothing: a compiled graph is traced once per bucket.
    func testGraphsAreTracedOncePerBucket() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 20, 28])
        let d = try direct(s, window: 2, compile: true, steps: 9)
        let c = d.carry.compiled!
        // Windows of two over six layers: the top window [4, 5], and layers 0 to 3 below it, each with a
        // forward (4) and a backward (4) function.
        let functions = 1 + 4 + 4
        XCTAssertEqual(c.traces, functions * 2, "one trace per function and bucket")
        XCTAssertEqual(c.calls, 9 * functions, "every layer of every step ran compiled")
    }

    /// A function compiled for one set of weights runs the weights it is given: two layers of the same
    /// shape through the same compiled function give each its own output.
    func testCompiledForwardReadsTheWeightsItIsGiven() throws {
        // Layers 0 and 2 of five are both sliding-attention layers: the same shapes, different weights.
        let m = try HealingFixture.model(layers: 5)
        let layers = [m.healableLayers[0], m.healableLayers[2]]
        let set = ExpertSet.all(HealingFixture.experts)
        let la = LayerAdapters(
            layer: layers[0], keptExperts: HealingFixture.experts, numExperts: HealingFixture.experts,
            configuration: HealingFixture.smallAdapters)
        eval(la)
        MLXRandom.seed(1)
        let x = MLXRandom.normal([1, 16, 64])
        eval(x)
        let step = CompiledLayerStep()
        for _ in 0 ..< 2 {
            for (i, layer) in layers.enumerated() {
                let compiled = step.forward(
                    index: 0, layer: layer, set: set, adapters: la, sliced: false, signature: "t", x: x)
                let plain = healingLayerForward(layer, x, set: set, adapters: la, sliced: false)
                assertClose(compiled, plain, atol: 1e-5, rtol: 1e-4, "layer \(i)")
            }
        }
        XCTAssertEqual(step.traces, 1, "two layers, one traced graph")
        // And the adapters: new values through the same graph.
        la.update(parameters: ModuleParameters.unflattened(la.trainableParameters().flattened().map { ($0.0, $0.1 + 0.3) }))
        eval(la)
        let compiled = step.forward(index: 0, layer: layers[0], set: set, adapters: la, sliced: false, signature: "t", x: x)
        assertClose(compiled, healingLayerForward(layers[0], x, set: set, adapters: la, sliced: false), atol: 1e-5, rtol: 1e-4, "new adapters")
        XCTAssertEqual(step.traces, 1)
    }

    /// A compiled call leaves the layer and the adapters holding their own arrays (not the traced inputs):
    /// after a compiled forward, top-window step and backward every parameter reads back, bit for bit, as it
    /// was (a module left holding a dead trace's arrays would crash or read garbage here).
    func testCompiledCallsLeaveTheModulesHoldingTheirOwnArrays() throws {
        let m = try HealingFixture.model(layers: 2)
        let layers = m.healableLayers
        let sets = layers.map { _ in ExpertSet.all(HealingFixture.experts) }
        let adapters = layers.map {
            LayerAdapters(
                layer: $0, keptExperts: HealingFixture.experts, numExperts: HealingFixture.experts,
                configuration: HealingFixture.smallAdapters)
        }
        MLXRandom.seed(11)
        for a in adapters {
            a.update(parameters: ModuleParameters.unflattened(
                a.parameters().flattened().map { ($0.0, $0.1 + 0.05 * MLXRandom.normal($0.1.shape)) }))
            eval(a)
        }
        func snapshot() -> [String: MLXArray] {
            var out: [String: MLXArray] = [:]
            for (i, l) in layers.enumerated() {
                for (k, v) in (l as! Module).parameters().flattened() { out["layer\(i).\(k)"] = v * 1 }
                for (k, v) in adapters[i].parameters().flattened() { out["adapter\(i).\(k)"] = v * 1 }
            }
            eval(Array(out.values))
            return out
        }
        let before = snapshot()
        MLXRandom.seed(2)
        let x = MLXRandom.normal([1, 16, 64]), target = MLXRandom.normal([1, 16, 64])
        let step = CompiledLayerStep()
        for _ in 0 ..< 2 {
            let h = step.forward(
                index: 0, layer: layers[0], set: sets[0], adapters: adapters[0], sliced: false, signature: "t", x: x)
            let top = step.top(
                first: 0, layers: layers, sets: sets, adapters: adapters, sliced: false, recomputed: layers.count, signature: "t",
                x: x, wantInputGrad: true, target: target, weights: MLXArray.ones([16]), length: MLXArray(Float(16)),
                gradScale: MLXArray(Float(1)))
            let back = step.back(
                index: 1, layer: layers[1], set: sets[1], adapters: adapters[1], sliced: false, signature: "t", x: h,
                cotangent: top.cotangent!, wantInputGrad: true, gradScale: MLXArray(Float(1)))
            eval(h, top.loss, back.cotangent!)
            eval(Array(top.grads.values), Array(back.grads.values))
        }
        let after = snapshot()
        XCTAssertEqual(Set(after.keys), Set(before.keys))
        for (k, v) in before { assertClose(after[k]!, v, atol: 0, rtol: 0, k) }
        // They are also still usable by the uncompiled path.
        let plain = healingLayerForward(layers[0], x, set: sets[0], adapters: adapters[0], sliced: false)
        assertClose(plain, step.forward(
            index: 0, layer: layers[0], set: sets[0], adapters: adapters[0], sliced: false, signature: "t", x: x),
            atol: 1e-5, rtol: 1e-4)
    }

    /// Peak MLX memory of one top-window step over eight layers, uncompiled and compiled, with and without
    /// per-layer checkpointing: whether the recompute wrapper still saves memory inside a compiled graph.
    /// Off by default; prints the table.
    ///   TEST_RUNNER_SHEARS_BENCH=1 xcodebuild test ... -only-testing:ShearsTests/CompiledStepTests/testCheckpointMemoryUnderCompile
    func testCheckpointMemoryUnderCompile() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        let count = 8
        let m = try MemoryBenchmarkTests.model(layers: count)
        let layers = m.healableLayers
        let sets = layers.map { _ in ExpertSet(keep: Array(0 ..< MemoryBenchmarkTests.keep), numExperts: MemoryBenchmarkTests.experts) }
        let adapters = layers.map {
            LayerAdapters(
                layer: $0, keptExperts: MemoryBenchmarkTests.keep, numExperts: MemoryBenchmarkTests.experts,
                configuration: MemoryBenchmarkTests.adapters)
        }
        for a in adapters { eval(a) }
        let t = Int(ProcessInfo.processInfo.environment["SHEARS_BENCH_SEQ"] ?? "") ?? 1024
        MLXRandom.seed(3)
        let x = MLXRandom.normal([1, t, MemoryBenchmarkTests.hidden]), target = MLXRandom.normal([1, t, MemoryBenchmarkTests.hidden])
        let w = MLXArray.ones([t])
        eval(x, target, w)
        let step = CompiledLayerStep()
        func uncompiled(checkpointed: Bool) -> Float {
            let model = GroupVJPModel(layers: adapters, input: x)
            let vg = valueAndGrad(model: model) { (mm: GroupVJPModel, _: [MLXArray]) in
                var h = mm.x!
                for (j, layer) in layers.enumerated() {
                    let set = sets[j]
                    if checkpointed {
                        h = checkpoint(mm.layers[j]) { (a: LayerAdapters, x: MLXArray) in
                            healingLayerForward(layer, x, set: set, adapters: a, sliced: false)
                        }(h)
                    } else {
                        h = healingLayerForward(layer, h, set: set, adapters: mm.layers[j], sliced: false)
                    }
                }
                return [HealingLoss.pinBatched([0: h], teacher: [0: target], weights: w)]
            }
            let (l, g) = vg(model, [])
            eval(l, Array(GradMath.flatten(g).values))
            return l[0].item(Float.self)
        }
        func compiled(checkpointed: Bool) -> Float {
            let r = step.top(
                first: 0, layers: layers, sets: sets, adapters: adapters, sliced: false, recomputed: checkpointed ? layers.count : 0,
                signature: "bench", x: x, wantInputGrad: true, target: target, weights: w, length: MLXArray(Float(t)),
                gradScale: MLXArray(Float(1)))
            eval(r.loss, r.cotangent!, Array(r.grads.values))
            return r.loss.item(Float.self)
        }
        var lines = ["peak MLX memory above the resident layers and adapters, \(count) layers, sequence \(t):"]
        for (name, run) in [("uncompiled", uncompiled), ("compiled", compiled)] {
            for ckpt in [false, true] {
                _ = run(ckpt)  // the first call traces
                Memory.clearCache()
                let base = Memory.activeMemory
                Memory.peakMemory = base
                let loss = run(ckpt)
                lines.append(String(format: "  %@ checkpoint=%@: %.1f MB (loss %.5f)", name, ckpt ? "on " : "off", Double(Memory.peakMemory - base) / 1_048_576, loss))
            }
        }
        print("\n" + lines.joined(separator: "\n") + "\n")
    }

    /// Steps that cannot be compiled run the old way: a skipped (over-`maxLoss`) step applies no update.
    func testASkippedStepAppliesNoUpdateWhenCompiled() throws {
        StreamedBlockHealer.sequenceBuckets = [16, 32]
        let s = try setup(lengths: [12, 28])
        StreamedBlockHealer.compileLayerSteps = true
        var c = HealingFixture.tinyConfiguration()
        c.residentLayers = 2
        c.maxLoss = 0
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: s.checkpoint), dtype: .float32, mapWeights: true)
        let h = StreamedBlockHealer(
            source: src, keep: s.keep, blockEnds: [Self.layers - 1], adapterConfiguration: HealingFixture.smallAdapters,
            configuration: c, initialAdapters: s.initial)
        let store = InMemoryBoundaryStore()
        try h.precomputeBoundaries(rows: s.rows, store: store, options: .init(storageDType: .float32))
        var applied: [Bool] = []
        let r = try h.train(rows: ShuffledRows(rows: s.rows, seed: 9), steps: 3, store: store, onStep: { _, rec in applied.append(rec.applied) })
        XCTAssertEqual(applied, [false, false, false])
        for (k, v) in r.adapters { assertClose(v, s.initial[k]!, atol: 0, rtol: 0, k) }
    }
}

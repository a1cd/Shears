import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The router correction as its own knob: on or off by its rank, its own learning-rate multiplier, and the
/// routing-agreement measure that says how far it moved the routing.
final class RouterAdapterTests: ShearsTestCase {
    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("router-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    struct Setup {
        var checkpoint: URL
        var keep: [[Int]]
        var rows: [HealingRow]
        var ends: [Int]
        var source: Gemma4LayerSource { get throws { Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: checkpoint), dtype: .float32) } }
    }

    func setup() throws -> Setup {
        let m = try HealingFixture.model()
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: 4), adapters: nil, to: dir, options: o)
        return Setup(
            checkpoint: dir, keep: HealingFixture.keepSets(count: 5),
            rows: (0 ..< 3).map { HealingFixture.row(length: 20, seed: UInt64(3 + $0), key: "row\($0)") }, ends: m.defaultBlockEnds)
    }

    func held() -> [HealingRow] { (0 ..< 2).map { HealingFixture.row(length: 20, seed: UInt64(50 + $0), key: "held\($0)") } }

    func run(
        _ s: Setup, steps: Int = 6, lr: Float = 5e-3, routerRank: Int = 4, routerLR: Float = 1, held: [HealingRow] = []
    ) throws -> BlockPipeline.Report {
        var c = HealingFixture.tinyConfiguration()
        c.learningRate = lr
        var o = BlockPipelineOptions()
        o.batchSteps = steps
        o.mix = .clean
        o.fixedKeep = s.keep
        o.storageDType = .float32
        o.maxBlocks = 1
        o.evalRows = held
        var a = HealingFixture.smallAdapters
        a.routerRank = routerRank
        a.routerLRMultiplier = routerLR
        let p = try BlockPipeline(
            source: try s.source, blockEnds: s.ends, adapterConfiguration: a, configuration: c, options: o, directory: try tempDir())
        return try p.run(rows: SequentialRowSource(rows: s.rows), steps: steps)
    }

    func maxDiff(_ a: [String: MLXArray], _ b: [String: MLXArray], only: String? = nil, except: String? = nil) -> Float {
        var m: Float = 0
        for (k, v) in a where (only == nil || k.contains(only!)) && (except == nil || !k.contains(except!)) {
            m = max(m, abs(v - b[k]!).max().item(Float.self))
        }
        return m
    }

    // MARK: on / off, rank, learning-rate multiplier

    func testTheRouterCorrectionIsOffOnItsOwn() throws {
        let s = try setup()
        let off = try run(s, routerRank: 0)
        XCTAssertFalse(off.adapters.keys.contains { $0.contains("router") })
        XCTAssertTrue(off.adapters.keys.contains { $0.contains("experts") } && off.adapters.keys.contains { $0.contains("mlp_gate") })
        let on = try run(s, routerRank: 4)
        XCTAssertTrue(on.adapters.keys.contains { $0.contains("rt_b") })
        // dense and expert initialisation do not depend on the router (it is created last)
        XCTAssertEqual(on.adapters.filter { !$0.key.contains("router") }.count, off.adapters.count)
        // the router's rank is its own
        let wide = try run(s, routerRank: 8)
        XCTAssertEqual(wide.adapters["layers.0.router.rt_a"]!.shape.last, 8)
        XCTAssertEqual(on.adapters["layers.0.router.rt_a"]!.shape.last, 4)
    }

    func testTheRouterLearningRateMultiplier() throws {
        let s = try setup()
        let base = try run(s)
        XCTAssertEqual(maxDiff(try run(s, routerLR: 1).adapters, base.adapters), 0, "1 is the default, exactly")
        let frozen = try run(s, routerLR: 0)
        for (k, v) in frozen.adapters where k.contains("rt_b") { XCTAssertEqual(abs(v).max().item(Float.self), 0, k) }
        XCTAssertGreaterThan(maxDiff(frozen.adapters, base.adapters, only: "rt_b"), 1e-4, "the router moves in the base run")
        let fast = try run(s, routerLR: 3)
        XCTAssertGreaterThan(maxDiff(fast.adapters, base.adapters, only: "rt_b"), 1e-4)
        // The multiplier touches the router's parameters and nothing else. Update 0 runs at a learning rate of 0 (the
        // warmup), so two steps are the shortest run in which anything moves, and the router is still zero when update
        // 1's gradients are taken: the other adapters match exactly between multipliers, and did move from their start.
        let one = try run(s, steps: 2, routerLR: 1), three = try run(s, steps: 2, routerLR: 3), start = try run(s, steps: 1, lr: 0)
        XCTAssertEqual(maxDiff(one.adapters, three.adapters, except: "router"), 0)
        XCTAssertGreaterThan(maxDiff(one.adapters, start.adapters, except: "router"), 1e-6, "the other adapters moved")
    }

    // MARK: the routing measure

    func layerAndAdapters() throws -> (any HealableMoELayer, ExpertSet, LayerAdapters, MLXArray) {
        let model = try HealingFixture.model()
        let layer = model.healableLayers[0]
        let set = ExpertSet(keep: [1, 3, 4, 6, 7], numExperts: HealingFixture.experts)
        MLXRandom.seed(21)
        let ad = LayerAdapters(layer: layer, keptExperts: set.keep.count, numExperts: HealingFixture.experts, configuration: HealingFixture.smallAdapters)
        ad.router!.update(parameters: ModuleParameters.unflattened([("rt_b", 0.7 * MLXRandom.normal(ad.router!.b.shape))]))
        eval(ad)
        let x = model.embed(HealingFixture.row(length: 12, seed: 4).tokenArray)
        eval(x)
        return (layer, set, ad, x)
    }

    func testTheMeasureEqualsAHandComputation() throws {
        let (layer, set, ad, x) = try layerAndAdapters()
        let parts = layer.attentionPart(x, adapters: ad)
        let (_, aux) = healingLayerForwardAux(layer, x, set: set, adapters: ad, sliced: false)
        eval(aux)
        let T = x.dim(1), E = HealingFixture.experts, K = set.keep.count, r = HealingFixture.smallAdapters.routerRank
        let D = parts.routerInput.dim(-1)
        let scores = parts.routerLogits.asType(.float32).asArray(Float.self)
        let rin = parts.routerInput.asType(.float32).asArray(Float.self)
        let a = ad.router!.a.asArray(Float.self), b = ad.router!.b.asArray(Float.self)
        let got = aux.asArray(Float.self)
        XCTAssertEqual(aux.shape, [1, T, 3])
        var anyNonzero = false
        for t in 0 ..< T {
            var low = [Double](repeating: 0, count: r)
            for j in 0 ..< r { for d in 0 ..< D { low[j] += Double(rin[t * D + d]) * Double(a[d * r + j]) } }
            var s0 = [Double](), s1 = [Double](), corr = [Double]()
            for e in set.keep {
                var c = 0.0
                for j in 0 ..< r { c += low[j] * Double(b[j * E + e]) }
                corr.append(c)
                s0.append(Double(scores[t * E + e]))
                s1.append(Double(scores[t * E + e]) + c)
            }
            func softmax(_ v: [Double]) -> [Double] {
                let m = v.max()!
                let ex = v.map { exp($0 - m) }
                let z = ex.reduce(0, +)
                return ex.map { $0 / z }
            }
            let (p, q) = (softmax(s0), softmax(s1))
            var kl = 0.0
            for i in 0 ..< K { kl += p[i] * (log(p[i]) - log(q[i])) }  // KL(original || corrected), original first
            let l2 = corr.map { $0 * $0 }.reduce(0, +) / Double(K)
            func top(_ v: [Double]) -> Set<Int> { Set(v.indices.sorted { v[$0] > v[$1] }.prefix(HealingFixture.topK)) }
            let overlap = Double(top(s0).intersection(top(s1)).count) / Double(HealingFixture.topK)
            XCTAssertEqual(Double(got[t * 3 + 0]), kl, accuracy: 1e-5 + 1e-4 * kl, "kl at \(t)")
            XCTAssertEqual(Double(got[t * 3 + 1]), l2, accuracy: 1e-5 + 1e-4 * l2, "size at \(t)")
            XCTAssertEqual(Double(got[t * 3 + 2]), overlap, accuracy: 1e-6, "overlap at \(t)")
            if kl > 1e-3 { anyNonzero = true }
        }
        XCTAssertTrue(anyNonzero)
    }

    func testMaskedAndSlicedStudentsAgreeAndTheOutputIsThePlainForward() throws {
        let (layer, set, ad, x) = try layerAndAdapters()
        let (o1, aux1) = healingLayerForwardAux(layer, x, set: set, adapters: ad, sliced: false)
        eval(o1, aux1)
        assertClose(o1, healingLayerForward(layer, x, set: set, adapters: ad, sliced: false), atol: 0, rtol: 0, "plain forward")
        sliceToKeptExperts(layer, set: set)
        let (o2, aux2) = healingLayerForwardAux(layer, x, set: set, adapters: ad, sliced: true)
        assertClose(o1, o2, atol: 1e-4, rtol: 1e-3, "outputs")
        assertClose(aux1, aux2, atol: 1e-5, rtol: 1e-4, "measure")
    }

    func testUnhealedAgreesPerfectlyTrainedDriftsAndOffReportsNothing() throws {
        let s = try setup()
        let zero = try run(s, steps: 1, lr: 0, held: held())
        let r0 = try XCTUnwrap(zero.blocks[0].finalRouting)
        XCTAssertEqual(r0.kl, 0, accuracy: 1e-9)
        XCTAssertEqual(r0.topKOverlap, 1, accuracy: 1e-6)
        let trained = try run(s, steps: 16, lr: 2e-2, held: held())
        let r1 = try XCTUnwrap(trained.blocks[0].finalRouting)
        XCTAssertGreaterThan(r1.kl, 1e-4)
        XCTAssertLessThanOrEqual(r1.topKOverlap, 1)
        XCTAssertNil(try run(s, steps: 1, routerRank: 0, held: held()).blocks[0].finalRouting)
        // a faster router moves the routing further
        let fast = try XCTUnwrap(try run(s, steps: 16, lr: 2e-2, routerLR: 3, held: held()).blocks[0].finalRouting)
        XCTAssertGreaterThan(fast.kl, r1.kl)
    }

    // MARK: the sweep's variant axis

    func testSweepVariantsCarryTheRouterKnobs() throws {
        let s = try setup()
        var base = HealingFixture.tinyConfiguration()
        base.learningRate = 2e-2
        var o = BlockPipelineOptions()
        o.fixedKeep = s.keep
        o.storageDType = .float32
        let sweep = LearningRateSweep(
            source: try s.source, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, base: base,
            directory: try tempDir(), options: o)
        let report = try sweep.run(
            trainRows: s.rows, heldOut: held(), learningRates: [2e-2], steps: [10],
            variants: [
                SweepVariant(name: "off", routerRank: 0), SweepVariant(name: "r4"),
                SweepVariant(name: "r8", routerRank: 8), SweepVariant(name: "r4x3", routerLRMultiplier: 3),
            ])
        let by = Dictionary(uniqueKeysWithValues: report.results.map { ($0.variant, $0) })
        XCTAssertNil(by["off"]!.routing)
        XCTAssertNotNil(by["r4"]!.routing)
        XCTAssertGreaterThan(by["r4x3"]!.routing!.kl, by["r4"]!.routing!.kl)
        XCTAssertNotEqual(by["r8"]!.routing!.kl, by["r4"]!.routing!.kl)
        XCTAssertTrue(report.table.contains("route KL"))
        XCTAssertNoThrow(try report.json())
    }
}

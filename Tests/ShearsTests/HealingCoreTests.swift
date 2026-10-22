import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class HealingCoreTests: ShearsTestCase {
    func tokens(_ n: Int = 24, seed: UInt64 = 3) -> MLXArray {
        MLXArray(HealingFixture.row(length: n, seed: seed).tokens)[.newAxis]
    }

    // MARK: the recipe's gate: the decomposition reproduces the model

    func testTeacherDecompositionReproducesTheModel() throws {
        let m = try HealingFixture.model()
        let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4))
        let t = tokens()
        assertClose(h.logits(tokens: t, student: false), m(t), atol: 1e-4)
    }

    // MARK: the recipe's own property test

    private func assertAllExpertsZeroAdaptersEqualTeacher(dtype: DType) throws {
        let m = try HealingFixture.model(dtype: dtype)
        let all = Array(0 ..< HealingFixture.experts)
        let h = try HealingFixture.healer(model: m, keep: Array(repeating: all, count: 4))
        let t = tokens()
        let teacher = h.logits(tokens: t, student: false)
        let student = h.logits(tokens: t, student: true)
        eval(teacher, student)
        XCTAssertEqual(abs(teacher.asType(.float32) - student.asType(.float32)).max().item(Float.self), 0, "\(dtype)")
        // The adapters are live: noise in them moves the student and never the teacher.
        HealingFixture.randomizeAdapters(h.adapters)
        let moved = h.logits(tokens: t, student: true)
        XCTAssertGreaterThan(abs(teacher.asType(.float32) - moved.asType(.float32)).max().item(Float.self), 1e-4)
        XCTAssertEqual(
            abs(h.logits(tokens: t, student: false).asType(.float32) - teacher.asType(.float32)).max().item(Float.self), 0)
    }

    func testAllExpertsMaskWithZeroAdaptersEqualsTeacherFloat32() throws {
        try assertAllExpertsZeroAdaptersEqualTeacher(dtype: .float32)
    }

    func testAllExpertsMaskWithZeroAdaptersEqualsTeacherBFloat16() throws {
        try assertAllExpertsZeroAdaptersEqualTeacher(dtype: .bfloat16)
    }

    func testMaskRestrictsRoutingToTheKeptSet() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 3)
        let h = try HealingFixture.healer(model: m, keep: keep)
        let layer = h.network.layers[0]
        let x = m.embed(tokens())
        let parts = layer.attentionPart(x, adapters: nil)
        let (idx, w) = route(
            layer, logits: parts.routerLogits, set: h.network.sets[0], tweak: nil, routerInput: nil)
        let kept = Set(keep[0])
        for e in idx.asArray(Int32.self) { XCTAssertTrue(kept.contains(Int(e))) }
        XCTAssertTrue(w.asArray(Float.self).allSatisfy { $0.isFinite })
        // Fewer kept experts than top-k: the surplus slots carry exactly zero weight.
        let one = ExpertSet(keep: [keep[0][0]], numExperts: HealingFixture.experts)
        let (_, w1) = route(layer, logits: parts.routerLogits, set: one, tweak: nil, routerInput: nil)
        let zeros = (w1[.ellipsis, 0 ..< 1] .== 0).all().item(Bool.self) || (w1[.ellipsis, 1 ..< 2] .== 0).all().item(Bool.self)
        XCTAssertTrue(zeros)
    }

    func testMasksAreNotParameters() throws {
        let m = try HealingFixture.model()
        let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4))
        let keys = h.adapters.parameters().flattened().map(\.0)
        XCTAssertFalse(keys.contains { $0.contains("mask") || $0.contains("slot") || $0.contains("keep") })
        XCTAssertFalse(m.parameters().flattened().contains { $0.0.contains("mask") })
    }

    func testAdapterParameterCounts() throws {
        let m = try HealingFixture.model()
        let a = AdapterConfiguration(expertRank: 8, denseRank: 4, routerRank: 2)
        let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4), adapters: a)
        let c = h.adapters.parameterCounts()
        let (d, hid, e, n, layers) = (64, 32, 8, 4, 4)
        let expert = layers * n * (2 * (d * 8 + 8 * hid) + (hid * 8 + 8 * d))
        let router = layers * (d * 2 + 2 * e)
        let mlpHidden = 96, attnIn = 4 * 16
        let dense = layers * ((d * 4 + 4 * mlpHidden) * 2 + (mlpHidden * 4 + 4 * d))
        var attn = 0
        for l in 0 ..< layers { attn += (l % 2 == 1 || l == 3 ? 4 * 32 : attnIn) * 4 + 4 * d }
        XCTAssertEqual(c["experts"], expert)
        XCTAssertEqual(c["router"], router)
        XCTAssertEqual(c["dense"], dense + attn)
        XCTAssertEqual(c["total"], expert + router + dense + attn)
    }

    // MARK: pin loss and weights

    func testPinLossMatchesAHandComputation() {
        // T = 2, D = 2. Teacher r = [[1, 2], [3, 0]]: per-dimension scale = sqrt(mean r^2)
        // = [sqrt(5), sqrt(2)]. h differs from r by [[1, 0], [0, 2]]:
        // d = [[(1/sqrt5)^2, 0], [0, (2/sqrt2)^2]] = [[0.2, 0], [0, 2]]; per-position means
        // [0.1, 1]. Weights [0.5, 1] (sum 1.5): (0.05 + 1) / 1.5.
        let r = MLXArray([1, 2, 3, 0] as [Float], [1, 2, 2])
        let h = r + MLXArray([1, 0, 0, 2] as [Float], [1, 2, 2])
        let w = MLXArray([0.5, 1] as [Float])
        XCTAssertEqual(HealingLoss.pin([3: h], teacher: [3: r], weights: w).item(Float.self), 1.05 / 1.5, accuracy: 1e-6)
        // Weights summing below one are not renormalised (the recipe's max(sum, 1) floor).
        let w2 = MLXArray([0.25, 0.25] as [Float])
        XCTAssertEqual(
            HealingLoss.pin([3: h], teacher: [3: r], weights: w2).item(Float.self), (0.025 + 0.25) / 1, accuracy: 1e-6)
        // Two layers average.
        let h2 = r
        XCTAssertEqual(
            HealingLoss.pin([3: h, 5: h2], teacher: [3: r, 5: r], weights: w).item(Float.self), 0.5 * 1.05 / 1.5, accuracy: 1e-6)
        // The 1e-3 floor on the scale keeps an all-zero teacher dimension finite.
        let z = MLXArray.zeros([1, 2, 2])
        XCTAssertTrue(HealingLoss.pin([0: z + 0.5], teacher: [0: z], weights: w).item(Float.self).isFinite)
    }

    func testRampedWeights() {
        let w = HealingLoss.rampedWeights(count: 10, boundary: 8, floor: 0.2, rampLength: 4).asArray(Float.self)
        XCTAssertEqual(w[0 ... 4].map { $0 }, [0.2, 0.2, 0.2, 0.2, 0.2])
        XCTAssertEqual(w[5], 0.2 + 0.8 * 0.25, accuracy: 1e-6)
        XCTAssertEqual(w[6], 0.2 + 0.8 * 0.5, accuracy: 1e-6)
        XCTAssertEqual(w[8], 1, accuracy: 1e-6)
        XCTAssertEqual(w[9], 1, accuracy: 1e-6)
    }

    func testBlockEndsFollowGlobalAttentionLayers() throws {
        let m = try HealingFixture.model()
        XCTAssertEqual(m.defaultBlockEnds, [1, 3])
        XCTAssertEqual(blockRanges(ends: [1, 3]).map { [$0.start, $0.end] }, [[0, 1], [2, 3]])
        let m6 = try HealingFixture.model(layers: 5)
        XCTAssertEqual(m6.defaultBlockEnds, [1, 3, 4])
    }

    // MARK: schedule

    func testLearningRateSchedule() {
        let s = LearningRateSchedule(peak: 1e-4, floor: 1e-5, warmup: 20, total: 100)
        XCTAssertEqual(s.value(at: 0), 0)  // the first update runs at zero
        XCTAssertEqual(s.value(at: 10), 5e-5, accuracy: 1e-9)
        XCTAssertEqual(s.value(at: 20), 1e-4, accuracy: 1e-9)
        // Cosine over 80 steps: halfway is the midpoint of peak and floor.
        XCTAssertEqual(s.value(at: 60), 5.5e-5, accuracy: 1e-9)
        // A quarter of the way through the cosine: 0.5 * (1 + cos(pi / 4)) of the way down.
        XCTAssertEqual(s.value(at: 40), 1e-5 + 0.5 * (1 + cos(Float.pi / 4)) * 9e-5, accuracy: 1e-9)
        XCTAssertEqual(s.value(at: 100), 1e-5, accuracy: 1e-9)
        XCTAssertEqual(s.value(at: 500), 1e-5, accuracy: 1e-9)  // constant past the end
        let q = LearningRateSchedule(peak: 5e-5, floor: 5e-6, warmup: 0, total: 10)
        XCTAssertEqual(q.value(at: 0), 5e-5, accuracy: 1e-10)
    }

    func testPlanParsing() throws {
        let p = try PhasePlan(parsing: "block@0,freerun@600,end@1400", total: 3000)
        XCTAssertEqual(p.phase(at: 599), .block)
        XCTAssertEqual(p.phase(at: 600), .freerun)
        XCTAssertEqual(p.phase(at: 2999), .end)
        XCTAssertEqual(p.text, "block@0,freerun@600,end@1400")
        XCTAssertThrowsError(try PhasePlan(parsing: "block@5", total: 10))
        XCTAssertThrowsError(try PhasePlan(parsing: "bogus@0", total: 10))
    }

    // MARK: each phase reduces its own loss

    func testEachPhaseReducesItsLoss() throws {
        for phase in HealingPhase.allCases {
            let m = try HealingFixture.model()
            var c = HealingFixture.tinyConfiguration()
            if phase == .answerKL { c.trainFromLayer = 1 }
            let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4), configuration: c)
            let row = HealingFixture.row()
            let before = h.probeLoss(phase, row: row)
            let steps = 40
            let plan = PhasePlan(entries: [(phase, 0)], total: steps)
            try h.run(rows: ShuffledRows(rows: [row]), plan: plan)
            let after = h.probeLoss(phase, row: row)
            XCTAssertGreaterThan(before, 0, "\(phase)")
            XCTAssertLessThan(after, before * 0.8, "\(phase): \(before) -> \(after)")
        }
    }

    func testDirtyPhaseWithZeroLambdaTargetsTheTeachersTrueBlockEnd() throws {
        let m = try HealingFixture.model()
        let all = Array(repeating: Array(0 ..< HealingFixture.experts), count: 4)
        let h = try HealingFixture.healer(model: m, keep: all)
        let row = HealingFixture.row()
        // With every expert kept and zero adapters the student equals the teacher, so the
        // student's block input is the teacher's, and the teacher's block on it (the optional
        // regulariser's target) equals the clean target.
        let (inputs, targets) = h.dirtyInputs(row, withTargets: true)
        let ends = h.teacherStates(row, at: Set(h.blockEnds))
        for (prev, next) in zip(h.blocks, h.blocks.dropFirst()) {
            assertClose(inputs[next.start]!, ends[prev.end]!, atol: 0)
        }
        for b in h.blocks { assertClose(targets[b.end]!, ends[b.end]!, atol: 0) }
        XCTAssertEqual(h.probeLoss(.dirty, row: row), 0, accuracy: 1e-9)
    }

    func testDirtyPhaseFeedsEachBlockTheStudentsOwnOutput() throws {
        let m = try HealingFixture.model()
        let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4))
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        let row = HealingFixture.row()
        let ends = h.teacherStates(row, at: Set(h.blockEnds))
        let (inputs, _) = h.dirtyInputs(row, withTargets: false)
        let b = h.blocks
        // The second block's input is the *student's* output of the first block ...
        let (_, st) = h.network.forward(h.embed(row), through: b[0].end, student: true, collect: [b[0].end])
        assertClose(inputs[b[1].start]!, st[b[0].end]!, atol: 1e-5)
        // ... which is not the teacher's, and so dirty and block steps are different objectives.
        XCTAssertGreaterThan(abs(inputs[b[1].start]! - ends[b[0].end]!).max().item(Float.self), 1e-3)
        XCTAssertNotEqual(h.probeLoss(.dirty, row: row), h.probeLoss(.block, row: row))
    }

    /// The recipe's gradient gates: checkpointed gradients equal plain ones, every zero-init
    /// half of every adapter gets a finite nonzero gradient in every layer, and every random
    /// half gets exactly zero at initialisation.
    func testEndToEndGradientsAreCheckpointInvariantAndReachEveryAdapter() throws {
        let m = try HealingFixture.model()
        let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4))
        let row = HealingFixture.row()
        let ends = h.teacherStates(row, at: Set(h.blockEnds))
        let x = h.embed(row)
        let w = row.weights
        func grads(checkpointed: Bool) -> [String: MLXArray] {
            let vg = valueAndGrad(model: h.adapters) { [network = h.network, ends, blockEnds = h.blockEnds] (_: ModelAdapters, _: [MLXArray]) in
                let (_, p) = network.forward(x, student: true, collect: Set(blockEnds), checkpointed: checkpointed)
                return [HealingLoss.pin(p, teacher: ends, weights: w)]
            }
            let g = GradMath.flatten(vg(h.adapters, []).1)
            eval(Array(g.values))
            return g
        }
        let plain = grads(checkpointed: false), ckpt = grads(checkpointed: true)
        XCTAssertEqual(Set(plain.keys), Set(ckpt.keys))
        XCTAssertEqual(plain.count, h.adapters.tensors().count)
        for (k, g) in plain {
            assertClose(ckpt[k]!, g, atol: 1e-5, rtol: 1e-4, k)
            XCTAssertTrue(isFinite(g).all().item(Bool.self), k)
            let magnitude = abs(g).max().item(Float.self)
            let zeroInit = k.hasSuffix(".b") || k.hasSuffix("_b") || k.hasSuffix("rt_b")
            if zeroInit { XCTAssertGreaterThan(magnitude, 0, "no gradient reaches \(k)") }
            else { XCTAssertEqual(magnitude, 0, "\(k): a-halves get no gradient while b = 0") }
        }
    }

    func testFrozenLayersBelowTrainFromLayerStayPut() throws {
        let m = try HealingFixture.model()
        var c = HealingFixture.tinyConfiguration()
        c.trainFromLayer = 2
        let h = try HealingFixture.healer(model: m, keep: HealingFixture.keepSets(count: 4), configuration: c)
        let before = HealingFixture.copyTensors(h.adapters.tensors())
        try h.run(
            rows: ShuffledRows(rows: [HealingFixture.row()]),
            plan: PhasePlan(entries: [(.answerKL, 0)], total: 4))
        let after = h.adapters.tensors()
        for (k, v) in after {
            let layer = Int(k.split(separator: ".")[1])!
            let diff = abs(v - before[k]!).max().item(Float.self)
            if layer < 2 { XCTAssertEqual(diff, 0, k) }
        }
        XCTAssertTrue(after.contains { k, v in
            Int(k.split(separator: ".")[1])! >= 2 && abs(v - before[k]!).max().item(Float.self) > 0
        })
    }

    // MARK: checkpoint and resume

    func testResumeContinuesIdentically() throws {
        let plan = try PhasePlan(parsing: "block@0,dirty@3,freerun@5,end@7", total: 9)
        let row1 = HealingFixture.row(seed: 3, key: "a")
        let row2 = HealingFixture.row(length: 20, seed: 4, key: "b")
        let source = ShuffledRows(rows: [row1, row2], seed: 9)
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let seed = try HealingFixture.healer(model: m, keep: keep)
        let initial = HealingFixture.copyTensors(seed.adapters.tensors())

        let straight = try HealingFixture.healer(model: m, keep: keep, initial: initial)
        try straight.run(rows: source, plan: plan)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("heal-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let first = try HealingFixture.healer(model: m, keep: keep, initial: initial)
        var seen = 0
        var cb = HealingCallbacks()
        cb.onStep = { _ in seen += 1 }
        cb.shouldStop = { seen >= 4 }
        let s1 = try first.run(rows: source, plan: plan, checkpointDirectory: dir, callbacks: cb)
        XCTAssertEqual(s1.outcome, .stopped(step: 4, reason: "stop requested"))

        let second = try HealingFixture.healer(model: m, keep: keep, initial: initial)
        let s2 = try second.run(rows: source, plan: plan, checkpointDirectory: dir)
        XCTAssertEqual(s2.outcome, .finished)
        XCTAssertEqual(s2.stepsRun, 5)
        let a = straight.adapters.tensors(), b = second.adapters.tensors()
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        var worst: Float = 0
        for (k, v) in a { worst = max(worst, abs(v - b[k]!).max().item(Float.self)) }
        // MLX's gather-matmul gradients are not bit-reproducible (two identical runs differ by
        // about 1e-5), so compare at a tolerance well under how far the run moves the adapters.
        var moved: Float = 0
        for (k, v) in a { moved = max(moved, abs(v - initial[k]!).max().item(Float.self)) }
        XCTAssertGreaterThan(moved, 0.01)
        XCTAssertLessThan(worst, 2e-3, "resumed run diverged (adapters moved by \(moved))")
    }

    func testAdapterLoadRejectsMismatchedFiles() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let h = try HealingFixture.healer(model: m, keep: keep)
        var t = h.adapters.tensors()
        let k = t.keys.first { $0.contains("experts") }!
        t[k] = nil
        XCTAssertThrowsError(try h.adapters.load(tensors: t))
        var u = h.adapters.tensors()
        let k2 = u.keys.first { $0.contains("rt_a") }!
        u[k2] = MLXArray.zeros([3, 3])
        XCTAssertThrowsError(try h.adapters.load(tensors: u))
    }
}

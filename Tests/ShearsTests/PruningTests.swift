import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class PruningTests: ShearsTestCase {
    func testPruningProducesExpectedShapes() {
        let block = MoEBlock(dims: 32, hiddenDims: 64, numExperts: 8, topK: 4)
        eval(block)
        ExpertPruner.prune(block, keep: [6, 1, 3])

        XCTAssertEqual(block.router.gateWeight.shape, [3, 32])
        XCTAssertEqual(block.router.topK, 3, "topK clamps to the experts left")
        XCTAssertEqual(block.experts.gateProj.weight.shape, [3, 64, 32])
        XCTAssertEqual(block.experts.upProj.weight.shape, [3, 64, 32])
        XCTAssertEqual(block.experts.downProj.weight.shape, [3, 32, 64])
        XCTAssertEqual(block(MLXRandom.normal([2, 5, 32])).shape, [2, 5, 32])
    }

    func testPruningSlicesTheRightRows() {
        let block = MoEBlock(dims: 8, hiddenDims: 8, numExperts: 5, topK: 2)
        eval(block)
        // Module.update rewrites parameter arrays in place, so snapshot by value.
        let router = block.router.gateWeight * 1
        let gate = block.experts.gateProj.weight * 1
        eval(router, gate)
        ExpertPruner.prune(block, keep: [4, 0, 2])
        assertClose(block.router.gateWeight, router[MLXArray([0, 2, 4] as [Int32])])
        assertClose(block.experts.gateProj.weight[1], gate[2])
        assertClose(block.experts.gateProj.weight[2], gate[4])
    }

    func testKeepingEveryExpertLeavesOutputUnchanged() {
        let block = MoEBlock(dims: 16, hiddenDims: 32, numExperts: 6, topK: 2, normalizeTopK: true)
        eval(block)
        let x = MLXRandom.normal([2, 12, 16])
        let before = block(x)
        ExpertPruner.prune(block, keep: Array(0 ..< 6))
        assertClose(block(x), before)
    }

    func testTokensWhoseExpertsSurviveAreUnchanged() {
        // With top-k renormalisation the softmax denominator cancels, so a token whose
        // whole top-k is kept must produce exactly the same output after pruning. A
        // router/expert misalignment breaks this even though shapes stay right.
        let block = MoEBlock(dims: 16, hiddenDims: 32, numExperts: 8, topK: 2, normalizeTopK: true)
        eval(block)
        let x = MLXRandom.normal([1, 256, 16])
        let before = block(x)
        let (indices, _) = block.router(x)

        let keep = [0, 2, 3, 5, 7]
        ExpertPruner.prune(block, keep: keep)
        let after = block(x)

        let idx = indices.reshaped([-1, 2]).asArray(UInt32.self)
        let keepSet = Set(keep.map(UInt32.init))
        var surviving: [Int32] = []
        for t in 0 ..< 256 where keepSet.contains(idx[2 * t]) && keepSet.contains(idx[2 * t + 1]) {
            surviving.append(Int32(t))
        }
        XCTAssertGreaterThan(surviving.count, 10, "test needs some fully-kept tokens")
        let sel = MLXArray(surviving)
        assertClose(after[0].take(sel, axis: 0), before[0].take(sel, axis: 0))
    }

    func testExpertStatsAccumulateAndRank() {
        let block = MoEBlock(dims: 16, hiddenDims: 32, numExperts: 8, topK: 2)
        eval(block)
        let stats = ExpertStats(numExperts: 8)
        block.stats = stats
        for x in Toy.batches(3, tokens: 64, dims: 16) { _ = block(x) }

        XCTAssertEqual(stats.tokens, 3 * 64)
        XCTAssertEqual(stats.pickCount.sum().item(Float.self), Float(3 * 64 * 2))
        // Normalised top-k scores sum to 1 per token.
        XCTAssertEqual(stats.weightMass.sum().item(Float.self), Float(3 * 64), accuracy: 1e-2)

        let keep = stats.expertsToKeep(count: 3)
        let mass = stats.weightMass.asArray(Float.self)
        let threshold = keep.map { mass[$0] }.min()!
        for e in 0 ..< 8 where !keep.contains(e) { XCTAssertLessThanOrEqual(mass[e], threshold) }
        XCTAssertEqual(stats.expertsToKeep(count: 3, by: .reap).count, 3)

        ExpertPruner.prune(block, using: stats, keeping: 3)
        XCTAssertEqual(block.numExperts, 3)
        XCTAssertNil(block.stats)
    }
}

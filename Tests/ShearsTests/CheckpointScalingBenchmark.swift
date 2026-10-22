import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import Shears

/// How whole-model backprop peak memory grows with depth, with and without per-layer
/// ``checkpoint(_:_:)``. Off by default; run like ``MemoryBenchmarkTests``.
final class CheckpointScalingBenchmark: ShearsTestCase {
    func testPeakMemoryVersusDepth() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1",
            "set TEST_RUNNER_SHEARS_BENCH=1 to run")
        var out = ""
        for useCkpt in [false, true] {
            for L in [2, 4, 8, 12] {
                let layers = (0 ..< L).map { i -> MoEResidualLayer in
                    MLXRandom.seed(UInt64(i))
                    let l = MoEResidualLayer(dims: 256, hiddenDims: 512, numExperts: 8, topK: 2)
                    eval(l); return l
                }
                final class S: Module, UnaryLayer {
                    let layers: [MoEResidualLayer]; let c: Bool
                    init(_ l: [MoEResidualLayer], _ c: Bool) { layers = l; self.c = c }
                    func callAsFunction(_ x: MLXArray) -> MLXArray {
                        layers.reduce(x) { h, l in c ? checkpoint(l) { $0($1) }(h) : l(h) }
                    }
                }
                let s = S(layers, useCkpt)
                let x = MLXRandom.normal([4, 512, 256]); eval(x)
                let vg = valueAndGrad(model: s) { (m: S, x, t) in m(x).square().mean() }
                Memory.clearCache(); Memory.peakMemory = 0
                let before = Memory.activeMemory
                let (l, g) = vg(s, x, x)
                eval(l, g)
                let peak = Memory.peakMemory - before
                out += "ckpt=\(useCkpt) L=\(L) peak-minus-resident=\(peak / 1_048_576) MB\n"
            }
        }
        print(out)
    }
}

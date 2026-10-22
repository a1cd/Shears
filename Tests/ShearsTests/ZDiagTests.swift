import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// TEMPORARY diagnostic (CI investigation of MergedExpertTests on Linux CPU). Removed once the investigation is over.
extension MergedExpertTests {
    func zdiagRun(_ t: BlockPipelineTests, _ s: BlockPipelineTests.Setup, merged: Bool, steps: Int) throws -> (losses: [Float], adapters: [String: MLXArray]) {
        ExpertCompute.merged = merged
        ExpertCompute.mergedMaxKeptRatio = 100
        defer { ExpertCompute.merged = true; ExpertCompute.mergedMaxKeptRatio = 1.5; ExpertCompute.tokenChunk = nil }
        var o = BlockPipelineOptions()
        o.batchSteps = steps
        o.mix = .clean
        o.fixedKeep = s.keep
        o.storageDType = .float32
        o.tokenChunk = nil
        var c = t.config()
        c.checkpointBlocks = true
        let dir = try tempDir()
        let p = try BlockPipeline(source: try s.source, blockEnds: s.ends, adapterConfiguration: HealingFixture.smallAdapters, configuration: c, options: o, directory: dir)
        let r = try p.run(rows: ShuffledRows(rows: s.rows, seed: 4), steps: steps)
        return (r.blocks[0].losses, r.adapters)
    }

    func testZDiagMergedVsRouted() throws {
        for steps in [1, 2, 4] {
            let t = BlockPipelineTests()
            let setup = try t.setup(layers: 4, ends: [3], keepCount: 2, rowCount: 3, length: 20)
            let routed = try zdiagRun(t, setup, merged: false, steps: steps)
            let merged = try zdiagRun(t, setup, merged: true, steps: steps)
            let routed2 = try zdiagRun(t, setup, merged: false, steps: steps)
            var worst = (0 as Float, "")
            var worstSame = (0 as Float, "")
            var over = 0
            for (k, v) in routed.adapters {
                let d = abs(merged.adapters[k]! - v)
                let m = d.max().item(Float.self)
                if m > worst.0 { worst = (m, "\(k) max|v|=\(abs(v).max().item(Float.self))") }
                let s2 = abs(routed2.adapters[k]! - v).max().item(Float.self)
                if s2 > worstSame.0 { worstSame = (s2, k) }
                if !allClose(merged.adapters[k]!, v, rtol: 5e-3, atol: 5e-4).item(Bool.self) { over += 1 }
            }
            print("ZDIAG steps=\(steps) losses routed=\(routed.losses) merged=\(merged.losses)")
            print("ZDIAG steps=\(steps) merged-vs-routed worst max|d|=\(worst.0) at \(worst.1); tensors over tol: \(over) of \(routed.adapters.count); routed-vs-routed again worst max|d|=\(worstSame.0) \(worstSame.1)")
        }
    }
}

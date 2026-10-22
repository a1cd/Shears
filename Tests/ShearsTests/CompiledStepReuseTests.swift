import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The compiled whole-block step is traced once per sequence bucket for the life of a run, however many rows of
/// different lengths pass through it: a step must never pay for a trace it has already paid for (a trace costs
/// seconds on the real model, so a regression would look like per-step overhead).
final class CompiledStepReuseTests: ShearsTestCase {
    func testRowsOfMixedLengthsTraceOncePerBucket() throws {
        let layers = 3
        let config = try SpeedMemoryBenchmark.configFor(hidden: 64, layers: layers, dense: 32, vocab: 64, experts: 8, moeHidden: 16, topK: 2)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("reuse-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = Gemma4TextModel(config)
        MLXRandom.seed(5)
        let p = model.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
        model.update(parameters: ModuleParameters.unflattened(p))
        model.train(false)
        eval(model)
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        let ckpt = dir.appendingPathComponent("ckpt")
        try MergeAndSliceExport.write(model: model, keep: Array(repeating: Array(0 ..< 8), count: layers), adapters: nil, to: ckpt, options: o)
        let source = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .float32)

        let lengths = [20, 40, 70]  // buckets 32, 64, 96
        let rows = lengths.enumerated().map { (i, n) -> HealingRow in
            var rng = SplitMix64(seed: UInt64(i + 1))
            return HealingRow(key: "r\(i)", tokens: (0 ..< n).map { _ in Int32.random(in: 0 ..< 64, using: &rng) })
        }
        var options = BlockPipelineOptions()
        options.fixedKeep = Array(repeating: Array(0 ..< 8), count: layers)
        options.storageDType = .float32
        options.mix = .clean
        options.batchSteps = 12
        options.maxBlocks = 1
        options.trainOnly = true
        var c = HealingConfiguration()
        c.warmupSteps = 0
        c.checkpointBlocks = true
        c.keepIntermediates = .lowMemory
        let pipeline = try BlockPipeline(
            source: source, blockEnds: [layers - 1], adapterConfiguration: AdapterConfiguration(expertRank: 4, denseRank: 4, routerRank: 2),
            configuration: c, options: options, directory: dir.appendingPathComponent("run"))
        let before = CompiledLayerStep.debugTraces
        var applied = 0
        _ = try pipeline.run(rows: ShuffledRows(rows: rows, seed: 1), steps: 12, onStep: { _, s in if s.applied { applied += 1 } })
        withExtendedLifetime(pipeline) {}
        XCTAssertEqual(applied, 12)
        // The whole block is one window, run through one compiled function (`top`); each bucket traces it once.
        XCTAssertEqual(CompiledLayerStep.debugTraces - before, Set(lengths.map { StreamedBlockHealer.sequenceBucket(for: $0) }).count)
    }
}

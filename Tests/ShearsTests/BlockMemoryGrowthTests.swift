import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Does anything carry over from one block of the pipeline to the next? Every block has the same shape, so
/// the memory MLX holds when a block starts, its training peak and the residency planner's estimate should not
/// grow with the block's index.
final class BlockMemoryGrowthTests: ShearsTestCase {
    func tempDir(_ name: String = "bmg") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    struct Block { var activeAtFirstStep = 0, peak = 0, estimate = 0, activeAtEnd = 0 }

    func run(blocks: Int, layersPerBlock: Int = 2, steps: Int = 4, batchSteps: Int = 2, quantized: Bool = false, configure: (inout BlockPipelineOptions) -> Void = { _ in }) throws -> [Block] {
        let layers = blocks * layersPerBlock
        let ckpt = try tempDir("bmg-ckpt")
        let cfg = try PrepareMemoryBenchmark.config(layers: layers)
        let m = Gemma4TextModel(cfg)
        MLXRandom.seed(5)
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        eval(m)
        var o = MergeAndSliceExport.Options()
        o.dtype = .bfloat16
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< PrepareMemoryBenchmark.experts), count: layers), adapters: nil, to: ckpt, options: o)
        Memory.clearCache()
        let source = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .bfloat16)
        var rng = SplitMix64(seed: 11)
        let rows = (0 ..< steps).map { i -> HealingRow in
            HealingRow(key: "r\(i)", tokens: (0 ..< 96).map { _ in Int32.random(in: 0 ..< 2048, using: &rng) })
        }
        var opts = BlockPipelineOptions()
        opts.fixedKeep = (0 ..< layers).map { _ in Array(0 ..< 8) }
        opts.mix = .both
        opts.batchSteps = batchSteps
        opts.storageDType = .bfloat16
        configure(&opts)
        let ends = (0 ..< blocks).map { ($0 + 1) * layersPerBlock - 1 }
        let pipeline = try BlockPipeline(
            source: source, blockEnds: ends, adapterConfiguration: AdapterConfiguration(expertRank: 8, denseRank: 8, routerRank: 4),
            configuration: { var c = HealingFixture.tinyConfiguration(); if ProcessInfo.processInfo.environment["BMG_CKPT"] == "0" { c.checkpointBlocks = false }; return c }(), options: opts, directory: try tempDir("bmg-run"))
        var out = Array(repeating: Block(), count: blocks)
        var seenFirst = Set<Int>()
        let before = Memory.activeMemory
        var report: BlockPipeline.Report? = try pipeline.run(
            rows: ShuffledRows(rows: rows, seed: 3), steps: steps,
            onStep: { b, r in
                if seenFirst.insert(b).inserted { out[b].activeAtFirstStep = Memory.activeMemory }
                out[b].peak = max(out[b].peak, r.peakMemoryBytes)
            },
            onBlockFinished: { b, _ in out[b].activeAtEnd = Memory.activeMemory; print("BLOCKGROWTH live compiled steps after block \(b + 1): \(CompiledLayerStep.live)") })
        let held = Memory.activeMemory
        for (b, info) in report!.blocks.enumerated() { out[b].estimate = info.decision?.estimate.total ?? 0 }
        report = nil
        Memory.clearCache()
        print("BLOCKGROWTH (inside run) before run \(before >> 10) KB, after run (report held) \(held >> 10) KB, report dropped \(Memory.activeMemory >> 10) KB")
        return out
    }

    func testBlocksDoNotGrow() throws {
        let compiled = ProcessInfo.processInfo.environment["BMG_COMPILE"] != "0"
        StreamedBlockHealer.compileLayerSteps = compiled
        let adam = ProcessInfo.processInfo.environment["BMG_ADAM"] != "0"
        HealingAdamW.compileStep = adam
        print("BLOCKGROWTH layer compile \(compiled), adam compile \(adam)")
        defer { StreamedBlockHealer.compileLayerSteps = true; HealingAdamW.compileStep = true }
        let base = Memory.activeMemory
        let blocks = try run(blocks: 6)
        Memory.clearCache()
        print("BLOCKGROWTH after run() returned: \(Memory.activeMemory >> 10) KB vs \(base >> 10) KB before")
        // Nothing of the run is kept once it returns (the old cached checkpoint closure kept every block's layers alive:
        // 75 MB after six tiny blocks).
        XCTAssertLessThan(Memory.activeMemory - base, 8 << 20, "memory held after the run returned")
        for (b, x) in blocks.enumerated() {
            print("BLOCKGROWTH compile=\(compiled) block \(b + 1): active at first step \(x.activeAtFirstStep >> 10) KB, peak \(x.peak >> 10) KB, estimate \(x.estimate >> 10) KB, active at end \(x.activeAtEnd >> 10) KB")
        }
        // Every block has the same shape: what is held when it starts, its training peak and the planner's estimate stay put.
        let mb = 1 << 20
        for (b, x) in blocks.enumerated().dropFirst(2) {
            XCTAssertLessThan(x.activeAtFirstStep - blocks[1].activeAtFirstStep, 8 * mb, "block \(b + 1) active at first step grew")
            XCTAssertLessThan(x.activeAtEnd - blocks[1].activeAtEnd, 8 * mb, "block \(b + 1) active at end grew")
            XCTAssertLessThan(x.peak - blocks[1].peak, 16 * mb, "block \(b + 1) training peak grew")
            XCTAssertLessThan(x.estimate - blocks[1].estimate, 1 * mb, "block \(b + 1) estimate grew")
        }
    }
}

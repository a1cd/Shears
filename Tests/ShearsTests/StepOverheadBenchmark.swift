import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Per-step time and memory of the normal (compiled) block-pipeline step on a toy with a MIX of row lengths (several
/// sequence buckets per run), whole block resident, one row per step. Off by default. Run with:
///   TEST_RUNNER_SHEARS_BENCH=1 xcodebuild test -scheme Shears -destination 'platform=macOS' \
///     -only-testing:ShearsTests/StepOverheadBenchmark
/// Prints a table and writes it to `shears-step-overhead.txt` in the temporary directory.
final class StepOverheadBenchmark: ShearsTestCase {
    static let big = ProcessInfo.processInfo.environment["SHEARS_BENCH_BIG"] == "1"
    static let hidden = big ? 1024 : 256, experts = 32, moeHidden = big ? 256 : 64, dense = big ? 512 : 192, vocab = 1024, layers = 6
    /// Row lengths, in tokens: six distinct sequence buckets.
    static let lengths = big ? [600, 1000, 1400, 1800, 2400, 3000] : [300, 520, 700, 900, 1200, 1500]

    func config() throws -> Gemma4Configuration {
        try SpeedMemoryBenchmark.configFor(
            hidden: Self.hidden, layers: Self.layers, dense: Self.dense, vocab: Self.vocab, experts: Self.experts,
            moeHidden: Self.moeHidden)
    }

    func checkpoint(in dir: URL) throws -> URL {
        let ckpt = dir.appendingPathComponent("ckpt")
        if FileManager.default.fileExists(atPath: ckpt.path) { return ckpt }
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        let m = Gemma4TextModel(try config())
        MLXRandom.seed(5)
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        eval(m)
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< Self.experts), count: Self.layers), adapters: nil, to: ckpt, options: o)
        return ckpt
    }

    struct Mode {
        var name: String
        var freeEarly = false
        var overlap = false
        var cacheMB: Int?
    }

    /// One pipeline run whose steps cycle through `modes` one epoch (every row once) at a time, so every mode sees
    /// the same rows in the same machine conditions. Per mode: mean s/step and the MLX peak over the step's start.
    func compare(keep: ActivationKeep, label: String, modes: [Mode], rounds: Int, dir: URL) throws -> [String] {
        let ckpt = try checkpoint(in: dir)
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .float32)
        let layerKeep = (0 ..< Self.layers).map { _ in Array(0 ..< Self.experts) }
        let rows = Self.lengths.enumerated().map { (i, n) -> HealingRow in
            var rng = SplitMix64(seed: UInt64(i + 1))
            return HealingRow(key: "r\(i)", tokens: (0 ..< n).map { _ in Int32.random(in: 0 ..< Int32(Self.vocab), using: &rng) })
        }
        let per = Self.lengths.count
        let steps = per * modes.count * rounds
        var opts = BlockPipelineOptions()
        opts.fixedKeep = layerKeep
        opts.storageDType = .float32
        opts.mix = .clean
        opts.batchSteps = steps
        opts.maxBlocks = 1
        opts.trainOnly = true
        ExpertCompute.merged = true
        ExpertCompute.mergedMaxKeptRatio = 100
        var c = HealingConfiguration()
        c.warmupSteps = 0
        c.checkpointBlocks = true
        c.keepIntermediates = keep
        let ac = AdapterConfiguration(expertRank: 32, denseRank: 16, routerRank: 8)
        let p = try BlockPipeline(
            source: src, blockEnds: [Self.layers - 1], adapterConfiguration: ac, configuration: c, options: opts,
            directory: dir.appendingPathComponent("run-\(UUID().uuidString)"))
        func apply(_ epoch: Int) {
            let m = modes[epoch % modes.count]
            StreamedBlockHealer.freeGradientsEarly = m.freeEarly
            StreamedBlockHealer.overlapLossCheck = m.overlap
            if let mb = m.cacheMB { Memory.cacheLimit = mb << 20 }
        }
        defer {
            StreamedBlockHealer.freeGradientsEarly = false
            StreamedBlockHealer.overlapLossCheck = false
        }
        apply(0)
        var secs: [Double] = [], peaks: [Int] = []
        Memory.clearCache()
        var base = 0
        let r = try p.run(
            rows: ShuffledRows(rows: rows, seed: 3), steps: steps, memory: .init(cacheLimitBytes: nil),
            onStep: { _, s in
                if secs.isEmpty { base = Memory.activeMemory }
                secs.append(s.seconds)
                peaks.append(max(0, s.peakMemoryBytes - base))
                Memory.peakMemory = Memory.activeMemory
                apply(secs.count / per)
            })
        withExtendedLifetime(p) {}
        var out = ["== keep \(label)  (resident \(r.blocks[0].residentLayers), \(steps) steps, epoch 0 excluded)"]
        for (k, m) in modes.enumerated() {
            let idx = secs.indices.filter { ($0 / per) % modes.count == k && $0 >= per }
            let t = idx.map { secs[$0] }, pk = idx.map { peaks[$0] }
            let sorted = t.sorted()
            out.append(String(format: "  %-34@ mean %.3f s/step  median %.3f  min %.3f   MLX step peak max %7.1f MB", m.name as NSString,
                t.reduce(0, +) / Double(max(1, t.count)), sorted[sorted.count / 2], sorted[0], Double(pk.max() ?? 0) / 1_048_576))
        }
        return out
    }

    func testStepOverhead() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("stepoverhead-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        Memory.memoryLimit = 14 << 30
        Memory.cacheLimit = 1 << 30
        let rounds = Int(ProcessInfo.processInfo.environment["SHEARS_BENCH_STEPS"] ?? "") ?? 3
        var out = ["toy: hidden \(Self.hidden), \(Self.experts) experts, \(Self.layers) layers, row lengths \(Self.lengths), fp32, \(rounds) rounds"]
        let cacheModes = [
            Mode(name: "cache 64 MB", cacheMB: 64), Mode(name: "cache 256 MB", cacheMB: 256), Mode(name: "cache 1 GB", cacheMB: 1024),
            Mode(name: "cache 4 GB", cacheMB: 4096), Mode(name: "cache 12 GB", cacheMB: 12288),
        ]
        if ProcessInfo.processInfo.environment["SHEARS_BENCH_CACHE"] == "1" {
            for (keep, label) in [(ActivationKeep.lowMemory, "lowMemory"), (.fast, "fast")] {
                out += try compare(keep: keep, label: label, modes: cacheModes, rounds: rounds, dir: dir)
            }
            let text = out.joined(separator: "\n")
            print("\n" + text + "\n")
            return
        }
        let modes = [
            Mode(name: "standard"), Mode(name: "freeGradientsEarly", freeEarly: true), Mode(name: "overlapLossCheck", overlap: true),
        ]
        for (keep, label) in [(ActivationKeep.lowMemory, "lowMemory"), (.layers(3), "3"), (.fast, "fast")] {
            out += try compare(keep: keep, label: label, modes: modes, rounds: rounds, dir: dir)
        }
        let text = out.joined(separator: "\n")
        print("\n" + text + "\n")
        try text.write(to: FileManager.default.temporaryDirectory.appendingPathComponent("shears-step-overhead.txt"), atomically: true, encoding: .utf8)
    }
}

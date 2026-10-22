import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Training-phase memory and seconds per step on the toy checkpoint of ``PrepareMemoryBenchmark`` (adapters sized so
/// their state is a real share of the footprint). Off by default; each configuration is one process:
///
///   TEST_RUNNER_SHEARS_BENCH=1 TEST_RUNNER_SHEARS_TRAIN_FREE=1 \
///     xcodebuild test -scheme Shears -destination 'platform=macOS' -only-testing:ShearsTests/TrainStepBenchmark
///
/// Variables: `SHEARS_TRAIN_FREE` (0/1, default 0, ``StreamedBlockHealer/freeGradientsEarly``), `SHEARS_TRAIN_OVERLAP` (0/1, default 0), `SHEARS_TRAIN_CKPT` (0/1), `SHEARS_TRAIN_KEEP` (auto, fast, lowMemory (default) or a layer count, see ``ActivationKeep``),
/// `SHEARS_TRAIN_LAYERS` (default 6, one block), `SHEARS_TRAIN_SEQ` (default 384), `SHEARS_TRAIN_STEPS` (default 12),
/// `SHEARS_TRAIN_RANK` (expert rank, default 128), `SHEARS_TRAIN_KEPT` (default 32).
/// The line is appended to `shears-train-bench.txt` in the temporary directory.
final class TrainStepBenchmark: ShearsTestCase {
    func testTrainStep() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        func int(_ k: String, _ d: Int) -> Int { Int(env[k] ?? "") ?? d }
        let layers = int("SHEARS_TRAIN_LAYERS", 6), seq = int("SHEARS_TRAIN_SEQ", 384), steps = int("SHEARS_TRAIN_STEPS", 12)
        let rank = int("SHEARS_TRAIN_RANK", 128), kept = int("SHEARS_TRAIN_KEPT", 32)
        StreamedBlockHealer.freeGradientsEarly = int("SHEARS_TRAIN_FREE", 0) == 1
        StreamedBlockHealer.overlapLossCheck = int("SHEARS_TRAIN_OVERLAP", 0) == 1
        let ckptOn = int("SHEARS_TRAIN_CKPT", 1) == 1
        let ckpt = try PrepareMemoryBenchmark.checkpoint(layers: layers)
        Memory.clearCache()
        Memory.memoryLimit = 8 << 30
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .bfloat16)
        let keep = (0 ..< layers).map { _ in Array(0 ..< kept) }
        let rows = (0 ..< 6).map { i -> HealingRow in
            var rng = SplitMix64(seed: UInt64(i + 1))
            return HealingRow(key: "r\(i)", tokens: (0 ..< seq).map { _ in Int32.random(in: 0 ..< 2048, using: &rng) })
        }
        var opts = BlockPipelineOptions()
        opts.fixedKeep = keep
        opts.mix = .clean
        opts.batchSteps = steps
        opts.maxBlocks = 1
        opts.trainOnly = true
        if let mbs = Int(env["SHEARS_TRAIN_CACHE_MB"] ?? "") { opts.cacheLimitBytes = mbs << 20 }
        var c = HealingConfiguration()
        c.warmupSteps = 0
        c.checkpointBlocks = ckptOn
        c.keepIntermediates = try ActivationKeep(parsing: env["SHEARS_TRAIN_KEEP"] ?? "lowMemory")
        func runOnce() throws -> (secs: [Double], footprint: Int, cache: Int, base: Int, report: BlockPipeline.Report) {
            let wd = FileManager.default.temporaryDirectory.appendingPathComponent("trainbench-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: wd) }
            let p = try BlockPipeline(
                source: src, blockEnds: [layers - 1], adapterConfiguration: AdapterConfiguration(expertRank: rank, denseRank: 32, routerRank: 8),
                configuration: c, options: opts, directory: wd)
            var secs: [Double] = []
            var footprint = 0
            var cache = 0
            let base = ProcessMemory.snapshot().footprint
            let r = try p.run(rows: ShuffledRows(rows: rows, seed: 3), steps: steps, onStep: { _, s in
                secs.append(s.seconds)
                footprint = max(footprint, s.peakFootprintBytes)
                cache = max(cache, Memory.cacheMemory)
                if s.step == 8, let hold = Double(env["SHEARS_TRAIN_HOLD"] ?? "") {
                    // Between steps, with the block's weights, adapters and optimizer state live: print MLX's counters next to the
                    // process footprint at this instant, then wait so `footprint <pid>` / `vmmap -summary <pid>` can be run.
                    let snap = ProcessMemory.snapshot()
                    let mbi = { (b: Int) in String(format: "%.1f", Double(b) / 1_048_576) }
                    print("HOLD pid \(getpid()): MLX active \(mbi(Memory.activeMemory)) MB, cache \(mbi(Memory.cacheMemory)) MB, peak \(mbi(Memory.peakMemory)) MB; footprint \(mbi(snap.footprint)) MB, anonymous \(mbi(snap.anonymous)) MB, file-backed \(mbi(snap.fileBacked)) MB")
                    Thread.sleep(forTimeInterval: hold)
                }
            })
            return (secs, footprint, cache, base, r)
        }
        if let ab = env["SHEARS_TRAIN_AB"] {
            // In-process A/B: the toggle named by `ab` alternates off/on, three times each; the means exclude each run's first 3 steps.
            var means: [Bool: [Double]] = [false: [], true: []]
            var feet: [Bool: [Int]] = [false: [], true: []]
            for _ in 0 ..< 3 {
                for on in [false, true] {
                    switch ab {
                    case "overlap": StreamedBlockHealer.overlapLossCheck = on
                    case "free": StreamedBlockHealer.freeGradientsEarly = on
                    case "cache": opts.cacheLimitBytes = on ? 128 << 20 : nil
                    default: break
                    }
                    let r = try runOnce()
                    let t = r.secs.dropFirst(3)
                    means[on]!.append(t.reduce(0, +) / Double(max(1, t.count)))
                    feet[on]!.append(r.footprint)
                }
            }
            let f = { (v: [Double]) in v.map { String(format: "%.4f", $0) }.joined(separator: " ") }
            let line = "A/B \(ab): off [\(f(means[false]!))] mean \(String(format: "%.4f", means[false]!.reduce(0, +) / 3)); on [\(f(means[true]!))] mean \(String(format: "%.4f", means[true]!.reduce(0, +) / 3)) s/step; footprint peak MB off \(feet[false]!.map { $0 >> 20 }) on \(feet[true]!.map { $0 >> 20 }) (layers \(layers) seq \(seq) kept \(kept) rank \(rank) ckpt \(ckptOn))\n"
            print("\n" + line)
            return
        }
        let (secs, footprint, cache, base, r) = try runOnce()
        let tail = secs.dropFirst(3)
        let per = tail.reduce(0, +) / Double(max(1, tail.count))
        let mb = { (b: Int) in String(format: "%.1f", Double(b) / 1_048_576) }
        let b = r.blocks[0]
        let line = "layers \(layers) seq \(seq) kept \(kept) rank \(rank) ckpt \(ckptOn) keep \(c.keepIntermediates) free \(StreamedBlockHealer.freeGradientsEarly) overlap \(StreamedBlockHealer.overlapLossCheck): \(String(format: "%.4f", per)) s/step, MLX peak \(mb(b.trainingPeakBytes)) MB, footprint peak \(mb(footprint)) MB (start \(mb(base))), cache max \(mb(cache)) MB, resident \(b.residentLayers), final loss \(b.losses.last ?? -1)\n"
        print("\n" + line)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("shears-train-bench.txt")
        if let h = try? FileHandle(forWritingTo: out) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try h.close() } else { try line.write(to: out, atomically: true, encoding: .utf8) }
    }
}

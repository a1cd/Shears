import Darwin
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Process-lifetime physical footprint of the stages that run before training (embedding, residency
/// measurement, the teacher walk), on a toy checkpoint whose size is set by environment variables. Off by
/// default. One configuration per process, because the number reported is `proc_pid_rusage`'s
/// `ri_lifetime_max_phys_footprint`, which cannot be reset:
///
///   TEST_RUNNER_SHEARS_BENCH=1 TEST_RUNNER_SHEARS_PREP_LAYERS=12 TEST_RUNNER_SHEARS_PREP_ROWS=48 \
///     xcodebuild test -scheme Shears -destination 'platform=macOS' -only-testing:ShearsTests/PrepareMemoryBenchmark
///
/// The line is appended to `shears-prepare-bench.txt` in the temporary directory.
final class PrepareMemoryBenchmark: ShearsTestCase {
    static let hidden = 512, experts = 64, moeHidden = 192, dense = 512

    static func config(layers: Int) throws -> Gemma4Configuration {
        let types = (0 ..< layers).map { $0 % 6 == 5 ? "full_attention" : "sliding_attention" }
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": dense, "num_attention_heads": 4, "head_dim": 64, "global_head_dim": 64,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "vocab_size": 2048, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 128, "final_logit_softcapping": 30.0,
            "enable_moe_block": true, "num_experts": experts, "top_k_experts": 8,
            "moe_intermediate_size": moeHidden, "layer_types": types, "tie_word_embeddings": true,
        ]
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        try c.validate()
        return c
    }

    /// The toy checkpoint with `layers` layers, written once and reused (bf16 on disk).
    static func checkpoint(layers: Int) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("prepbench")
        let ckpt = root.appendingPathComponent("ckpt-\(layers)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: ckpt.path) {
            var o = MergeAndSliceExport.Options()
            o.dtype = .bfloat16
            let m = Gemma4TextModel(try config(layers: layers))
            MLXRandom.seed(5)
            let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
            m.update(parameters: ModuleParameters.unflattened(p))
            m.train(false)
            eval(m)
            try MergeAndSliceExport.write(
                model: m, keep: Array(repeating: Array(0 ..< experts), count: layers), adapters: nil, to: ckpt, options: o)
        }
        return ckpt
    }

    static func lifetimeMaxFootprint() -> Int {
        var info = rusage_info_v6()
        let rc = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V6, $0) }
        }
        return rc == 0 ? Int(info.ri_lifetime_max_phys_footprint) : 0
    }

    func testPrepareFootprint() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        let layers = Int(env["SHEARS_PREP_LAYERS"] ?? "") ?? 6
        let nRows = Int(env["SHEARS_PREP_ROWS"] ?? "") ?? 24
        let blockLayers = Int(env["SHEARS_PREP_BLOCK"] ?? "") ?? 3
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("prepbench")
        let ckpt = try Self.checkpoint(layers: layers)
        Memory.clearCache()
        Memory.memoryLimit = 8 << 30
        Memory.cacheLimit = 8 << 30  // the library's own default for a long run is whatever it sets; start from "unset"
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .bfloat16)
        let keep = (0 ..< layers).map { _ in Array(0 ..< 24) }
        var rng = SplitMix64(seed: 11)
        let rows = (0 ..< nRows).map { i -> HealingRow in
            let len = Int.random(in: 150 ... 700, using: &rng)
            return HealingRow(key: "r\(i)", tokens: (0 ..< len).map { _ in Int32.random(in: 0 ..< 2048, using: &rng) })
        }
        var opts = BlockPipelineOptions()
        opts.fixedKeep = keep
        opts.mix = .clean
        opts.batchSteps = nRows
        opts.maxBlocks = Int(env["SHEARS_PREP_MAXBLOCKS"] ?? "") ?? 2
        opts.prepareOnly = true
        opts.maxBatchRows = 4
        var blockEnds: [Int] = []
        var e = blockLayers - 1
        while e < layers - 1 { blockEnds.append(e); e += blockLayers }
        blockEnds.append(layers - 1)
        let wd = root.appendingPathComponent("run-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: wd) }
        let baseline = Self.lifetimeMaxFootprint()
        let sampler = FootprintSampler()
        var maxCache = 0
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "bench.cache"))
        let lock = NSLock()
        timer.schedule(deadline: .now(), repeating: 0.005)
        timer.setEventHandler { let c = Memory.cacheMemory; lock.lock(); maxCache = max(maxCache, c); lock.unlock() }
        timer.resume()
        let p = try BlockPipeline(
            source: src, blockEnds: blockEnds, adapterConfiguration: AdapterConfiguration(expertRank: 8, denseRank: 4, routerRank: 4),
            configuration: HealingConfiguration(), options: opts, directory: wd)
        _ = try p.run(rows: ShuffledRows(rows: rows, seed: 3), steps: nRows)
        timer.cancel()
        let life = Self.lifetimeMaxFootprint()
        lock.lock(); let mc = maxCache; lock.unlock()
        let mb = { (b: Int) in String(format: "%.0f", Double(b) / 1_048_576) }
        let line = "layers \(layers) rows \(nRows) blockLayers \(blockLayers): lifetime max phys footprint \(mb(life)) MB (at start \(mb(baseline))), sampled footprint peak \(mb(sampler.peak)) MB, MLX peak \(mb(Memory.peakMemory)) MB, MLX cache max sampled \(mb(mc)) MB, ckpt on disk \(mb(directoryBytes(ckpt))) MB\n"
        print("\n" + line)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("shears-prepare-bench.txt")
        if let h = try? FileHandle(forWritingTo: out) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try h.close() } else { try line.write(to: out, atomically: true, encoding: .utf8) }
    }

    func directoryBytes(_ url: URL) -> Int {
        guard let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var n = 0
        for case let f as URL in en { n += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        return n
    }
}

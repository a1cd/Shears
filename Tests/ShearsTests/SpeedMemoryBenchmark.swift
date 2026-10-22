import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Seconds per step and memory of a block-pipeline run on a toy shaped like the real model's proportions, for the
/// residency, merged-layout, token-chunk and cache settings. Off by default. Run with:
///   TEST_RUNNER_SHEARS_BENCH=1 xcodebuild test -scheme Shears -destination 'platform=macOS' \
///     -only-testing:ShearsTests/SpeedMemoryBenchmark
/// The table is printed and written to `shears-speed-bench.txt` in the temporary directory.
final class SpeedMemoryBenchmark: ShearsTestCase {
    static let hidden = 176, experts = 32, moeHidden = 44, dense = 132, seq = 512, vocab = 1024, layers = 6

    static func config(topK: Int) throws -> Gemma4Configuration {
        try configFor(hidden: hidden, layers: layers, dense: dense, vocab: vocab, experts: experts, moeHidden: moeHidden, topK: topK)
    }

    static func configFor(
        hidden: Int, layers: Int, dense: Int, vocab: Int, experts: Int, moeHidden: Int, topK: Int = 8
    ) throws -> Gemma4Configuration {
        let types = (0 ..< layers).map { $0 % 6 == 5 ? "full_attention" : "sliding_attention" }
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": dense, "num_attention_heads": 4, "head_dim": 32, "global_head_dim": 64,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "vocab_size": vocab, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 128, "final_logit_softcapping": 30.0,
            "enable_moe_block": true, "num_experts": experts, "top_k_experts": topK,
            "moe_intermediate_size": moeHidden, "layer_types": types, "tie_word_embeddings": true,
        ]
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        try c.validate()
        return c
    }

    struct Variant {
        var name: String
        var topK = 8
        var kept = 8
        var forceResident: Int?
        var merged = true
        var shared = false
        var expertRank = 32
        var tokenChunk: Int?
        var cacheLimit: Int?
    }

    func mb(_ b: Int) -> String { String(format: "%6.1f", Double(b) / 1_048_576) }

    func checkpoint(topK: Int, in dir: URL) throws -> URL {
        let ckpt = dir.appendingPathComponent("ckpt-\(topK)")
        if FileManager.default.fileExists(atPath: ckpt.path) { return ckpt }
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        let m = Gemma4TextModel(try Self.config(topK: topK))
        MLXRandom.seed(5)
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        eval(m)
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< Self.experts), count: Self.layers), adapters: nil, to: ckpt, options: o)
        return ckpt
    }

    func run(_ v: Variant, dir: URL, steps: Int) throws -> String {
        let ckpt = try checkpoint(topK: v.topK, in: dir)
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: ckpt), dtype: .float32)
        let keep = (0 ..< Self.layers).map { _ in Array(0 ..< v.kept) }
        let rows = (1 ... 6).map { seed -> HealingRow in
            var rng = SplitMix64(seed: UInt64(seed))
            return HealingRow(key: "r\(seed)", tokens: (0 ..< Self.seq).map { _ in Int32.random(in: 0 ..< Int32(Self.vocab), using: &rng) })
        }
        var opts = BlockPipelineOptions()
        opts.fixedKeep = keep
        opts.storageDType = .float32
        opts.mix = .clean
        opts.batchSteps = steps
        opts.maxBlocks = 1
        opts.trainOnly = true
        opts.forcedResidentLayers = v.forceResident
        opts.tokenChunk = v.tokenChunk
        ExpertCompute.merged = v.merged
        ExpertCompute.mergedMaxKeptRatio = 100
        defer { ExpertCompute.merged = true; ExpertCompute.mergedMaxKeptRatio = 1.5 }
        var c = HealingConfiguration()
        c.warmupSteps = 0
        c.checkpointBlocks = true
        var ac = AdapterConfiguration(expertRank: v.expertRank, denseRank: 16, routerRank: 8)
        ac.sharedExperts = v.shared
        let wd = dir.appendingPathComponent("run-\(UUID().uuidString)")
        let p = try BlockPipeline(
            source: src, blockEnds: [Self.layers - 1], adapterConfiguration: ac, configuration: c, options: opts, directory: wd)
        var secs: [Double] = []
        var cache = 0, footprint = 0
        Memory.clearCache()
        let baseline = ProcessMemory.snapshot().footprint
        let active0 = Memory.activeMemory
        Memory.peakMemory = 0
        let r = try p.run(
            rows: ShuffledRows(rows: rows, seed: 3), steps: steps, memory: .init(cacheLimitBytes: v.cacheLimit),
            onStep: { _, s in
                secs.append(s.seconds)
                cache = max(cache, Memory.cacheMemory)
                footprint = max(footprint, s.peakFootprintBytes)
            })
        let peak = max(0, Memory.peakMemory - active0)
        let tail = secs.dropFirst(3)
        let per = tail.reduce(0, +) / Double(max(1, tail.count))
        let b = r.blocks[0]
        withExtendedLifetime(p) {}
        return "\(v.name.padding(toLength: 40, withPad: " ", startingAt: 0)) \(String(format: "%7.4f", per)) s/step  resident \(b.residentLayers)  MLX peak over start \(mb(peak)) MB (start held \(mb(active0)))  cache \(mb(cache)) MB  footprint +\(mb(max(0, footprint - baseline))) MB  gap \(mb(max(0, footprint - baseline - peak))) MB"
    }

    func testSpeedAndMemory() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("speedbench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        Memory.memoryLimit = 8 << 30
        Memory.cacheLimit = 1 << 30
        var out = ["toy: hidden \(Self.hidden), moe width \(Self.moeHidden), \(Self.experts) experts, top-8, \(Self.layers) layers (one block), seq \(Self.seq), fp32, expert rank 32 / dense 16 / router 8, 14 steps (first 3 excluded)"]
        out.append("-- residency")
        out.append(try run(Variant(name: "1 resident layer (old auto), routed", forceResident: 1, merged: false), dir: dir, steps: 14))
        out.append(try run(Variant(name: "whole block, routed", merged: false), dir: dir, steps: 14))
        out.append("-- merged layout")
        out.append(try run(Variant(name: "whole block, merged"), dir: dir, steps: 14))
        out.append(try run(Variant(name: "whole block, merged, shared LoRA r64", shared: true, expertRank: 64), dir: dir, steps: 14))
        out.append("-- token chunk (merged, whole block)")
        for chunk in [1024, 512, 256, 128] {
            out.append(try run(Variant(name: "chunk \(chunk)", tokenChunk: chunk), dir: dir, steps: 14))
        }
        out.append("-- buffer cache limit (merged, whole block)")
        for limit in [16 << 20, 64 << 20, 1 << 30] {
            out.append(try run(Variant(name: "cache limit \(limit >> 20) MB", cacheLimit: limit), dir: dir, steps: 14))
        }
        out.append("-- merged vs routed by kept count (top-8)")
        for k in [8, 9, 12, 16] {
            out.append(try run(Variant(name: "K=\(k) routed", kept: k, merged: false), dir: dir, steps: 14))
            out.append(try run(Variant(name: "K=\(k) merged", kept: k), dir: dir, steps: 14))
        }
        let text = out.joined(separator: "\n")
        print("\n" + text + "\n")
        try text.write(to: FileManager.default.temporaryDirectory.appendingPathComponent("shears-speed-bench.txt"), atomically: true, encoding: .utf8)
    }
}

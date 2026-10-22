import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Peak memory of the healing phases on a toy Gemma 4 MoE: the main (whole model resident)
/// path, with and without per-layer checkpointing, against the streamed block phase.
///
/// Off by default. Run with:
///   TEST_RUNNER_SHEARS_BENCH=1 xcodebuild test -scheme Shears -destination 'platform=macOS' \
///     -only-testing:ShearsTests/MemoryBenchmarkTests
final class MemoryBenchmarkTests: ShearsTestCase {
    // hidden 256, 16 experts of 256 -> 256 (kept 8), top-2, dense MLP 512, sequence 512, fp32.
    static let hidden = 256, experts = 16, keep = 8, moeHidden = 256, seq = 512, vocab = 1024

    static func config(layers: Int) throws -> Gemma4Configuration {
        let types = (0 ..< layers).map { $0 % 2 == 1 ? "full_attention" : "sliding_attention" }
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": 512, "num_attention_heads": 4, "head_dim": 64, "global_head_dim": 128,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "vocab_size": vocab, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 128, "final_logit_softcapping": 30.0,
            "enable_moe_block": true, "num_experts": experts, "top_k_experts": 2,
            "moe_intermediate_size": moeHidden, "layer_types": types, "tie_word_embeddings": true,
        ]
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        try c.validate()
        return c
    }

    static func model(layers: Int) throws -> Gemma4TextModel {
        MLXRandom.seed(5)
        let m = Gemma4TextModel(try config(layers: layers))
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        eval(m)
        return m
    }

    static let adapters = AdapterConfiguration(expertRank: 32, denseRank: 16, routerRank: 8)

    func row(_ seed: UInt64) -> HealingRow {
        var rng = SplitMix64(seed: seed)
        let t = (0 ..< Self.seq).map { _ in Int32.random(in: 0 ..< Int32(Self.vocab), using: &rng) }
        return HealingRow(key: "r\(seed)", tokens: t)
    }

    func mb(_ b: Int) -> String { String(format: "%8.1f MB", Double(b) / 1_048_576) }

    func testPeakMemory() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1",
            "set TEST_RUNNER_SHEARS_BENCH=1 to run")
        var lines = [
            "toy Gemma 4 MoE: hidden \(Self.hidden), \(Self.experts)->\(Self.keep) experts (top-2, width \(Self.moeHidden)), "
                + "dense 512, sequence \(Self.seq), fp32, adapters ranks 32/16/8",
            "arm                                   resident-before    peak          peak-minus-resident",
        ]
        let rows = [row(1), row(2)]
        let source = ShuffledRows(rows: rows, seed: 1)

        // Main path: whole model resident.
        var mainBlock = 0
        for (layers, arms) in [
            (8, [
                ("main: block phase", HealingPhase.block, false),
                ("main: block phase, checkpointed", .block, true),
                ("main: freerun (checkpointed)", .freerun, true),
                ("main: end-to-end KL (checkpointed)", .end, true),
            ]),
            (16, [("main: block phase, 16 layers", HealingPhase.block, false)]),
        ] {
            let m = try Self.model(layers: layers)
            let keep = Array(repeating: Array(0 ..< Self.keep), count: layers)
            for (name, phase, ckptBlocks) in arms {
                var c = HealingConfiguration()
                c.checkpointBlocks = ckptBlocks
                c.endSamplePositions = 16
                var r = row(1)
                r.klAnchors = [KLAnchor(position: Self.seq - 8, weight: 5)]
                r.klSampleRange = (Self.seq - 64) ..< (Self.seq - 1)
                let h = try PruneHealer(model: m, keep: keep, adapterConfiguration: Self.adapters, configuration: c)
                eval(h.adapters)
                Memory.clearCache()
                Memory.peakMemory = 0
                let resident = Memory.activeMemory
                _ = h.probeLoss(phase, row: r)
                let peak = Memory.peakMemory
                if name == "main: block phase" { mainBlock = peak }
                lines.append("\(name.padding(toLength: 38, withPad: " ", startingAt: 0))\(mb(resident))   \(mb(peak))   \(mb(peak - resident))")
            }
        }

        // Streamed: layers read one block at a time from a checkpoint directory.
        func streamed(layerCount: Int) throws -> (resident: Int, peak: Int) {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("bench-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dir) }
            do {
                let m = try Self.model(layers: layerCount)
                var o = MergeAndSliceExport.Options()
                o.dtype = .float32
                try MergeAndSliceExport.write(
                    model: m, keep: Array(repeating: Array(0 ..< Self.experts), count: layerCount),
                    adapters: nil, to: dir, options: o)
            }
            Memory.clearCache()
            let cp = try Gemma4Checkpoint(directory: dir)
            let src = Gemma4LayerSource(checkpoint: cp, dtype: .float32)
            var c = HealingConfiguration()
            c.warmupSteps = 0
            let keep = Array(repeating: Array(0 ..< Self.keep), count: layerCount)
            let s = StreamedBlockHealer(
                source: src, keep: keep, blockEnds: stride(from: 1, to: layerCount, by: 2).map { $0 },
                adapterConfiguration: Self.adapters, configuration: c)
            let store = try DiskBoundaryStore(directory: dir.appendingPathComponent("bounds"))
            Memory.peakMemory = 0
            let resident = Memory.activeMemory
            try s.precomputeBoundaries(rows: rows, store: store)
            let report = try s.train(rows: source, steps: 2, store: store)
            return (resident, max(report.peakMemoryBytes, Memory.peakMemory))
        }
        let s8 = try streamed(layerCount: 8)
        lines.append("\("streamed block phase, 8 layers".padding(toLength: 38, withPad: " ", startingAt: 0))\(mb(s8.resident))   \(mb(s8.peak))   \(mb(s8.peak - s8.resident))")
        let s16 = try streamed(layerCount: 16)
        lines.append("\("streamed block phase, 16 layers".padding(toLength: 38, withPad: " ", startingAt: 0))\(mb(s16.resident))   \(mb(s16.peak))   \(mb(s16.peak - s16.resident))")

        let report = lines.joined(separator: "\n")
        print("\n" + report + "\n")
        try report.write(
            to: FileManager.default.temporaryDirectory.appendingPathComponent("shears-bench.txt"),
            atomically: true, encoding: .utf8)
        XCTAssertLessThan(s8.peak, mainBlock, "streaming should peak below the whole-model block phase")
        XCTAssertLessThan(Double(s16.peak), Double(s8.peak) * 1.15, "streamed peak grew with depth")
    }
}

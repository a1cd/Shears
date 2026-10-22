import Darwin
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Memory and time of the streamed block phase by number of resident layers, on a toy Gemma 4 MoE.
///
/// Off by default. One arm per process (the process footprint's peak cannot be reset):
///   TEST_RUNNER_SHEARS_BENCH_ARM=<resident layers, 0 = whole block>[,nomap] \
///   [TEST_RUNNER_SHEARS_BENCH_COMPILE=<0 or 1, default: the library's setting>] \
///   [TEST_RUNNER_SHEARS_BENCH_SEQ=<sequence length, default by size>] \
///   [TEST_RUNNER_SHEARS_BENCH_CKPT=<0 or 1: per-layer checkpointing, default 1>] \
///   xcodebuild test -scheme Shears -destination 'platform=macOS' -only-testing:ShearsTests/ResidencyBenchmarkTests
final class ResidencyBenchmarkTests: ShearsTestCase {
    static let layers = 6
    func cpuSeconds() -> Double {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        func s(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1e6 }
        return s(u.ru_utime) + s(u.ru_stime)
    }

    func testArm() throws {
        guard let arm = ProcessInfo.processInfo.environment["SHEARS_BENCH_ARM"] else { throw XCTSkip("set TEST_RUNNER_SHEARS_BENCH_ARM") }
        let parts = arm.split(separator: ",").map(String.init)
        let resident = Int(parts[0]).flatMap { $0 == 0 ? nil : $0 }
        let map = !parts.contains("nomap")
        let big = ProcessInfo.processInfo.environment["SHEARS_BENCH_SIZE"] == "big"
        if let v = ProcessInfo.processInfo.environment["SHEARS_BENCH_COMPILE"] { StreamedBlockHealer.compileLayerSteps = v != "0" }
        let hidden = big ? 512 : MemoryBenchmarkTests.hidden, moeHidden = big ? 512 : MemoryBenchmarkTests.moeHidden
        let dense = big ? 1024 : 512
        let seq = ProcessInfo.processInfo.environment["SHEARS_BENCH_SEQ"].flatMap { Int($0) } ?? (big ? 1024 : MemoryBenchmarkTests.seq)
        let types = (0 ..< Self.layers).map { $0 % 2 == 1 ? "full_attention" : "sliding_attention" }
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": hidden, "num_hidden_layers": Self.layers,
            "intermediate_size": dense, "num_attention_heads": 4, "head_dim": 64, "global_head_dim": 128,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "vocab_size": 1024, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 128, "final_logit_softcapping": 30.0,
            "enable_moe_block": true, "num_experts": 16, "top_k_experts": 2,
            "moe_intermediate_size": moeHidden, "layer_types": types, "tie_word_embeddings": true,
        ]
        let cfg = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("rbench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var rows: [HealingRow] = []
        for i in 0 ..< 2 {
            var rng = SplitMix64(seed: UInt64(i + 1))
            rows.append(HealingRow(key: "r\(i)", tokens: (0 ..< seq).map { _ in Int32.random(in: 0 ..< Int32(MemoryBenchmarkTests.vocab), using: &rng) }))
        }
        do {
            MLXRandom.seed(5)
            let m = Gemma4TextModel(cfg)
            let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.1 * MLXRandom.normal($0.1.shape)) }
            m.update(parameters: ModuleParameters.unflattened(p))
            m.train(false)
            eval(m)
            var o = MergeAndSliceExport.Options()
            o.dtype = .float32
            try MergeAndSliceExport.write(
                model: m, keep: Array(repeating: Array(0 ..< MemoryBenchmarkTests.experts), count: Self.layers), adapters: nil, to: dir, options: o)
        }
        Memory.clearCache()
        let keep = Array(repeating: Array(0 ..< MemoryBenchmarkTests.keep), count: Self.layers)
        let src = Gemma4LayerSource(checkpoint: try Gemma4Checkpoint(directory: dir), dtype: .float32, mapWeights: map)
        var c = HealingConfiguration()
        c.warmupSteps = 0
        c.residentLayers = resident
        c.checkpointBlocks = ProcessInfo.processInfo.environment["SHEARS_BENCH_CKPT"] != "0"
        let h = StreamedBlockHealer(
            source: src, keep: keep, blockEnds: [Self.layers - 1], adapterConfiguration: MemoryBenchmarkTests.adapters, configuration: c)
        let store = try DiskBoundaryStore(directory: dir.appendingPathComponent("bounds"))
        try h.precomputeBoundaries(rows: rows, store: store, options: .init(storageDType: .float32))
        Memory.clearCache()
        let steps = 8
        LayerLoadStats.reset()
        var seconds: [Double] = [], cpu: [Double] = []
        var last = cpuSeconds()
        let wallStart = Date()
        let r = try h.train(rows: ShuffledRows(rows: rows, seed: 1), steps: steps, store: store, onStep: { _, rec in
            let now = self.cpuSeconds()
            seconds.append(rec.seconds)
            cpu.append(now - last)
            last = now
        })
        let wall = Date().timeIntervalSince(wallStart)
        let sorted = seconds.dropFirst().sorted()
        let median = sorted[sorted.count / 2]
        let meanCPU = cpu.dropFirst().reduce(0, +) / Double(cpu.count - 1)
        func mb(_ b: Int) -> String { String(format: "%.1f", Double(b) / 1_048_576) }
        let fileBacked = ProcessMemory.snapshot().fileBacked
        print(String(
            format: "BENCH arm=%@ seq=%d ckpt=%d resident=%d compile=%d mlxPeakMB=%@ footprintPeakMB=%@ median_s_per_step=%.3f first_s=%.3f cpu_s_per_step=%.3f wall_total_s=%.2f fileBackedAtEndMB=%@ loads=%d load_s_per_step=%.3f",
            arm, seq, c.checkpointBlocks ? 1 : 0, r.residentLayers, StreamedBlockHealer.compileLayerSteps ? 1 : 0, mb(r.peakMemoryBytes), mb(r.peakFootprintBytes), median, seconds[0], meanCPU, wall, mb(fileBacked), LayerLoadStats.count, LayerLoadStats.seconds / Double(steps)))
    }

    /// The memory model at the shape of a 26B-A4B-class model (hidden 2816, 30 layers in six-layer
    /// blocks, 8 kept experts of width 704, dense MLP 2112, sequence 2700, 4-bit base), from byte counts
    /// worked out by hand; no weights are loaded. Printed, not asserted.
    func testEstimateAtLargeModelShape() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1")
        let d = 2816, h = 704, dense = 2112, kept = 8, t = 2700, heads = 16
        let c = AdapterConfiguration()  // ranks 128 / 64 / 16
        let bits = 4.5  // 4-bit weights with a bf16 scale and bias per group of 64
        let attention = 35_000_000.0  // q, k, v, o of the assumed 16 x 256 attention
        let weightParams = Double(kept * 3 * h * d) + Double(3 * d * dense) + attention
        let weightBytes = Int(weightParams * bits / 8) + 128 * d * 2  // + the bf16 router
        let adapterParams = 3 * kept * c.expertRank * (d + h)
            + c.denseRank * (3 * (d + dense) + (heads * 256 + d)) + c.routerRank * (d + 128)
        let full = 128 * 3 * h * d * 9 / 16 + Int(3 * Double(d * dense) * bits / 8 + attention * bits / 8)
        var i = ResidencyInputs(
            blockLayers: 6, layerWeightBytes: weightBytes, adapterBytesPerLayer: adapterParams * 4, sequence: t, hidden: d,
            workspaceBytesPerToken: (14 * d + 4 * heads * 256 + 3 * dense + 4 * 2 * h) * 4,
            attentionBytes: 2 * heads * 512 * t * 4, loadTransientBytes: full, baselineBytes: 0)
        i.slackBytes = 64 << 20
        func g(_ b: Int) -> String { String(format: "%.2f", Double(b) / 1_073_741_824) }
        print("EST per layer: sliced weights \(g(weightBytes)) GB, adapters fp32 \(g(adapterParams * 4)) GB (x3 with Adam moments), stored uncut \(g(full)) GB")
        for r in [6, 3, 2, 1] {
            let e = ResidencyPlanner.estimate(i, resident: r)
            print("EST R=\(r): weights \(g(e.weights)) adapters+opt \(g(e.adapterState)) grads \(g(e.gradients)) activations \(g(e.activations)) workspace \(g(e.workspace)) load-transient \(g(i.loadTransientBytes)) -> total \(g(e.total)) GB")
        }
    }
}

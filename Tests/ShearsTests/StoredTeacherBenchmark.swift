import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Peak memory of the pre-computed teacher and the sliced student on a toy whose proportions
/// follow the real 26B (hidden : expert width : dense width = 2816 : 704 : 2112, 16x expert
/// pruning), against the masked, live-teacher modes.
///
/// Off by default. Run with:
///   TEST_RUNNER_SHEARS_BENCH=1 xcodebuild test -scheme Shears -destination 'platform=macOS' \
///     -only-testing:ShearsTests/StoredTeacherBenchmark
final class StoredTeacherBenchmark: ShearsTestCase {
    static let hidden = 176, experts = 32, keep = 2, moeHidden = 44, dense = 132, seq = 512, vocab = 1024
    static let layers = 12

    static func config() throws -> Gemma4Configuration {
        let types = (0 ..< layers).map { $0 % 6 == 5 ? "full_attention" : "sliding_attention" }
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": dense, "num_attention_heads": 4, "head_dim": 32, "global_head_dim": 64,
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

    static func model() throws -> Gemma4TextModel {
        let m = Gemma4TextModel(try config())
        MLXRandom.seed(5)
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
        var r = HealingRow(key: "r\(seed)", tokens: t)
        r.klAnchors = [KLAnchor(position: Self.seq - 8, weight: 5)]
        r.klSampleRange = (Self.seq - 64) ..< (Self.seq - 1)
        return r
    }

    func mb(_ b: Int) -> String { String(format: "%7.1f MB", Double(b) / 1_048_576) }
    func line(_ name: String, _ cols: [Int]) -> String {
        name.padding(toLength: 44, withPad: " ", startingAt: 0) + cols.map(mb).joined(separator: "  ")
    }

    func bytes(_ layers: [any HealableMoELayer]) -> Int {
        layers.reduce(0) { acc, l in
            acc + ((l as? Module)?.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } ?? 0)
        }
    }

    func testPeakMemory() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SHEARS_BENCH"] == "1", "set TEST_RUNNER_SHEARS_BENCH=1 to run")
        var out = [
            "toy: hidden \(Self.hidden), moe width \(Self.moeHidden), dense \(Self.dense), \(Self.experts) -> \(Self.keep) experts, "
                + "\(Self.layers) layers (blocks of 6), seq \(Self.seq), fp32, adapter ranks 32/16/8",
        ]
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("storedbench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = dir.appendingPathComponent("ckpt")
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        do {
            let m = try Self.model()
            try MergeAndSliceExport.write(
                model: m, keep: Array(repeating: Array(0 ..< Self.experts), count: Self.layers),
                adapters: nil, to: ckpt, options: o)
        }
        Memory.clearCache()
        let cp = try Gemma4Checkpoint(directory: ckpt)
        let src = Gemma4LayerSource(checkpoint: cp, dtype: .float32)
        let keep = (0 ..< Self.layers).map { _ in Array(0 ..< Self.keep) }
        let ends = [5, 11]
        let rows = (1 ... 8).map { row(UInt64($0)) }

        // Weights resident for one block (6 layers).
        func blockWeights(_ k: [[Int]]?) throws -> Int {
            Memory.clearCache()
            let before = Memory.activeMemory
            let l = try src.loadLayers(0 ... 5, keeping: k)
            eval(l.compactMap { $0 as? Module })
            let used = Memory.activeMemory - before
            withExtendedLifetime(l) {}
            return used
        }
        let wFull = try blockWeights(nil)
        let wSliced = try blockWeights(Array(keep[0 ... 5]))
        out.append("block weights resident (6 layers): all experts \(mb(wFull)), kept only \(mb(wSliced)), ratio \(String(format: "%.1f", Double(wFull) / Double(wSliced)))x")

        // Pre-pass peak against batch size.
        out.append("\npre-pass (peak of the block stages; weights = one full block)")
        out.append("rows per batch".padding(toLength: 44, withPad: " ", startingAt: 0) + "     peak  peak-weights  per token")
        for b in [1, 2, 4, 8] {
            let store = try DiskBoundaryStore(directory: dir.appendingPathComponent("pre\(b)"))
            Memory.clearCache()
            let r = try TeacherPrepass(
                source: src, blockEnds: ends,
                options: .init(storageDType: .float32, maxBatchTokens: b * Self.seq, maxBatchRows: b)
            ).run(rows: rows, store: store)
            let peak = r.stagePeakBytes.filter { $0.key > 0 }.values.max()!
            out.append(line("\(b)", [peak, peak - wFull]) + String(format: "  %8.1f KB", Double(peak - wFull) / Double(b * Self.seq) / 1024))
        }

        // Sequence-length sweeps, one row: pre-pass peak and sliced block-training peak, with the
        // attention scores scored whole (chunk off) and in query chunks of 512.
        out.append("\nby sequence length, one row (peak minus one block's weights; sliced training: minus kept weights)")
        out.append("tokens  chunk".padding(toLength: 44, withPad: " ", startingAt: 0) + "pre-pass       per token     train peak    per token")
        for chunked in [false, true] {
            for t in [512, 1024, 2048, 4096] {
                Gemma4Attention.queryChunk = chunked ? 512 : 0
                var rng = SplitMix64(seed: UInt64(t))
                let long = HealingRow(key: "long\(t)", tokens: (0 ..< t).map { _ in Int32.random(in: 0 ..< Int32(Self.vocab), using: &rng) })
                let store = try DiskBoundaryStore(directory: dir.appendingPathComponent("len\(t)-\(chunked)"))
                Memory.clearCache()
                let r = try TeacherPrepass(source: src, blockEnds: ends, options: .init(storageDType: .float32, maxBatchRows: 1))
                    .run(rows: [long], store: store)
                let pre = r.stagePeakBytes.filter { $0.key > 0 }.values.max()! - wFull
                var c = HealingConfiguration()
                c.warmupSteps = 0
                let s = StreamedBlockHealer(source: src, keep: keep, blockEnds: ends, adapterConfiguration: Self.adapters, configuration: c)
                Memory.clearCache()
                let rep = try s.train(rows: ShuffledRows(rows: [long]), steps: 1, store: store)
                let train = rep.peakMemoryBytes - wSliced
                out.append(line("\(t)  \(chunked ? "512" : "off")", [pre]) + String(format: "  %6.1f KB ", Double(pre) / Double(t) / 1024) + mb(train) + String(format: "  %6.1f KB", Double(train) / Double(t) / 1024))
            }
        }
        Gemma4Attention.queryChunk = 512

        // Streamed block-phase training: masked (all experts resident) vs sliced.
        let store = try DiskBoundaryStore(directory: dir.appendingPathComponent("pre1"))
        out.append("\nstreamed block training, one block (2 steps)")
        out.append("student".padding(toLength: 44, withPad: " ", startingAt: 0) + "  weights   adapters(+grad+Adam)  peak")
        var peaks: [Bool: Int] = [:]
        for sliced in [false, true] {
            var c = HealingConfiguration()
            c.warmupSteps = 0
            var s = StreamedBlockHealer(
                source: src, keep: keep, blockEnds: ends, adapterConfiguration: Self.adapters, configuration: c)
            s.slicedStudent = sliced
            Memory.clearCache()
            let report = try s.train(rows: ShuffledRows(rows: rows, seed: 1), steps: 2, store: store)
            let ad = report.adapters.values.reduce(0) { $0 + $1.nbytes } / 2  // per block
            peaks[sliced] = report.peakMemoryBytes
            out.append(line(sliced ? "sliced (stored teacher)" : "masked, all experts", [sliced ? wSliced : wFull, 4 * ad, report.peakMemoryBytes]))
        }

        // Training against the shipping format: the merged weights are materialised and rounded.
        out.append("\nsliced block training, 512 tokens, rounded merged weights (QAT) against plain")
        for (name, q) in [("plain", nil), ("Q8_0 dense + experts", QATConfiguration(.q8_0)), ("affine 4-bit", QATConfiguration(.affine(bits: 4, groupSize: 32)))] as [(String, QATConfiguration?)] {
            var c = HealingConfiguration()
            c.warmupSteps = 0
            var ad = Self.adapters
            ad.qat = q
            let s = StreamedBlockHealer(source: src, keep: keep, blockEnds: ends, adapterConfiguration: ad, configuration: c)
            Memory.clearCache()
            let rep = try s.train(rows: ShuffledRows(rows: [rows[0]]), steps: 2, store: store)
            out.append(line(name, [rep.peakMemoryBytes]))
        }

        // Whole-student phases: masked with a live teacher (today) vs sliced with the store.
        out.append("\nwhole student, one step (resident = model + adapters)")
        out.append("arm".padding(toLength: 44, withPad: " ", startingAt: 0) + "resident        peak   peak-resident")
        var whole: [String: Int] = [:]
        for phase in [HealingPhase.freerun, .end] {
            var c = HealingConfiguration()
            c.endSamplePositions = 16
            for sliced in [false, true] {
                Memory.clearCache()
                let base = Memory.activeMemory
                let m = try Gemma4Checkpoint(directory: ckpt).loadModel(dtype: .float32)
                m.train(false)
                let h: PruneHealer
                if sliced {
                    let cut = try Gemma4Checkpoint(directory: ckpt).loadModel(keeping: keep, dtype: .float32)
                    cut.train(false)
                    h = try PruneHealer(
                        model: cut, keep: keep, adapterConfiguration: Self.adapters, configuration: c,
                        student: .preSliced(numExperts: Self.experts), teacher: store)
                } else {
                    h = try PruneHealer(model: m, keep: keep, adapterConfiguration: Self.adapters, configuration: c)
                }
                eval(h.adapters)
                Memory.clearCache()
                let resident = Memory.activeMemory - base - (sliced ? bytes(m.healableLayers) + 0 : 0)
                Memory.peakMemory = 0
                _ = h.probeLoss(phase, row: rows[0])
                let peak = Memory.peakMemory - base - (sliced ? bytes(m.healableLayers) : 0)
                whole["\(phase)-\(sliced)"] = peak
                out.append(line("\(phase.rawValue), \(sliced ? "sliced + stored teacher" : "masked + live teacher")", [resident, peak, peak - resident]))
            }
        }
        defer { Gemma4Attention.queryChunk = 512 }
        let text = out.joined(separator: "\n")
        print("\n" + text + "\n")
        try text.write(to: FileManager.default.temporaryDirectory.appendingPathComponent("shears-stored-bench.txt"), atomically: true, encoding: .utf8)
        XCTAssertLessThan(wSliced * 4, wFull)
        XCTAssertLessThan(peaks[true]!, peaks[false]!)
    }
}

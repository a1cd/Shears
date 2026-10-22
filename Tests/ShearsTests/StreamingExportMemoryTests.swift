import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Peak memory of the streaming export (see ``StreamingExportTests`` for what it must equal).
final class StreamingExportMemoryTests: ShearsTestCase {
    func tempDir(_ name: String = "stream-export-mem") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func exportedShape(_ dir: URL, _ key: String) throws -> [Int]? {
        let idx = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
        guard let f = (idx["weight_map"] as! [String: String])[key] else { return nil }
        return try loadArrays(url: dir.appendingPathComponent(f))[key]?.shape
    }

    /// A 4-bit checkpoint with `layers` layers of 64 experts (hidden 512, expert width 192), written layer by
    /// layer so building it does not itself hold the model.
    static func writeQuantizedCheckpoint(layers: Int, to dir: URL) throws {
        let cfg = try PrepareMemoryBenchmark.config(layers: layers)
        let m = Gemma4TextModel(cfg)
        MLXRandom.seed(5)
        var tensors: [String: MLXArray] = [:]
        for (key, w) in m.parameters().flattened() {
            let name = "language_model." + key
            let v = (w + 0.1 * MLXRandom.normal(w.shape))
            let quantize = (key.contains("switch_glu.") || key.contains("self_attn.") && key.hasSuffix("_proj.weight")
                || key == "model.embed_tokens.weight") && key.hasSuffix(".weight")
            if quantize {
                let (wq, s, b) = MLX.quantized(v, groupSize: 64, bits: 4)
                let base = String(name.dropLast(".weight".count))
                tensors[name] = wq
                tensors[base + ".scales"] = s.asType(.bfloat16)
                tensors[base + ".biases"] = b!.asType(.bfloat16)
            } else {
                tensors[name] = v.asType(.bfloat16)
            }
            eval(Array(tensors.values))
        }
        try save(arrays: tensors, url: dir.appendingPathComponent("model.safetensors"))
        let quant: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
        let json: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": PrepareMemoryBenchmark.hidden, "num_hidden_layers": layers,
            "intermediate_size": PrepareMemoryBenchmark.dense, "num_attention_heads": 4, "head_dim": 64, "global_head_dim": 64,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "vocab_size": 2048, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 128, "final_logit_softcapping": 30.0,
            "enable_moe_block": true, "num_experts": PrepareMemoryBenchmark.experts, "top_k_experts": 8,
            "moe_intermediate_size": PrepareMemoryBenchmark.moeHidden, "layer_types": cfg.layerTypes, "tie_word_embeddings": true,
        ]
        let root: [String: Any] = ["model_type": "gemma4", "text_config": json, "quantization": quant]
        try JSONSerialization.data(withJSONObject: root).write(to: dir.appendingPathComponent("config.json"))
    }

    /// Peak MLX memory of exporting a 4-bit checkpoint to bf16: bounded by about one layer, not by the model.
    ///
    /// The old export dequantized each layer's whole 64-expert bank before cutting it, a shard of five layers at
    /// once, and held every global tensor throughout; its peak was several full-precision layers and grew with
    /// the shard. The bound here is a layer's bf16 bank bytes times a small constant, and the same for a model
    /// of 6 and of 24 layers.
    func testExportPeakIsAboutOneLayerWhateverTheDepth() throws {
        let experts = PrepareMemoryBenchmark.experts, d = PrepareMemoryBenchmark.hidden, h = PrepareMemoryBenchmark.moeHidden
        let bankBF16 = experts * 3 * h * d * 2  // one layer's full-precision expert bank
        var peaks: [Int: Int] = [:]
        for layers in [6, 24] {
            let ckpt = try tempDir("q-big")
            try Self.writeQuantizedCheckpoint(layers: layers, to: ckpt)
            Memory.clearCache()
            let cp = try Gemma4Checkpoint(directory: ckpt)
            let keep = (0 ..< layers).map { _ in Array(0 ..< 8) }
            let out = try tempDir("q-out")
            var o = MergeAndSliceExport.Options()
            o.dtype = .bfloat16
            Memory.peakMemory = Memory.activeMemory
            let base = Memory.activeMemory
            try MergeAndSliceExport.write(checkpoint: cp, keep: keep, adapters: nil, to: out, options: o)
            peaks[layers] = Memory.peakMemory - base
            // Not vacuous: the experts really were expanded and the model is much bigger than the peak.
            XCTAssertEqual(try exportedShape(out, "model.layers.0.experts.gate_up_proj"), [8, 2 * h, d])
        }
        let (small, big) = (peaks[6]!, peaks[24]!)
        print("STREAMING-EXPORT-PEAK bank=\(bankBF16) 6 layers=\(small) 24 layers=\(big)")
        // Exporting 24 layers costs no more than 6 (plus slack for allocator rounding), and a full-precision bank
        // of one layer is the most it may need on top of the embedding table.
        let embedding = 2048 * d * 2
        XCTAssertLessThan(big, bankBF16 + 3 * embedding, "peak \(big) vs one bank \(bankBF16)")
        XCTAssertLessThan(Double(big), Double(small) * 1.25 + 4_000_000, "peak grows with depth: \(small) -> \(big)")
        XCTAssertLessThan(big, 24 * bankBF16 / 8)
    }
}

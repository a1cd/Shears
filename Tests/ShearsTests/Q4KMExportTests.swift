import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Shears

/// A tiny Gemma 4 MoE with the dimensions K-quants need (hidden 256) and enough layers for the
/// Q4_K_M mix to vary, in Hugging Face layout.
struct KQuantFixture {
    var layers = 12
    var experts = 8
    let hidden = 256, heads = 4, headDim = 64, globalHeadDim = 128, kvHeads = 2, globalKV = 1
    /// 384 and 128 are not multiples of 256: the matching `ffn_down` rows must fall back.
    let ff = 384, moeFF = 128
    static let prefix = "model.language_model."

    var layerTypes: [String] { (0 ..< layers).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" } }

    func config() -> [String: Any] {
        let text: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": ff, "num_attention_heads": heads, "head_dim": headDim, "global_head_dim": globalHeadDim,
            "num_key_value_heads": kvHeads, "num_global_key_value_heads": globalKV, "attention_k_eq_v": true,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0, "use_double_wide_mlp": false, "sliding_window": 16,
            "final_logit_softcapping": 30.0, "enable_moe_block": true, "layer_types": layerTypes,
            "tie_word_embeddings": true, "num_experts": experts, "top_k_experts": 2, "moe_intermediate_size": moeFF,
            "vocab_size": GGUFTinyGemma4.vocabCount + 4,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention": ["rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional"],
            ],
        ]
        return ["architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "eos_token_id": [1, 106], "text_config": text]
    }

    func write(to dir: URL, seed: UInt64 = 1) throws {
        MLXRandom.seed(seed)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        func rnd(_ shape: [Int]) -> MLXArray {
            // Weights with some structure: per-row scales and the odd outlier, as trained weights have.
            let n = MLXRandom.normal(shape) * 0.05
            let rowScale = exp(MLXRandom.normal(shape.dropLast() + [1]) * 0.7)
            let outliers = MLX.where(MLXRandom.uniform(0.0 ..< 1.0, shape) .< 0.002, MLXRandom.normal(shape) * 0.6, MLXArray(Float(0)))
            return (n * rowScale + outliers).asType(.bfloat16)
        }
        let p = Self.prefix
        var t: [String: MLXArray] = [:]
        let rows = GGUFTinyGemma4.vocabCount + 4
        t[p + "embed_tokens.weight"] = rnd([rows, hidden])
        t[p + "norm.weight"] = rnd([hidden])
        for i in 0 ..< layers {
            let l = "\(p)layers.\(i)."
            let full = layerTypes[i] == "full_attention"
            let hd = full ? globalHeadDim : headDim
            let kv = (full ? globalKV : kvHeads) * hd
            let q = heads * hd
            for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
                      "post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"] {
                t[l + n + ".weight"] = rnd([hidden])
            }
            t[l + "self_attn.q_proj.weight"] = rnd([q, hidden])
            t[l + "self_attn.k_proj.weight"] = rnd([kv, hidden])
            if !full { t[l + "self_attn.v_proj.weight"] = rnd([kv, hidden]) }
            t[l + "self_attn.k_norm.weight"] = rnd([hd])
            t[l + "self_attn.q_norm.weight"] = rnd([hd])
            t[l + "self_attn.o_proj.weight"] = rnd([hidden, q])
            t[l + "mlp.gate_proj.weight"] = rnd([ff, hidden])
            t[l + "mlp.up_proj.weight"] = rnd([ff, hidden])
            t[l + "mlp.down_proj.weight"] = rnd([hidden, ff])
            t[l + "experts.gate_up_proj"] = rnd([experts, 2 * moeFF, hidden])
            t[l + "experts.down_proj"] = rnd([experts, hidden, moeFF])
            t[l + "router.proj.weight"] = rnd([experts, hidden])
            t[l + "router.scale"] = rnd([hidden])
            t[l + "router.per_expert_scale"] = rnd([experts])
            t[l + "layer_scalar"] = rnd([1])
        }
        try save(arrays: t, url: dir.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: config(), options: [.sortedKeys]).write(to: dir.appendingPathComponent("config.json"))
        try GGUFTinyGemma4.writeTokenizer(to: dir)
    }
}

/// `Q4_K_M` exports against the llama.cpp build that produced the shipped files.
///
/// Needs `llama-quantize` from llama.cpp commit a3a1c47; set `TEST_RUNNER_SHEARS_LLAMA_BIN` to the
/// directory holding it. Everything runs on a tiny scratch model.
final class Q4KMExportTests: XCTestCase {
    var bin: URL!

    override func setUpWithError() throws {
        guard let d = ProcessInfo.processInfo.environment["SHEARS_LLAMA_BIN"] else {
            throw XCTSkip("set TEST_RUNNER_SHEARS_LLAMA_BIN to the directory holding llama-quantize")
        }
        bin = URL(fileURLWithPath: d)
    }

    func tempDir(_ name: String) throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func run(_ args: [String]) throws -> String {
        let p = Process()
        p.executableURL = bin.appendingPathComponent("llama-quantize")
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()
        let out = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        let text = String(decoding: out, as: UTF8.self)
        XCTAssertEqual(p.terminationStatus, 0, "llama-quantize failed:\n\(text.suffix(2000))")
        return text
    }

    func compare(experts: Int, imatrix: Bool) throws {
        let dir = try tempDir("kfix")
        var fx = KQuantFixture()
        fx.experts = experts
        try fx.write(to: dir)
        let f32 = dir.appendingPathComponent("f32.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: f32, type: .f32)

        var args: [String] = []
        var matrix: GGUFImatrix?
        if imatrix {
            let src = try GGUFReader(url: f32)
            var rng = SplitMix64(seed: 5)
            var entries: [String: [Float]] = [:]
            for t in src.tensors where t.shape.count >= 2 && t.name.hasSuffix(".weight") && !t.name.contains("_norm") && !t.name.contains("ffn_gate_inp") {
                if t.name == "token_embd.weight" || t.name == "blk.3.attn_q.weight" { continue }  // some tensors without data
                let n = t.shape.last! * (t.shape.count == 3 ? t.shape[0] : 1)
                entries[t.name] = (0 ..< n).map { _ in
                    Float.random(in: 0.01 ... 1, using: &rng) * (Float.random(in: 0 ... 1, using: &rng) < 0.1 ? 8 : 1)
                }
            }
            let m = GGUFImatrix(entries: entries)
            let file = dir.appendingPathComponent("imatrix.dat")
            try m.write(to: file, callCount: 3)
            matrix = try GGUFImatrix.load(file)
            XCTAssertEqual(matrix!.entries.count, entries.count)
            args += ["--imatrix", file.path]
        }
        let ref = dir.appendingPathComponent("ref.gguf")
        args += [f32.path, ref.path, "Q4_K_M", "2"]
        _ = try run(args)

        let mine = dir.appendingPathComponent("mine.gguf")
        try GGUFExport.gemma4(hfDirectory: dir, output: mine, type: .q4_k_m, importance: matrix)

        let a = try GGUFReader(url: ref), b = try GGUFReader(url: mine)
        XCTAssertEqual(Set(a.tensors.map(\.name)), Set(b.tensors.map(\.name)))
        XCTAssertEqual(a["general.file_type"], b["general.file_type"])
        var typeDiffs: [String] = [], byteDiffs: [String] = []
        var types: [String: Int] = [:]
        for t in a.tensors {
            guard let u = b.tensorInfo(t.name) else { continue }
            types["\(t.type.name)", default: 0] += 1
            if t.type != u.type || t.shape != u.shape { typeDiffs.append("\(t.name): llama \(t.type.name) vs \(u.type.name)"); continue }
            if try a.tensorData(t.name) != b.tensorData(t.name) { byteDiffs.append(t.name) }
        }
        print("Q4KM experts=\(experts) imatrix=\(imatrix) types=\(types)")
        XCTAssertEqual(typeDiffs, [], "per-tensor types differ")
        XCTAssertEqual(byteDiffs, [], "bytes differ for \(byteDiffs.count) tensors")
    }

    func testQ4KMEightExpertsMatchesLlamaQuantize() throws { try compare(experts: 8, imatrix: false) }
    func testQ4KMFourExpertsMatchesLlamaQuantize() throws { try compare(experts: 4, imatrix: false) }
    func testQ4KMEightExpertsWithImatrixMatchesLlamaQuantize() throws { try compare(experts: 8, imatrix: true) }
    func testQ4KMFourExpertsWithImatrixMatchesLlamaQuantize() throws { try compare(experts: 4, imatrix: true) }
}

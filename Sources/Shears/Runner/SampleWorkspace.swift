import Foundation
import MLX
import MLXNN
import MLXRandom

/// The sample workspace behind `shears-cli sample`: a tiny random Gemma 4 mixture-of-experts model (four
/// layers, eight experts, a 293-token byte-fallback tokenizer) in Hugging Face layout, plus a row file
/// to heal it on. It exists so the pipeline can be tried without downloading anything.
/// `Examples/tiny-config.json` is the config written for it.
///
/// The model is random, so it generates gibberish. The weights are seeded, so two runs write the same model.
public enum SampleWorkspace {
    /// Tokenizer vocabulary size: 5 specials, a few letters, 256 byte tokens, 3 added tokens.
    public static let tokenizerVocabCount = 5 + 1 + 26 + 2 + 256 + 3
    /// Embedding rows: the vocabulary plus padding, as real checkpoints pad theirs.
    public static let vocabSize = tokenizerVocabCount + 4
    public static let experts = 8
    public static let topK = 2
    public static let layers = 4
    public static let rowCount = 4

    public static let chatTemplate =
        "{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}"

    /// The text model's `config.json` fields.
    public static func configJSON(layers: Int = SampleWorkspace.layers) -> [String: Any] {
        let types = (0 ..< layers).map { ($0 % 2 == 1 || $0 == layers - 1) ? "full_attention" : "sliding_attention" }
        return [
            "model_type": "gemma4_text",
            "hidden_size": 64, "num_hidden_layers": layers, "intermediate_size": 96,
            "num_attention_heads": 4, "head_dim": 16, "global_head_dim": 32,
            "num_key_value_heads": 2, "num_global_key_value_heads": 1,
            "attention_k_eq_v": true, "rms_norm_eps": 1e-6, "vocab_size": vocabSize,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false, "sliding_window": 8,
            "final_logit_softcapping": 30.0, "enable_moe_block": true,
            "num_experts": experts, "top_k_experts": topK, "moe_intermediate_size": 32,
            "max_position_embeddings": 512,
            "layer_types": types, "tie_word_embeddings": true,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention": [
                    "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional",
                ],
            ],
        ]
    }

    /// A byte-fallback BPE tokenizer in the layout of the real one: special tokens that are also
    /// listed as added tokens at low ids, a base vocabulary, and added tokens above it.
    public static func writeTokenizer(to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var vocab: [String: Int] = [:]
        let specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
        for (i, s) in specials.enumerated() { vocab[s] = i }
        vocab["▁"] = 5
        for (i, c) in "abcdefghijklmnopqrstuvwxyz".enumerated() { vocab[String(c)] = 6 + i }
        vocab["▁a"] = 32
        vocab["ab"] = 33
        for b in 0 ..< 256 { vocab[String(format: "<0x%02X>", b)] = 34 + b }
        let added = ["<|turn>", "<turn|>", "<|channel>"]
        func entry(_ id: Int, _ content: String) -> [String: Any] {
            [
                "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            ]
        }
        var addedTokens = specials.enumerated().map { entry($0.offset, $0.element) }
        for (i, s) in added.enumerated() { addedTokens.append(entry(290 + i, s)) }
        let tokenizerJSON: [String: Any] = [
            "version": "1.0", "added_tokens": addedTokens,
            "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "▁"],
            "model": [
                "type": "BPE", "vocab": vocab, "merges": [["▁", "a"], ["a", "b"]],
                "byte_fallback": true,
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerJSON).write(
            to: dir.appendingPathComponent("tokenizer.json"))
        let tokenizerConfig: [String: Any] = [
            "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>",
            "mask_token": "<mask>", "eot_token": "<turn|>", "soc_token": "<|channel>",
            "extra_special_tokens": ["<|turn>"], "add_eos_token": false,
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(
            to: dir.appendingPathComponent("tokenizer_config.json"))
        try chatTemplate.write(
            to: dir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
    }

    /// Writes `model/` (the random model, float32, every expert kept) and `rows.jsonl` into `directory`.
    /// Pair it with `Examples/tiny-config.json`, which names exactly these paths.
    public static func write(to directory: URL, seed: UInt64 = 7) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: configJSON()))
        try config.validate()
        let model = Gemma4TextModel(config)
        MLXRandom.seed(seed)
        // Perturb every parameter so norms, scales and router scales all matter.
        let p = model.parameters().flattened().map {
            ($0.0, ($0.1 + 0.2 * MLXRandom.normal($0.1.shape)).asType(.float32))
        }
        model.update(parameters: ModuleParameters.unflattened(p))
        model.train(false)
        eval(model)

        let tok = fm.temporaryDirectory.appendingPathComponent("shears-sample-tokenizer-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: tok) }
        try writeTokenizer(to: tok)
        var options = MergeAndSliceExport.Options()
        options.dtype = .float32
        options.sourceDirectory = tok
        let modelDirectory = directory.appendingPathComponent("model")
        try? fm.removeItem(at: modelDirectory)
        try MergeAndSliceExport.write(
            model: model, keep: Array(repeating: Array(0 ..< experts), count: layers), adapters: nil,
            to: modelDirectory, options: options)

        var lines: [String] = []
        for i in 0 ..< rowCount {
            var rng = SplitMix64(seed: UInt64(10 + i))
            let length = 20 + 2 * (i % 2)
            let tokens = (0 ..< length).map { _ in Int(Int32.random(in: 0 ..< Int32(vocabSize), using: &rng)) }
            let record: [String: Any] = [
                "key": "row-\(i)", "tokens": tokens, "answerStart": length - 6,
                "klAnchors": [["position": length - 7, "weight": 5]], "klSampleRange": [length - 6, length - 1],
                "validation": i == rowCount - 1,
            ]
            lines.append(String(data: try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]), encoding: .utf8)!)
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: directory.appendingPathComponent("rows.jsonl"), atomically: true, encoding: .utf8)
    }
}

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A tiny random Gemma 4 MoE (four layers, window 8, softcap, global layers without a value projection) that
/// shares its vocabulary size with the tiny tokenizer of `InferenceTokenizerFixtures`.
enum InferenceFixture {
    static let vocab = 352
    static let endOfTurn = 351
    static let endOfSequence = 1

    static func config(layers: Int = 4) throws -> Gemma4Configuration {
        var json = HealingFixture.configJSON(layers: layers)
        json["vocab_size"] = vocab
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: try JSONSerialization.data(withJSONObject: json))
        try c.validate()
        return c
    }

    /// Every parameter perturbed so norms, scales and router scales all matter.
    static func model(layers: Int = 4, seed: UInt64 = 7) throws -> Gemma4TextModel {
        // Seed before building: the layers draw their initial weights from the global random state.
        MLXRandom.seed(seed)
        let m = Gemma4TextModel(try config(layers: layers))
        let p = m.parameters().flattened().map { ($0.0, $0.1 + 0.2 * MLXRandom.normal($0.1.shape)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        eval(m)
        return m
    }

    static let chatTemplate =
        "{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}"

    /// tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json.
    static func writeTokenizerFiles(to dir: URL, sampling: Bool = false) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try InferenceTokenizerFixtures.tokenizerJSON.write(to: dir.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try InferenceTokenizerFixtures.configJSON.write(to: dir.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        try chatTemplate.write(to: dir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let gen = sampling
            ? #"{"bos_token_id": 2, "do_sample": true, "eos_token_id": [1, 351], "pad_token_id": 0, "temperature": 0.7, "top_k": 20, "top_p": 0.9}"#
            : #"{"bos_token_id": 2, "eos_token_id": [1, 351], "pad_token_id": 0}"#
        try gen.write(to: dir.appendingPathComponent("generation_config.json"), atomically: true, encoding: .utf8)
    }

    static func tempDirectory(_ name: String = "inf") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        return u
    }

    /// A deterministic prompt of `n` ordinary tokens.
    static func prompt(_ n: Int, seed: UInt64 = 3) -> [Int] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< n).map { _ in Int.random(in: 6 ..< 100, using: &rng) }
    }

    static func tokenizer() throws -> BPETokenizer {
        try BPETokenizer(
            tokenizerJSON: Data(InferenceTokenizerFixtures.tokenizerJSON.utf8),
            configuration: Data(InferenceTokenizerFixtures.configJSON.utf8))
    }

    /// An exported HF-layout directory (Shears' own export format) of `model` with `keep`/`adapters` merged.
    static func exportDirectory(
        model: Gemma4TextModel, keep: [[Int]], adapters: ModelAdapters, tokenizerDirectory: URL, in dir: URL
    ) throws {
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        o.sourceDirectory = tokenizerDirectory
        try MergeAndSliceExport.write(model: model, keep: keep, adapters: adapters, to: dir, options: o)
    }
}

extension ShearsTestCase {
    /// Largest value in `logits` minus the value at `token`.
    func gap(_ logits: MLXArray, to token: Int) -> Float {
        (logits.max() - logits[token]).item(Float.self)
    }
}

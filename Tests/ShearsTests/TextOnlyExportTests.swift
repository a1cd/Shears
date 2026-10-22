import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A multimodal source config (`google/gemma-4-26B-A4B-it`, and every mlx-community conversion of it, has a
/// `text_config` and a `vision_config`) exported as a text-only model, so that transformers loads the export as
/// `Gemma4ForCausalLM`. Before ``MergeAndSliceExport/Options/textOnly`` the export copied the multimodal config and
/// named the tensors `model.language_model.*`; transformers then built `Gemma4ForConditionalGeneration`, whose
/// vision tower had no weights in the export and was initialized at random.
final class TextOnlyExportTests: ShearsTestCase {
    func tempDir(_ name: String = "textonly") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    /// The shape of the real source's `config.json`: nested text config, a vision config, a root `eos_token_id`
    /// list that differs from the text config's, special-token ids at the root, and (mlx-community's) a
    /// `quantization` block.
    static func multimodalConfig() -> [String: Any] {
        var text = SampleWorkspace.configJSON()
        text["eos_token_id"] = 1
        text["bos_token_id"] = 2
        text["pad_token_id"] = 0
        return [
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "eos_token_id": [1, 106, 50], "boi_token_id": 255_999, "image_token_id": 258_880,
            "tie_word_embeddings": true, "dtype": "bfloat16", "transformers_version": "5.6.2",
            "audio_config": NSNull(), "quantization": ["group_size": 64, "bits": 4],
            "text_config": text, "vision_config": ["model_type": "gemma4_vision", "hidden_size": 8],
        ]
    }

    /// A random model, a multimodal source directory for it, and the export of `keep` from it.
    func export(textOnly: Bool, flatSource: Bool = false) throws -> (dir: URL, model: Gemma4TextModel, keep: [[Int]]) {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: SampleWorkspace.configJSON()))
        MLXRandom.seed(5)
        let model = Gemma4TextModel(config)
        let p = model.parameters().flattened().map {
            ($0.0, ($0.1 + 0.2 * MLXRandom.normal($0.1.shape)).asType(.float32))
        }
        model.update(parameters: ModuleParameters.unflattened(p))
        model.train(false)
        eval(model)

        let src = try tempDir("src")
        try SampleWorkspace.writeTokenizer(to: src)
        let json = flatSource ? SampleWorkspace.configJSON() : Self.multimodalConfig()
        try JSONSerialization.data(withJSONObject: json).write(to: src.appendingPathComponent("config.json"))
        let keep = (0 ..< SampleWorkspace.layers).map { l in [0, 2, 3, 5 + (l % 3)] }
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        o.sourceDirectory = src
        o.textOnly = textOnly
        try MergeAndSliceExport.write(model: model, keep: keep, adapters: nil, to: dir, options: o)
        return (dir, model, keep)
    }

    func config(_ dir: URL) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as! [String: Any]
    }

    func tensors(in dir: URL) throws -> [String: MLXArray] {
        let idx = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
        var out: [String: MLXArray] = [:]
        for f in Set((idx["weight_map"] as! [String: String]).values) {
            out.merge(try loadArrays(url: dir.appendingPathComponent(f))) { $1 }
        }
        return out
    }

    func testTextOnlyConfigIsFlatAndNamesTheCausalLM() throws {
        let c = try config(try export(textOnly: true).dir)
        XCTAssertNil(c["text_config"])
        XCTAssertNil(c["vision_config"])
        XCTAssertNil(c["audio_config"])
        XCTAssertNil(c["quantization"], "the source's MLX quantization block does not describe the bf16 export")
        XCTAssertEqual(c["architectures"] as? [String], ["Gemma4ForCausalLM"])
        XCTAssertEqual(c["model_type"] as? String, "gemma4_text")
        XCTAssertEqual(c["num_experts"] as? Int, 4)
        XCTAssertEqual(c["top_k_experts"] as? Int, 2)
        XCTAssertEqual(c["hidden_size"] as? Int, 64, "the text config's fields are lifted to the top level")
        XCTAssertEqual(c["eos_token_id"] as? [Int], [1, 106, 50], "the root's end-of-turn list wins over the text config's single id")
        XCTAssertEqual(c["image_token_id"] as? Int, 258_880, "special-token ids the GGUF converter reads stay")
        XCTAssertEqual(c["tie_word_embeddings"] as? Bool, true)
    }

    func testMultimodalLayoutStaysAvailable() throws {
        let c = try config(try export(textOnly: false).dir)
        XCTAssertNotNil(c["text_config"])
        XCTAssertNotNil(c["vision_config"])
        XCTAssertEqual((c["text_config"] as? [String: Any])?["num_experts"] as? Int, 4)
    }

    func testTextOnlyTensorsAreTheMultimodalOnesWithoutLanguageModelInTheName() throws {
        let t = try tensors(in: try export(textOnly: true).dir)
        let m = try tensors(in: try export(textOnly: false).dir)
        XCTAssertFalse(t.keys.contains { $0.contains("language_model") })
        XCTAssertTrue(t.keys.allSatisfy { $0.hasPrefix("model.") || $0 == "lm_head.weight" })
        XCTAssertEqual(
            Set(t.keys), Set(m.keys.map { $0.replacingOccurrences(of: "model.language_model.", with: "model.") }))
        for (k, v) in m { assertClose(t[k.replacingOccurrences(of: "model.language_model.", with: "model.")]!, v, atol: 0, rtol: 0, k) }
    }

    /// Shears' own readers take the text-only layout: the checkpoint loader gives the same logits from it as from
    /// the multimodal layout, and as the source model with the experts masked.
    func testShearsLoadsTheTextOnlyExport() throws {
        let t = try export(textOnly: true)
        let reloaded = try Gemma4Checkpoint(directory: t.dir).loadModel(dtype: .float32)
        reloaded.train(false)
        XCTAssertEqual(reloaded.configuration.numExperts, 4)
        let m = try Gemma4Checkpoint(directory: try export(textOnly: false).dir).loadModel(dtype: .float32)
        m.train(false)
        let tokens = MLXArray((0 ..< 12).map { Int32($0 * 7 % 200) })[.newAxis]
        assertClose(reloaded(tokens), m(tokens), atol: 0, rtol: 0, "text-only vs multimodal layout")
    }

    func testFlatSourceIsUnaffectedByTheOption() throws {
        let a = try config(try export(textOnly: true, flatSource: true).dir)
        let b = try config(try export(textOnly: false, flatSource: true).dir)
        XCTAssertEqual(NSDictionary(dictionary: a), NSDictionary(dictionary: b))
        XCTAssertNil(a["text_config"])
    }

    func testTextOnlyExportGoesThroughGGUFExport() throws {
        let dir = try export(textOnly: true).dir
        let out = dir.appendingPathComponent("pruned.gguf")
        let summary = try GGUFExport.gemma4(hfDirectory: dir, output: out, type: .bf16)
        XCTAssertEqual(summary.expertCount, 4)
        XCTAssertEqual(summary.expertsUsed, 2)
    }

    // MARK: transformers

    /// Loads the export with transformers' `AutoModelForCausalLM` and checks what the Mac test run cannot: that the
    /// class is `Gemma4ForCausalLM`, that no checkpoint key is missing or unexpected (so no weight is random), and
    /// that the logits are finite. Needs a Python with torch and transformers >= 5.5; set `SHEARS_HF_PYTHON` to its
    /// path (the Linux CI notebook covers the tiny sample model without this). The multimodal layout is the negative
    /// control: it must fail the same check, or the check proves nothing.
    func testTransformersLoadsTheTextOnlyExportAsGemma4ForCausalLM() throws {
        guard let python = ProcessInfo.processInfo.environment["SHEARS_HF_PYTHON"], !python.isEmpty else {
            throw XCTSkip("set SHEARS_HF_PYTHON to a Python with torch and transformers >= 5.5 to run this")
        }
        let script = try tempDir("py").appendingPathComponent("load.py")
        try """
        import json, sys, torch
        from transformers import AutoModelForCausalLM
        model, info = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32, output_loading_info=True)
        ids = torch.tensor([[2, 5, 9, 14, 20]])
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        print(json.dumps({"class": type(model).__name__, "missing": sorted(info["missing_keys"]),
                          "unexpected": sorted(info["unexpected_keys"]), "finite": bool(torch.isfinite(logits).all())}))
        """.write(to: script, atomically: true, encoding: .utf8)

        func load(_ dir: URL) throws -> [String: Any] {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: python)
            p.arguments = ["-I", script.path, dir.path]
            let pipe = Pipe()
            p.standardOutput = pipe
            try p.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            XCTAssertEqual(p.terminationStatus, 0, "python failed")
            let last = String(decoding: data, as: UTF8.self).split(separator: "\n").last ?? ""
            return try JSONSerialization.jsonObject(with: Data(last.utf8)) as! [String: Any]
        }

        let good = try load(try export(textOnly: true).dir)
        XCTAssertEqual(good["class"] as? String, "Gemma4ForCausalLM")
        XCTAssertEqual(good["missing"] as? [String], [])
        XCTAssertEqual(good["unexpected"] as? [String], [])
        XCTAssertEqual(good["finite"] as? Bool, true)

        let control = try load(try export(textOnly: false).dir)
        XCTAssertEqual(control["class"] as? String, "Gemma4ForConditionalGeneration")
        XCTAssertFalse((control["missing"] as? [String] ?? []).isEmpty, "the multimodal layout leaves the vision tower random")
    }
}

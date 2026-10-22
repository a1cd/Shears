import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Shears

/// Defaults for absent config keys follow Hugging Face transformers; keys that are present win.
final class Gemma4ConfigurationTests: ShearsTestCase {
    func decode(_ json: [String: Any]) throws -> Gemma4Configuration {
        try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    /// Only what the model needs to be a valid MoE-less text model.
    func minimal(layers: Int) -> [String: Any] {
        ["model_type": "gemma4_text", "num_hidden_layers": layers, "num_kv_shared_layers": 0,
         "hidden_size_per_layer_input": 0]
    }

    func testAbsentRopeParametersGiveAQuarterRotaryFactorOnFullAttention() throws {
        let c = try decode(minimal(layers: 6))
        XCTAssertEqual(c.fullPartialRotaryFactor, 0.25)
        XCTAssertEqual(c.fullRopeType, "proportional")
        XCTAssertEqual(c.fullRopeTheta, 1_000_000)
        XCTAssertEqual(c.slidingRopeTheta, 10_000)
        // Present but without the key: Hugging Face's proportional default of 1.0.
        var j = minimal(layers: 6)
        j["rope_parameters"] = ["full_attention": ["rope_type": "proportional", "rope_theta": 5.0e5]]
        let d = try decode(j)
        XCTAssertEqual(d.fullPartialRotaryFactor, 1.0)
        XCTAssertEqual(d.fullRopeTheta, 5.0e5)
    }

    func testAbsentLayerTypesFollowTheFiveToOnePatternAndEndOnFullAttention() throws {
        let full = { (c: Gemma4Configuration) in c.layerTypes.enumerated().filter { $0.element == "full_attention" }.map(\.offset) }
        XCTAssertEqual(full(try decode(minimal(layers: 12))), [5, 11])
        XCTAssertEqual(full(try decode(minimal(layers: 8))), [5, 7], "the last layer is forced to full attention")
        var j = minimal(layers: 8)
        j["sliding_window_pattern"] = 4
        XCTAssertEqual(full(try decode(j)), [3, 7], "an explicit pattern is the period")
        // Listed layer types are kept, except that the last layer is forced to full attention.
        j = minimal(layers: 3)
        j["layer_types"] = ["full_attention", "sliding_attention", "sliding_attention"]
        XCTAssertEqual(try decode(j).layerTypes, ["full_attention", "sliding_attention", "full_attention"])
    }

    func testAbsentSoftcappingMeansNoCap() throws {
        XCTAssertNil(try decode(minimal(layers: 2)).finalLogitSoftcapping)
        var j = minimal(layers: 2)
        j["final_logit_softcapping"] = 30.0
        XCTAssertEqual(try decode(j).finalLogitSoftcapping, 30)
        j["final_logit_softcapping"] = NSNull()
        XCTAssertNil(try decode(j).finalLogitSoftcapping)
    }

    func testRopeFactorAndTypeAreHonoured() throws {
        var j = minimal(layers: 6)
        j["rope_parameters"] = [
            "sliding_attention": ["rope_type": "default", "rope_theta": 10000.0],
            "full_attention": ["rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1.0e6, "factor": 8.0],
        ]
        let c = try decode(j)
        XCTAssertEqual(c.fullRopeFactor, 8)
        try c.validate()
        // The factor reaches the rotation: the same input rotates differently.
        let base = Gemma4ProportionalRoPE(dims: 64, base: 1.0e6, partialRotaryFactor: 0.25, factor: 1)
        let scaled = Gemma4ProportionalRoPE(dims: 64, base: 1.0e6, partialRotaryFactor: 0.25, factor: c.fullRopeFactor)
        MLXRandom.seed(1)
        let x = MLXRandom.normal([1, 2, 5, 64])
        XCTAssertGreaterThan(abs(base(x, offset: 3) - scaled(x, offset: 3)).max().item(Float.self), 1e-3)
        // And a config round trip keeps it.
        let back = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONEncoder().encode(c))
        XCTAssertEqual(back.fullRopeFactor, 8)
        XCTAssertEqual(back.fullPartialRotaryFactor, 0.25)

        var bad = j
        bad["rope_parameters"] = ["full_attention": ["rope_type": "yarn"]]
        XCTAssertThrowsError(try decode(bad).validate())
        bad["rope_parameters"] = ["sliding_attention": ["rope_type": "proportional"]]
        XCTAssertThrowsError(try decode(bad).validate())
    }

    /// The `text_config` of the published gemma-4-26B-A4B QAT 4-bit conversion's `config.json`, as
    /// shipped (every key it has). It sets every key that has a default, so nothing may fall back:
    /// parsing it must give exactly what it says.
    static let real26BTextConfig = #"""
    {"attention_bias":false,"attention_dropout":0.0,"attention_k_eq_v":true,"bos_token_id":2,"dtype":"bfloat16","enable_moe_block":true,"eos_token_id":1,"final_logit_softcapping":30.0,"global_head_dim":512,"head_dim":256,"hidden_activation":"gelu_pytorch_tanh","hidden_size":2816,"hidden_size_per_layer_input":0,"initializer_range":0.02,"intermediate_size":2112,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],"max_position_embeddings":262144,"model_type":"gemma4_text","moe_intermediate_size":704,"num_attention_heads":16,"num_experts":128,"num_global_key_value_heads":2,"num_hidden_layers":30,"num_key_value_heads":8,"num_kv_shared_layers":0,"pad_token_id":0,"rms_norm_eps":1e-06,"rope_parameters":{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},"sliding_window":1024,"tie_word_embeddings":true,"top_k_experts":8,"use_bidirectional_attention":"vision","use_cache":true,"use_double_wide_mlp":false,"vocab_size":262144,"vocab_size_per_layer_input":262144}
    """#

    func testTheRealGemma426BConfigParsesToExactlyWhatItSays() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("cfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let text = try JSONSerialization.jsonObject(with: Data(Self.real26BTextConfig.utf8))
        let url = dir.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: ["model_type": "gemma4", "text_config": text]).write(to: url)
        let c = try Gemma4Configuration.load(from: url)
        let listed = (text as! [String: Any])["layer_types"] as! [String]
        XCTAssertEqual(c.layerTypes, listed)
        XCTAssertEqual(c.layerTypes.enumerated().filter { $0.element == "full_attention" }.map(\.offset), [5, 11, 17, 23, 29])
        XCTAssertEqual(c.finalLogitSoftcapping, 30)
        XCTAssertEqual(c.fullPartialRotaryFactor, 0.25)
        XCTAssertEqual(c.fullRopeFactor, 1)
        XCTAssertEqual(c.fullRopeTheta, 1_000_000)
        XCTAssertEqual(c.slidingRopeTheta, 10_000)
        XCTAssertEqual(c.fullRopeType, "proportional")
        XCTAssertEqual(c.slidingRopeType, "default")
        XCTAssertEqual(c.slidingWindow, 1024)
        XCTAssertEqual([c.hiddenSize, c.numExperts!, c.topKExperts!, c.moeIntermediateSize!, c.globalHeadDim, c.headDim], [2816, 128, 8, 704, 512, 256])
        XCTAssertEqual([c.numAttentionHeads, c.numKeyValueHeads, c.numGlobalKeyValueHeads!, c.numHiddenLayers, c.intermediateSize], [16, 8, 2, 30, 2112])
        XCTAssertTrue(c.attentionKEqV)
        XCTAssertTrue(c.tieWordEmbeddings)
        XCTAssertFalse(c.useDoubleWideMlp)
    }

    /// The same real config with each defaulted key removed, one at a time: it must still build,
    /// and the keys whose defaults matter report the new (Hugging Face) defaults, not the real values.
    func testRemovingKeysFromTheRealConfigFallsBackToHuggingFaceDefaults() throws {
        let text = try JSONSerialization.jsonObject(with: Data(Self.real26BTextConfig.utf8)) as! [String: Any]
        func parse(removing key: String) throws -> Gemma4Configuration {
            var t = text
            t[key] = nil
            return try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: t))
        }
        XCTAssertNil(try parse(removing: "final_logit_softcapping").finalLogitSoftcapping)
        XCTAssertEqual(try parse(removing: "rope_parameters").fullPartialRotaryFactor, 0.25)
        let noTypes = try parse(removing: "layer_types")
        XCTAssertEqual(noTypes.layerTypes.enumerated().filter { $0.element == "full_attention" }.map(\.offset), [5, 11, 17, 23, 29],
                       "30 layers in a 5:1 pattern end on full attention: the same layers as the real config")
    }
}

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A tiny random Gemma 4 MoE with four layers (full attention at layers 1 and 3, so two
/// blocks), a sliding window shorter than the test sequences, and the vocabulary of the
/// tokenizer the GGUF tests write.
enum HealingFixture {
    static let vocab = GGUFTinyGemma4.vocabCount + 4
    static let experts = 8
    static let topK = 2

    static func configJSON(layers: Int = 4) -> [String: Any] { SampleWorkspace.configJSON(layers: layers) }

    static func config(layers: Int = 4) throws -> Gemma4Configuration {
        let data = try JSONSerialization.data(withJSONObject: configJSON(layers: layers))
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        try c.validate()
        return c
    }

    /// Every parameter perturbed so norms, scales and router scales all matter.
    static func model(layers: Int = 4, seed: UInt64 = 7, dtype: DType = .float32) throws -> Gemma4TextModel {
        let m = Gemma4TextModel(try config(layers: layers))
        MLXRandom.seed(seed)
        let p = m.parameters().flattened().map {
            ($0.0, ($0.1 + 0.2 * MLXRandom.normal($0.1.shape)).asType(dtype))
        }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.train(false)
        eval(m)
        return m
    }

    /// Distinct random kept sets (`count` per layer).
    static func keepSets(layers: Int = 4, count: Int, seed: UInt64 = 5) -> [[Int]] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< layers).map { _ in Array(Array(0 ..< experts).shuffled(using: &rng).prefix(count)).sorted() }
    }

    static func row(length: Int = 24, seed: UInt64 = 3, key: String = "r") -> HealingRow {
        var rng = SplitMix64(seed: seed)
        let tokens = (0 ..< length).map { _ in Int32.random(in: 0 ..< Int32(vocab), using: &rng) }
        return HealingRow(
            key: key, tokens: tokens,
            pinWeights: HealingLoss.rampedWeights(count: length, boundary: length - 6, floor: 0.15, rampLength: 8),
            klAnchors: [KLAnchor(position: length - 7, weight: 5)], klSampleRange: (length - 6) ..< (length - 1))
    }

    static let smallAdapters = AdapterConfiguration(expertRank: 8, denseRank: 8, routerRank: 4)

    static func healer(
        model: Gemma4TextModel, keep: [[Int]], adapters: AdapterConfiguration = smallAdapters,
        configuration: HealingConfiguration = tinyConfiguration(), initial: [String: MLXArray]? = nil
    ) throws -> PruneHealer {
        try PruneHealer(
            model: model, keep: keep, adapterConfiguration: adapters, configuration: configuration,
            initialAdapters: initial)
    }

    static func tinyConfiguration() -> HealingConfiguration {
        var c = HealingConfiguration()
        c.learningRate = 5e-3
        c.learningRateFloor = 5e-4
        c.warmupSteps = 2
        c.endSamplePositions = 3
        c.checkpointEvery = 1000
        c.evaluateEvery = 1000
        c.milestoneEvery = 0
        return c
    }

    /// Replaces every adapter's zero-initialised half with noise, so adapters really act.
    static func randomizeAdapters(_ a: ModelAdapters, scale: Float = 0.05, seed: UInt64 = 11) {
        MLXRandom.seed(seed)
        let p = a.parameters().flattened().map { ($0.0, $0.1 + scale * MLXRandom.normal($0.1.shape)) }
        a.update(parameters: ModuleParameters.unflattened(p))
        eval(a)
    }

    static func copyTensors(_ t: [String: MLXArray]) -> [String: MLXArray] {
        let c = t.mapValues { $0 * 1 }
        eval(Array(c.values))
        return c
    }
}

extension HealingFixture {
    /// Rounds every parameter to a value bfloat16 holds exactly, so an export in the default dtype
    /// changes nothing it did not mean to.
    static func makeBF16Exact(_ m: Gemma4TextModel) {
        let p = m.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16).asType(.float32)) }
        m.update(parameters: ModuleParameters.unflattened(p))
        eval(m)
    }

    /// A copy of an exported model's weights as a float32 model, with `transform` applied to each
    /// tensor by Hugging Face name (nil keeps the tensor): what a converter's output loads as.
    static func shippedModel(
        exported hf: URL, scratch: URL, transform: (String, MLXArray) throws -> MLXArray?
    ) throws -> Gemma4TextModel {
        var tensors: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: hf, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            for (k, v) in try loadArrays(url: file) { tensors[k] = try transform(k, v) ?? v }
        }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        try save(arrays: tensors, url: scratch.appendingPathComponent("model.safetensors"))
        try FileManager.default.copyItem(at: hf.appendingPathComponent("config.json"), to: scratch.appendingPathComponent("config.json"))
        let model = try Gemma4Checkpoint(directory: scratch).loadModel(dtype: .float32)
        model.train(false)
        return model
    }

    /// Whether a Hugging Face tensor name is one quantization-aware healing rounds.
    static func isCovered(_ name: String) -> Bool {
        ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.o_proj.weight",
         "experts.gate_up_proj", "experts.down_proj"].contains { name.hasSuffix($0) }
    }
}

extension HealingFixture {
    /// Makes every expert bank take the sorted gather path whatever the batch size (the default
    /// switches at 64 token-expert rows, which these small sequences never reach, although real
    /// runs always do).
    static func forceSortedGather(_ m: any HealableMoEModel, threshold: Int = 1) {
        for l in m.healableLayers { l.expertBank.sortThreshold = threshold }
    }
}

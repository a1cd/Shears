import Foundation
import MLX
import MLXNN

public enum MoELoaderError: Error, CustomStringConvertible {
    case missingFile(URL)
    case missingKey(String)
    case badConfig(String)

    public var description: String {
        switch self {
        case .missingFile(let u): "missing file: \(u.path)"
        case .missingKey(let k): "missing tensor: \(k)"
        case .badConfig(let s): "bad config.json: \(s)"
        }
    }
}

/// The parts of a Qwen3-MoE `config.json` the MoE layers need.
public struct Qwen3MoEConfiguration: Sendable {
    public var hiddenSize: Int
    public var moeIntermediateSize: Int
    public var numExperts: Int
    public var numExpertsPerToken: Int
    public var normTopKProb: Bool
    public var numHiddenLayers: Int
    public var rmsNormEps: Float
    public var decoderSparseStep: Int
    public var mlpOnlyLayers: Set<Int>
    /// mlx-lm `quantization` block: default group size / bits, plus per-module overrides
    /// keyed by module path (e.g. `model.layers.0.mlp.gate`).
    public var quantization: (groupSize: Int, bits: Int)?
    public var quantizationOverrides: [String: (groupSize: Int, bits: Int)]

    public init(json: [String: Any]) throws {
        func int(_ k: String) throws -> Int {
            guard let v = json[k] as? Int else { throw MoELoaderError.badConfig("no \(k)") }
            return v
        }
        hiddenSize = try int("hidden_size")
        moeIntermediateSize = try int("moe_intermediate_size")
        numExperts = try int("num_experts")
        numExpertsPerToken = try int("num_experts_per_tok")
        numHiddenLayers = try int("num_hidden_layers")
        normTopKProb = json["norm_topk_prob"] as? Bool ?? false
        rmsNormEps = (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        decoderSparseStep = json["decoder_sparse_step"] as? Int ?? 1
        mlpOnlyLayers = Set(json["mlp_only_layers"] as? [Int] ?? [])
        quantizationOverrides = [:]
        if let q = json["quantization"] as? [String: Any],
            let gs = q["group_size"] as? Int, let bits = q["bits"] as? Int
        {
            quantization = (gs, bits)
            for (k, v) in q {
                if let o = v as? [String: Any], let ogs = o["group_size"] as? Int,
                    let ob = o["bits"] as? Int
                {
                    quantizationOverrides[k] = (ogs, ob)
                }
            }
        } else {
            quantization = nil
        }
    }

    public func isMoELayer(_ layer: Int) -> Bool {
        numExperts > 0 && !mlpOnlyLayers.contains(layer) && (layer + 1) % decoderSparseStep == 0
    }

    func quantization(for module: String) throws -> (groupSize: Int, bits: Int) {
        guard let q = quantizationOverrides[module] ?? quantization else {
            throw MoELoaderError.badConfig(
                "\(module) is quantized but config.json has no quantization block")
        }
        return q
    }
}

/// Maps a Qwen3-MoE checkpoint's sparse MLP layers onto ``MoEBlock``.
///
/// Reads three layouts:
/// - Hugging Face: `mlp.experts.{e}.{gate,up,down}_proj.weight`, stacked here into
///   `[E, out, in]`;
/// - mlx-lm (sanitised): `mlp.switch_mlp.{gate,up,down}_proj.weight`;
/// - mlx-lm quantized: the same plus `.scales` / `.biases`, loaded as frozen
///   ``QuantizedSwitchLinear`` (a quantized router is dequantized to full precision, since
///   it is small and is what trains).
///
/// Loading is per layer: only the shards holding that layer's tensors are opened, and
/// MLX reads safetensors lazily, so peak memory is one layer's MLP rather than the model.
/// Attention is not mapped; ``loadResidualLayer(_:)`` gives the MLP half of a decoder
/// layer (`x + mlp(post_attention_layernorm(x))`).
public final class Qwen3MoECheckpoint {
    public let directory: URL
    public let configuration: Qwen3MoEConfiguration
    /// Tensor name to shard file name.
    private let weightMap: [String: String]
    private var shardCache: [String: [String: MLXArray]] = [:]

    public init(directory: URL) throws {
        self.directory = directory
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MoELoaderError.missingFile(configURL)
        }
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL))
        guard let dict = json as? [String: Any] else { throw MoELoaderError.badConfig("not an object") }
        configuration = try Qwen3MoEConfiguration(json: dict)

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let idx = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
            weightMap = idx?["weight_map"] as? [String: String] ?? [:]
        } else {
            let single = directory.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: single.path) else {
                throw MoELoaderError.missingFile(single)
            }
            let keys = try loadArrays(url: single).keys
            weightMap = Dictionary(uniqueKeysWithValues: keys.map { ($0, "model.safetensors") })
        }
    }

    /// Drop open shards. The load methods do this when they return, so a block's tensors
    /// are freed with the block.
    public func releaseShards() { shardCache.removeAll() }

    private func tensor(_ name: String) throws -> MLXArray? {
        guard let file = weightMap[name] else { return nil }
        if shardCache[file] == nil {
            shardCache[file] = try loadArrays(url: directory.appendingPathComponent(file))
        }
        // A fresh object over the same data: `Module.update` rewrites parameter arrays in
        // place, so handing out the cached object would let one loaded block (say, a
        // pruned student) silently rewrite another (its teacher) and the cache itself.
        return shardCache[file]?[name].map { handle(to: $0) }
    }

    private func require(_ name: String) throws -> MLXArray {
        guard let t = try tensor(name) else { throw MoELoaderError.missingKey(name) }
        return t
    }

    /// - Parameter dtype: cast full-precision weights to this (e.g. `.float32` for
    ///   training); quantized weights are left as stored.
    public func loadMoEBlock(layer: Int, dtype: DType? = nil) throws -> MoEBlock {
        defer { releaseShards() }
        let c = configuration
        guard c.isMoELayer(layer) else {
            throw MoELoaderError.badConfig("layer \(layer) is a dense MLP layer")
        }
        let p = "model.layers.\(layer).mlp."
        func cast(_ a: MLXArray) -> MLXArray { dtype.map { a.asType($0) } ?? a }

        // Router.
        var gate = try require(p + "gate.weight")
        if let scales = try tensor(p + "gate.scales") {
            let q = try c.quantization(for: p + "gate")
            gate = dequantized(
                gate, scales: scales, biases: try tensor(p + "gate.biases"),
                groupSize: q.groupSize, bits: q.bits)
        }
        let router = TopKRouter(
            gateWeight: cast(gate), topK: c.numExpertsPerToken, normalizeTopK: c.normTopKProb)

        // Experts.
        func projection(_ name: String, inDims: Int, outDims: Int) throws -> SwitchLinear {
            let mlx = p + "switch_mlp.\(name)."
            if let w = try tensor(mlx + "weight") {
                if let scales = try tensor(mlx + "scales") {
                    let q = try c.quantization(for: p + "switch_mlp.\(name)")
                    return QuantizedSwitchLinear(
                        weight: w, scales: scales, biases: try tensor(mlx + "biases"),
                        inputDims: inDims, outputDims: outDims,
                        groupSize: q.groupSize, bits: q.bits)
                }
                return SwitchLinear(weight: cast(w))
            }
            let stacked = try (0 ..< c.numExperts).map {
                try require(p + "experts.\($0).\(name).weight")
            }
            return SwitchLinear(weight: cast(MLX.stacked(stacked)))
        }

        let d = c.hiddenSize, h = c.moeIntermediateSize
        let experts = SwitchGLU(
            gateProj: try projection("gate_proj", inDims: d, outDims: h),
            upProj: try projection("up_proj", inDims: d, outDims: h),
            downProj: try projection("down_proj", inDims: h, outDims: d))
        let block = MoEBlock(router: router, experts: experts)
        eval(block)
        return block
    }

    /// The MLP half of decoder layer `layer`: its `post_attention_layernorm` plus the
    /// ``MoEBlock``.
    public func loadResidualLayer(_ layer: Int, dtype: DType? = nil) throws -> MoEResidualLayer {
        defer { releaseShards() }
        let moe = try loadMoEBlock(layer: layer, dtype: dtype)
        var w = try require("model.layers.\(layer).post_attention_layernorm.weight")
        if let dtype { w = w.asType(dtype) }
        let norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        norm.update(parameters: ModuleParameters.unflattened([("weight", w)]))
        let layer = MoEResidualLayer(norm: norm, moe: moe)
        eval(layer)
        return layer
    }
}

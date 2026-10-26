import Foundation
import MLX
import MLXNN

/// Writes a Gemma 4 checkpoint as an mlx-community-layout affine quantized one (`weight`, `scales`, `biases`),
/// which the loader reads back as `QuantizedLinear`, `QuantizedSwitchLinear` and `QuantizedEmbedding`.
///
/// This is a way to get a 4-bit model for the inference path to run and measure, from a Shears export in
/// bf16: plain affine rounding of every weight, group size 64, one layer in memory at a time. It is not
/// quantization-aware, not tuned (the router stays in full precision, everything else gets the same bits), and
/// it is not Q4_K_M. The embedding table is quantized too.
public enum CheckpointQuantizer {
    public static func write(
        from source: URL, to directory: URL, bits: Int = 4, groupSize: Int = 64,
        log: (String) -> Void = { _ in }
    ) throws {
        let ckpt = try Gemma4Checkpoint(directory: source)
        guard ckpt.quantization == nil else { throw InferenceError.invalid("the source is already quantized") }
        let cfg = ckpt.configuration
        guard cfg.hiddenSize % groupSize == 0 else {
            throw InferenceError.invalid("the hidden size \(cfg.hiddenSize) is not a multiple of the group size \(groupSize)")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        var weightMap: [String: String] = [:]
        var bytes = 0

        func shard(_ name: String, _ arrays: [String: MLXArray]) throws {
            try save(arrays: arrays, url: directory.appendingPathComponent(name))
            for (k, v) in arrays {
                weightMap[k] = name
                bytes += v.nbytes
            }
        }

        // Embedding table and final norm.
        let globals = try ckpt.loadGlobalTensors()
        guard let table = globals["model.embed_tokens.weight"], let norm = globals["model.norm.weight"] else {
            throw MoELoaderError.missingKey("model.embed_tokens.weight")
        }
        if cfg.tieWordEmbeddings == false { throw InferenceError.invalid("untied embeddings are not supported") }
        let (qw, qs, qb) = MLX.quantized(table, groupSize: groupSize, bits: bits)
        try shard("model-global.safetensors", [
            "language_model.model.embed_tokens.weight": qw,
            "language_model.model.embed_tokens.scales": qs,
            "language_model.model.embed_tokens.biases": qb!,
            "language_model.model.norm.weight": norm,
        ])
        log("embedding and norm written")
        Memory.clearCache()

        for i in 0 ..< cfg.numHiddenLayers {
            let layer = try ckpt.loadDecoderLayer(i)
            quantize(
                model: layer, groupSize: groupSize, bits: bits,
                filter: { path, m in
                    if path.hasPrefix("router") || m is Quantized { return false }
                    if let l = m as? Linear { return l.weight.dim(-1) % groupSize == 0 }
                    if let s = m as? SwitchLinear { return s.inputDims % groupSize == 0 }
                    return false
                },
                apply: { m, g, b, mode in
                    if let s = m as? SwitchLinear { return QuantizedSwitchLinear(s, groupSize: g, bits: b, mode: mode) }
                    return quantizeSingle(layer: m, groupSize: g, bits: b, mode: mode)
                })
            var arrays: [String: MLXArray] = [:]
            for (k, v) in layer.parameters().flattened() { arrays["language_model.model.layers.\(i).\(k)"] = v }
            eval(Array(arrays.values))
            try shard(String(format: "model-layers-%02d.safetensors", i), arrays)
            Memory.clearCache()
            log("layer \(i) written")
        }

        let index: [String: Any] = ["metadata": ["total_size": bytes], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        // The config gains the quantization entry; the tokenizer and template files are copied as regular files.
        let data = try Data(contentsOf: source.appendingPathComponent("config.json"))
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw InferenceError.invalid("config.json is not an object")
        }
        root["quantization"] = ["group_size": groupSize, "bits": bits]
        try Data(ConfigJSON.text(root).utf8).write(to: directory.appendingPathComponent("config.json"))
        for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "special_tokens_map.json"] {
            let from = source.appendingPathComponent(f)
            guard fm.fileExists(atPath: from.path) else { continue }
            let to = directory.appendingPathComponent(f)
            try? fm.removeItem(at: to)
            try Data(contentsOf: from.resolvingSymlinksInPath()).write(to: to)
        }
    }
}

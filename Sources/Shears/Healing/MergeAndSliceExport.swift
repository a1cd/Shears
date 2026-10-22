import Foundation
import MLX
import MLXNN

/// Folds healed adapters into the weights and slices every layer to its kept experts,
/// writing a Hugging Face-layout model that ``GGUFExport/gemma4(hfDirectory:output:type:tokenizer:name:)``
/// and ``Gemma4Checkpoint`` read.
///
/// - Dense LoRA: `W += scale * (a @ b)^T` on the MLP projections and `o_proj`.
/// - Per-expert LoRA: the same fold into the kept expert's gate, up and down weights.
/// - Router correction: `proj.weight += (a @ b)^T`, exact because the correction acts on the
///   router's own normalised input.
/// - Experts, router rows and `per_expert_scale` are sliced to the kept ids (ascending), and
///   `gate_up_proj` is `[gate; up]` along axis 1.
///
/// A model run with every layer's routing restricted to its kept set and the adapters on is
/// exactly (to float rounding) the sliced model this writes: that is the property the export
/// tests check. `top_k_experts` is written as `min(native top-k, kept count)`.
///
/// Hugging Face configs carry one expert count, so every layer must keep the same number.
public enum MergeAndSliceExport {
    public struct Options {
        public var dtype: DType = .bfloat16
        /// Merged layers per output file. The default writes every layer to its own file as soon as it is merged,
        /// which is what bounds the export's memory to one layer; a larger number holds that many merged
        /// (kept-expert) layers at a time.
        public var layersPerShard = 1
        /// The tensor name prefix. `nil` follows the config that is written: `model.language_model.` when the
        /// config is multimodal (the source `config.json` has a `text_config`, as `google/gemma-4-26B-A4B-it` does
        /// and ``textOnly`` is off), and `model.` for a text-only config, which is what transformers'
        /// `Gemma4ForCausalLM` expects.
        public var prefix: String?
        /// Write a text-only model when the source `config.json` is multimodal: a flat `gemma4_text` config
        /// (`architectures: ["Gemma4ForCausalLM"]`, no `vision_config`) and `model.` tensor names. transformers then
        /// loads the export as `Gemma4ForCausalLM`. The multimodal layout (`false`) makes it build
        /// `Gemma4ForConditionalGeneration`, whose vision tower has no weights in the export and is initialized at
        /// random: about 0.6 billion parameters of noise that no text prompt touches, and a load report full of
        /// MISSING keys. Shears' own readers (checkpoint, inference, GGUF) take either layout. A source whose config
        /// is already flat is unaffected.
        public var textOnly = false
        /// A directory whose `config.json` and tokenizer files are carried over.
        public var sourceDirectory: URL?
        /// Progress lines: one per merged layer, with the MLX memory peak so far.
        public var log: ((String) -> Void)?
        public init() {}
    }

    /// The module's parameters with every quantized weight replaced by its dequantized
    /// full-precision weight and the quantization side tensors (`scales`, `biases`) removed, so
    /// nothing downstream can copy packed integers or stray scales. Expert banks are left to
    /// the expert path (which dequantizes after slicing, so only kept experts are expanded).
    static func fullPrecisionParameters(_ module: Module) -> [String: MLXArray] {
        var params = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        for (path, m) in module.namedModules() {
            let prefix = path.isEmpty ? "" : path + "."
            switch m {
            case let q as QuantizedLinear:
                params[prefix + "weight"] = dequantized(
                    q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
                params[prefix + "scales"] = nil
                params[prefix + "biases"] = nil
            case let q as QuantizedEmbedding:
                params[prefix + "weight"] = dequantized(
                    q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
                params[prefix + "scales"] = nil
                params[prefix + "biases"] = nil
            default: break
            }
        }
        return params
    }

    /// One layer's tensors under Hugging Face names, adapters merged and experts sliced.
    public static func mergedLayerTensors(
        _ layer: Gemma4MoEDecoderLayer, index: Int, set: ExpertSet, adapters: LayerAdapters?,
        scale: Float, dtype: DType, prefix: String
    ) -> [String: MLXArray] {
        let pre = "\(prefix)layers.\(index)."
        var out: [String: MLXArray] = [:]
        let params = fullPrecisionParameters(layer)
        let keep = MLXArray(set.keep.map(Int32.init))

        func fold(_ w: MLXArray, _ pair: LoRAPair?) -> MLXArray {
            var m = w.asType(.float32)
            if let pair { m = m + pair.scale * matmul(pair.a, pair.b).T }
            // Never rounded here, whatever the format a model was healed against: the converter
            // that ships it (GGUF, MLX) quantizes this very tensor, and a pre-rounded one would
            // not re-quantize to itself once cast to a narrow dtype. See ``QATConfiguration``.
            return m.asType(dtype)
        }
        let dense: [(String, DenseSite?)] = [
            ("mlp.gate_proj.weight", .mlpGate), ("mlp.up_proj.weight", .mlpUp),
            ("mlp.down_proj.weight", .mlpDown), ("self_attn.o_proj.weight", .attentionOutput),
        ]
        var handled = Set<String>()
        for (name, site) in dense {
            out[pre + name] = fold(params[name]!, site.flatMap { adapters?.dense($0) })
            handled.insert(name)
        }

        let bank = layer.experts.switchGLU
        // The kept experts only are ever expanded: a quantized bank's packed weights, scales and biases are cut to
        // the kept rows first and dequantized after (dequantizing is per row, so this is the same numbers), so a
        // layer of 128 experts never exists in full precision.
        func expert(_ p: SwitchLinear, _ a: MLXArray?, _ b: MLXArray?) -> MLXArray {
            var w: MLXArray
            if let q = p as? QuantizedSwitchLinear {
                w = dequantized(
                    q.weight.take(keep, axis: 0), scales: q.scales.take(keep, axis: 0),
                    biases: q.biases?.take(keep, axis: 0), groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            } else {
                w = p.weight.take(keep, axis: 0)
            }
            w = w.asType(.float32)
            if let a, let b, let lora = adapters?.experts {
                w = w + lora.scale * matmul(a, b).swappedAxes(-1, -2)
            }
            return w.asType(dtype)
        }
        let lora = adapters?.experts
        // A shared expert LoRA is folded as the per-expert LoRA it equals.
        let pe = lora?.perExpertTensors
        let gate = expert(bank.gateProj, pe?.gateA, pe?.gateB)
        let up = expert(bank.upProj, pe?.upA, pe?.upB)
        out[pre + "experts.gate_up_proj"] = concatenated([gate, up], axis: 1)
        out[pre + "experts.down_proj"] = expert(bank.downProj, pe?.downA, pe?.downB)
        for k in params.keys where k.hasPrefix("experts.") { handled.insert(k) }

        var proj = params["router.proj.weight"]!.asType(.float32)
        if let t = adapters?.router { proj = proj + matmul(t.a, t.b).T }
        out[pre + "router.proj.weight"] = proj.take(keep, axis: 0).asType(dtype)
        out[pre + "router.per_expert_scale"] = layer.router.perExpertScale.take(keep, axis: 0).asType(dtype)
        handled.formUnion(["router.proj.weight", "router.per_expert_scale"])

        for (k, v) in params where !handled.contains(k) { out[pre + k] = v.asType(dtype) }
        return out
    }

    /// The adapters of model layer `index`, given the layer (all its experts) they are merged into and its
    /// expert set; `nil` merges nothing into that layer.
    public typealias LayerAdaptersProvider = (_ index: Int, _ layer: Gemma4MoEDecoderLayer, _ set: ExpertSet) throws -> LayerAdapters?

    /// Writes the merged, sliced model.
    ///
    /// The export streams: the layers come from `source` **one at a time**, each is merged and sliced and then
    /// evaluated, and its source layer is released before the next is loaded; the merged layers are written
    /// ``Options/layersPerShard`` at a time (default one). The peak is therefore one source layer plus one shard of
    /// merged (kept-expert) layers, whatever the model's depth, and never the whole model.
    public static func write(
        source: LayerSource, globals: [String: MLXArray], configuration: Gemma4Configuration,
        keep: [[Int]], adapters: ModelAdapters?, scale: Float? = nil, to directory: URL,
        options: Options = .init()
    ) throws {
        try write(
            source: source, globals: { globals }, configuration: configuration, keep: keep,
            adapterConfiguration: adapters?.configuration,
            layerAdapters: adapters.map { a in { i, _, _ in a.layers[i - a.firstLayer] } },
            scale: scale, to: directory, options: options)
    }

    /// As ``write(source:globals:configuration:keep:adapters:scale:to:options:)``, with the adapters supplied one
    /// layer at a time (`layerAdapters`; `adapterConfiguration` is their configuration) and the global tensors
    /// (embedding table, final norm, head) produced by a closure, written first and released before any layer is
    /// loaded. A caller reading the adapters from a file can then build each layer's from that file's tensors
    /// and hold none of the others.
    public static func write(
        source: LayerSource, globals: () throws -> [String: MLXArray], configuration: Gemma4Configuration,
        keep: [[Int]], adapterConfiguration: AdapterConfiguration?, layerAdapters: LayerAdaptersProvider?,
        scale: Float? = nil, to directory: URL, options: Options = .init()
    ) throws {
        let n = configuration.numHiddenLayers
        guard keep.count == n, let kept = keep.first?.count, keep.allSatisfy({ $0.count == kept }) else {
            throw HealingError.invalid("every layer must keep the same number of experts")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let scale = scale ?? adapterConfiguration?.scale ?? 2
        // The healed forward pass rounded the merged weight cast to the QAT export dtype; a file in
        // another dtype is a different tensor and would be quantized to different codes.
        if let qat = adapterConfiguration?.qat, qat.exportDType != options.dtype {
            throw HealingError.invalid(
                "the model was healed against weights exported as \(qat.exportDType) but this export writes \(options.dtype)")
        }
        let p = options.prefix ?? layoutPrefix(options.sourceDirectory, textOnly: options.textOnly)
        var weightMap: [String: String] = [:]
        var bytes = 0
        func flush(_ name: String, _ tensors: [String: MLXArray]) throws {
            eval(Array(tensors.values))
            try MLX.save(arrays: tensors, metadata: ["format": "pt"], url: directory.appendingPathComponent(name))
            for (k, v) in tensors { weightMap[k] = name; bytes += v.nbytes }
        }
        do {
            // Written and released before the first layer is read: the embedding table is the largest single tensor.
            var g: [String: MLXArray] = [:]
            for (k, v) in try globals() {
                let hf = k.hasPrefix("model.") ? p + k.dropFirst("model.".count) : k
                g[hf] = v.asType(options.dtype)
            }
            try flush("model-global.safetensors", g)
        }
        Memory.clearCache()
        var formats: [String: QuantFormat] = [:]
        var group: [String: MLXArray] = [:]
        var inGroup = 0, shard = 0
        for i in 0 ..< n {
            do {
                guard let l = try source.loadLayers(i ... i)[0] as? Gemma4MoEDecoderLayer else {
                    throw HealingError.invalid("only Gemma 4 MoE layers can be exported")
                }
                let set = ExpertSet(keep: keep[i], numExperts: l.numExperts)
                let la = try layerAdapters?(i, l, set)
                let merged = mergedLayerTensors(
                    l, index: i, set: set, adapters: la, scale: scale, dtype: options.dtype, prefix: p)
                // Evaluated now, so nothing lazy still points at the source layer when it is released.
                eval(Array(merged.values))
                group.merge(merged) { $1 }
                if let la { formats.merge(ExportRecord.formats(of: la, layerIndex: i)) { $1 } }
            }
            Memory.clearCache()
            options.log?("layer \(i + 1)/\(n) merged, MLX peak so far \(String(format: "%.2f", Double(Memory.peakMemory) / 1_073_741_824)) GB")
            inGroup += 1
            if inGroup >= options.layersPerShard || i == n - 1 {
                try flush(String(format: "model-layers-%02d.safetensors", shard), group)
                group = [:]
                inGroup = 0
                shard += 1
                Memory.clearCache()
            }
        }
        try ExportRecord(dtype: options.dtype, formats: formats).write(to: directory)
        let index: [String: Any] = ["metadata": ["total_size": bytes], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try writeConfiguration(configuration, kept: kept, to: directory, options: options)
    }

    /// In-memory model convenience.
    public static func write(
        model: Gemma4TextModel, keep: [[Int]], adapters: ModelAdapters?, to directory: URL,
        options: Options = .init()
    ) throws {
        func full(_ m: Module) -> MLXArray {
            fullPrecisionParameters(m)["weight"]!
        }
        var globals = [
            "model.embed_tokens.weight": full(model.model.embedTokens),
            "model.norm.weight": model.model.norm.weight,
        ]
        if let head = model.lmHead { globals["lm_head.weight"] = full(head) }
        try write(
            source: ModelLayerSource(model: model), globals: globals,
            configuration: model.configuration, keep: keep, adapters: adapters, to: directory,
            options: options)
    }

    /// Streaming convenience: layers and globals read from a checkpoint directory.
    public static func write(
        checkpoint: Gemma4Checkpoint, loadDType: DType? = nil, keep: [[Int]], adapters: ModelAdapters?,
        to directory: URL, options: Options = .init()
    ) throws {
        try write(
            source: Gemma4LayerSource(checkpoint: checkpoint, dtype: loadDType),
            globals: { try checkpoint.loadGlobalTensors(dtype: loadDType) },
            configuration: checkpoint.configuration, keep: keep, adapterConfiguration: adapters?.configuration,
            layerAdapters: adapters.map { a in { i, _, _ in a.layers[i - a.firstLayer] } }, to: directory,
            options: options)
    }

    /// `model.language_model.` for a multimodal source config (unless the export is ``Options/textOnly``),
    /// `model.` for a text-only or absent one.
    static func layoutPrefix(_ sourceDirectory: URL?, textOnly: Bool = false) -> String {
        guard !textOnly, let src = sourceDirectory,
            let data = try? Data(contentsOf: src.appendingPathComponent("config.json")),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            json["text_config"] != nil
        else { return "model." }
        return "model.language_model."
    }

    /// The flat config transformers loads as `Gemma4ForCausalLM`: the multimodal config's `text_config` lifted to the
    /// top, plus the root's settings that still mean something without a vision tower. The vision and audio configs
    /// are dropped. The root's `eos_token_id` (the list that includes the end-of-turn token) wins over the text
    /// config's single one, and the root's `*_token_id` / `*_token_index` entries stay, because the GGUF converter
    /// reads the special-token ids from the config.
    static func textOnlyConfig(root: [String: Any], text: [String: Any]) -> [String: Any] {
        var flat = text
        flat["architectures"] = ["Gemma4ForCausalLM"]
        flat["model_type"] = flat["model_type"] ?? "gemma4_text"
        for (k, v) in root where k != "text_config" && k != "vision_config" && k != "audio_config" {
            if k == "eos_token_id" || k == "dtype" || k == "transformers_version" || k == "tie_word_embeddings"
                || ((k.hasSuffix("_token_id") || k.hasSuffix("_token_index")) && flat[k] == nil)
            {
                if !(v is NSNull) { flat[k] = v }
            }
        }
        return flat
    }

    private static func writeConfiguration(
        _ c: Gemma4Configuration, kept: Int, to directory: URL, options: Options
    ) throws {
        let topK = min(c.topKExperts ?? kept, kept)
        let fm = FileManager.default
        var root: [String: Any]
        if let src = options.sourceDirectory,
            let data = try? Data(contentsOf: src.appendingPathComponent("config.json")),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            root = json
            func patch(_ d: inout [String: Any]) {
                d["num_experts"] = kept
                d["top_k_experts"] = topK
            }
            if var text = root["text_config"] as? [String: Any] {
                patch(&text)
                root["text_config"] = text
                if options.textOnly { root = textOnlyConfig(root: root, text: text) }
            } else {
                patch(&root)
            }
            root.removeValue(forKey: "quantization")
            root.removeValue(forKey: "quantization_config")
        } else {
            var cfg = c
            cfg.numExperts = kept
            cfg.topKExperts = topK
            let data = try JSONEncoder().encode(cfg)
            root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        }
        try Data(ConfigJSON.text(root).utf8).write(to: directory.appendingPathComponent("config.json"))
        if let src = options.sourceDirectory {
            for f in [
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "generation_config.json", "special_tokens_map.json",
            ] {
                let from = src.appendingPathComponent(f)
                let to = directory.appendingPathComponent(f)
                if fm.fileExists(atPath: from.path) {
                    try? fm.removeItem(at: to)
                    try fm.copyItem(at: from, to: to)
                }
            }
        }
    }
}

/// Writes `config.json` with floats kept as floats. `JSONSerialization` writes `30.0` as `30`, and
/// Hugging Face transformers validates `final_logit_softcapping` (and the other float settings) as a
/// float, so a config with `30` fails to load. The settings in `floatKeys` are always written with a
/// decimal point, even when they arrive as integers (the `Codable` path encodes `30.0` as `30`).
enum ConfigJSON {
    static let floatKeys: Set<String> = [
        "final_logit_softcapping", "attn_logit_softcapping", "rms_norm_eps", "rope_theta",
        "partial_rotary_factor", "factor", "initializer_range",
    ]

    static func text(_ root: [String: Any]) -> String {
        render(root, key: nil, indent: 0) + "\n"
    }

    private static func fragment(_ value: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]) else {
            return "null"
        }
        return String(decoding: data, as: UTF8.self)
    }

    private static func render(_ value: Any, key: String?, indent: Int) -> String {
        let pad = String(repeating: " ", count: indent)
        let inner = String(repeating: " ", count: indent + 2)
        switch value {
        case let d as [String: Any]:
            if d.isEmpty { return "{}" }
            let items = d.keys.sorted().map { k in
                "\(inner)\(fragment(k)): \(render(d[k]!, key: k, indent: indent + 2))"
            }
            return "{\n" + items.joined(separator: ",\n") + "\n\(pad)}"
        case let a as [Any]:
            if a.isEmpty { return "[]" }
            let items = a.map { inner + render($0, key: key, indent: indent + 2) }
            return "[\n" + items.joined(separator: ",\n") + "\n\(pad)]"
        case let n as NSNumber:
            if n.isBoolean { return n.boolValue ? "true" : "false" }
            if n.isFloatingPoint || (key.map(floatKeys.contains) ?? false) { return float(n.doubleValue) }
            return n.stringValue
        case is NSNull:
            return "null"
        default:
            return fragment(value)
        }
    }

    /// The shortest text that reads back as the same value, always with a decimal point or exponent.
    private static func float(_ d: Double) -> String {
        guard d.isFinite else { return "null" }
        let f = Float(d)
        let text = Double(f) == d ? f.description : d.description  // 1e-06, not 9.999999974752427e-07
        return text
    }
}

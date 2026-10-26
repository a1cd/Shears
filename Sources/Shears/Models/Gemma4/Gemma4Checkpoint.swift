// Key remapping ported from ml-explore/mlx-swift-lm, MIT License, Copyright (c) 2024 ml-explore
// (`sanitize` in Libraries/MLXLLM/Models/Gemma4Text.swift and Gemma4.swift).
// See THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

/// Loads a Gemma 4 text model from a Hugging Face safetensors directory, whole or one
/// decoder layer at a time, and maps a model back to Hugging Face tensor names.
///
/// Accepted tensor layouts:
/// - Hugging Face multimodal (`model.language_model.…`, as `google/gemma-4-26B-A4B-it`
///   ships) and text-only (`model.…`). Experts are `experts.gate_up_proj` `[E, 2H, D]`
///   (gate rows first) and `experts.down_proj` `[E, D, H]`, with no `.weight` suffix;
///   `gate_up_proj` is split into `gate_proj` and `up_proj` here.
/// - mlx-lm sanitised (`language_model.model.…`, `experts.switch_glu.*.weight`), full
///   precision only.
///
/// Quantized checkpoints in the mlx-community layout are supported: a module with
/// `weight`/`scales`/`biases` tensors loads as `QuantizedLinear`, `QuantizedEmbedding` or
/// (experts) ``QuantizedSwitchLinear``, with group size and bits from the config's
/// `quantization` entry, its per-module overrides, or failing that inferred from the shapes.
/// Quantized tensors are never cast; `dtype` applies to floating-point tensors only.
///
/// Vision, audio and multimodal-projector tensors are skipped.
///
/// Loading is lazy per layer as in ``Qwen3MoECheckpoint``: only the shards holding the
/// requested tensors are opened, and they are dropped when the load returns.
public final class Gemma4Checkpoint {
    public let directory: URL
    public let configuration: Gemma4Configuration
    /// Module key (mlx-lm sanitised, e.g. `model.layers.3.router.proj.weight`) to the
    /// source tensor name and shard.
    private var sources: [String: [(name: String, file: String)]] = [:]
    private var shardCache: [String: [String: MLXArray]] = [:]
    /// The config's `quantization` entry, if any.
    public let quantization: Gemma4Quantization?

    public init(directory: URL) throws {
        self.directory = directory
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MoELoaderError.missingFile(configURL)
        }
        configuration = try Gemma4Configuration.load(from: configURL)
        quantization = try Gemma4Quantization.load(from: configURL)

        let weightMap: [String: String]
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
        for (name, file) in weightMap {
            guard let key = Self.moduleKey(forSourceName: name) else { continue }
            sources[key, default: []].append((name, file))
        }
    }

    // MARK: Name mapping

    /// Strips the checkpoint's prefix down to the module layout, or returns `nil` for a
    /// tensor the text model doesn't use. Expert tensors keep their source suffix
    /// (`experts.gate_up_proj`); ``sanitize(_:)`` splits them.
    static func moduleKey(forSourceName name: String) -> String? {
        var k = name
        for (from, to) in [
            ("model.language_model.", "model."),
            ("language_model.model.", "model."),
            ("language_model.lm_head.", "lm_head."),
        ] where k.hasPrefix(from) {
            k = to + k.dropFirst(from.count)
            break
        }
        guard k.hasPrefix("model.") || k.hasPrefix("lm_head.") else { return nil }
        let body = k.dropFirst(k.hasPrefix("model.") ? 6 : 0)
        for skip in [
            "vision_tower", "multi_modal_projector", "audio_tower", "embed_audio",
            "embed_vision", "vision_embedder",
        ] where body.hasPrefix(skip) {
            return nil
        }
        for skip in ["self_attn.rotary_emb", "input_max", "input_min", "output_max", "output_min"]
        where k.contains(skip) {
            return nil
        }
        return k
    }

    /// Hugging Face expert tensors to ``SwitchGLU`` keys:
    /// `experts.gate_up_proj` → `experts.switch_glu.{gate,up}_proj.weight` (split on
    /// axis -2) and `experts.down_proj` → `experts.switch_glu.down_proj.weight`.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.hasSuffix(".experts.down_proj") {
                out[k.replacingOccurrences(
                    of: ".experts.down_proj", with: ".experts.switch_glu.down_proj.weight")] = v
            } else if k.hasSuffix(".experts.gate_up_proj") {
                let mid = v.dim(-2) / 2
                out[k.replacingOccurrences(
                    of: ".experts.gate_up_proj", with: ".experts.switch_glu.gate_proj.weight")] =
                    v[.ellipsis, ..<mid, 0...]
                out[k.replacingOccurrences(
                    of: ".experts.gate_up_proj", with: ".experts.switch_glu.up_proj.weight")] =
                    v[.ellipsis, mid..., 0...]
            } else {
                out[k] = v
            }
        }
        return out
    }

    /// A model's parameters under Hugging Face names: the inverse of loading. Gate and up
    /// projections are fused back into `experts.gate_up_proj`, expert tensors lose their
    /// `.weight` suffix, and every backbone key gets `prefix` (`model.language_model.` for
    /// the multimodal layout, `model.` for text-only). An untied `lm_head.weight` is
    /// written as is.
    ///
    /// A pruned model writes its pruned expert count; the matching `config.json` needs
    /// `num_experts` updated to it, which Hugging Face can only express when every layer
    /// keeps the same number.
    public static func huggingFaceWeights(
        of model: Gemma4TextModel, prefix: String = "model.language_model."
    ) -> [String: MLXArray] {
        huggingFaceWeights(model.parameters().flattened(), prefix: prefix)
    }

    /// One decoder layer's parameters under Hugging Face names, as layer `index`.
    public static func huggingFaceWeights(
        of layer: Gemma4DecoderLayer, index: Int, prefix: String = "model.language_model."
    ) -> [String: MLXArray] {
        huggingFaceWeights(
            layer.parameters().flattened().map { ("model.layers.\(index).\($0.0)", $0.1) },
            prefix: prefix)
    }

    static func huggingFaceWeights(_ flat: [(String, MLXArray)], prefix: String) -> [String: MLXArray] {
        var params = Dictionary(uniqueKeysWithValues: flat)
        var out: [String: MLXArray] = [:]
        for key in params.keys.sorted() where key.hasSuffix(".experts.switch_glu.gate_proj.weight") {
            let base = String(key.dropLast(".switch_glu.gate_proj.weight".count))
            let upKey = base + ".switch_glu.up_proj.weight"
            let downKey = base + ".switch_glu.down_proj.weight"
            guard let gate = params[key], let up = params[upKey], let down = params[downKey] else {
                continue
            }
            out[base + ".gate_up_proj"] = concatenated([gate, up], axis: -2)
            out[base + ".down_proj"] = down
            params[key] = nil
            params[upKey] = nil
            params[downKey] = nil
        }
        for (k, v) in params { out[k] = v }
        var renamed: [String: MLXArray] = [:]
        for (k, v) in out {
            if k.hasPrefix("model.") {
                renamed[prefix + k.dropFirst("model.".count)] = v
            } else {
                renamed[k] = v
            }
        }
        return renamed
    }

    // MARK: Loading

    /// Drop open shards. The load methods do this when they return.
    public func releaseShards() { shardCache.removeAll() }

    private var headerCache: [String: [String: MappedSafetensors.Entry]] = [:]
    private func header(_ file: String) throws -> [String: MappedSafetensors.Entry] {
        if let h = headerCache[file] { return h }
        let h = try MappedSafetensors.index(of: directory.appendingPathComponent(file))
        headerCache[file] = h
        return h
    }

    /// Asks the system to start reading the tensors of layers `range` into the page cache, so a
    /// following ``loadDecoderLayer(_:dtype:mapped:)`` finds them there. Returns at once; a failure
    /// only means the later load does its own reading.
    public func prefetchLayers(_ range: ClosedRange<Int>) {
        var byFile: [String: [MappedSafetensors.Entry]] = [:]
        for i in range {
            let prefix = "model.layers.\(i)."
            for (key, srcs) in sources where key.hasPrefix(prefix) {
                guard let src = srcs.first, let e = try? header(src.file)[src.name] else { continue }
                byFile[src.file, default: []].append(e)
            }
        }
        for (file, entries) in byFile {
            let fd = open(directory.appendingPathComponent(file).path, O_RDONLY)
            guard fd >= 0 else { continue }
            MappedSafetensors.prefetch(entries, fd: fd)
            close(fd)
        }
    }

    private func mappedTensors(withPrefix prefix: String) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var fds: [String: Int32] = [:]
        defer { for fd in fds.values { close(fd) } }
        for (key, srcs) in sources where key.hasPrefix(prefix) {
            guard let src = srcs.first else { continue }
            guard let e = try header(src.file)[src.name] else { throw MoELoaderError.missingKey(src.name) }
            let url = directory.appendingPathComponent(src.file)
            if fds[src.file] == nil {
                let fd = open(url.path, O_RDONLY)
                guard fd >= 0 else { throw MoELoaderError.missingFile(url) }
                fds[src.file] = fd
            }
            out[key] = try MappedSafetensors.tensor(e, fd: fds[src.file]!, in: url)
        }
        return out
    }

    private func tensors(withPrefix prefix: String, excluding: (String) -> Bool = { _ in false }) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, srcs) in sources where key.hasPrefix(prefix) && !excluding(key) {
            guard let src = srcs.first else { continue }
            if shardCache[src.file] == nil {
                shardCache[src.file] = try loadArrays(url: directory.appendingPathComponent(src.file))
            }
            guard let t = shardCache[src.file]?[src.name] else {
                throw MoELoaderError.missingKey(src.name)
            }
            // A fresh object over the same data, so loaded modules never share arrays
            // (``Module/update(parameters:)`` rewrites them in place).
            out[key] = handle(to: t)
        }
        return out
    }

    private func apply(
        _ weights: [String: MLXArray], to module: Module, stripping prefix: String, dtype: DType?,
        verify: Module.VerifyUpdate = .all
    ) throws {
        var local: [String: MLXArray] = [:]
        for (k, v) in Self.sanitize(weights) { local[String(k.dropFirst(prefix.count))] = v }
        try installQuantizedModules(local, in: module, absolutePrefix: prefix)
        let flat = local.map { (k, v) -> (String, MLXArray) in
            // Packed integer weights must never be cast; scales and biases are floating point.
            (k, (dtype != nil && v.dtype.isFloatingPoint) ? v.asType(dtype!) : v)
        }
        try module.update(parameters: ModuleParameters.unflattened(flat), verify: verify)
        eval(module)
    }

    /// Swaps each module that has `scales` among `local` for its quantized counterpart, built
    /// from the loaded tensors. `local` keys are relative to `module`; `absolutePrefix` is what
    /// to prepend to get the checkpoint's module path (for looking up per-module overrides).
    private func installQuantizedModules(
        _ local: [String: MLXArray], in module: Module, absolutePrefix: String
    ) throws {
        let scaleKeys = local.keys.filter { $0.hasSuffix(".scales") }
        guard !scaleKeys.isEmpty else { return }
        let named = Dictionary(uniqueKeysWithValues: module.namedModules().map { ($0.0, $0.1) })
        for sk in scaleKeys {
            let path = String(sk.dropLast(".scales".count))
            guard let weight = local[path + ".weight"], let scales = local[sk] else {
                throw MoELoaderError.missingKey(path + ".weight")
            }
            let biases = local[path + ".biases"]
            // The unquantized module the checkpoint replaces holds the logical input width.
            let logicalIn: Int
            switch named[path] {
            case let sl as SwitchLinear: logicalIn = sl.weight.dim(-1)
            case let e as Embedding: logicalIn = e.weight.dim(-1)
            case let l as Linear: logicalIn = l.weight.dim(-1)
            default: throw MoELoaderError.badConfig("no quantizable module at \(absolutePrefix + path)")
            }
            let (groupSize, bits, mode) = try quantizationParameters(
                forModule: absolutePrefix + path, weight: weight, scales: scales, logicalIn: logicalIn)
            let replacement: Module
            switch named[path] {
            case is SwitchLinear:
                let inDims = logicalIn
                replacement = QuantizedSwitchLinear(
                    weight: weight, scales: scales, biases: biases, bias: local[path + ".bias"],
                    inputDims: inDims, outputDims: weight.dim(1), groupSize: groupSize, bits: bits,
                    mode: mode)
            case is Embedding:
                replacement = QuantizedEmbedding(
                    weight: weight, scales: scales, biases: biases, groupSize: groupSize,
                    bits: bits, mode: mode)
            case is Linear:
                replacement = QuantizedLinear(
                    weight: weight, bias: local[path + ".bias"], scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode)
            default:
                throw MoELoaderError.badConfig("no quantizable module at \(absolutePrefix + path)")
            }
            let dot = path.lastIndex(of: ".")
            let parentPath = dot.map { String(path[..<$0]) } ?? ""
            let leaf = dot.map { String(path[path.index(after: $0)...]) } ?? path
            let parent = parentPath.isEmpty ? module : named[parentPath]
            guard let parent else { throw MoELoaderError.badConfig("no parent module for \(path)") }
            parent.update(modules: ModuleChildren.unflattened([(leaf, replacement)]))
        }
    }

    /// Group size, bits and mode for the module at checkpoint path `path` (module-key naming,
    /// e.g. `model.layers.3.mlp.gate_proj`), checked against the tensor shapes.
    ///
    /// `logicalIn` is the module's real input width. The tensors fix group size and bits from it
    /// (`group = in / scale columns`, `bits = 32 * packed words / in`); when the config names them
    /// they must agree, and when it does not they are inferred, or this throws. A group size is
    /// never assumed.
    private func quantizationParameters(
        forModule path: String, weight: MLXArray, scales: MLXArray, logicalIn: Int
    ) throws -> (Int, Int, QuantizationMode) {
        let scaleCols = scales.dim(-1), packed = weight.dim(-1)
        guard scaleCols > 0, logicalIn % scaleCols == 0, (32 * packed) % logicalIn == 0 else {
            throw MoELoaderError.badConfig(
                "\(path): scales \(scales.shape) and weight \(weight.shape) do not fit an input width of \(logicalIn)")
        }
        let inferredGroup = logicalIn / scaleCols, inferredBits = 32 * packed / logicalIn
        guard [32, 64, 128].contains(inferredGroup), [2, 3, 4, 5, 6, 8].contains(inferredBits) else {
            throw MoELoaderError.badConfig(
                "\(path): shapes imply group size \(inferredGroup) and \(inferredBits) bits, which is not a supported affine quantization")
        }
        let spec = quantization?.parameters(for: path)
        if let spec {
            guard spec.groupSize == inferredGroup else {
                throw MoELoaderError.badConfig(
                    "\(path): config says group size \(spec.groupSize) but the tensor shapes imply \(inferredGroup)")
            }
            guard spec.bits == inferredBits else {
                throw MoELoaderError.badConfig(
                    "\(path): config says \(spec.bits) bits but the tensor shapes imply \(inferredBits)")
            }
        }
        return (inferredGroup, inferredBits, spec?.mode ?? .affine)
    }

    /// The whole text model. Every model parameter must be present and every loaded
    /// tensor used, or this throws.
    ///
    /// - Parameter dtype: cast every weight to this (e.g. `.float32` for training).
    public func loadModel(dtype: DType? = nil) throws -> Gemma4TextModel {
        defer { releaseShards() }
        let model = Gemma4TextModel(configuration)
        var weights = try tensors(withPrefix: "model.")
        if !configuration.tieWordEmbeddings {
            weights.merge(try tensors(withPrefix: "lm_head.")) { $1 }
        }
        try apply(weights, to: model, stripping: "", dtype: dtype)
        return model
    }

    /// The whole text model with every MoE layer already cut to its kept experts (`keep[i]` for
    /// layer `i`). Layers are loaded and cut one at a time, so the full expert banks are never
    /// all resident: the peak is one full layer on top of the sliced model.
    public func loadModel(keeping keep: [[Int]], dtype: DType? = nil) throws -> Gemma4TextModel {
        guard keep.count == configuration.numHiddenLayers else {
            throw HealingError.invalid("keep has \(keep.count) entries for \(configuration.numHiddenLayers) layers")
        }
        let model = Gemma4TextModel(configuration)
        var layers: [Gemma4DecoderLayer] = []
        for i in 0 ..< configuration.numHiddenLayers {
            let l = try loadDecoderLayer(i, dtype: dtype)
            guard let m = l as? Gemma4MoEDecoderLayer else {
                throw HealingError.invalid("layer \(i) has no MoE block")
            }
            sliceToKeptExperts(m, set: ExpertSet(keep: keep[i], numExperts: m.numExperts))
            layers.append(m)
        }
        model.model.update(modules: ModuleChildren.unflattened(layers.enumerated().map { ("layers.\($0.offset)", $0.element) }))
        defer { releaseShards() }
        var weights = try tensors(withPrefix: "model.").filter { !$0.key.contains(".layers.") }
        if !configuration.tieWordEmbeddings {
            weights.merge(try tensors(withPrefix: "lm_head.")) { $1 }
        }
        // The layers are already installed and verified as they loaded; the rest gets the same full
        // check as ``loadModel(dtype:)``: every global present, shapes equal, no tensor unused.
        let provided = Set(Self.sanitize(weights).keys)
        try apply(weights, to: model, stripping: "", dtype: dtype, verify: [.noUnusedKeys, .shapeMismatch])
        let missing = model.parameters().flattened().map(\.0)
            .filter { !$0.hasPrefix("model.layers.") && !provided.contains($0) }
        guard missing.isEmpty else { throw MoELoaderError.missingKey(missing.sorted().joined(separator: ", ")) }
        return model
    }

    /// Decoder layer `index` alone, for streaming calibration or healing.
    ///
    /// With `mapped`, the layer's tensors are read-only views of the safetensors files' own pages
    /// (see ``MappedSafetensors``): nothing is copied or read until the layer is used, and the
    /// system can drop the pages under memory pressure. A cast (`dtype` other than the stored one)
    /// or slicing experts still produces ordinary memory for the result.
    ///
    /// With `keeping`, the layer comes back already cut to those experts and **only their rows are ever
    /// read**: the router rows, per-expert scale and every expert tensor (fused or split, quantized
    /// scales and biases included) are `pread` row by row from the file into a `[K, ...]` buffer,
    /// so the unkept experts are never mapped, copied or counted. The result equals loading the layer
    /// and calling ``sliceToKeptExperts(_:set:)``.
    public func loadDecoderLayer(
        _ index: Int, dtype: DType? = nil, mapped: Bool = false, keeping keep: [Int]?
    ) throws -> Gemma4DecoderLayer {
        guard let keep else { return try loadDecoderLayer(index, dtype: dtype, mapped: mapped) }
        defer { releaseShards() }
        guard let total = configuration.numExperts, configuration.enableMoEBlock else {
            throw HealingError.invalid("layer \(index) has no MoE block to cut")
        }
        let rows = Array(Set(keep)).sorted()
        guard let first = rows.first, let last = rows.last, first >= 0, last < total else {
            throw HealingError.invalid("kept experts out of range 0..<\(total)")
        }
        var cut = configuration
        cut.numExperts = rows.count
        cut.topKExperts = min(configuration.topKExperts ?? rows.count, rows.count)
        let layer = Gemma4TextModel.makeLayer(cut, layer: index)
        let prefix = "model.layers.\(index)."
        var weights: [String: MLXArray] = [:]
        var fds: [String: Int32] = [:]
        defer { for fd in fds.values { close(fd) } }
        for (key, srcs) in sources where key.hasPrefix(prefix) {
            guard let src = srcs.first else { continue }
            guard let e = try header(src.file)[src.name] else { throw MoELoaderError.missingKey(src.name) }
            let url = directory.appendingPathComponent(src.file)
            if fds[src.file] == nil {
                let fd = open(url.path, O_RDONLY)
                guard fd >= 0 else { throw MoELoaderError.missingFile(url) }
                fds[src.file] = fd
            }
            let fd = fds[src.file]!
            let expertRows = e.shape.first == total
                && (key.contains(".experts.") || key.contains(".router.proj.") || key.hasSuffix(".router.per_expert_scale"))
            if expertRows {
                weights[key] = try MappedSafetensors.gatherRows(e, rows: rows, fd: fd, in: url)
            } else if mapped {
                weights[key] = try MappedSafetensors.tensor(e, fd: fd, in: url)
            } else {
                weights[key] = try MappedSafetensors.read(e, fd: fd, in: url)
            }
        }
        try apply(weights, to: layer, stripping: prefix, dtype: dtype)
        return layer
    }

    /// Asks the system to read only the kept experts' rows of layers `range` (and the layers' other
    /// tensors in full) into the page cache.
    public func prefetchLayers(_ range: ClosedRange<Int>, keeping: [[Int]]) {
        guard let total = configuration.numExperts else { return prefetchLayers(range) }
        var byFile: [String: [MappedSafetensors.Entry]] = [:]
        for (n, i) in range.enumerated() {
            let rows = Array(Set(keeping[n])).sorted()
            let prefix = "model.layers.\(i)."
            for (key, srcs) in sources where key.hasPrefix(prefix) {
                guard let src = srcs.first, let e = try? header(src.file)[src.name] else { continue }
                if e.shape.first == total, e.end > e.begin, (e.end - e.begin) % total == 0 {
                    let rb = (e.end - e.begin) / total
                    for r in rows {
                        byFile[src.file, default: []].append(.init(dtype: e.dtype, shape: [], begin: e.begin + r * rb, end: e.begin + (r + 1) * rb))
                    }
                } else { byFile[src.file, default: []].append(e) }
            }
        }
        for (file, entries) in byFile {
            let fd = open(directory.appendingPathComponent(file).path, O_RDONLY)
            guard fd >= 0 else { continue }
            MappedSafetensors.prefetch(entries, fd: fd)
            close(fd)
        }
    }

    public func loadDecoderLayer(_ index: Int, dtype: DType? = nil, mapped: Bool = false) throws -> Gemma4DecoderLayer {
        defer { releaseShards() }
        let layer = Gemma4TextModel.makeLayer(configuration, layer: index)
        let prefix = "model.layers.\(index)."
        let weights = mapped ? try mappedTensors(withPrefix: prefix) : try tensors(withPrefix: prefix)
        try apply(weights, to: layer, stripping: prefix, dtype: dtype)
        return layer
    }

    /// Every tensor outside the decoder layers (`model.embed_tokens.weight`,
    /// `model.norm.weight`, and `lm_head.weight` when untied), under module-key names.
    public func loadGlobalTensors(dtype: DType? = nil) throws -> [String: MLXArray] {
        defer { releaseShards() }
        var raw: [String: MLXArray] = [:]
        for (key, srcs) in sources where !key.contains(".layers.") {
            guard let src = srcs.first else { continue }
            if shardCache[src.file] == nil {
                shardCache[src.file] = try loadArrays(url: directory.appendingPathComponent(src.file))
            }
            guard let t = shardCache[src.file]?[src.name] else { throw MoELoaderError.missingKey(src.name) }
            raw[key] = handle(to: t)
        }
        var out: [String: MLXArray] = [:]
        for (key, t) in raw {
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") { continue }
            var w = t
            let base = String(key.dropLast(".weight".count))
            if key.hasSuffix(".weight"), let scales = raw[base + ".scales"] {
                let (g, b, m) = try quantizationParameters(
                    forModule: base, weight: t, scales: scales, logicalIn: configuration.hiddenSize)
                w = dequantized(t, scales: scales, biases: raw[base + ".biases"], groupSize: g, bits: b, mode: m)
            }
            out[key] = dtype.map { w.asType($0) } ?? w
        }
        return out
    }

    /// The token embedding, for computing layer 0's input (scale its output by
    /// `sqrt(hidden_size)`, as ``Gemma4TextModel/embed(_:)`` does).
    public func loadEmbedding(dtype: DType? = nil) throws -> Embedding {
        defer { releaseShards() }
        let prefix = "model.embed_tokens."
        let weights = try tensors(withPrefix: prefix)
        if let scales = weights[prefix + "scales"], let w = weights[prefix + "weight"] {
            let (g, b, m) = try quantizationParameters(
                forModule: "model.embed_tokens", weight: w, scales: scales, logicalIn: configuration.hiddenSize)
            return QuantizedEmbedding(
                weight: w, scales: scales, biases: weights[prefix + "biases"], groupSize: g, bits: b, mode: m)
        }
        let e = Embedding(embeddingCount: configuration.vocabSize, dimensions: configuration.hiddenSize)
        try apply(weights, to: e, stripping: prefix, dtype: dtype)
        return e
    }
}

/// The `quantization` entry of an mlx-community `config.json`: defaults plus per-module
/// overrides keyed by module path (`language_model.model.layers.0.mlp.gate_proj`).
public struct Gemma4Quantization: Sendable {
    public struct Parameters: Sendable, Equatable {
        public var groupSize: Int
        public var bits: Int
        public var mode: QuantizationMode
    }
    public var defaults: Parameters
    public var overrides: [String: Parameters]

    /// Parameters for a module path in module-key naming (`model.layers.3.mlp.gate_proj`).
    public func parameters(for path: String) -> Parameters {
        overrides[path] ?? defaults
    }

    /// `nil` when the config has no quantization entry.
    static func load(from url: URL) throws -> Gemma4Quantization? {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let q = (root["quantization"] ?? root["quantization_config"]) as? [String: Any]
        else { return nil }
        func mode(_ d: [String: Any]) throws -> QuantizationMode {
            switch d["mode"] as? String ?? "affine" {
            case "affine": return .affine
            case let other:
                throw MoELoaderError.badConfig("quantization mode '\(other)' is not supported")
            }
        }
        guard let g = q["group_size"] as? Int, let b = q["bits"] as? Int else {
            throw MoELoaderError.badConfig("quantization needs group_size and bits")
        }
        let defaults = Parameters(groupSize: g, bits: b, mode: try mode(q))
        var overrides: [String: Parameters] = [:]
        for (k, v) in q {
            guard let d = v as? [String: Any] else { continue }
            let key = Gemma4Checkpoint.moduleKey(forSourceName: k + ".weight").map { String($0.dropLast(".weight".count)) } ?? k
            overrides[key] = Parameters(
                groupSize: d["group_size"] as? Int ?? g, bits: d["bits"] as? Int ?? b, mode: try mode(d))
        }
        return Gemma4Quantization(defaults: defaults, overrides: overrides)
    }
}

extension Gemma4Checkpoint {
    /// Identifies the model on disk and how it is loaded: `config.json` (which carries the
    /// quantization entry), the safetensors index if there is one, and for every `.safetensors`
    /// file its name, size and header (tensor names, dtypes, shapes and offsets), plus the load
    /// dtype. It reads a few kilobytes per shard, not the weights, so two checkpoints of one
    /// architecture whose tensors differ only in their *values* (same sizes) are not told apart.
    public func fingerprint(loadDType: DType?) throws -> String {
        var fp = Fingerprint()
        fp.add("gemma4 load=\(loadDType.map { "\($0)" } ?? "stored")")
        fp.add(try Data(contentsOf: directory.appendingPathComponent("config.json")))
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: index.path) { fp.add(try Data(contentsOf: index)) }
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for file in files {
            let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            fp.add("\(file.lastPathComponent) \(size)")
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            let lengthBytes = try handle.read(upToCount: 8) ?? Data()
            guard lengthBytes.count == 8 else { throw MoELoaderError.badConfig("\(file.lastPathComponent) is not a safetensors file") }
            let n = lengthBytes.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
            guard n < 1 << 28 else { throw MoELoaderError.badConfig("\(file.lastPathComponent) has an implausible header") }
            fp.add(try handle.read(upToCount: n) ?? Data())
        }
        return fp.hex
    }
}

extension Gemma4Checkpoint {
    /// Where the embedding table's rows can be read from, on disk, a row at a time (see ``VocabularyWindow``).
    /// Nothing of the table is read here.
    ///
    /// - Parameter dtype: cast floating-point rows to this when they are read.
    public func embeddingRowSource(dtype: DType? = nil) throws -> EmbeddingRowSource {
        func tensor(_ suffix: String) throws -> EmbeddingRowSource.Tensor? {
            guard let src = sources["model.embed_tokens." + suffix]?.first else { return nil }
            guard let e = try header(src.file)[src.name] else { throw MoELoaderError.missingKey(src.name) }
            return .init(entry: e, url: directory.appendingPathComponent(src.file))
        }
        guard let weight = try tensor("weight") else { throw MoELoaderError.missingKey("model.embed_tokens.weight") }
        let scales = try tensor("scales"), biases = try tensor("biases")
        let v = configuration.vocabSize, d = configuration.hiddenSize
        guard weight.entry.shape.first == v else {
            throw MoELoaderError.badConfig(
                "embed_tokens has \(weight.entry.shape.first ?? 0) rows but the config's vocab_size is \(v)")
        }
        var quantization: EmbeddingRowSource.Quantization?
        if let scales {
            let scaleCols = scales.entry.shape.last ?? 0, packed = weight.entry.shape.last ?? 0
            guard scaleCols > 0, d % scaleCols == 0, (32 * packed) % d == 0 else {
                throw MoELoaderError.badConfig("embed_tokens: scales \(scales.entry.shape) and weight \(weight.entry.shape) do not fit width \(d)")
            }
            let group = d / scaleCols, bits = 32 * packed / d
            let spec = self.quantization?.parameters(for: "model.embed_tokens")
            if let spec, spec.groupSize != group || spec.bits != bits {
                throw MoELoaderError.badConfig("embed_tokens: the config's quantization disagrees with the tensor shapes")
            }
            quantization = .init(groupSize: group, bits: bits, mode: spec?.mode ?? .affine)
        } else {
            guard weight.entry.shape == [v, d] else {
                throw MoELoaderError.badConfig("embed_tokens has shape \(weight.entry.shape), expected [\(v), \(d)]")
            }
        }
        return EmbeddingRowSource(
            weight: weight, scales: scales, biases: biases, vocabularySize: v, hiddenSize: d,
            quantization: quantization, dtype: dtype)
    }

    /// The whole text model **without its embedding table**: the table stays on disk and the model runs on a
    /// ``VocabularyWindow`` of it (the rows of `vocabulary` now, those of each request's context as it comes), for
    /// both embedding the prompt and scoring the next token. Everything else loads as in ``loadModel(dtype:)``.
    ///
    /// Needs tied embeddings (Gemma 4's are). The window's vectors have the model's compute type.
    public func loadModel(dtype: DType? = nil, vocabulary: StaticVocabulary) throws -> Gemma4TextModel {
        defer { releaseShards() }
        guard configuration.tieWordEmbeddings else {
            throw HealingError.invalid("a vocabulary window needs tied embeddings (this model has an lm_head)")
        }
        let model = Gemma4TextModel(configuration)
        // A one-row stand-in, so the full table is never built, loaded or evaluated.
        model.model.update(modules: ModuleChildren.unflattened([("embed_tokens", Embedding(weight: MLXArray.zeros([1, configuration.hiddenSize])))]))
        var weights = try tensors(withPrefix: "model.", excluding: { $0.hasPrefix("model.embed_tokens.") })
        weights["model.embed_tokens.weight"] = MLXArray.zeros([1, configuration.hiddenSize])
        try apply(weights, to: model, stripping: "", dtype: dtype)
        let source = try embeddingRowSource(dtype: dtype)
        model.vocabularyWindow = try VocabularyWindow(
            source: source, staticIDs: vocabulary.ids, dtype: model.model.norm.weight.dtype)
        return model
    }
}

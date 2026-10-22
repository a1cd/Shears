import Foundation
import MLX

/// The tensor type an export stores its 2-D and larger weights in. Vectors, norms, the router
/// weight and scales always stay float32.
public enum GGUFOutputType: Sendable {
    /// bfloat16 or float16 to match the checkpoint, as the reference converter's default does.
    case auto
    case f32
    case f16
    case bf16
    case q8_0
    /// `llama-quantize ... Q4_K_M`: Q4_K for most weights, Q6_K / Q5_K / Q8_0 for the sensitive
    /// ones, Q5_0 or Q8_0 where a row is not a multiple of 256 (see ``GGUFQuantMix``).
    case q4_k_m

    var tensorType: GGUFTensorType? {
        switch self {
        case .auto: nil
        case .f32: .f32
        case .f16: .f16
        case .bf16: .bf16
        case .q8_0: .q8_0
        case .q4_k_m: .q4_k
        }
    }
}

public enum GGUFExport {
    /// Where the `tokenizer.*` metadata and the vocabulary size come from.
    public enum TokenizerSource {
        /// `tokenizer.json`, `tokenizer_config.json` and chat template files in the model directory.
        case fromDirectory
        /// No tokenizer metadata. The file will not load in llama.cpp until some is added, but the
        /// model tensors and hyperparameters are complete. Embedding rows are left as they are.
        case none
        /// Metadata built elsewhere. Its `tokenCount` sets the vocabulary size.
        case custom(GGUFTokenizerMetadata)
    }

    public struct Summary: Sendable {
        public var tensorCount: Int
        public var fileType: GGUFTensorType
        public var expertCount: Int?
        public var expertsUsed: Int?
        public var expertFeedForwardLength: Int?
        public var tokenCount: Int?
        /// Tensors stored per type name, e.g. `["BF16": 40, "F32": 31]`.
        public var tensorsByType: [String: Int]
        public var bytes: Int
        /// Disagreements between `config.json` and the tensors that the export resolved in
        /// favour of the tensors.
        public var warnings: [String]
    }

    /// Tokens llama.cpp's chat parser must be able to see as text.
    static let gemma4VisibleTokens: Set<String> = [
        "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>",
        "<tool_response|>", "<|\"|>",
    ]

    /// Exports a Hugging Face Gemma 4 checkpoint (safetensors) to a GGUF file the `gemma4`
    /// architecture of llama.cpp loads.
    ///
    /// The checkpoint is read shard by shard and tensor by tensor: planning reads only the
    /// safetensors headers, and each tensor is converted and written before the next is read.
    /// The expert count, expert width and per-token expert count come from the tensors and
    /// config together, so a pruned checkpoint exports as pruned.
    ///
    /// - Parameters:
    ///   - hfDirectory: directory with `config.json`, safetensors weights, and (for tokenizer
    ///     metadata) `tokenizer.json`.
    ///   - output: GGUF file to write. Replaced if it exists.
    ///   - type: storage type for weights of two or more dimensions.
    ///   - tokenizer: where tokenizer metadata comes from.
    ///   - name: `general.name`; defaults to the directory name.
    ///   - importance: an importance matrix for the K-quants (as `llama-quantize --imatrix`).
    ///   - copyGenerationConfig: write the `general.sampling.*` defaults from `generation_config.json`
    ///     when there is one, as the reference converter does (see ``GGUFGenerationConfig``).
    @discardableResult
    public static func gemma4(
        hfDirectory: URL,
        output: URL,
        type: GGUFOutputType = .auto,
        tokenizer: TokenizerSource = .fromDirectory,
        name: String? = nil,
        importance: GGUFImatrix? = nil,
        copyGenerationConfig: Bool = true
    ) throws -> Summary {
        let config = try GGUFGemma4Config.load(directory: hfDirectory)
        let index = try GGUFSafetensorsIndex(directory: hfDirectory)

        let tokenizerMetadata: GGUFTokenizerMetadata?
        switch tokenizer {
        case .fromDirectory:
            tokenizerMetadata = try GGUFTokenizer.load(
                directory: hfDirectory, model: "gemma4", userDefinedTokens: gemma4VisibleTokens)
        case .none: tokenizerMetadata = nil
        case .custom(let m): tokenizerMetadata = m
        }

        // 1. Map names.
        var mapped: [(gguf: String, entry: GGUFSafetensorsEntry)] = []
        var unknown: [String] = []
        var seen: [String: String] = [:]
        for hf in index.entries.keys.sorted() {
            switch GGUFGemma4Name.map(hf) {
            case .skip: continue
            case .unknown: unknown.append(hf)
            case .mapped(let g):
                if let other = seen[g] {
                    throw GGUFError.other("tensors \(other) and \(hf) both map to \(g)")
                }
                seen[g] = hf
                mapped.append((g, index.entries[hf]!))
            }
        }
        guard unknown.isEmpty else {
            throw GGUFError.other(
                "cannot map \(unknown.count) tensor(s) to gemma4: \(unknown.prefix(5).joined(separator: ", "))")
        }
        let byName = Dictionary(uniqueKeysWithValues: mapped.map { ($0.gguf, $0.entry) })
        let record = try ExportRecord.read(from: hfDirectory)
        if let record, importance != nil, !record.formats.isEmpty {
            throw GGUFError.other(
                "this model was healed against a weight format (\(ExportRecord.fileName)); healing never used an importance matrix, so one would change the rounding. Export without it.")
        }

        // 2. Facts that come from the tensors.
        var shape = GGUFGemma4Shape()
        try resolveExperts(config: config, tensors: byName, into: &shape)

        guard let embed = byName["token_embd.weight"], embed.shape.count == 2 else {
            throw GGUFError.other("checkpoint has no 2-D embedding tensor")
        }
        let vocab = tokenizerMetadata?.tokenCount
        if let vocab, embed.shape[0] < vocab {
            throw GGUFError.other(
                "embedding has \(embed.shape[0]) rows but the tokenizer has \(vocab) tokens")
        }
        // The vocabulary size llama.cpp will see; rows beyond it are padding and are dropped.
        let rows = vocab ?? embed.shape[0]

        // 3. Check the layout against what the loader requires.
        try validate(config: config, tensors: byName, expert: shape, vocabRows: rows)

        // 4. Choose storage types.
        let requested: GGUFTensorType
        if let t = type.tensorType {
            requested = t
        } else {
            requested = mapped.sorted(by: { $0.gguf < $1.gguf }).lazy
                .filter { $0.entry.shape.count >= 2 }
                .compactMap { $0.entry.nativeGGUFType }
                .first(where: { $0 == .bf16 || $0 == .f16 }) ?? .f16
        }

        // The Q4_K_M preset picks a type per tensor; anything else uses one type for all.
        let plan: [String: GGUFTensorType]?
        if case .q4_k_m = type {
            let half: GGUFTensorType = mapped.lazy.filter { $0.entry.shape.count >= 2 }
                .compactMap { $0.entry.nativeGGUFType }.first(where: { $0 == .bf16 || $0 == .f16 }) ?? .f16
            var infos: [(name: String, shape: [Int])] = []
            for (g, e) in mapped {
                var sh = e.shape.isEmpty ? [1] : e.shape
                if ["token_embd.weight", "per_layer_token_embd.weight", "output.weight"].contains(g), sh[0] != rows { sh[0] = rows }
                infos.append((g, sh))
            }
            plan = GGUFQuantMix.q4KM(infos, layerCount: config.layerCount, expertCount: shape.expertCount, half: half)
        } else {
            plan = nil
        }

        // 5. Metadata.
        let writer = try GGUFWriter()
        var kv = try GGUFGemma4Metadata.build(
            config: config, shape: shape, fileType: requested,
            name: name ?? hfDirectory.lastPathComponent)
        if copyGenerationConfig {
            // The converter writes the sampling defaults ahead of `general.name`.
            let sampling = GGUFGenerationConfig.samplingEntries(directory: hfDirectory)
            let at = kv.firstIndex { $0.0 == "general.name" } ?? kv.count
            kv.insert(contentsOf: sampling, at: at)
            if config.architectures.contains("Gemma4UnifiedForConditionalGeneration"),
                let suppress = GGUFGenerationConfig.suppressTokens(directory: hfDirectory)
            {
                kv.append(("tokenizer.ggml.suppress_tokens", suppress))
            }
        }
        if let tokenizerMetadata {
            kv.append(contentsOf: tokenizerMetadata.entries.map { ($0.key, $0.value) })
            kv.append(("tokenizer.ggml.add_space_prefix", .bool(false)))
            kv.append(("tokenizer.ggml.add_bos_token", .bool(true)))
        }
        for (k, v) in kv { try writer.set(k, v) }

        // 6. Tensors: generated first, then the checkpoint's in layer order.
        var summaryTypes: [String: Int] = [:]
        func count(_ t: GGUFTensorType) { summaryTypes[t.name, default: 0] += 1 }

        if config.hasFullAttention() {
            let factors = GGUFGemma4Metadata.ropeFrequencyFactors(config)
            let data = factors.withUnsafeBufferPointer { Data(buffer: $0) }
            try writer.addTensor(
                name: "rope_freqs.weight", shape: [factors.count], type: .f32, source: .encoded({ data }))
            count(.f32)
        }

        let ordered = mapped.sorted { a, b in
            let la = GGUFGemma4Name.layer(of: a.gguf) ?? -1
            let lb = GGUFGemma4Name.layer(of: b.gguf) ?? -1
            return la != lb ? la < lb : a.gguf < b.gguf
        }
        let rowTruncated: Set<String> = ["token_embd.weight", "per_layer_token_embd.weight", "output.weight"]
        for (gguf, entry) in ordered {
            var tensorShape = entry.shape.isEmpty ? [1] : entry.shape
            var keepRows: Int?
            if rowTruncated.contains(gguf), tensorShape[0] != rows {
                tensorShape[0] = rows
                keepRows = rows
            }
            let storage = plan?[gguf] ?? GGUFGemma4Precision.type(ggufName: gguf, shape: tensorShape, requested: requested)
            if let record { try checkAgainstRecord(record, gguf: gguf, entry: entry, storage: storage, shape: tensorShape) }
            let source: GGUFWriter.TensorSource
            if entry.nativeGGUFType == storage {
                // Same byte layout as the checkpoint: copy the bytes (a prefix of rows if the
                // vocabulary is shorter than the table) without going through MLX.
                let bytes = storage.byteCount(shape: tensorShape)
                guard bytes <= entry.byteCount, keepRows != nil || bytes == entry.byteCount else {
                    throw GGUFError.other("tensor \(entry.name): header size \(entry.byteCount) does not match shape \(entry.shape)")
                }
                source = .file(entry.file, offset: entry.offset, count: bytes)
            } else {
                let scalar = entry.shape.isEmpty
                let firstRows = keepRows
                var weights: [Float]?
                if let importance, let w = importance.entries[gguf], storage.blockSize > 1, storage != .q8_0 {
                    let columns = tensorShape[tensorShape.count - 1] * (tensorShape.count == 3 ? tensorShape[0] : 1)
                    if w.count == columns {
                        weights = w
                    } else if !["token_embd.weight", "per_layer_token_embd.weight"].contains(gguf) {
                        throw GGUFError.other("importance matrix has \(w.count) values for \(gguf), expected \(columns)")
                    }
                }
                let produce: () throws -> MLXArray = {
                    let a = try entry.load(firstRows: firstRows)
                    return scalar ? a.reshaped([1]) : a
                }
                source = weights.map { .weightedArray(produce, importance: $0) } ?? .array(produce)
            }
            try writer.addTensor(name: gguf, shape: tensorShape, type: storage, source: source)
            count(storage)
        }

        try writer.write(to: output)

        let total = writer.entries.reduce(0) { $0 + $1.byteCount }
        return Summary(
            tensorCount: writer.tensorCount, fileType: requested, expertCount: shape.expertCount,
            expertsUsed: shape.expertCount.flatMap { e in config.expertsPerToken.map { min($0, e) } },
            expertFeedForwardLength: shape.expertFeedForwardLength, tokenCount: vocab,
            tensorsByType: summaryTypes, bytes: total, warnings: shape.warnings)
    }

    // MARK: Planning

    /// A tensor healing rounded must be written in the type it was rounded for, from the dtype it
    /// was rounded in; anything else ships a model other than the one that was healed.
    static func checkAgainstRecord(
        _ record: ExportRecord, gguf: String, entry: GGUFSafetensorsEntry, storage: GGUFTensorType,
        shape: [Int]
    ) throws {
        guard let wanted = record.formats[gguf] else { return }
        let block: Int
        switch wanted {
        case let w where w.hasPrefix("affine:"):
            throw GGUFError.other(
                "\(gguf) was healed against \(w), an MLX format; a GGUF file would hold different weights")
        default:
            block = GGUFTensorType.allCases.first { $0.name == wanted }?.blockSize ?? 1
        }
        // A row that is not a whole number of blocks was never rounded (QuantFormat leaves it alone).
        guard let last = shape.last, last % block == 0 else { return }
        guard storage.name == wanted else {
            throw GGUFError.other(
                "\(gguf) was healed against \(wanted) but this export would write it as \(storage.name)")
        }
        guard entry.dtype == record.dtype else {
            throw GGUFError.other(
                "\(gguf) was healed in \(record.dtype) but the checkpoint holds \(entry.dtype)")
        }
    }

    /// Reads the expert count and width from the tensors and reconciles them with the config.
    static func resolveExperts(
        config: GGUFGemma4Config, tensors: [String: GGUFSafetensorsEntry], into shape: inout GGUFGemma4Shape
    ) throws {
        var count: Int?
        var width: Int?
        for layer in 0 ..< config.layerCount {
            let prefix = "blk.\(layer)."
            let router = tensors[prefix + "ffn_gate_inp.weight"]
            let fused = tensors[prefix + "ffn_gate_up_exps.weight"]
            let down = tensors[prefix + "ffn_down_exps.weight"]
            if router == nil && fused == nil && down == nil { continue }
            guard let router, let fused, let down else {
                throw GGUFError.other(
                    "layer \(layer): router, ffn_gate_up_exps and ffn_down_exps must all be present or all absent")
            }
            guard router.shape.count == 2, fused.shape.count == 3, down.shape.count == 3 else {
                throw GGUFError.other("layer \(layer): unexpected expert tensor ranks")
            }
            let e = router.shape[0]
            guard fused.shape[0] == e, down.shape[0] == e else {
                throw GGUFError.other(
                    "layer \(layer): expert counts disagree (router \(e), gate_up \(fused.shape[0]), down \(down.shape[0]))")
            }
            if let scale = tensors[prefix + "ffn_down_exps.scale"], scale.shape != [e] {
                throw GGUFError.other("layer \(layer): per_expert_scale has shape \(scale.shape), expected [\(e)]")
            }
            let w = down.shape[2]
            guard fused.shape[1] == 2 * w else {
                throw GGUFError.other(
                    "layer \(layer): gate_up width \(fused.shape[1]) is not twice the down width \(w)")
            }
            if let count, count != e {
                throw GGUFError.other("layer \(layer) has \(e) experts but earlier layers have \(count)")
            }
            if let width, width != w {
                throw GGUFError.other("layer \(layer) has expert width \(w) but earlier layers have \(width)")
            }
            count = e
            width = w
        }
        shape.expertCount = count
        shape.expertFeedForwardLength = width
        guard let count, let width else { return }
        if let configured = config.expertCount, configured != count {
            shape.warnings.append("config.json says \(configured) experts; the tensors have \(count). Using \(count).")
        }
        if let configured = config.expertIntermediateSize, configured != width {
            shape.warnings.append("config.json says expert width \(configured); the tensors have \(width). Using \(width).")
        }
        guard let used = config.expertsPerToken else {
            throw GGUFError.other("the checkpoint has experts but config.json has no top_k_experts")
        }
        if used > count {
            shape.warnings.append("top_k_experts is \(used) but only \(count) experts exist. Using \(count).")
        }
    }

    /// Checks every tensor's presence and shape against what llama.cpp's `gemma4` loader
    /// allocates, so a mismatch fails here and not as a load error later.
    static func validate(
        config c: GGUFGemma4Config, tensors: [String: GGUFSafetensorsEntry], expert: GGUFGemma4Shape,
        vocabRows: Int
    ) throws {
        let h = c.hiddenSize
        let p = c.perLayerInputSize
        var problems: [String] = []

        func check(_ name: String, _ expected: [Int], required: Bool) {
            guard let e = tensors[name] else {
                if required { problems.append("missing \(name)") }
                return
            }
            var shape = e.shape
            var want = expected
            // Embedding-like tensors may carry extra padding rows.
            if ["token_embd.weight", "per_layer_token_embd.weight", "output.weight"].contains(name),
                shape.count == 2, shape[0] >= want[0]
            {
                shape[0] = want[0]
            }
            if shape.isEmpty { shape = [1] }
            if want.isEmpty { want = [1] }
            if shape != want { problems.append("\(name): shape \(e.shape), expected \(expected)") }
        }

        check("token_embd.weight", [vocabRows, h], required: true)
        check("output_norm.weight", [h], required: true)
        check("output.weight", [vocabRows, h], required: false)
        if p > 0 {
            check("per_layer_token_embd.weight", [vocabRows, c.layerCount * p], required: true)
            check("per_layer_model_proj.weight", [c.layerCount * p, h], required: true)
            check("per_layer_proj_norm.weight", [p], required: true)
        }
        for i in 0 ..< c.layerCount {
            let b = "blk.\(i)."
            let hd = c.headDim(layer: i)
            let q = c.headCount * hd
            let kv = c.kvHeads(layer: i) * hd
            let shared = i >= c.firstKVSharedLayer
            let ff = c.denseFeedForwardLength(layer: i)
            check(b + "attn_norm.weight", [h], required: true)
            check(b + "attn_q.weight", [q, h], required: true)
            check(b + "attn_k.weight", [kv, h], required: !shared)
            check(b + "attn_v.weight", [kv, h], required: false)
            check(b + "attn_output.weight", [h, q], required: true)
            check(b + "attn_q_norm.weight", [hd], required: true)
            check(b + "attn_k_norm.weight", [hd], required: !shared)
            check(b + "post_attention_norm.weight", [h], required: true)
            check(b + "ffn_norm.weight", [h], required: true)
            check(b + "ffn_gate.weight", [ff, h], required: true)
            check(b + "ffn_up.weight", [ff, h], required: true)
            check(b + "ffn_down.weight", [h, ff], required: true)
            check(b + "post_ffw_norm.weight", [h], required: true)
            check(b + "layer_output_scale.weight", [1], required: false)
            if let e = expert.expertCount, let w = expert.expertFeedForwardLength,
                tensors[b + "ffn_gate_inp.weight"] != nil
            {
                check(b + "ffn_gate_inp.weight", [e, h], required: true)
                check(b + "ffn_gate_inp.scale", [h], required: true)
                check(b + "pre_ffw_norm_2.weight", [h], required: true)
                check(b + "post_ffw_norm_1.weight", [h], required: true)
                check(b + "post_ffw_norm_2.weight", [h], required: true)
                check(b + "ffn_gate_up_exps.weight", [e, 2 * w, h], required: true)
                check(b + "ffn_down_exps.weight", [e, h, w], required: true)
                check(b + "ffn_down_exps.scale", [e], required: false)
            }
            if p > 0 {
                check(b + "inp_gate.weight", [p, h], required: true)
                check(b + "proj.weight", [h, p], required: true)
                check(b + "post_norm.weight", [h], required: true)
            }
        }
        for (name, e) in tensors where e.dtype != "F32" && e.dtype != "F16" && e.dtype != "BF16" {
            problems.append("\(name): dtype \(e.dtype) is not supported")
        }
        guard problems.isEmpty else {
            throw GGUFError.other(
                "checkpoint does not match the gemma4 layout:\n  " + problems.prefix(12).joined(separator: "\n  ")
                    + (problems.count > 12 ? "\n  ... and \(problems.count - 12) more" : ""))
        }
    }
}

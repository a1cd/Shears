import Foundation
import MLX
import MLXNN

/// Everything needed to chat with one model: weights, tokenizer, chat template, the model's generation
/// defaults, and a ``Generator``.
public final class InferenceRuntime {
    public let model: InferenceModel
    public let tokenizer: BPETokenizer
    public let chatTemplate: ChatTemplate?
    public let defaults: GenerationDefaults
    public let generator: Generator
    /// Where the weights came from, when they came from a directory.
    public let directory: URL?

    public init(
        model: InferenceModel, tokenizer: BPETokenizer, chatTemplate: ChatTemplate?, defaults: GenerationDefaults,
        directory: URL? = nil
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.defaults = defaults
        self.directory = directory
        generator = Generator(model: model, tokenizer: tokenizer)
    }

    /// Loads a Hugging Face layout directory: `config.json`, safetensors weights (full precision,
    /// a Shears export, or a 4-bit quantized checkpoint), `tokenizer.json`, `tokenizer_config.json`, and
    /// optionally `chat_template.jinja` and `generation_config.json`.
    ///
    /// - Parameters:
    ///   - dtype: cast floating-point weights to this (quantized tensors are never cast); nil keeps the stored type.
    ///   - tokenizerDirectory: read the tokenizer files from here instead of `directory`.
    ///   - vocabulary: when given, the embedding table is not loaded: the model runs on a ``VocabularyWindow`` of it
    ///     (these rows plus each request's context tokens), read from the files as needed. See
    ///     `docs/vocabulary-subset.md`.
    public static func load(
        directory: URL, dtype: DType? = nil, tokenizerDirectory: URL? = nil, vocabulary: StaticVocabulary? = nil
    ) throws -> InferenceRuntime {
        let checkpoint = try Gemma4Checkpoint(directory: directory)
        let tokenizerDirectory = tokenizerDirectory ?? directory
        let tokenizer = try BPETokenizer(directory: tokenizerDirectory)
        let module: Gemma4TextModel
        if let vocabulary {
            try vocabulary.validate(for: tokenizer)
            module = try checkpoint.loadModel(dtype: dtype, vocabulary: vocabulary)
        } else {
            module = try checkpoint.loadModel(dtype: dtype)
        }
        eval(module)
        module.vocabularyWindow?.contextVariants = tokenizer.spaceVariantMap()
        return try make(
            model: try InferenceModel(model: module), tokenizerDirectory: tokenizerDirectory, directory: directory,
            tokenizer: tokenizer)
    }

    /// A runtime over a model already in memory, optionally with its healing adapters left unmerged.
    public static func make(
        model: InferenceModel, tokenizerDirectory: URL, directory: URL? = nil, tokenizer prebuilt: BPETokenizer? = nil
    ) throws -> InferenceRuntime {
        let tokenizer = try prebuilt ?? BPETokenizer(directory: tokenizerDirectory)
        let template = try ChatTemplate.load(
            directory: tokenizerDirectory, bosToken: tokenizer.bosToken, eosToken: tokenizer.eosToken)
        var defaults = try GenerationDefaults.load(directory: tokenizerDirectory)
        if defaults.eosTokenIDs.isEmpty, let eos = tokenizer.eosTokenID { defaults.eosTokenIDs = [eos] }
        return InferenceRuntime(
            model: model, tokenizer: tokenizer, chatTemplate: template, defaults: defaults, directory: directory)
    }

    /// The prompt token ids for a chat: the template's rendering, encoded without any extra special tokens
    /// (the template writes `<bos>` itself).
    ///
    /// Message text is data: a control token written in it (`<turn|>`, `<|channel>`, `<bos>`, ...) from the
    /// system prompt, a user, an assistant turn, a tool call or a tool reply is encoded as the plain characters it
    /// spells, never as the special token, so no message can forge a turn boundary. Only the template's own
    /// markup produces special tokens.
    public func promptTokens(
        messages: [[String: Any]], tools: [[String: Any]]? = nil, addGenerationPrompt: Bool = true,
        templateArguments: [String: Any] = [:]
    ) throws -> [Int] {
        guard let chatTemplate else { throw InferenceError.missing("this model has no chat template") }
        let text = try chatTemplate.render(
            messages: messages.map(escaped), tools: tools?.map(escaped), addGenerationPrompt: addGenerationPrompt,
            extra: escaped(templateArguments))
        return tokenizer.encode(templateOutput: text)
    }

    /// `value` with every string in it (and every dictionary key) passed through
    /// ``BPETokenizer/escapingSpecialTokens(in:)``.
    private func escaped(_ value: Any) -> Any {
        switch value {
        case let s as String: return tokenizer.escapingSpecialTokens(in: s)
        case let d as [String: Any]: return escaped(d)
        case let a as [Any]: return a.map(escaped)
        default: return value
        }
    }

    private func escaped(_ dictionary: [String: Any]) -> [String: Any] {
        var out: [String: Any] = [:]
        for (k, v) in dictionary { out[tokenizer.escapingSpecialTokens(in: k)] = escaped(v) }
        return out
    }

    /// A request with this model's defaults: its sampling settings and end-of-generation tokens.
    public func request(
        promptTokens: [Int], maxTokens: Int = 256, sampling: SamplingParameters? = nil, stopStrings: [String] = [],
        logprobs: Int? = nil, constraint: GenerationConstraint? = nil
    ) -> GenerationRequest {
        GenerationRequest(
            promptTokens: promptTokens, maxTokens: maxTokens, sampling: sampling ?? defaults.sampling,
            stopTokenIDs: Set(defaults.eosTokenIDs), stopStrings: stopStrings, logprobs: logprobs, constraint: constraint)
    }
}

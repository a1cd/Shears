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
    public static func load(directory: URL, dtype: DType? = nil, tokenizerDirectory: URL? = nil) throws -> InferenceRuntime {
        let checkpoint = try Gemma4Checkpoint(directory: directory)
        let module = try checkpoint.loadModel(dtype: dtype)
        eval(module)
        return try make(
            model: try InferenceModel(model: module), tokenizerDirectory: tokenizerDirectory ?? directory,
            directory: directory)
    }

    /// A runtime over a model already in memory, optionally with its healing adapters left unmerged.
    public static func make(
        model: InferenceModel, tokenizerDirectory: URL, directory: URL? = nil
    ) throws -> InferenceRuntime {
        let tokenizer = try BPETokenizer(directory: tokenizerDirectory)
        let template = try ChatTemplate.load(
            directory: tokenizerDirectory, bosToken: tokenizer.bosToken, eosToken: tokenizer.eosToken)
        var defaults = try GenerationDefaults.load(directory: tokenizerDirectory)
        if defaults.eosTokenIDs.isEmpty, let eos = tokenizer.eosTokenID { defaults.eosTokenIDs = [eos] }
        return InferenceRuntime(
            model: model, tokenizer: tokenizer, chatTemplate: template, defaults: defaults, directory: directory)
    }

    /// The prompt token ids for a chat: the template's rendering, encoded without any extra special tokens
    /// (the template writes `<bos>` itself).
    public func promptTokens(
        messages: [[String: Any]], tools: [[String: Any]]? = nil, addGenerationPrompt: Bool = true,
        templateArguments: [String: Any] = [:]
    ) throws -> [Int] {
        guard let chatTemplate else { throw InferenceError.missing("this model has no chat template") }
        let text = try chatTemplate.render(
            messages: messages, tools: tools, addGenerationPrompt: addGenerationPrompt, extra: templateArguments)
        return tokenizer.encode(text, addSpecialTokens: false)
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

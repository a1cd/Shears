// The prefill-then-decode structure follows ml-explore/mlx-swift-lm (Libraries/MLXLMCommon/Evaluate.swift),
// MIT License, Copyright (c) 2024 ml-explore. See THIRD_PARTY_NOTICES.md.

import Foundation
import MLX

/// Narrows which tokens may come next, for structured output.
public protocol GenerationConstraint: AnyObject {
    /// A boolean mask `[vocab]` of the token ids allowed after the tokens accepted so far, given the
    /// model's `logits` `[vocab]`; nil allows every token.
    func allowedMask(logits: MLXArray) -> MLXArray?
    /// Called with each token that was chosen (stop tokens included).
    func accept(_ token: Int)
}

public struct GenerationRequest {
    public var promptTokens: [Int]
    public var maxTokens: Int
    public var sampling: SamplingParameters
    /// Token ids that end generation; the token itself is not part of the output.
    public var stopTokenIDs: Set<Int>
    /// Strings that end generation; output stops before the earliest one.
    public var stopStrings: [String]
    /// nil: no log probabilities. Otherwise the number of most likely alternatives to report with
    /// each generated token (0 reports only the chosen token's).
    public var logprobs: Int?
    public var constraint: GenerationConstraint?

    public init(
        promptTokens: [Int], maxTokens: Int = 256, sampling: SamplingParameters = .greedy,
        stopTokenIDs: Set<Int> = [], stopStrings: [String] = [], logprobs: Int? = nil,
        constraint: GenerationConstraint? = nil
    ) {
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.stopTokenIDs = stopTokenIDs
        self.stopStrings = stopStrings
        self.logprobs = logprobs
        self.constraint = constraint
    }
}

public enum FinishReason: String, Sendable {
    /// A stop token or stop string.
    case stop
    /// `maxTokens` reached.
    case length
    /// The caller's event handler returned false.
    case cancelled
}

/// One generated token as the handler sees it.
public struct GenerationEvent {
    public var tokenID: Int
    /// Text this token released: empty while a possible stop string is held back or a character is
    /// incomplete, and it may include text held from earlier tokens.
    public var text: String
    public var logprob: TokenLogprob?
    public var topLogprobs: [TokenLogprob]
    /// The last event of a generation that ended with text still held back: `tokenID` is -1 and only
    /// `text` is meaningful.
    public var isTail = false
}

public struct GenerationResult {
    /// The generated text (stop strings excluded), special tokens kept.
    public var text: String
    /// Every generated token id, stop token excluded.
    public var tokenIDs: [Int]
    public var finishReason: FinishReason
    public var promptTokenCount: Int
    /// Prompt tokens served from the previous request's cache rather than computed.
    public var cachedPromptTokenCount: Int
    public var completionTokenCount: Int
    /// Wall time to process the prompt (to the first token's logits).
    public var promptSeconds: Double
    /// Wall time from the first token's logits to the last token.
    public var decodeSeconds: Double
    public var events: [GenerationEvent]
}

/// Prefill and streaming decode over a KV cache, with sampling, stop conditions and log probabilities.
///
/// The cache of the last request is kept, and a request whose prompt starts with the same tokens
/// computes only the rest. One request at a time.
public final class Generator {
    public let model: InferenceModel
    public let tokenizer: BPETokenizer
    /// Prompt tokens processed per forward pass during prefill.
    public var prefillChunk = 512
    /// Keep the cache between requests for prefix reuse.
    public var reusesPromptCache = true

    private var cache: [Gemma4KVCache]?
    private var cachedTokens: [Int] = []

    public init(model: InferenceModel, tokenizer: BPETokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public var contextLength: Int { model.configuration.maxPositionEmbeddings }

    public func resetCache() {
        cache = nil
        cachedTokens = []
    }

    /// Runs `request`, calling `onEvent` for every generated token (return false to stop early).
    @discardableResult
    public func generate(
        _ request: GenerationRequest, onEvent: ((GenerationEvent) -> Bool)? = nil
    ) throws -> GenerationResult {
        let prompt = request.promptTokens
        guard !prompt.isEmpty else { throw InferenceError.invalid("empty prompt") }
        guard prompt.count + 1 <= contextLength else {
            throw InferenceError.invalid("prompt of \(prompt.count) tokens does not fit the \(contextLength)-token context")
        }
        let maxNew = max(0, min(request.maxTokens, contextLength - prompt.count))
        try model.prepareVocabulary(promptTokens: prompt)

        // Reuse the longest common prefix of the previous sequence, always leaving one token to run.
        var reused = 0
        var layers: [Gemma4KVCache]
        if reusesPromptCache, let old = cache {
            var common = 0
            let limit = min(cachedTokens.count, prompt.count - 1)
            while common < limit, cachedTokens[common] == prompt[common] { common += 1 }
            if common > 0 {
                layers = old.map { $0.prefix(common) }
                reused = common
            } else {
                layers = model.newCache()
            }
        } else {
            layers = model.newCache()
        }
        cache = nil
        cachedTokens = []

        let start = Date()
        var logits = try prefill(Array(prompt[reused...]), cache: layers)
        eval(logits)
        let promptSeconds = Date().timeIntervalSince(start)

        let sampler = Sampler(request.sampling)
        var decoder = tokenizer.makeStreamingDecoder(skipSpecialTokens: false)
        var text = ""
        var pending = ""
        var tokenIDs: [Int] = []
        var events: [GenerationEvent] = []
        var fed = prompt
        var finish = FinishReason.length
        var stoppedByString = false
        let decodeStart = Date()

        func release(_ s: String) { text += s }
        step: for i in 0 ..< maxNew {
            let row = logits[0]
            let mask = request.constraint?.allowedMask(logits: row)
            if model.vocabularyWindow != nil {
                // A window scores only its tokens; a constraint that allows none of them would leave every logit at
                // -inf and the sampler would return token 0. That is an error, not an answer.
                let usable = mask.map { MLX.where($0, row, MLXArray(-Float.infinity)) } ?? row
                if usable.asType(.float32).max().item(Float.self) == -Float.infinity {
                    throw InferenceError.invalid("no token the constraint allows is in the vocabulary window")
                }
            }
            let token = sampler.sample(row, allowed: mask)
            var lp: TokenLogprob?
            var top: [TokenLogprob] = []
            if let n = request.logprobs {
                let r = Logprobs.compute(row, token: token, top: n)
                lp = TokenLogprob(id: token, logprob: r.chosen)
                top = r.top
            }
            request.constraint?.accept(token)
            if request.stopTokenIDs.contains(token) {
                finish = .stop
                break step
            }
            tokenIDs.append(token)
            let delta = decoder.append(token)
            pending += delta
            var released = ""
            var stopped = false
            if !request.stopStrings.isEmpty {
                if let cut = Self.earliestStop(in: pending, stops: request.stopStrings) {
                    released = String(pending[..<cut])
                    pending = ""
                    stopped = true
                } else {
                    let hold = Self.heldSuffixLength(of: pending, stops: request.stopStrings)
                    let end = pending.index(pending.endIndex, offsetBy: -hold)
                    released = String(pending[..<end])
                    pending = String(pending[end...])
                }
            } else {
                released = pending
                pending = ""
            }
            release(released)
            let event = GenerationEvent(tokenID: token, text: released, logprob: lp, topLogprobs: top)
            events.append(event)
            if let onEvent, !onEvent(event) {
                finish = .cancelled
                break step
            }
            if stopped {
                finish = .stop
                stoppedByString = true
                break step
            }
            if i == maxNew - 1 { break step }
            fed.append(token)
            logits = model.nextLogits(MLXArray([Int32(token)])[.newAxis], cache: layers)
            eval(logits)
        }
        if !stoppedByString, finish != .cancelled {
            // Text still held back (an incomplete character, or the start of a stop string that never
            // completed) belongs to the output.
            let tail = pending + decoder.finish()
            if !tail.isEmpty {
                release(tail)
                let event = GenerationEvent(tokenID: -1, text: tail, logprob: nil, topLogprobs: [], isTail: true)
                events.append(event)
                _ = onEvent?(event)
            }
        }
        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        if reusesPromptCache {
            cache = layers
            cachedTokens = fed
        }
        return GenerationResult(
            text: text, tokenIDs: tokenIDs, finishReason: finish, promptTokenCount: prompt.count,
            cachedPromptTokenCount: reused, completionTokenCount: tokenIDs.count, promptSeconds: promptSeconds,
            decodeSeconds: decodeSeconds, events: events)
    }

    /// Runs `tokens` through the model in chunks and returns the logits `[1, vocab]` after the last.
    func prefill(_ tokens: [Int], cache: [Gemma4KVCache]) throws -> MLXArray {
        precondition(!tokens.isEmpty)
        var logits: MLXArray?
        var i = 0
        while i < tokens.count {
            let end = min(tokens.count, i + max(1, prefillChunk))
            let chunk = MLXArray(tokens[i ..< end].map(Int32.init))[.newAxis]
            let out = model.nextLogits(chunk, cache: cache)
            if end < tokens.count {
                // Materialize the cache so the graph does not grow with the prompt.
                eval(cache.compactMap { $0.keys } + cache.compactMap { $0.values })
            } else {
                logits = out
            }
            i = end
        }
        return logits!
    }

    // MARK: Stop strings

    /// The index of the earliest stop string's start in `text`.
    static func earliestStop(in text: String, stops: [String]) -> String.Index? {
        var best: String.Index?
        for s in stops where !s.isEmpty {
            if let r = text.range(of: s), best == nil || r.lowerBound < best! { best = r.lowerBound }
        }
        return best
    }

    /// Characters at the end of `text` that begin some stop string and so cannot be released yet.
    static func heldSuffixLength(of text: String, stops: [String]) -> Int {
        var hold = 0
        for s in stops where !s.isEmpty {
            let maxLen = min(s.count - 1, text.count)
            guard maxLen > 0 else { continue }
            for len in stride(from: maxLen, through: 1, by: -1) where len > hold {
                if s.hasPrefix(String(text.suffix(len))) { hold = len; break }
            }
        }
        return hold
    }
}

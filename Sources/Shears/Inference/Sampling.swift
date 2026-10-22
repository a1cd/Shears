// The sampling chain (temperature, top-k, top-p) follows mlx-swift-lm's MLXLMCommon/Evaluate.swift
// `TopPSampler` / `CategoricalSampler`, MIT License, Copyright (c) 2024 ml-explore. See
// THIRD_PARTY_NOTICES.md.

import Foundation
import MLX

/// How the next token is chosen from the logits.
public struct SamplingParameters: Sendable, Equatable {
    /// 0 picks the largest logit (greedy); otherwise logits are divided by this before sampling.
    public var temperature: Float
    /// Keep only the `topK` largest logits; 0 keeps all.
    public var topK: Int
    /// Keep the smallest set of tokens whose probability mass reaches `topP`; 1 keeps all.
    public var topP: Float
    /// Seeds the random stream; nil draws a fresh seed.
    public var seed: UInt64?

    public init(temperature: Float = 0, topK: Int = 0, topP: Float = 1, seed: UInt64? = nil) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.seed = seed
    }

    public static let greedy = SamplingParameters()
}

/// A model's own generation defaults: `generation_config.json`.
public struct GenerationDefaults: Sendable, Equatable {
    public var sampling: SamplingParameters
    /// Token ids that end generation (`eos_token_id`, a number or a list).
    public var eosTokenIDs: [Int]
    public var bosTokenID: Int?
    public var padTokenID: Int?

    public init(sampling: SamplingParameters = .greedy, eosTokenIDs: [Int] = [], bosTokenID: Int? = nil, padTokenID: Int? = nil) {
        self.sampling = sampling
        self.eosTokenIDs = eosTokenIDs
        self.bosTokenID = bosTokenID
        self.padTokenID = padTokenID
    }

    /// Reads `generation_config.json` from `directory`. With `do_sample: true` the defaults sample with the
    /// temperature, top-k and top-p it names (Hugging Face's own 1.0, 50 and 1.0 where it names none);
    /// otherwise, and with no file, they are greedy.
    public static func load(directory: URL) throws -> GenerationDefaults {
        guard let json = try GGUFJSON.load(directory.appendingPathComponent("generation_config.json")) else {
            return GenerationDefaults()
        }
        var sampling = SamplingParameters.greedy
        if json["do_sample"]?.bool == true {
            sampling.temperature = Float(json["temperature"]?.double ?? 1.0)
            sampling.topK = json["top_k"]?.int ?? 50
            sampling.topP = Float(json["top_p"]?.double ?? 1.0)
        }
        var eos: [Int] = []
        if let list = json["eos_token_id"]?.array { eos = list.compactMap { $0.int } } else if let one = json["eos_token_id"]?.int { eos = [one] }
        return GenerationDefaults(
            sampling: sampling, eosTokenIDs: eos, bosTokenID: json["bos_token_id"]?.int, padTokenID: json["pad_token_id"]?.int)
    }
}

/// Chooses tokens from logits `[vocab]`.
public final class Sampler {
    public let parameters: SamplingParameters
    private var key: MLXArray?

    public init(_ parameters: SamplingParameters) {
        self.parameters = parameters
        if parameters.temperature > 0 {
            key = MLXRandom.key(parameters.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max)))
        }
    }

    /// The chosen token id for `logits` (`[vocab]`).
    public func sample(_ logits: MLXArray) -> Int {
        sample(logits, allowed: nil)
    }

    /// As ``sample(_:)``, restricted to the token ids in `allowed` when given (everything else is
    /// removed before top-k and top-p are applied).
    public func sample(_ logits: MLXArray, allowed: MLXArray?) -> Int {
        var l = logits.asType(.float32)
        if let allowed {
            l = MLX.where(allowed, l, MLXArray(-Float.infinity))
        }
        guard parameters.temperature > 0 else { return argMax(l, axis: -1).item(Int.self) }
        l = l / parameters.temperature
        let vocab = l.dim(-1)
        if parameters.topK > 0, parameters.topK < vocab {
            let kth = argPartition(l, kth: vocab - parameters.topK, axis: -1)[vocab - parameters.topK]
            let threshold = l[kth]
            l = MLX.where(l .< threshold, MLXArray(-Float.infinity), l)
        }
        let (next, sub) = MLXRandom.split(key: key!)
        key = next
        if parameters.topP > 0, parameters.topP < 1 {
            // Work in ascending order: drop the tail whose mass is at most 1 - topP.
            let order = argSort(l, axis: -1)
            let sorted = l[order]
            let probs = softmax(sorted, axis: -1)
            let cumulative = cumsum(probs, axis: -1)
            let kept = MLX.where(cumulative .> (1 - parameters.topP), sorted, MLXArray(-Float.infinity))
            let pick = categorical(kept, key: sub)
            return order[pick].item(Int.self)
        }
        return categorical(l, key: sub).item(Int.self)
    }
}

/// Log probabilities of the model's raw next-token distribution (a softmax of the logits, before
/// temperature, top-k and top-p), the way `llama-server` reports them.
public struct TokenLogprob: Sendable, Equatable {
    public var id: Int
    public var logprob: Float
}

public enum Logprobs {
    /// `logprob` of `token` and the `top` most likely tokens, most likely first.
    public static func compute(_ logits: MLXArray, token: Int, top: Int) -> (chosen: Float, top: [TokenLogprob]) {
        let l = logits.asType(.float32)
        let lse = logSumExp(l, axis: -1)
        let chosen = (l[token] - lse)
        var out: [TokenLogprob] = []
        if top > 0 {
            let n = l.dim(-1)
            let k = min(top, n)
            let idx = argPartition(l, kth: n - k, axis: -1)[(n - k)...]
            let values = l[idx] - lse
            eval(chosen, idx, values)
            let ids = idx.asArray(Int32.self)
            let vals = values.asArray(Float.self)
            out = zip(ids, vals).map { TokenLogprob(id: Int($0), logprob: $1) }
                .sorted { $0.logprob != $1.logprob ? $0.logprob > $1.logprob : $0.id < $1.id }
        } else {
            eval(chosen)
        }
        return (chosen.item(Float.self), out)
    }
}

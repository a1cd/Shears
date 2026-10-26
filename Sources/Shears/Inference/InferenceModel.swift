// The cached decode loop follows ml-explore/mlx-swift-lm (Libraries/MLXLMCommon/Evaluate.swift,
// KVCache.swift), MIT License, Copyright (c) 2024 ml-explore. See THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

/// A pruned model's healing state, kept unmerged: the adapters, each layer's kept experts, and whether
/// the layers have been cut to those experts (`sliced`) or still hold every expert behind a routing mask.
public struct AdapterOverlay {
    public var adapters: ModelAdapters
    public var sets: [ExpertSet]
    public var sliced: Bool

    public init(adapters: ModelAdapters, sets: [ExpertSet], sliced: Bool) {
        self.adapters = adapters
        self.sets = sets
        self.sliced = sliced
    }
}

/// A Gemma 4 text model ready for generation: the forward pass over a KV cache, with or without
/// unmerged healing adapters.
///
/// Without an overlay this is ``Gemma4TextModel`` as loaded (a checkpoint, a merged export, a 4-bit
/// QAT checkpoint). With one, each decoder layer runs the student the healer trained
/// (masked or sliced routing, the router correction, dense and expert LoRAs), through
/// the same routing and expert code, so the output equals the healer's own forward pass and, up to
/// rounding, the model that merge-and-slice export writes.
///
/// A model is not thread-safe: one generation at a time.
public final class InferenceModel {
    public let model: Gemma4TextModel
    public let overlay: AdapterOverlay?

    public var configuration: Gemma4Configuration { model.configuration }

    public init(model: Gemma4TextModel, overlay: AdapterOverlay? = nil) throws {
        self.model = model
        self.overlay = overlay
        model.train(false)
        if let overlay {
            let n = model.layers.count
            guard overlay.adapters.firstLayer == 0, overlay.adapters.layers.count == n, overlay.sets.count == n else {
                throw InferenceError.invalid("adapters must cover all \(n) layers starting at layer 0")
            }
            guard model.layers.allSatisfy({ $0 is Gemma4MoEDecoderLayer }) else {
                throw InferenceError.invalid("adapters need a model with MoE decoder layers")
            }
            overlay.adapters.refreshRounding(layers: model.healableLayers, sets: overlay.sets, sliced: overlay.sliced)
            eval(overlay.adapters)
        }
    }

    public func newCache() -> [Gemma4KVCache] { model.newCache() }

    /// Logits `[1, vocab]` for the position after the last of `tokens` (`[1, L]` ids), which continue the
    /// sequence held in `cache`. Only the last position is projected to the vocabulary.
    public func nextLogits(_ tokens: MLXArray, cache: [Gemma4KVCache]) -> MLXArray {
        model.logits(fromHidden: lastHidden(tokens, cache: cache))[0..., 0, 0...]
    }

    /// The final-layer hidden state `[1, 1, D]` at the last of `tokens` (before the final norm).
    public func lastHidden(_ tokens: MLXArray, cache: [Gemma4KVCache]) -> MLXArray {
        let h = hidden(tokens, cache: cache)
        return h[0..., (h.dim(1) - 1)..., 0...]
    }

    /// The vocabulary window, when the model runs on one.
    public var vocabularyWindow: VocabularyWindow? { model.vocabularyWindow }

    /// Gets the model ready for a request whose prompt is `promptTokens`: with a vocabulary window, loads the
    /// rows of the prompt's tokens the window lacks. A no-op for a model with its whole table.
    public func prepareVocabulary(promptTokens: [Int]) throws {
        try model.vocabularyWindow?.prepare(contextTokens: promptTokens)
    }

    /// The final-layer hidden states `[1, L, D]` (before the final norm).
    public func hidden(_ tokens: MLXArray, cache: [Gemma4KVCache]?) -> MLXArray {
        var h = model.embed(tokens)
        let n = tokens.dim(1)
        let offset = cache?.first?.offset ?? 0
        let sliding = gemma4AttentionMask(n: n, offset: offset, window: configuration.slidingWindow)
        let full = gemma4AttentionMask(n: n, offset: offset)
        for (i, layer) in model.layers.enumerated() {
            let mask = layer.isSliding ? sliding : full
            if let overlay, let moe = layer as? Gemma4MoEDecoderLayer {
                h = Self.adaptedLayer(
                    moe, h, mask: mask, cache: cache?[i], set: overlay.sets[i],
                    adapters: overlay.adapters.layers[i], sliced: overlay.sliced)
            } else {
                h = layer(h, mask: mask, cache: cache?[i])
            }
        }
        return h
    }

    /// Logits `[1, L, vocab]` for a whole sequence from position 0 with no cache: the reference the cached
    /// path is tested against.
    public func fullLogits(_ tokens: MLXArray) -> MLXArray {
        model.logits(fromHidden: hidden(tokens, cache: nil))
    }

    // MARK: Adapted layer

    /// One MoE decoder layer as the healed student runs it, over a KV cache.
    static func adaptedLayer(
        _ layer: Gemma4MoEDecoderLayer, _ x: MLXArray, mask: MLXArray?, cache: Gemma4KVCache?,
        set: ExpertSet, adapters: LayerAdapters, sliced: Bool
    ) -> MLXArray {
        let raw = layer.selfAttn.attended(layer.inputLayernorm(x), mask: mask, cache: cache)
        let parts = attentionParts(layer, x, raw: raw, adapters: adapters)
        return healingLayerRemainder(layer, parts, set: set, adapters: adapters, sliced: sliced)
    }

    /// A dense linear with its LoRA, or the rounded merged weight when the adapters train against a
    /// shipping format (the same two cases the healer's forward pass has).
    private static func dense(
        _ l: Linear, _ x: MLXArray, _ pair: LoRAPair?, _ adapters: LayerAdapters, _ site: DenseSite
    ) -> MLXArray {
        if let format = adapters.denseFormats[site] {
            return quantizationAwareLinear(
                l, x, pair: pair, format: format, rounded: adapters.roundedDense[site], dtype: adapters.exportDType)
        }
        let y = l(x)
        return pair.map { $0.apply(x, to: y) } ?? y
    }

    /// The attention output projection, residual and dense branch from the attention output `raw` (before
    /// `o_proj`), as the training-time forward pass computes them.
    private static func attentionParts(
        _ layer: Gemma4MoEDecoderLayer, _ x: MLXArray, raw: MLXArray, adapters: LayerAdapters
    ) -> MoELayerParts {
        let o = dense(layer.selfAttn.oProj, raw, adapters.attentionOutput, adapters, .attentionOutput)
        let h = x + layer.postAttentionLayernorm(o)
        let n = layer.preFeedforwardLayernorm(h)
        let gate = dense(layer.mlp.gateProj, n, adapters.mlpGate, adapters, .mlpGate)
        let up = dense(layer.mlp.upProj, n, adapters.mlpUp, adapters, .mlpUp)
        let hidden = geluApproximate(gate) * up
        let down = dense(layer.mlp.downProj, hidden, adapters.mlpDown, adapters, .mlpDown)
        let denseBranch = layer.postFeedforwardLayernorm1(down)
        let rin = layer.router.normalizedInput(h)
        return MoELayerParts(
            residual: h, dense: denseBranch, routerLogits: layer.router.proj(rin), routerInput: rin,
            expertInput: layer.preFeedforwardLayernorm2(h))
    }
}

public enum InferenceError: Error, CustomStringConvertible {
    case invalid(String)
    case missing(String)

    public var description: String {
        switch self {
        case .invalid(let s), .missing(let s): s
        }
    }
}

extension Gemma4KVCache {
    /// A copy holding only the first `length` positions.
    func prefix(_ length: Int) -> Gemma4KVCache {
        let c = Gemma4KVCache()
        guard length > 0, let k = keys, let v = values else { return c }
        _ = c.update(keys: k[0..., 0..., ..<length, 0...], values: v[0..., 0..., ..<length, 0...])
        return c
    }
}

extension AdapterOverlay {
    /// The healer's adapters and expert sets, unmerged.
    public init(healer: PruneHealer) {
        self.init(adapters: healer.adapters, sets: healer.network.sets, sliced: healer.network.sliced)
    }
}

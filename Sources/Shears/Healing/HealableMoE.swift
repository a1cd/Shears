import Foundation
import MLX
import MLXNN

// MARK: - Interfaces

/// The pieces of a decoder layer between the layer input and output that healing needs to
/// reach into: everything up to the routing decision, and everything after the expert sum.
public struct MoELayerParts {
    /// Residual stream after attention (what the router reads in Gemma 4).
    public var residual: MLXArray
    /// The dense branch's output, already post-normed.
    public var dense: MLXArray
    /// Raw router logits `[..., E]`.
    public var routerLogits: MLXArray
    /// The router's own normalised input, which a low-rank router correction reads.
    public var routerInput: MLXArray
    /// What the experts read.
    public var expertInput: MLXArray
}

/// The attention keys and values of a sequence's first `P` positions (`[1, kvHeads, P, headDim]`,
/// after any norms and positional encoding), kept so later positions of the same sequence can attend
/// to them without the prefix being run again. See ``healingLayerForwardShared(_:prefix:suffix:set:adapters:sliced:)``.
public struct PrefixKV {
    public var keys: MLXArray
    public var values: MLXArray
    public init(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
    }
    /// Number of prefix positions.
    public var length: Int { keys.dim(2) }
}

/// A mixture-of-experts decoder layer that healing can drive piece by piece.
///
/// ``Gemma4MoEDecoderLayer`` conforms. Another family (a Qwen-style MoE, say) needs to say how
/// its layer splits into the attention half, the routing decision and the combine step, and
/// which dense linears accept an adapter; the pruned forward pass, adapters, losses, trainer,
/// selection and merge-and-slice export are generic over this protocol.
public protocol HealableMoELayer: MixtureOfExperts {
    var hiddenSize: Int { get }
    var expertHiddenSize: Int { get }
    /// Input and output sizes of the dense linears that take a LoRA.
    var denseSiteDimensions: [DenseSite: (input: Int, output: Int)] { get }
    /// Everything up to the routing decision, over a whole sequence from position 0.
    /// With `adapters`, dense LoRAs are applied.
    func attentionPart(_ x: MLXArray, adapters: LayerAdapters?) -> MoELayerParts
    /// Weight the routed expert's output gets in the sum: the layer's per-expert scale.
    func expertScale(_ indices: MLXArray) -> MLXArray
    /// Folds the expert branch back into the stream and returns the layer output.
    func combine(_ parts: MoELayerParts, expertBranch y: MLXArray) -> MLXArray
    /// The map applied to the summed expert output before it joins the stream. Selection
    /// scores candidate sets after this map.
    func normalizeExpertBranch(_ y: MLXArray) -> MLXArray
    /// Whether the layer has a full-attention (global) pattern; block ends default to these.
    var isGlobalAttention: Bool { get }
    /// Whether the layer implements ``prefixAttentionPart(_:adapters:)`` and
    /// ``suffixAttentionPart(_:after:adapters:)``. A layer that does not still works with
    /// ``healingLayerForwardShared(_:prefix:suffix:set:adapters:sliced:)``, which then runs each
    /// row whole.
    var supportsSharedPrefix: Bool { get }
    /// ``attentionPart(_:adapters:)`` for a prefix (`[1, P, D]`) that later positions will attend to,
    /// also returning the keys and values those positions need.
    func prefixAttentionPart(_ x: MLXArray, adapters: LayerAdapters?) -> (parts: MoELayerParts, prefix: PrefixKV)
    /// ``attentionPart(_:adapters:)`` for the `[R, S, D]` positions that follow `prefix` (positions
    /// `P ..< P + S` of each of `R` rows). Row `r`'s parts equal the corresponding positions of
    /// ``attentionPart(_:adapters:)`` on the whole row `prefix + x[r]`.
    func suffixAttentionPart(_ x: MLXArray, after prefix: PrefixKV, adapters: LayerAdapters?) -> MoELayerParts
}

extension HealableMoELayer {
    public var supportsSharedPrefix: Bool { false }
    public func prefixAttentionPart(_ x: MLXArray, adapters: LayerAdapters?) -> (parts: MoELayerParts, prefix: PrefixKV) {
        preconditionFailure("this layer does not support shared-prefix forward passes")
    }
    public func suffixAttentionPart(_ x: MLXArray, after prefix: PrefixKV, adapters: LayerAdapters?) -> MoELayerParts {
        preconditionFailure("this layer does not support shared-prefix forward passes")
    }
}

/// A model made of ``HealableMoELayer`` layers.
public protocol HealableMoEModel: AnyObject {
    var healableLayers: [any HealableMoELayer] { get }
    func embed(_ tokens: MLXArray) -> MLXArray
    func logits(fromHidden h: MLXArray) -> MLXArray
}

extension HealableMoEModel {
    /// Block ends: the global-attention layers, plus the last layer if it isn't one. For
    /// Gemma 4 26B-A4B that is 5, 11, 17, 23, 29: five blocks of six layers.
    public var defaultBlockEnds: [Int] {
        let layers = healableLayers
        var ends = layers.indices.filter { layers[$0].isGlobalAttention }
        if ends.last != layers.count - 1 { ends.append(layers.count - 1) }
        return ends
    }
}

/// Contiguous `(start, end)` layer ranges (inclusive) from a list of block ends.
public func blockRanges(ends: [Int]) -> [(start: Int, end: Int)] {
    var start = 0
    return ends.map { e in
        defer { start = e + 1 }
        return (start, e)
    }
}

// MARK: - Expert sets

/// One layer's kept experts, as the routing mask and the slot table the per-expert LoRA is
/// gathered with.
public struct ExpertSet {
    /// Kept expert ids, ascending. Slot `j` of a per-expert adapter belongs to `keep[j]`.
    public let keep: [Int]
    public let numExperts: Int
    /// `keep` as `[K]` int32: the rows a sliced layer was cut to.
    public let keepArray: MLXArray
    /// `[E]` bool.
    public let mask: MLXArray
    /// `[E]` int32, expert id to kept slot. Monotone non-decreasing (a pruned expert maps to
    /// the next kept slot, and its adapter delta is gated to zero), so slot indices come out
    /// sorted whenever expert indices do, which lets the gather take `sortedIndices`.
    public let slotTable: MLXArray

    public init(keep ids: [Int], numExperts: Int) {
        let keep = Array(Set(ids)).sorted()
        precondition(!keep.isEmpty, "an expert set must not be empty")
        precondition(keep.first! >= 0 && keep.last! < numExperts, "expert id out of range")
        self.keep = keep
        self.numExperts = numExperts
        keepArray = MLXArray(keep.map(Int32.init))
        let kept = Set(keep)
        mask = MLXArray((0 ..< numExperts).map { kept.contains($0) })
        var table: [Int32] = []
        var seen = 0
        for e in 0 ..< numExperts {
            table.append(Int32(min(seen, keep.count - 1)))
            if kept.contains(e) { seen += 1 }
        }
        slotTable = MLXArray(table)
    }

    /// Every expert kept.
    public static func all(_ numExperts: Int) -> ExpertSet {
        ExpertSet(keep: Array(0 ..< numExperts), numExperts: numExperts)
    }
}

// MARK: - Routing and the expert branch

let routingMaskedLogit: Float = -1e9

/// The layer's native routing: top-k over the raw logits, softmax among the selected, times
/// the per-expert scale. Used for the teacher and by selection.
func routeNative(_ layer: any HealableMoELayer, _ logits: MLXArray) -> (MLXArray, MLXArray) {
    route(layer, logits: logits, set: nil, tweak: nil, routerInput: nil)
}

/// Routing restricted to `set`.
///
/// Disallowed experts get a logit of -1e9, and the top `topK` is always taken, so a token
/// fires exactly the experts the full router would pick among the kept ones (with fewer
/// kept experts than `topK` the surplus slots carry exactly zero weight). With `set == nil`
/// this is the layer's own routing.
///
/// With a router `tweak`, the selection and the softmax run in float32 and the weights are
/// built as the native-precision softmax plus the float32 difference between the corrected
/// and uncorrected softmax. At zero correction the difference is exactly zero, so the weights
/// are bit-identical to the uncorrected path; a plain float32 softmax cast back is not.
func route(
    _ layer: any HealableMoELayer, logits scores: MLXArray, set: ExpertSet?, tweak: RouterTweak?,
    routerInput: MLXArray?
) -> (MLXArray, MLXArray) {
    let k = layer.expertRouter.topK
    var s = scores
    var corrected: MLXArray?
    if let tweak, let routerInput, set != nil {
        let corr = matmul(matmul(routerInput.asType(.float32), tweak.a), tweak.b)
        corrected = scores.asType(.float32) + corr
        s = corrected!
    }
    if let set {
        s = MLX.where(set.mask, s, MLXArray(routingMaskedLogit, dtype: s.dtype))
    }
    let idx = stopGradient(topKIndices(s, k))
    let weights: MLXArray
    if let corrected, let set {
        let s0 = MLX.where(set.mask, scores, MLXArray(routingMaskedLogit, dtype: scores.dtype))
        let p = softmax(takeAlong(s0, idx, axis: -1), axis: -1, precise: true)
        let s0f = stopGradient(s0.asType(.float32))
        let diff = softmax(takeAlong(s, idx, axis: -1), axis: -1, precise: true)
            - softmax(takeAlong(s0f, idx, axis: -1), axis: -1, precise: true)
        weights = (p.asType(.float32) + diff).asType(scores.dtype)
    } else {
        weights = softmax(takeAlong(s, idx, axis: -1), axis: -1, precise: true)
    }
    return (idx, weights * layer.expertScale(idx).asType(weights.dtype))
}

/// ``route(_:logits:set:tweak:routerInput:)`` for a layer cut down to its kept experts.
///
/// The layer's router already has only the kept rows, so `scores` is `[..., K]` and the top-k
/// runs over those directly, which picks exactly the experts the masked formulation picks. The
/// router correction's `b` stays `[r, E]` (so adapters keep one format whatever the student's
/// form); its kept columns are gathered here. The weights are built as in the masked path.
func routeSliced(
    _ layer: any HealableMoELayer, logits scores: MLXArray, keep: MLXArray, tweak: RouterTweak?,
    routerInput: MLXArray?
) -> (MLXArray, MLXArray) {
    let k = layer.expertRouter.topK
    var s = scores
    var corrected: MLXArray?
    if let tweak, let routerInput {
        let b = tweak.b.take(keep, axis: 1)
        let corr = matmul(matmul(routerInput.asType(.float32), tweak.a), b)
        corrected = scores.asType(.float32) + corr
        s = corrected!
    }
    let idx = stopGradient(topKIndices(s, k))
    let weights: MLXArray
    if let corrected {
        let p = softmax(takeAlong(scores, idx, axis: -1), axis: -1, precise: true)
        let s0f = stopGradient(scores.asType(.float32))
        let diff = softmax(takeAlong(corrected, idx, axis: -1), axis: -1, precise: true)
            - softmax(takeAlong(s0f, idx, axis: -1), axis: -1, precise: true)
        weights = (p.asType(.float32) + diff).asType(scores.dtype)
    } else {
        weights = softmax(takeAlong(s, idx, axis: -1), axis: -1, precise: true)
    }
    return (idx, weights * layer.expertScale(idx).asType(weights.dtype))
}

/// Cuts `layer` down to `set`'s experts in place: the router's rows and per-expert scale and
/// the expert bank, in `set.keep` order (ascending), so slot `j` is expert `set.keep[j]`.
/// `topK` is clamped to what remains. Everything else about the layer is untouched.
///
/// This rewrites the layer's own parameters, so a layer that is also somebody's teacher cannot
/// be sliced; slice layers that were loaded for the student.
public func sliceToKeptExperts(_ layer: any HealableMoELayer, set: ExpertSet) {
    precondition(layer.numExperts == set.numExperts, "layer is already sliced or the set does not match it")
    layer.expertRouter.keepExperts(set.keepArray)
    layer.expertBank.keepExperts(set.keepArray)
    if let m = layer as? Module { eval(m) }
}

/// `sum_k w_k * expert_k(x)`, with per-expert LoRA deltas when `lora` is given.
///
/// This is the bank's `callAsFunction` op for op (the same sort decision, the same gathers)
/// plus a low-rank delta added to each selected expert's gate and up pre-activations and to
/// its down output. The delta runs in float32 and is added to the float32 upcast base output
/// before the cast back, so a zero `b` is exactly the base path.
func expertBranch(
    _ layer: any HealableMoELayer, x: MLXArray, indices: MLXArray, weights: MLXArray,
    set: ExpertSet?, lora: ExpertLoRA?, sliced: Bool = false
) -> MLXArray {
    let bank = layer.expertBank
    if let set, mergedExpertsApplicable(layer, set: set, lora: lora) {
        return mergedExpertBranch(layer, x: x, indices: indices, weights: weights, set: set, lora: lora, sliced: sliced)
    }
    let y: MLXArray
    let sort = indices.size >= bank.sortThreshold
    if let lora, let set, lora.gateUpFormat != nil || lora.downFormat != nil {
        y = expertsRounded(bank, x: x, indices: indices, set: set, lora: lora, sort: sort, sliced: sliced)
    } else if let lora, let set {
        y = expertsWithDelta(bank, x: x, indices: indices, set: set, lora: lora, sort: sort, sliced: sliced)
    } else {
        y = bank(x, indices, sort: sort)
    }
    return (expandedDimensions(weights.asType(y.dtype), axis: -1) * y).sum(axis: -2)
}

/// The expert sum with the kept experts' merged weights rounded to `format` (straight-through),
/// in place of base plus a separate low-rank delta.
private func expertsRounded(
    _ bank: SwitchGLU, x input: MLXArray, indices: MLXArray, set: ExpertSet, lora: ExpertLoRA,
    sort: Bool, sliced: Bool
) -> MLXArray {
    var x = expandedDimensions(input, axes: [-2, -3])
    var idx = stopGradient(indices)
    var slot = sliced ? idx : set.slotTable[idx]
    var inverseOrder: MLXArray?
    if sort {
        let k = indices.dim(-1)
        let flat = idx.flattened()
        let order = argSort(flat)
        inverseOrder = argSort(order)
        x = x.flattened(start: 0, end: -3)[floorDivide(order, k)]
        idx = flat[order]
        slot = slot.flattened()[order]
    }
    let keep: MLXArray? = sliced ? nil : set.keepArray
    func merged(
        _ p: SwitchLinear, _ a: MLXArray, _ b: MLXArray, _ format: QuantFormat?, _ rounded: MLXArray?
    ) -> MLXArray {
        let w = mergedExpertWeight(p, keep: keep, a: a, b: b, scale: lora.scale)
        guard let format else { return w.asType(x.dtype).swappedAxes(-1, -2) }
        return format.fakeQuantized(w, rounded: rounded, dtype: lora.exportDType).asType(x.dtype).swappedAxes(-1, -2)
    }
    let up = gatherMM(x, merged(bank.upProj, lora.upA, lora.upB, lora.gateUpFormat, lora.roundedUp), rhsIndices: slot, sortedIndices: sort)
    let g = gatherMM(x, merged(bank.gateProj, lora.gateA, lora.gateB, lora.gateUpFormat, lora.roundedGate), rhsIndices: slot, sortedIndices: sort)
    let hidden = bank.activation(g) * up
    var out = gatherMM(hidden, merged(bank.downProj, lora.downA, lora.downB, lora.downFormat, lora.roundedDown), rhsIndices: slot, sortedIndices: sort)
    if let inverseOrder {
        out = out[inverseOrder].reshaped(indices.shape + Array(out.shape.dropFirst()))
    }
    return out.squeezed(axis: -2)
}

private func expertsWithDelta(
    _ bank: SwitchGLU, x input: MLXArray, indices: MLXArray, set: ExpertSet, lora: ExpertLoRA,
    sort: Bool, sliced: Bool
) -> MLXArray {
    var x = expandedDimensions(input, axes: [-2, -3])
    var idx = stopGradient(indices)
    // Sliced: indices already are slots and every one is a kept expert, so the gate is 1.
    var slot = sliced ? idx : set.slotTable[idx]
    var gate = sliced ? MLXArray.ones(idx.shape, dtype: .float32) : set.mask[idx].asType(.float32)
    var inverseOrder: MLXArray?
    if sort {
        let k = indices.dim(-1)
        let flat = idx.flattened()
        let order = argSort(flat)
        inverseOrder = argSort(order)
        x = x.flattened(start: 0, end: -3)[floorDivide(order, k)]
        idx = flat[order]
        slot = slot.flattened()[order]
        gate = gate.flattened()[order][0..., .newAxis, .newAxis]
    } else {
        gate = gate[.ellipsis, .newAxis, .newAxis]
    }

    let pe = lora.perExpertTensors
    func lowRank(_ x: MLXArray, _ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let xa = gatherMM(x, a.asType(x.dtype), rhsIndices: slot, sortedIndices: sort)
            .asType(.float32)
        return gatherMM(xa, b, rhsIndices: slot, sortedIndices: sort)
    }
    func add(_ base: MLXArray, _ delta: MLXArray) -> MLXArray {
        (base.asType(.float32) + lora.scale * delta * gate).asType(base.dtype)
    }

    let up = add(bank.upProj(x, idx, sortedIndices: sort), lowRank(x, pe.upA, pe.upB))
    let g = add(bank.gateProj(x, idx, sortedIndices: sort), lowRank(x, pe.gateA, pe.gateB))
    let hidden = bank.activation(g) * up
    var out = add(bank.downProj(hidden, idx, sortedIndices: sort), lowRank(hidden, pe.downA, pe.downB))
    if let inverseOrder {
        out = out[inverseOrder].reshaped(indices.shape + Array(out.shape.dropFirst()))
    }
    return out.squeezed(axis: -2)
}

/// One decoder layer: the teacher when `set` and `adapters` are `nil`, otherwise the pruned,
/// adapted student.
/// How many layer forwards have been built, in this process: a recompute under `checkpoint` builds one
/// again. Tests compare it between paths to show they do the same amount of compute.
nonisolated(unsafe) var healingLayerForwardCount = 0

func healingLayerForward(
    _ layer: any HealableMoELayer, _ x: MLXArray, set: ExpertSet?, adapters: LayerAdapters?,
    sliced: Bool = false
) -> MLXArray {
    healingLayerForwardCount += 1
    let parts = layer.attentionPart(x, adapters: adapters)
    return healingLayerRemainder(layer, parts, set: set, adapters: adapters, sliced: sliced)
}

/// Everything in a layer after its attention half: routing, the expert branch and the combine.
/// Position-wise, so it runs on any shape of `parts`.
func healingLayerRemainder(
    _ layer: any HealableMoELayer, _ parts: MoELayerParts, set: ExpertSet?, adapters: LayerAdapters?,
    sliced: Bool
) -> MLXArray {
    if sliced {
        guard let set else { preconditionFailure("a sliced layer needs its expert set") }
        let (idx, w) = routeSliced(
            layer, logits: parts.routerLogits, keep: set.keepArray, tweak: adapters?.router,
            routerInput: parts.routerInput)
        let y = expertBranch(
            layer, x: parts.expertInput, indices: idx, weights: w, set: set,
            lora: adapters?.experts, sliced: true)
        return layer.combine(parts, expertBranch: y)
    }
    let (idx, w) = route(
        layer, logits: parts.routerLogits, set: set, tweak: adapters?.router,
        routerInput: parts.routerInput)
    let y = expertBranch(
        layer, x: parts.expertInput, indices: idx, weights: w, set: set, lora: adapters?.experts)
    return layer.combine(parts, expertBranch: y)
}

/// ``healingLayerForward(_:_:set:adapters:sliced:)`` for a student layer with a router correction, also
/// returning the correction's per-position routing tensor `[..., 3]` (see ``routingChannels(scoresKept:routerInput:tweak:keep:topK:)``).
/// The output is the plain forward's, computed by the same code.
func healingLayerForwardAux(
    _ layer: any HealableMoELayer, _ x: MLXArray, set: ExpertSet, adapters: LayerAdapters, sliced: Bool
) -> (MLXArray, MLXArray) {
    healingLayerForwardCount += 1
    let parts = layer.attentionPart(x, adapters: adapters)
    return healingLayerRemainderAux(layer, parts, set: set, adapters: adapters, sliced: sliced)
}

/// ``healingLayerRemainder(_:_:set:adapters:sliced:)`` plus the routing tensor of the router correction.
func healingLayerRemainderAux(
    _ layer: any HealableMoELayer, _ parts: MoELayerParts, set: ExpertSet, adapters: LayerAdapters, sliced: Bool
) -> (MLXArray, MLXArray) {
    let out = healingLayerRemainder(layer, parts, set: set, adapters: adapters, sliced: sliced)
    guard let tweak = adapters.router else {
        return (out, MLXArray.zeros(Array(parts.routerLogits.shape.dropLast()) + [RoutingChannel.count]))
    }
    let kept = sliced ? parts.routerLogits : parts.routerLogits.take(set.keepArray, axis: -1)
    let aux = routingChannels(
        scoresKept: kept, routerInput: parts.routerInput, tweak: tweak, keep: set.keepArray,
        topK: layer.expertRouter.topK)
    return (out, aux)
}

// MARK: - Gemma 4

extension Gemma4MoEDecoderLayer: HealableMoELayer {
    public var hiddenSize: Int { preFeedforwardLayernorm.weight.dim(0) }
    public var expertHiddenSize: Int { experts.switchGLU.gateProj.outputDims }
    public var isGlobalAttention: Bool { !isSliding }

    public var denseSiteDimensions: [DenseSite: (input: Int, output: Int)] {
        // Logical sizes: a quantized weight's stored width is packed.
        let d = hiddenSize
        let h = mlp.gateProj.shape.0
        let a = selfAttn.oProj.shape.1
        return [
            .mlpGate: (d, h), .mlpUp: (d, h), .mlpDown: (h, d), .attentionOutput: (a, d),
        ]
    }

    /// A dense linear with its LoRA: the base output plus the adapter, or, when the adapters
    /// train against a shipping format, the rounded merged weight.
    private func dense(
        _ l: Linear, _ x: MLXArray, _ pair: LoRAPair?, _ adapters: LayerAdapters?, _ site: DenseSite
    ) -> MLXArray {
        if let format = adapters?.denseFormats[site] {
            return quantizationAwareLinear(l, x, pair: pair, format: format, rounded: adapters?.roundedDense[site],
                dtype: adapters?.exportDType ?? .float32)
        }
        let y = l(x)
        return pair.map { $0.apply(x, to: y) } ?? y
    }

    public func attentionPart(_ x: MLXArray, adapters: LayerAdapters?) -> MoELayerParts {
        let raw = selfAttn.attendedWhole(inputLayernorm(x), window: isSliding ? slidingWindow : nil)
        return finishAttentionPart(x, raw: raw, adapters: adapters)
    }

    public var supportsSharedPrefix: Bool { true }

    public func prefixAttentionPart(_ x: MLXArray, adapters: LayerAdapters?) -> (parts: MoELayerParts, prefix: PrefixKV) {
        let a = selfAttn.attendedWholeKeepingKV(inputLayernorm(x), window: isSliding ? slidingWindow : nil)
        return (finishAttentionPart(x, raw: a.out, adapters: adapters), PrefixKV(keys: a.keys, values: a.values))
    }

    public func suffixAttentionPart(_ x: MLXArray, after prefix: PrefixKV, adapters: LayerAdapters?) -> MoELayerParts {
        let raw = selfAttn.attended(
            after: (prefix.keys, prefix.values), inputLayernorm(x), window: isSliding ? slidingWindow : nil)
        return finishAttentionPart(x, raw: raw, adapters: adapters)
    }

    /// The attention output projection, residual and dense branch, from the attention output `raw`
    /// (before `o_proj`) of `x`.
    private func finishAttentionPart(_ x: MLXArray, raw: MLXArray, adapters: LayerAdapters?) -> MoELayerParts {
        let o = dense(selfAttn.oProj, raw, adapters?.attentionOutput, adapters, .attentionOutput)
        let h = x + postAttentionLayernorm(o)

        let n = preFeedforwardLayernorm(h)
        let gate = dense(mlp.gateProj, n, adapters?.mlpGate, adapters, .mlpGate)
        let up = dense(mlp.upProj, n, adapters?.mlpUp, adapters, .mlpUp)
        let hidden = geluApproximate(gate) * up
        let down = dense(mlp.downProj, hidden, adapters?.mlpDown, adapters, .mlpDown)
        let dense = postFeedforwardLayernorm1(down)

        let rin = router.normalizedInput(h)
        return MoELayerParts(
            residual: h, dense: dense, routerLogits: router.proj(rin), routerInput: rin,
            expertInput: preFeedforwardLayernorm2(h))
    }

    public func expertScale(_ indices: MLXArray) -> MLXArray { router.perExpertScale[indices] }

    public func normalizeExpertBranch(_ y: MLXArray) -> MLXArray { postFeedforwardLayernorm2(y) }

    public func combine(_ parts: MoELayerParts, expertBranch y: MLXArray) -> MLXArray {
        (parts.residual + postFeedforwardLayernorm(parts.dense + postFeedforwardLayernorm2(y)))
            * layerScalar
    }
}

extension Gemma4TextModel: HealableMoEModel {
    public var healableLayers: [any HealableMoELayer] {
        layers.map {
            guard let l = $0 as? Gemma4MoEDecoderLayer else {
                preconditionFailure("healing needs MoE decoder layers (enable_moe_block)")
            }
            return l
        }
    }
}

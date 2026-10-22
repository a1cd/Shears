import Foundation
import MLX
import MLXNN
import MLXRandom

/// Sizes and scale of the adapters healing trains.
///
/// The defaults are the recipe's: per-expert LoRA rank 128 on every kept expert's gate, up and
/// down projections, dense LoRA rank 64 on the dense MLP and the attention output projection,
/// and a rank 16 low-rank correction to the router logits. A rank of 0 (or `nil`) turns that
/// family off; the router correction is switched on and off by `routerRank` alone, whatever the others are.
public struct AdapterConfiguration: Sendable, Equatable {
    public var expertRank: Int
    public var denseRank: Int
    public var routerRank: Int
    /// Multiplies every LoRA product (dense and expert). The router correction has no scale.
    public var scale: Float
    /// Which dense linears get a LoRA.
    public var denseSites: Set<DenseSite>
    /// Train against the rounded merged weights of the shipping format (see ``QATConfiguration``).
    public var qat: QATConfiguration?
    /// Multiplies the learning rate of the router correction's parameters (and, with it, their decoupled weight
    /// decay `1 - lr * wd`). 1 is the same rate as every other adapter, exactly.
    public var routerLRMultiplier: Float
    /// One LoRA on the merged expert MLP (gate and up read the layer input, down reads the weighted hidden
    /// layer; see ``ExpertLoRA``) of `expertRank` instead of one LoRA per kept expert. It does not combine with `qat`.
    /// Export folds it back into the per-expert weights exactly.
    public var sharedExperts: Bool

    public init(
        expertRank: Int = 128, denseRank: Int = 64, routerRank: Int = 16, scale: Float = 2,
        denseSites: Set<DenseSite> = Set(DenseSite.allCases), qat: QATConfiguration? = nil,
        routerLRMultiplier: Float = 1, sharedExperts: Bool = false
    ) {
        self.sharedExperts = sharedExperts
        self.routerLRMultiplier = routerLRMultiplier
        self.qat = qat
        self.expertRank = expertRank
        self.denseRank = denseRank
        self.routerRank = routerRank
        self.scale = scale
        self.denseSites = denseSites
    }
}

extension AdapterConfiguration {
    /// Refuses combinations the adapters cannot represent.
    public func validate() throws {
        if sharedExperts, expertRank > 0, qat != nil {
            throw HealingError.invalid("a shared expert LoRA cannot be trained against a quantization format (qat)")
        }
    }
}

/// A dense linear layer inside a decoder layer that can carry a LoRA.
public enum DenseSite: String, CaseIterable, Sendable {
    case mlpGate = "mlp_gate"
    case mlpUp = "mlp_up"
    case mlpDown = "mlp_down"
    case attentionOutput = "attn_out"
}

/// `y + scale * (x @ a) @ b` with `a` `[in, r]` (normal / sqrt(in)) and `b` `[r, out]` (zeros).
///
/// The product runs in float32 whatever the base dtype and is cast back to the base output's
/// dtype, so a zero `b` reproduces the base output bit for bit.
public final class LoRAPair: Module {
    @ParameterInfo(key: "a") public var a: MLXArray
    @ParameterInfo(key: "b") public var b: MLXArray
    public let scale: Float

    public init(inputDims: Int, outputDims: Int, rank: Int, scale: Float) {
        self.scale = scale
        _a.wrappedValue = MLXRandom.normal([inputDims, rank]) * (1 / Float(inputDims).squareRoot())
        _b.wrappedValue = MLXArray.zeros([rank, outputDims])
        super.init()
    }

    /// `y + scale * ((x @ a) @ b)`, where `y` is the frozen layer's output for `x`.
    public func apply(_ x: MLXArray, to y: MLXArray) -> MLXArray {
        y + (scale * matmul(matmul(x, a), b)).asType(y.dtype)
    }
}

/// The LoRA on a layer's expert MLP, in one of two forms.
///
/// **Per expert** (default): one LoRA per kept expert for each of gate, up and down. `*_a` is `[n, in, r]` and
/// `*_b` is `[n, r, out]`, where `n` is the number of kept experts; slot `j` belongs to the `j`-th kept expert in
/// ascending id order.
///
/// **Shared**: one LoRA on the merged expert MLP, whose gate and up are the kept experts' stacked into
/// `hidden -> n * width` projections and whose down is the `n * width -> hidden` projection reading the hidden
/// layer after each expert's chunk is scaled by that expert's routing weight. `gate_a`, `up_a` are `[d, r]`,
/// `gate_b`, `up_b` are `[r, n * width]`, `down_a` is `[n * width, r]`, `down_b` is `[r, d]`. This is exactly a
/// per-expert LoRA whose gate/up `a` is the same for every expert (and whose `b` is the expert's column chunk)
/// and whose down `b` is the same for every expert (and whose `a` is the expert's row chunk), which is how
/// export folds it (see ``perExpertTensors``).
public final class ExpertLoRA: Module {
    @ParameterInfo(key: "gate_a") public var gateA: MLXArray
    @ParameterInfo(key: "gate_b") public var gateB: MLXArray
    @ParameterInfo(key: "up_a") public var upA: MLXArray
    @ParameterInfo(key: "up_b") public var upB: MLXArray
    @ParameterInfo(key: "down_a") public var downA: MLXArray
    @ParameterInfo(key: "down_b") public var downB: MLXArray
    public let scale: Float
    public let slots: Int
    /// Whether this is the shared form.
    public let shared: Bool
    /// Formats the merged expert weights are rounded to in the forward pass (gate and up share a
    /// tensor, so share a format).
    public var gateUpFormat: QuantFormat?
    public var downFormat: QuantFormat?
    /// The dtype the merged weights are exported in (see ``QATConfiguration/exportDType``).
    public var exportDType: DType = .float32
    /// The CPU-rounded merged weights for the current adapters, when the formats need them
    /// (see ``ModelAdapters/refreshRounding(layers:sets:sliced:)``).
    var roundedGate: MLXArray?
    var roundedUp: MLXArray?
    var roundedDown: MLXArray?

    public init(slots n: Int, inputDims d: Int, hiddenDims h: Int, rank r: Int, scale: Float, shared: Bool = false) {
        self.scale = scale
        self.slots = n
        self.shared = shared
        if shared {
            func a(_ i: Int, _ rows: Int) -> MLXArray { MLXRandom.normal([rows, r]) * (1 / Float(i).squareRoot()) }
            _gateA.wrappedValue = a(d, d)
            _gateB.wrappedValue = MLXArray.zeros([r, n * h])
            _upA.wrappedValue = a(d, d)
            _upB.wrappedValue = MLXArray.zeros([r, n * h])
            _downA.wrappedValue = a(n * h, n * h)
            _downB.wrappedValue = MLXArray.zeros([r, d])
        } else {
            func a(_ i: Int) -> MLXArray { MLXRandom.normal([n, i, r]) * (1 / Float(i).squareRoot()) }
            _gateA.wrappedValue = a(d)
            _gateB.wrappedValue = MLXArray.zeros([n, r, h])
            _upA.wrappedValue = a(d)
            _upB.wrappedValue = MLXArray.zeros([n, r, h])
            _downA.wrappedValue = a(h)
            _downB.wrappedValue = MLXArray.zeros([n, r, d])
        }
        super.init()
    }

    /// The six tensors as a per-expert LoRA (`a` `[n, in, r]`, `b` `[n, r, out]`). A per-expert LoRA returns its
    /// own; a shared one is expanded exactly (copies of the shared factor, slices of the other), so anything that
    /// consumes per-expert LoRAs, such as the export fold and the gathered forward path, takes either form.
    public var perExpertTensors: (
        gateA: MLXArray, gateB: MLXArray, upA: MLXArray, upB: MLXArray, downA: MLXArray, downB: MLXArray
    ) {
        guard shared else { return (gateA, gateB, upA, upB, downA, downB) }
        let n = slots
        func tile(_ a: MLXArray) -> MLXArray { broadcast(expandedDimensions(a, axis: 0), to: [n, a.dim(0), a.dim(1)]) }
        // b [r, n*h] -> [n, r, h]
        func chunks(_ b: MLXArray) -> MLXArray { b.reshaped(b.dim(0), n, b.dim(1) / n).transposed(1, 0, 2) }
        // a [n*h, r] -> [n, h, r]
        let h = downA.dim(0) / n
        return (tile(gateA), chunks(gateB), tile(upA), chunks(upB), downA.reshaped(n, h, downA.dim(1)), tile(downB))
    }
}

/// `(rmsNorm(h) @ a) @ b` added to the router logits: `a` `[D, r]`, `b` `[r, E]` (zeros).
/// `E` is the full (unpruned) expert count, so the correction can be folded into the
/// router's projection exactly.
public final class RouterTweak: Module {
    @ParameterInfo(key: "rt_a") public var a: MLXArray
    @ParameterInfo(key: "rt_b") public var b: MLXArray

    public init(inputDims: Int, numExperts: Int, rank: Int) {
        _a.wrappedValue = MLXRandom.normal([inputDims, rank]) * (1 / Float(inputDims).squareRoot())
        _b.wrappedValue = MLXArray.zeros([rank, numExperts])
        super.init()
    }
}

/// The sizes of a layer that adapters are built from, so adapters can be created without the layer
/// (and without loading it between random draws).
public struct LayerDimensions: Sendable {
    public var hiddenSize: Int
    public var expertHiddenSize: Int
    public var denseSites: [DenseSite: (input: Int, output: Int)]
    public init(_ layer: any HealableMoELayer) {
        hiddenSize = layer.hiddenSize
        expertHiddenSize = layer.expertHiddenSize
        denseSites = layer.denseSiteDimensions
    }
}

/// Every adapter of one decoder layer.
public final class LayerAdapters: Module {
    @ModuleInfo(key: "mlp_gate") public var mlpGate: LoRAPair?
    @ModuleInfo(key: "mlp_up") public var mlpUp: LoRAPair?
    @ModuleInfo(key: "mlp_down") public var mlpDown: LoRAPair?
    @ModuleInfo(key: "attn_out") public var attentionOutput: LoRAPair?
    @ModuleInfo(key: "experts") public var experts: ExpertLoRA?
    @ModuleInfo(key: "router") public var router: RouterTweak?
    /// The formats dense weights are rounded to in the forward pass, by site.
    public let denseFormats: [DenseSite: QuantFormat]
    /// The dtype the merged weights are exported in (see ``QATConfiguration/exportDType``).
    public let exportDType: DType
    /// CPU-rounded merged dense weights for the current adapters, by site.
    var roundedDense: [DenseSite: MLXArray] = [:]

    public convenience init(
        layer: any HealableMoELayer, layerIndex: Int = 0, keptExperts: Int, numExperts: Int? = nil,
        configuration c: AdapterConfiguration
    ) {
        self.init(
            dimensions: LayerDimensions(layer), layerIndex: layerIndex, keptExperts: keptExperts,
            numExperts: numExperts ?? layer.numExperts, configuration: c)
    }

    public init(
        dimensions layer: LayerDimensions, layerIndex: Int = 0, keptExperts: Int, numExperts: Int,
        configuration c: AdapterConfiguration
    ) {
        // The router correction spans the unpruned expert count, so the layer must not be
        // asked (a sliced layer reports only its kept experts).
        let fullExperts = numExperts
        var formats: [DenseSite: QuantFormat] = [:]
        if let qat = c.qat {
            for site in DenseSite.allCases { formats[site] = qat.format(site, layer: layerIndex) }
        }
        denseFormats = formats
        exportDType = c.qat?.exportDType ?? .float32
        let dims = layer.denseSites
        func site(_ s: DenseSite) -> LoRAPair? {
            guard c.denseRank > 0, c.denseSites.contains(s), let d = dims[s] else { return nil }
            return LoRAPair(inputDims: d.input, outputDims: d.output, rank: c.denseRank, scale: c.scale)
        }
        // Creation order fixes the random draws: dense first, then experts, then router.
        _mlpGate.wrappedValue = site(.mlpGate)
        _mlpUp.wrappedValue = site(.mlpUp)
        _mlpDown.wrappedValue = site(.mlpDown)
        _attentionOutput.wrappedValue = site(.attentionOutput)
        _experts.wrappedValue = c.expertRank > 0
            ? ExpertLoRA(
                slots: keptExperts, inputDims: layer.hiddenSize, hiddenDims: layer.expertHiddenSize,
                rank: c.expertRank, scale: c.scale, shared: c.sharedExperts)
            : nil
        _router.wrappedValue = c.routerRank > 0
            ? RouterTweak(
                inputDims: layer.hiddenSize, numExperts: fullExperts, rank: c.routerRank)
            : nil
        super.init()
        experts?.gateUpFormat = c.qat?.format(expertGateUp: true, layer: layerIndex)
        experts?.downFormat = c.qat?.format(expertGateUp: false, layer: layerIndex)
        experts?.exportDType = exportDType
    }

    public func dense(_ s: DenseSite) -> LoRAPair? {
        switch s {
        case .mlpGate: mlpGate
        case .mlpUp: mlpUp
        case .mlpDown: mlpDown
        case .attentionOutput: attentionOutput
        }
    }
}

/// The adapters for a contiguous range of decoder layers (all of them, or one block when
/// streaming). Masks and slot tables are deliberately not stored here: they are not
/// parameters, so nothing in this module's tree can be trained or saved by accident.
public final class ModelAdapters: Module {
    @ModuleInfo(key: "layers") public var layers: [LayerAdapters]
    /// Index of the first model layer these adapters cover.
    public let firstLayer: Int
    public let configuration: AdapterConfiguration

    public init(
        layers: [any HealableMoELayer], firstLayer: Int = 0, sets: [ExpertSet],
        configuration: AdapterConfiguration
    ) {
        precondition(layers.count == sets.count, "one expert set per layer")
        self.firstLayer = firstLayer
        self.configuration = configuration
        _layers.wrappedValue = zip(layers, sets).enumerated().map { i, pair in
            LayerAdapters(
                layer: pair.0, layerIndex: firstLayer + i, keptExperts: pair.1.keep.count,
                numExperts: pair.1.numExperts, configuration: configuration)
        }
        super.init()
        eval(self)
    }

    /// Builds the adapters of the layers whose sizes are `dimensions`, starting at `firstLayer`, without
    /// holding the layers. The random draws are made in layer order, as in
    /// ``init(layers:firstLayer:sets:configuration:)``, so the result is the same.
    public init(
        dimensions: [LayerDimensions], firstLayer: Int, sets: [ExpertSet], configuration: AdapterConfiguration
    ) {
        precondition(dimensions.count == sets.count, "one expert set per layer")
        self.firstLayer = firstLayer
        self.configuration = configuration
        _layers.wrappedValue = dimensions.enumerated().map { i, d in
            LayerAdapters(
                dimensions: d, layerIndex: firstLayer + i, keptExperts: sets[i].keep.count,
                numExperts: sets[i].numExperts, configuration: configuration)
        }
        super.init()
        eval(self)
    }

    /// Whether any covered module uses a format that is rounded on the CPU.
    public var needsRefresh: Bool {
        layers.contains { la in
            la.denseFormats.values.contains { $0.needsRefresh }
                || (la.experts.map { ($0.gateUpFormat?.needsRefresh ?? false) || ($0.downFormat?.needsRefresh ?? false) } ?? false)
        }
    }

    /// Rounds the merged weights of every module whose format runs on the CPU (the GGUF K-quants)
    /// for the *current* adapters, so the next traced forward pass can use the result. Call it
    /// whenever the adapters have changed; ``PruneHealer`` and ``StreamedBlockHealer`` do so before
    /// each step. A model with only in-graph formats (or none) has nothing to do.
    public func refreshRounding(layers: [any HealableMoELayer], sets: [ExpertSet], sliced: Bool) {
        guard needsRefresh else { return }
        for i in self.layers.indices { refreshRounding(layerAt: i, layer: layers[i], set: sets[i], sliced: sliced) }
    }

    /// ``refreshRounding(layers:sets:sliced:)`` for the layer at block index `i` alone, for a run that
    /// holds only a window of the block's layers.
    public func refreshRounding(layerAt i: Int, layer: any HealableMoELayer, set: ExpertSet, sliced: Bool) {
        let sets = [i: set]
        let la = self.layers[i]
        do {
            guard needsRefresh else { return }
            guard let l = layer as? Gemma4MoEDecoderLayer else { return }
            for (site, format) in la.denseFormats where format.needsRefresh {
                let lin: Linear
                switch site {
                case .mlpGate: lin = l.mlp.gateProj
                case .mlpUp: lin = l.mlp.upProj
                case .mlpDown: lin = l.mlp.downProj
                case .attentionOutput: lin = l.selfAttn.oProj
                }
                la.roundedDense[site] = format.roundTrip(mergedDenseWeight(lin, la.dense(site)), dtype: la.exportDType)
            }
            if let lora = la.experts {
                let bank = l.experts.switchGLU
                let keep: MLXArray? = sliced ? nil : sets[i]!.keepArray
                if let f = lora.gateUpFormat, f.needsRefresh {
                    lora.roundedGate = f.roundTrip(mergedExpertWeight(bank.gateProj, keep: keep, a: lora.gateA, b: lora.gateB, scale: lora.scale), dtype: lora.exportDType)
                    lora.roundedUp = f.roundTrip(mergedExpertWeight(bank.upProj, keep: keep, a: lora.upA, b: lora.upB, scale: lora.scale), dtype: lora.exportDType)
                }
                if let f = lora.downFormat, f.needsRefresh {
                    lora.roundedDown = f.roundTrip(mergedExpertWeight(bank.downProj, keep: keep, a: lora.downA, b: lora.downB, scale: lora.scale), dtype: lora.exportDType)
                }
            }
        }
    }

    /// Parameter counts by family: `dense`, `experts`, `router`, `total`.
    public func parameterCounts() -> [String: Int] {
        var out = ["dense": 0, "experts": 0, "router": 0]
        for (k, v) in parameters().flattened() {
            if k.contains(".experts.") { out["experts", default: 0] += v.size }
            else if k.contains(".router.") { out["router", default: 0] += v.size }
            else { out["dense", default: 0] += v.size }
        }
        out["total"] = out.values.reduce(0, +)
        return out
    }

    /// Every adapter tensor under global names (`layers.<model layer>.…`).
    public func tensors() -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in parameters().flattened() { out[globalName(k)] = v }
        return out
    }

    func globalName(_ local: String) -> String {
        let parts = local.split(separator: ".", maxSplits: 2).map(String.init)
        guard parts.count == 3, parts[0] == "layers", let i = Int(parts[1]) else { return local }
        return "layers.\(i + firstLayer).\(parts[2])"
    }

    func localName(_ global: String) -> String? {
        let parts = global.split(separator: ".", maxSplits: 2).map(String.init)
        guard parts.count == 3, parts[0] == "layers", let i = Int(parts[1]) else { return nil }
        let local = i - firstLayer
        guard local >= 0, local < layers.count else { return nil }
        return "layers.\(local).\(parts[2])"
    }

    /// Replaces the adapters with `tensors`. Throws unless the key set and every shape match
    /// exactly; a dense-only file can never load as a full adapter set.
    public func load(tensors: [String: MLXArray]) throws {
        let mine = Dictionary(uniqueKeysWithValues: parameters().flattened().map { (globalName($0.0), $0.1) })
        let missing = Set(mine.keys).subtracting(tensors.keys)
        let extra = Set(tensors.keys).subtracting(mine.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw HealingError.adapterMismatch(
                "adapter keys differ: \(extra.count) unexpected (e.g. \(extra.sorted().prefix(3))), "
                    + "\(missing.count) missing (e.g. \(missing.sorted().prefix(3)))")
        }
        for (k, v) in mine where tensors[k]!.shape != v.shape {
            throw HealingError.adapterMismatch("shape of \(k): \(tensors[k]!.shape) vs \(v.shape)")
        }
        let flat = tensors.compactMap { (k, v) in localName(k).map { ($0, v) } }
        update(parameters: ModuleParameters.unflattened(flat))
        eval(self)
    }

    /// Like ``load(tensors:)`` for a subset: only the keys that belong to this range are taken
    /// from `tensors`, and all of them must be present.
    public func load(subsetOf tensors: [String: MLXArray]) throws {
        let mine = Set(parameters().flattened().map { globalName($0.0) })
        try load(tensors: tensors.filter { mine.contains($0.key) })
    }
}

public enum HealingError: Error, CustomStringConvertible {
    case adapterMismatch(String)
    case invalid(String)
    /// A memory budget the run cannot meet, found before any training.
    case budget(String)
    public var description: String {
        switch self {
        case .budget(let s): s
        case .adapterMismatch(let s): "adapter mismatch: \(s)"
        case .invalid(let s): s
        }
    }
}

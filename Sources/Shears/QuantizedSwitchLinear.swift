import MLX
import MLXNN
import MLXRandom

/// A ``SwitchLinear`` whose expert weights are frozen and quantized, computed with
/// `gatherQuantizedMM`, with an optional per-expert LoRA adapter.
///
/// MLX's VJP for `gatherQuantizedMM` gives a gradient for the activations only, which is
/// exactly what's needed: the quantized base is a constant, gradients flow through it to
/// earlier layers, and only the router and LoRA adapters train.
///
/// LoRA here is per expert: `loraA` is `[E, r, in]`, `loraB` is `[E, out, r]`, and the
/// adapter output is `scale * B_e (A_e x)` for the token's expert `e`. `loraB` starts at
/// zero, so an adapter does not change the layer's output until it is trained.
open class QuantizedSwitchLinear: SwitchLinear, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    @ParameterInfo(key: "scales") public var scales: MLXArray
    @ParameterInfo(key: "biases") public var biases: MLXArray?
    @ParameterInfo(key: "lora_a") public var loraA: MLXArray?
    @ParameterInfo(key: "lora_b") public var loraB: MLXArray?
    public let loraScale: Float

    /// Quantize an existing ``SwitchLinear``.
    public convenience init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine, loraRank: Int? = nil, loraScale: Float = 1
    ) {
        let (wq, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)
        self.init(
            weight: wq, scales: scales, biases: biases, bias: other.bias,
            inputDims: other.inputDims, outputDims: other.outputDims,
            groupSize: groupSize, bits: bits, mode: mode,
            loraRank: loraRank, loraScale: loraScale)
    }

    /// From already-quantized arrays (an mlx-community checkpoint's `weight`, `scales`,
    /// `biases`).
    public init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?, bias: MLXArray? = nil,
        inputDims: Int, outputDims: Int,
        groupSize: Int, bits: Int, mode: QuantizationMode = .affine,
        loraRank: Int? = nil, loraScale: Float = 1
    ) {
        precondition(weight.ndim == 3, "quantized SwitchLinear weight must be [experts, out, packed]")
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.loraScale = loraScale
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        let e = weight.dim(0)
        if let r = loraRank {
            let s = 1 / Float(inputDims).squareRoot()
            self._loraA.wrappedValue = MLXRandom.uniform(low: -s, high: s, [e, r, inputDims])
            self._loraB.wrappedValue = MLXArray.zeros([e, outputDims, r])
        } else {
            self._loraA.wrappedValue = nil
            self._loraB.wrappedValue = nil
        }
        super.init(storedWeight: weight, bias: bias, inputDims: inputDims, outputDims: outputDims)
        freeze()
        if hasLoRA { unfreeze(recursive: false, keys: ["lora_a", "lora_b"]) }
    }

    public var hasLoRA: Bool { loraA != nil && loraB != nil }

    /// A copy sharing this layer's quantized weights, with fresh LoRA adapters of `rank`.
    public func withLoRA(rank: Int, scale: Float = 1) -> QuantizedSwitchLinear {
        QuantizedSwitchLinear(
            weight: weight, scales: scales, biases: biases, bias: bias,
            inputDims: inputDims, outputDims: outputDims,
            groupSize: groupSize, bits: bits, mode: mode, loraRank: rank, loraScale: scale)
    }

    open override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var y = gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices,
            transpose: true, groupSize: groupSize, bits: bits, mode: mode,
            sortedIndices: sortedIndices)
        if let bias {
            y = y + expandedDimensions(bias[indices], axis: -2)
        }
        if let loraA, let loraB {
            let h = gatherMM(
                x, loraA.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sortedIndices)
            let delta = gatherMM(
                h, loraB.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sortedIndices)
            y = y + loraScale * delta
        }
        return y
    }

    open override func keepExperts(_ keep: MLXArray) {
        var p: [(String, MLXArray)] = [
            ("weight", weight.take(keep, axis: 0)),
            ("scales", scales.take(keep, axis: 0)),
        ]
        if let biases { p.append(("biases", biases.take(keep, axis: 0))) }
        if let bias { p.append(("bias", bias.take(keep, axis: 0))) }
        if let loraA { p.append(("lora_a", loraA.take(keep, axis: 0))) }
        if let loraB { p.append(("lora_b", loraB.take(keep, axis: 0))) }
        update(parameters: ModuleParameters.unflattened(p))
    }

    /// The dequantized base weight plus the merged LoRA delta, `[E, out, in]`.
    public func dequantizedWeight() -> MLXArray {
        var w = dequantized(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: mode)
        if let loraA, let loraB {
            w = w + loraScale * matmul(loraB, loraA)
        }
        return w
    }
}

public enum ExpertQuantizer {
    /// Replace a block's expert projections with frozen quantized ones, optionally with
    /// LoRA adapters. The router stays full precision. Projections that are already
    /// quantized (e.g. loaded from a 4-bit checkpoint) are kept, and get adapters if
    /// `loraRank` is set and they have none.
    public static func quantizeExperts(
        _ block: any MixtureOfExperts, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine, loraRank: Int? = nil, loraScale: Float = 1
    ) {
        func q(_ l: SwitchLinear) -> SwitchLinear {
            if let q = l as? QuantizedSwitchLinear {
                if let r = loraRank, !q.hasLoRA { return q.withLoRA(rank: r, scale: loraScale) }
                return q
            }
            return QuantizedSwitchLinear(
                l, groupSize: groupSize, bits: bits, mode: mode,
                loraRank: loraRank, loraScale: loraScale)
        }
        let g = block.expertBank
        g.update(modules: ModuleChildren.unflattened([
            ("gate_proj", q(g.gateProj)),
            ("up_proj", q(g.upProj)),
            ("down_proj", q(g.downProj)),
        ]))
        eval(g)
    }
}

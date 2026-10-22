import MLX
import MLXNN
import MLXRandom

/// A bank of `E` linear layers selected per token by an expert index.
///
/// Port of mlx-lm's `SwitchLinear` (`mlx_lm/models/switch_layers.py`). `weight` is
/// `[E, out, in]`, matching mlx-lm's `switch_mlp.*.weight` checkpoint layout.
open class SwitchLinear: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    @ParameterInfo(key: "bias") public var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = false) {
        let scale = (1 / Float(inputDims)).squareRoot()
        self.inputDims = inputDims
        self.outputDims = outputDims
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [numExperts, outputDims, inputDims])
        self._bias.wrappedValue = bias ? MLXArray.zeros([numExperts, outputDims]) : nil
        super.init()
    }

    public init(weight: MLXArray, bias: MLXArray? = nil) {
        precondition(weight.ndim == 3, "SwitchLinear weight must be [experts, out, in]")
        self.inputDims = weight.dim(2)
        self.outputDims = weight.dim(1)
        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
        super.init()
    }

    /// For subclasses that store `weight` in another layout (quantized).
    init(storedWeight: MLXArray, bias: MLXArray?, inputDims: Int, outputDims: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self._weight.wrappedValue = storedWeight
        self._bias.wrappedValue = bias
        super.init()
    }

    open var numExperts: Int { weight.dim(0) }

    /// - Parameters:
    ///   - x: `[..., 1, in]` (or `[N, 1, in]` on the sorted path)
    ///   - indices: expert ids, broadcast against `x`'s batch dims
    open func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var y = gatherMM(
            x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sortedIndices)
        if let bias {
            y = y + expandedDimensions(bias[indices], axis: -2)
        }
        return y
    }

    /// Keep only the experts in `keep` (sorted int32 ids), along axis 0.
    open func keepExperts(_ keep: MLXArray) {
        var p: [(String, MLXArray)] = [("weight", weight.take(keep, axis: 0))]
        if let bias { p.append(("bias", bias.take(keep, axis: 0))) }
        update(parameters: ModuleParameters.unflattened(p))
    }
}

/// `down(activation(gate(x)) * up(x))` over a bank of experts. Port of mlx-lm's `SwitchGLU`.
///
/// `activation` defaults to SiLU (Qwen3-MoE, Mixtral); Gemma 4 uses `geluApproximate`.
open class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    /// mlx-lm sorts tokens by expert once `indices.size` reaches this, so `gatherMM`
    /// reads each expert's weights in one contiguous run.
    public var sortThreshold = 64

    /// Applied to the gate projection. Not a parameter, so not part of the checkpoint.
    public let activation: (MLXArray) -> MLXArray

    public init(
        inputDims: Int, hiddenDims: Int, numExperts: Int, bias: Bool = false,
        activation: @escaping (MLXArray) -> MLXArray = silu
    ) {
        self.activation = activation
        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
        super.init()
    }

    public init(
        gateProj: SwitchLinear, upProj: SwitchLinear, downProj: SwitchLinear,
        activation: @escaping (MLXArray) -> MLXArray = silu
    ) {
        self.activation = activation
        self._gateProj.wrappedValue = gateProj
        self._upProj.wrappedValue = upProj
        self._downProj.wrappedValue = downProj
        super.init()
    }

    public var numExperts: Int { gateProj.numExperts }

    /// - Parameters:
    ///   - x: `[..., D]`
    ///   - indices: `[..., k]` expert ids per token
    /// - Returns: `[..., k, D]`, one output per selected expert
    ///
    /// While the module is in training mode (`Module.training`, the MLXNN default) this
    /// always takes the sorted path. `GatherMM`'s weight gradient on unsorted indices
    /// materialises an `[out, in]` matrix per token before scatter-adding them, which is
    /// unusable at real sizes; sorted indices use a segmented matmul instead. Call
    /// `train(false)` for inference to get mlx-lm's size-based choice.
    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        callAsFunction(x, indices, sort: training || indices.size >= sortThreshold)
    }

    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sort: Bool) -> MLXArray {
        var x = expandedDimensions(x, axes: [-2, -3])
        var idx = stopGradient(indices)
        var inverseOrder: MLXArray?

        if sort {
            // _gather_sort: one row per (token, slot), ordered by expert id.
            let k = indices.dim(-1)
            let flat = idx.flattened()
            let order = argSort(flat)
            inverseOrder = argSort(order)
            x = x.flattened(start: 0, end: -3)[floorDivide(order, k)]
            idx = flat[order]
        }

        let up = upProj(x, idx, sortedIndices: sort)
        let gate = gateProj(x, idx, sortedIndices: sort)
        x = downProj(activation(gate) * up, idx, sortedIndices: sort)

        if let inverseOrder {
            // _scatter_unsort: undo the permutation, then restore [..., k, 1, D].
            x = x[inverseOrder].reshaped(indices.shape + Array(x.shape.dropFirst()))
        }
        return x.squeezed(axis: -2)
    }

    open func keepExperts(_ keep: MLXArray) {
        gateProj.keepExperts(keep)
        upProj.keepExperts(keep)
        downProj.keepExperts(keep)
    }
}

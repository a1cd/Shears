import Foundation
import MLX
import MLXNN

/// A weight format healing can train against: the format the healed model will ship in.
///
/// Only formats whose rounding is reproduced exactly are offered. MLX's affine quantizer is the
/// one the model is loaded and exported with; `q8_0` is GGUF's, matching `quantize_row_q8_0_ref`
/// and checked against this package's own writer. GGUF's K-quants (`Q4_K` and the `Q4_K_M` mix)
/// are not offered: their quantizer is an iterative search whose reference source was not
/// available to compare against, and a close imitation would make "the export equals the healed
/// model" untrue.
public enum QuantFormat: Sendable, Equatable {
    /// A GGUF K-quant or `Q5_0`, rounded exactly as `llama-quantize` writes it (see
    /// ``GGUFQ4K``). These quantizers search over scales and run on the CPU, so a format of this
    /// kind is rounded once per step *before* the traced forward pass (see
    /// ``ModelAdapters/refreshRounding(layers:sets:sliced:)``), and the forward uses that result.
    case gguf(GGUFTensorType)
    /// MLX affine quantization along the last axis (`bits` 2 to 8, `groupSize` 32, 64 or 128).
    case affine(bits: Int, groupSize: Int)
    /// GGUF `Q8_0`: blocks of 32 along the last axis, a half-precision scale `max|x| / 127`, int8 codes.
    case q8_0

    /// Last-axis block size; a tensor whose last axis is not a multiple of it ships in a wider
    /// format (the GGUF writer falls back to F16), so it is left unrounded here.
    var block: Int {
        switch self {
        case .gguf(let t): t.blockSize
        case .affine(_, let g): g
        case .q8_0: 32
        }
    }

    /// The weight as it will be after shipping and loading: quantize, then dequantize, in float32.
    /// Exactly the format's own round trip.
    ///
    /// `dtype` is the dtype the merged weight is exported in. The quantizer sees exactly the values
    /// that file holds (`w` cast to `dtype`, which widens exactly to float32), so the codes it picks
    /// are the codes a converter picks when it quantizes the exported tensor. An affine quantizer
    /// runs on the `dtype` array itself, so its scales and biases come out in `dtype`, as MLX's own
    /// conversion of a bf16 checkpoint gives.
    public func roundTrip(_ w: MLXArray, dtype: DType = .float32) -> MLXArray {
        let exported = w.asType(dtype)
        let w = exported.asType(.float32)
        guard w.ndim >= 2, w.dim(-1) % block == 0 else { return w }
        switch self {
        case .gguf(let type):
            return Self.roundTripOnCPU(w, type: type)
        case .affine(let bits, let groupSize):
            let flat = exported.reshaped(-1, exported.dim(-1))
            let (q, s, b) = MLX.quantized(flat, groupSize: groupSize, bits: bits)
            return dequantized(q, scales: s, biases: b, groupSize: groupSize, bits: bits)
                .asType(.float32).reshaped(w.shape)
        case .q8_0:
            let blocks = w.reshaped(-1, 32)
            let amax = abs(blocks).max(axis: -1, keepDims: true)
            let d = amax / 127
            let id = MLX.where(d .== 0, MLXArray(Float(0)), 1 / d)
            // The scale is stored as a half; the codes use the unrounded one, as in ggml.
            let stored = d.asType(.float16).asType(.float32)
            let scaled = abs(blocks * id)
            // roundf: halves go away from zero. Taking the integer part and testing the fraction
            // is exact, unlike floor(x + 0.5).
            let whole = floor(scaled)
            let q = minimum(whole + MLX.where(scaled - whole .>= 0.5, MLXArray(Float(1)), MLXArray(Float(0))), 127)
            return (sign(blocks) * q * stored).reshaped(w.shape)
        }
    }

    /// Whether the rounding has to be computed outside the traced forward pass.
    var needsRefresh: Bool {
        if case .gguf = self { return true }
        return false
    }

    /// Straight-through estimator: the value is exactly `roundTrip(w)`; the gradient is the
    /// identity, so it reaches whatever `w` was built from.
    ///
    /// A CPU-rounded format cannot round a traced array, so it takes `rounded`, the round trip of
    /// this very `w` computed beforehand, and adds the zero-valued `w - stopGradient(w)` to it.
    public func fakeQuantized(_ w: MLXArray, rounded: MLXArray? = nil, dtype: DType = .float32) -> MLXArray {
        if needsRefresh {
            guard let rounded else {
                preconditionFailure("\(self) rounds on the CPU: refresh the rounding before the forward pass")
            }
            return rounded + (w.asType(.float32) - stopGradient(w.asType(.float32)))
        }
        return roundTrip(stopGradient(w), dtype: dtype) + (w.asType(.float32) - stopGradient(w.asType(.float32)))
    }

    /// Quantize and dequantize `w` with the reference K-quant code, rows spread over the cores.
    static func roundTripOnCPU(_ w: MLXArray, type: GGUFTensorType) -> MLXArray {
        let shape = w.shape
        let perRow = shape[shape.count - 1]
        let values = w.reshaped(-1).asArray(Float.self)
        let rows = values.count / perRow
        var out = [Float](repeating: 0, count: values.count)
        let chunkRows = max(1, min(64, rows / (ProcessInfo.processInfo.activeProcessorCount * 2) + 1))
        let chunks = (rows + chunkRows - 1) / chunkRows
        values.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                let dstBase = dst.baseAddress!
                DispatchQueue.concurrentPerform(iterations: chunks) { c in
                    let r0 = c * chunkRows, r1 = min(rows, r0 + chunkRows)
                    let slice = UnsafeBufferPointer(rebasing: src[r0 * perRow ..< r1 * perRow])
                    let bytes: Data
                    let back: [Float]
                    switch type {
                    case .q4_k: bytes = GGUFQ4K.quantize(slice, perRow: perRow); back = GGUFQ4K.dequantize(bytes)
                    case .q5_k: bytes = GGUFQ5K.quantize(slice, perRow: perRow); back = GGUFQ5K.dequantize(bytes)
                    case .q6_k: bytes = GGUFQ6K.quantize(slice, perRow: perRow); back = GGUFQ6K.dequantize(bytes)
                    case .q5_0: bytes = GGUFQ5_0.quantize(slice, perRow: perRow); back = GGUFQ5_0.dequantize(bytes)
                    default: preconditionFailure("\(type.name) is not rounded on the CPU")
                    }
                    for (k, v) in back.enumerated() { dstBase[r0 * perRow + k] = v }
                }
            }
        }
        return MLXArray(out, shape)
    }
}

/// Which parts of the model are rounded to the shipping format during healing.
///
/// For each rounded module the forward pass uses `fakeQuantized(base + scale * (a @ b)^T)`:
/// the merged weight as it will be exported, rounded, with gradients passed straight through to
/// the full-precision adapters. Covered: the dense MLP and attention output projections (`dense`)
/// and the kept experts' gate, up and down projections (`experts`). Not covered: the router (it
/// ships as float32), norms, the query, key and value projections and the embeddings, which are
/// frozen (a frozen weight's rounding is a constant, not something healing can reduce).
/// The teacher is never rounded.
///
/// The rounding is applied to the merged weight *as exported*: ``exportDType`` is the dtype
/// ``MergeAndSliceExport`` writes (it refuses a different one), and the quantizer is run on the
/// merged weight cast to it. The export writes that unrounded tensor and leaves the quantizing to
/// the converter (GGUF's, or MLX's affine), which sees the same bits and so picks the same codes.
/// Pre-rounding in the export would not do: rounding and then casting to bfloat16 is not a fixed
/// point of the quantizer.
public struct QATConfiguration: Sendable, Equatable {
    /// The dtype the healed model is exported in; the training rounding is applied to the merged
    /// weight cast to it. Defaults to the export default, bfloat16.
    public var exportDType: DType = .bfloat16
    /// One format for every covered dense module (unless ``formats`` names it).
    public var dense: QuantFormat?
    /// One format for every kept expert (unless ``formats`` names it).
    public var experts: QuantFormat?
    /// Per-tensor formats under their GGUF names (`blk.3.ffn_down.weight`,
    /// `blk.3.ffn_gate_up_exps.weight`, ...), as ``q4KM(for:keptExperts:)`` fills them. A tensor
    /// named here uses this format, a tensor not named falls back to `dense` / `experts`.
    public var formats: [String: QuantFormat]

    public init(
        dense: QuantFormat? = nil, experts: QuantFormat? = nil, formats: [String: QuantFormat] = [:],
        exportDType: DType = .bfloat16
    ) {
        self.exportDType = exportDType
        self.dense = dense
        self.experts = experts
        self.formats = formats
    }
    /// The same format everywhere.
    public init(_ format: QuantFormat, exportDType: DType = .bfloat16) {
        self.init(dense: format, experts: format, exportDType: exportDType)
    }

    func format(_ site: DenseSite, layer: Int) -> QuantFormat? {
        formats["blk.\(layer).\(site.ggufName).weight"] ?? dense
    }
    func format(expertGateUp: Bool, layer: Int) -> QuantFormat? {
        formats["blk.\(layer).\(expertGateUp ? "ffn_gate_up_exps" : "ffn_down_exps").weight"] ?? experts
    }

    /// Train against what `llama-quantize ... Q4_K_M` will write for this model once exported with
    /// `keptExperts` experts per layer: Q4_K mostly, Q5_K / Q6_K / Q8_0 for the sensitive tensors,
    /// Q5_0 or Q8_0 where a row is not a multiple of 256 (see ``GGUFQuantMix``).
    public static func q4KM(
        for model: Gemma4TextModel, keptExperts: Int, exportDType: DType = .bfloat16
    ) -> QATConfiguration {
        let c = model.configuration
        var tensors: [(name: String, shape: [Int])] = [("token_embd.weight", [c.vocabSize, c.hiddenSize])]
        if !c.tieWordEmbeddings { tensors.append(("output.weight", [c.vocabSize, c.hiddenSize])) }
        for (i, layer) in model.layers.enumerated() {
            let h = c.hiddenSize
            let a = layer.selfAttn
            let q = a.numHeads * a.headDim, kv = a.numKVHeads * a.headDim
            let p = "blk.\(i)."
            tensors += [(p + "attn_q.weight", [q, h]), (p + "attn_k.weight", [kv, h]), (p + "attn_output.weight", [h, q])]
            if a.vProj != nil { tensors.append((p + "attn_v.weight", [kv, h])) }
            let ff = layer.mlp.gateProj.shape.0
            tensors += [(p + "ffn_gate.weight", [ff, h]), (p + "ffn_up.weight", [ff, h]), (p + "ffn_down.weight", [h, ff])]
            if let moe = layer as? Gemma4MoEDecoderLayer {
                let w = moe.experts.switchGLU.gateProj.outputDims
                tensors += [(p + "ffn_gate_up_exps.weight", [keptExperts, 2 * w, h]), (p + "ffn_down_exps.weight", [keptExperts, h, w])]
            }
        }
        let plan = GGUFQuantMix.q4KM(tensors, layerCount: c.numHiddenLayers, expertCount: keptExperts, half: .bf16)
        var formats: [String: QuantFormat] = [:]
        for (name, type) in plan {
            switch type {
            case .q8_0: formats[name] = .q8_0
            case .q4_k, .q5_k, .q6_k, .q5_0: formats[name] = .gguf(type)
            default: break
            }
        }
        return QATConfiguration(formats: formats, exportDType: exportDType)
    }
}

extension DenseSite {
    var ggufName: String {
        switch self {
        case .mlpGate: "ffn_gate"
        case .mlpUp: "ffn_up"
        case .mlpDown: "ffn_down"
        case .attentionOutput: "attn_output"
        }
    }
}

/// A linear layer's weight as float32, dequantized if it is stored quantized.
func fullPrecisionWeight(_ l: Linear) -> MLXArray {
    if let q = l as? QuantizedLinear {
        return dequantized(q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            .asType(.float32)
    }
    return l.weight.asType(.float32)
}

/// `W + scale * (a @ b)^T`, the weight a dense linear has once its LoRA is merged.
func mergedDenseWeight(_ l: Linear, _ pair: LoRAPair?) -> MLXArray {
    var w = fullPrecisionWeight(l)
    if let pair { w = w + pair.scale * matmul(pair.a, pair.b).T }
    return w
}

/// `x @ fakeQuantized(W + scale * (a @ b)^T)^T` for a dense linear with an optional LoRA pair.
func quantizationAwareLinear(
    _ l: Linear, _ x: MLXArray, pair: LoRAPair?, format: QuantFormat, rounded: MLXArray?, dtype: DType
) -> MLXArray {
    let q = format.fakeQuantized(mergedDenseWeight(l, pair), rounded: rounded, dtype: dtype).asType(x.dtype)
    return matmul(x, q.T)
}

/// The kept experts' merged weights `[K, out, in]`: base plus `scale * (a @ b)^T` per expert.
func mergedExpertWeight(_ p: SwitchLinear, keep: MLXArray?, a: MLXArray, b: MLXArray, scale: Float) -> MLXArray {
    keptExpertWeight(p, keep: keep) + scale * matmul(a, b).swappedAxes(-1, -2)
}

/// One expert projection's weights `[K, out, in]` (the kept experts only) in float32.
func keptExpertWeight(_ p: SwitchLinear, keep: MLXArray?) -> MLXArray {
    if let q = p as? QuantizedSwitchLinear {
        var (w, s, b) = (q.weight, q.scales, q.biases)
        if let keep {
            w = w.take(keep, axis: 0)
            s = s.take(keep, axis: 0)
            b = b?.take(keep, axis: 0)
        }
        return dequantized(w, scales: s, biases: b, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            .asType(.float32)
    }
    let w = keep.map { p.weight.take($0, axis: 0) } ?? p.weight
    return w.asType(.float32)
}

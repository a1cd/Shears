import MLX

/// Per-expert routing statistics accumulated over calibration batches.
///
/// Attach one to a ``MoEBlock`` (`block.stats = ExpertStats(numExperts:)`), run
/// calibration data through, then ask for ``expertsToKeep(count:by:)``. Each batch is
/// reduced to `[E]` sums and evaluated immediately, so the memory cost is constant no
/// matter how much data goes through.
public final class ExpertStats {
    public enum Criterion {
        /// How many times the expert was in a token's top-k.
        case pickCount
        /// Total routing weight the expert received.
        case weightMass
        /// REAP-style saliency: the mean over tokens routed to the expert of
        /// `score * ||expert(x)||`, so an expert that is picked often but contributes
        /// little ranks low.
        case reap
    }

    public let numExperts: Int
    public private(set) var tokens = 0
    public private(set) var pickCount: MLXArray
    public private(set) var weightMass: MLXArray
    public private(set) var saliencySum: MLXArray

    public init(numExperts: Int) {
        self.numExperts = numExperts
        self.pickCount = MLXArray.zeros([numExperts], dtype: .float32)
        self.weightMass = MLXArray.zeros([numExperts], dtype: .float32)
        self.saliencySum = MLXArray.zeros([numExperts], dtype: .float32)
    }

    /// - Parameters:
    ///   - indices: `[..., k]` expert ids
    ///   - scores: `[..., k]` routing weights
    ///   - expertOutputs: `[..., k, D]` per-slot expert outputs (for ``Criterion/reap``)
    public func record(indices: MLXArray, scores: MLXArray, expertOutputs: MLXArray? = nil) {
        let flatIdx = stopGradient(indices).flattened()
        let flatScores = stopGradient(scores).flattened().asType(.float32)
        // [N*k, E] one-hot of which expert each (token, slot) went to.
        let oneHot = (expandedDimensions(flatIdx, axis: -1)
            .== MLXArray.arange(numExperts, dtype: flatIdx.dtype)).asType(.float32)

        pickCount = pickCount + oneHot.sum(axis: 0)
        let weighted = oneHot * expandedDimensions(flatScores, axis: -1)
        weightMass = weightMass + weighted.sum(axis: 0)

        if let expertOutputs {
            let y = stopGradient(expertOutputs).asType(.float32)
            let norms = sqrt((y * y).sum(axis: -1)).flattened()
            saliencySum = saliencySum + (weighted * expandedDimensions(norms, axis: -1)).sum(axis: 0)
            eval(pickCount, weightMass, saliencySum)
        } else {
            eval(pickCount, weightMass)
        }
        tokens += flatIdx.size / max(indices.dim(-1), 1)
    }

    public func scores(by criterion: Criterion) -> MLXArray {
        switch criterion {
        case .pickCount: pickCount
        case .weightMass: weightMass
        case .reap: saliencySum / maximum(pickCount, 1)
        }
    }

    /// The `count` highest-scoring expert ids, ascending.
    public func expertsToKeep(count: Int, by criterion: Criterion = .weightMass) -> [Int] {
        precondition(count > 0 && count <= numExperts, "keep count out of range")
        let s = scores(by: criterion).asArray(Float.self)
        let ranked = s.indices.sorted { s[$0] != s[$1] ? s[$0] > s[$1] : $0 < $1 }
        return ranked.prefix(count).sorted()
    }

    public func reset() {
        tokens = 0
        pickCount = MLXArray.zeros([numExperts], dtype: .float32)
        weightMass = MLXArray.zeros([numExperts], dtype: .float32)
        saliencySum = MLXArray.zeros([numExperts], dtype: .float32)
    }
}

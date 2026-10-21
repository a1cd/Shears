import MLX

/// `MLX.gatherMM`, except on a CPU default device with inputs that are not float32.
///
/// MLX's CPU backend computes `gatherMM` in float32 only and stops the process on anything else
/// ("[GatherMM::eval] Currently only supports float32"); Metal and CUDA take half types. Declared in
/// this module, it is what every unqualified `gatherMM` call here resolves to. On a CPU device it
/// computes in float32 and casts the result to the type MLX would have returned.
func gatherMM(
    _ a: MLXArray, _ b: MLXArray, lhsIndices: MLXArray? = nil, rhsIndices: MLXArray? = nil,
    sortedIndices: Bool = false
) -> MLXArray {
    guard Device.defaultDevice().deviceType == .cpu, a.dtype != .float32 || b.dtype != .float32 else {
        return MLX.gatherMM(a, b, lhsIndices: lhsIndices, rhsIndices: rhsIndices, sortedIndices: sortedIndices)
    }
    let resultType = a.dtype == b.dtype
        ? a.dtype : (MLXArray.zeros([0], dtype: a.dtype) + MLXArray.zeros([0], dtype: b.dtype)).dtype
    return MLX.gatherMM(
        a.asType(.float32), b.asType(.float32), lhsIndices: lhsIndices, rhsIndices: rhsIndices,
        sortedIndices: sortedIndices
    ).asType(resultType)
}

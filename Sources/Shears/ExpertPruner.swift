import MLX

/// Removes experts from a ``MixtureOfExperts`` (a ``MoEBlock``, a Gemma 4 MoE layer) in place.
public enum ExpertPruner {
    /// Slices every per-expert router parameter (router rows, and for Gemma 4 also
    /// `per_expert_scale`) and every expert weight (axis 0) down to `keep`, and clamps the
    /// router's `topK` to the experts that remain. Expert ids are renumbered
    /// `0..<keep.count` in ascending order of their old ids.
    public static func prune(_ block: any MixtureOfExperts, keep: [Int]) {
        let sortedKeep = Array(Set(keep)).sorted()
        precondition(!sortedKeep.isEmpty, "must keep at least one expert")
        precondition(
            sortedKeep.first! >= 0 && sortedKeep.last! < block.numExperts,
            "expert id out of range 0..<\(block.numExperts)")
        let idx = MLXArray(sortedKeep.map(Int32.init))
        block.expertRouter.keepExperts(idx)
        block.expertBank.keepExperts(idx)
        eval(block.expertRouter, block.expertBank)
        block.stats = nil
    }

    /// Calibrated pruning: keep the `count` best experts according to `stats`.
    public static func prune(
        _ block: any MixtureOfExperts, using stats: ExpertStats, keeping count: Int,
        by criterion: ExpertStats.Criterion = .weightMass
    ) {
        prune(block, keep: stats.expertsToKeep(count: count, by: criterion))
    }
}

import MLX
import MLXNN

/// Freezing helpers: which parameters of a MoE block or decoder layer train.
///
/// Healing a pruned model (``PruneHealer``) trains adapters only; these are for training the
/// routers and expert weights themselves, as small experiments do.
public enum MoETraining {
    /// Train every router and expert weight in the block; freeze everything else
    /// (norms, attention, embeddings).
    ///
    /// Quantized expert weights can't train; for a ``QuantizedSwitchLinear`` only its LoRA
    /// adapters (if any) are unfrozen.
    public static func trainRouterAndExperts(_ block: Module) {
        block.freeze()
        for moe in mixtures(in: block) {
            moe.expertRouter.unfreeze()
            for proj in projections(moe) {
                if let q = proj as? QuantizedSwitchLinear {
                    if q.hasLoRA { q.unfreeze(recursive: false, keys: ["lora_a", "lora_b"]) }
                } else {
                    proj.unfreeze()
                }
            }
        }
    }

    /// Train only the routers.
    public static func trainRouterOnly(_ block: Module) {
        block.freeze()
        for moe in mixtures(in: block) { moe.expertRouter.unfreeze() }
    }

    /// Train routers plus any LoRA adapters on quantized experts; base weights stay frozen.
    public static func trainRouterAndLoRA(_ block: Module) {
        block.freeze()
        for moe in mixtures(in: block) {
            moe.expertRouter.unfreeze()
            for proj in projections(moe) {
                if let q = proj as? QuantizedSwitchLinear, q.hasLoRA {
                    q.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
                }
            }
        }
    }

    static func projections(_ moe: any MixtureOfExperts) -> [SwitchLinear] {
        let g = moe.expertBank
        return [g.gateProj, g.upProj, g.downProj]
    }

    /// Every ``MixtureOfExperts`` in `module`'s tree, including `module` itself: each
    /// ``MoEBlock``, and each MoE decoder layer whose router and experts are direct
    /// children (Gemma 4).
    public static func mixtures(in module: Module) -> [any MixtureOfExperts] {
        var found: [any MixtureOfExperts] = []
        if let m = module as? MixtureOfExperts { found.append(m) }
        for (_, child) in module.namedModules() {
            if let m = child as? MixtureOfExperts, child !== module { found.append(m) }
        }
        return found
    }

    /// Every ``MoEBlock`` in `module`'s tree, including `module` itself.
    public static func moeBlocks(in module: Module) -> [MoEBlock] {
        var found: [MoEBlock] = []
        if let m = module as? MoEBlock { found.append(m) }
        for (_, child) in module.namedModules() {
            if let m = child as? MoEBlock, m !== module { found.append(m) }
        }
        return found
    }
}

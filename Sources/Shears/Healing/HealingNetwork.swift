import Foundation
import MLX
import MLXNN

/// A run of decoder layers that can be driven as the unpruned teacher or as the pruned,
/// adapted student, from the same weights.
///
/// There is no second copy of the model. The teacher is these layers with the routing mask
/// and every adapter switched off; the student restricts routing to each layer's kept set and
/// applies the adapters. Pruning is therefore a mask: weights are sliced only when the result
/// is exported.
///
/// With `sliced` the layers have been cut to their kept experts (``sliceToKeptExperts(_:set:)``)
/// and the same student computation runs on them directly: no mask, no unkept expert resident.
/// A sliced network has no teacher, and says so loudly if asked for one.
public final class HealingNetwork {
    public let layers: [any HealableMoELayer]
    public let firstLayer: Int
    public let sets: [ExpertSet]
    public var adapters: ModelAdapters?
    /// The layers hold only their kept experts; there is no teacher to run.
    public let sliced: Bool

    public init(
        layers: [any HealableMoELayer], firstLayer: Int = 0, sets: [ExpertSet],
        adapters: ModelAdapters?, sliced: Bool = false
    ) {
        precondition(layers.count == sets.count, "one expert set per layer")
        for (l, s) in zip(layers, sets) {
            if sliced {
                precondition(l.numExperts == s.keep.count, "a sliced network needs layers cut to their kept experts")
            } else {
                precondition(l.numExperts == s.numExperts, "layers were sliced; build the network with sliced: true")
            }
        }
        self.sliced = sliced
        self.layers = layers
        self.firstLayer = firstLayer
        self.sets = sets
        self.adapters = adapters
    }

    /// Rounds the merged weights of CPU-rounded QAT formats for the current adapters; a no-op
    /// otherwise. Call before a traced student forward pass.
    public func refreshRounding() {
        adapters?.refreshRounding(layers: layers, sets: sets, sliced: sliced)
    }

    public var lastLayer: Int { firstLayer + layers.count - 1 }

    /// Runs layers `from...through` (model layer indices) on `h` (`[1, T, D]`, whole sequence
    /// from position 0).
    ///
    /// - Parameters:
    ///   - student: `false` runs the unpruned teacher with no adapters.
    ///   - collect: layers whose output to return.
    ///   - checkpointed: recompute each layer's activations in the backward pass.
    ///   - keptLayers: with `checkpointed`, the last this many layers of the run keep their activations instead (see ``ActivationKeep``).
    ///   - detachBefore: stop gradients at the input of this layer.
    public func forward(
        _ h0: MLXArray, from: Int? = nil, through: Int? = nil, student: Bool,
        collect: Set<Int> = [], checkpointed: Bool = false, keptLayers: Int = 0, detachBefore: Int? = nil
    ) -> (MLXArray, [Int: MLXArray]) {
        var h = h0
        var out: [Int: MLXArray] = [:]
        let lastRun = through ?? lastLayer
        for L in (from ?? firstLayer) ... lastRun {
            let i = L - firstLayer
            let layer = layers[i]
            if detachBefore == L { h = stopGradient(h) }
            if student {
                let set = sets[i]
                if let ad = adapters {
                    let la = ad.layers[i]
                    if checkpointed, L <= lastRun - keptLayers, detachBefore == nil || L >= detachBefore! {
                        let sl = sliced
                        h = checkpoint(la) { (m: LayerAdapters, x: MLXArray) in
                            healingLayerForward(layer, x, set: set, adapters: m, sliced: sl)
                        }(h)
                    } else {
                        h = healingLayerForward(layer, h, set: set, adapters: la, sliced: sliced)
                    }
                } else {
                    h = healingLayerForward(layer, h, set: set, adapters: nil, sliced: sliced)
                }
            } else {
                precondition(!sliced, "a sliced student holds only its kept experts and cannot run the teacher")
                h = healingLayerForward(layer, h, set: nil, adapters: nil)
            }
            if collect.contains(L) { out[L] = h }
        }
        return (h, out)
    }
}

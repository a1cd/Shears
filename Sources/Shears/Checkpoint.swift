import MLX
import MLXNN

/// Gradient checkpointing (rematerialisation) built on `CustomFunction`.
///
/// mlx-c exposes `mlx_checkpoint`, but mlx-swift does not wrap it and `Cmlx` is not a
/// public product, so this rebuilds it the way MLX core does (`transforms.cpp`,
/// `checkpoint`): the forward pass keeps only `f`'s inputs and outputs, and the VJP
/// re-runs `f` under `vjp` to recover the intermediates.
///
/// Every array `f` differentiates through must be an explicit input. Anything `f`
/// captures from its environment is treated as a constant and gets no gradient. Use
/// ``checkpoint(_:_:)`` for a `Module`, which passes its trainable parameters in.
///
/// Two details that are easy to get wrong:
/// - MLX's `CustomTransforms::vjp` indexes the returned list by input position and then
///   appends the cotangents, so the VJP must return exactly one gradient per primal.
///   `vjp(_:primals:cotangents:)` does (zeros for inputs the output does not depend on).
/// - Core makes the recomputation `depends` on the forward outputs so the scheduler cannot
///   hoist it into the forward pass, which would keep the intermediates alive and defeat
///   the point. mlx-swift's `VJP` closure is not given the outputs, so the recomputation
///   is made to depend on the cotangents instead: they only exist once the backward pass
///   has reached this function, which is later still.
public func checkpoint(
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray] {
    CustomFunction {
        Forward { f($0) }
        VJP { primals, cotangents in
            let scheduled = depends(inputs: primals, dependencies: cotangents)
            let grads = vjp(f, primals: scheduled, cotangents: cotangents).1
            precondition(
                grads.count == primals.count,
                "checkpoint: VJP returned \(grads.count) gradients for \(primals.count) primals")
            return grads
        }
    }
}

/// Checkpoints a module's forward pass, threading its trainable parameters through as
/// explicit inputs so they receive gradients.
///
/// The returned closure reads the module's current trainable parameters on every call,
/// so inside `valueAndGrad(model:)` it picks up the tracer arrays MLX installed. The
/// forward pass and the recomputation each install their own inputs and then put the
/// originals back.
public func checkpoint<M: Module>(
    _ module: M, _ f: @escaping (M, MLXArray) -> MLXArray
) -> (MLXArray) -> MLXArray {
    { x in
        let current = module.trainableParameters().flattened()
        let keys = current.map(\.0)
        // `Module.update` rewrites the module's existing MLXArray objects in place, so
        // holding those objects doesn't remember the old values. Take separate handles.
        let originals = current.map { handle(to: $0.1) }

        func withParameters(_ arrays: ArraySlice<MLXArray>, _ body: () -> MLXArray) -> MLXArray {
            module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, arrays))))
            defer {
                module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, originals))))
            }
            return body()
        }

        let cp = checkpoint { inputs in
            [withParameters(inputs.dropFirst()) { f(module, inputs[0]) }]
        }
        return cp([x] + originals)[0]
    }
}

/// ``checkpoint(_:_:)`` applied to `x` at once, for a graph that is cached: the recomputation closure that MLX keeps
/// in the traced graph (and a compiled function keeps in its cache, after the trace is done) holds `module`
/// **weakly**, and `f` must not capture anything that holds weights either. `module` has to be alive while this
/// runs, which is when the gradient is taken; the closure is not called after that. A graph that
/// held the module (and, through `f`, the layer) strongly kept every block's layers alive for as long as the
/// process ran: the pipeline's memory grew by a block's worth of weights per block.
func checkpointApplied<M: Module>(_ module: M, _ x: MLXArray, _ f: @escaping (M, MLXArray) -> MLXArray) -> MLXArray {
    let current = module.trainableParameters().flattened()
    let keys = current.map(\.0)
    let originals = current.map { handle(to: $0.1) }
    let cp = checkpoint { [weak module] inputs in
        guard let module else { fatalError("checkpoint: the module was released before its recomputation") }
        module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, inputs.dropFirst()))))
        defer { module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, originals)))) }
        return [f(module, inputs[0])]
    }
    return cp([x] + originals)[0]
}

/// A new MLXArray object backed by the same array as `a` (mlx-swift's internal
/// `copyContext`, built from public API).
func handle(to a: MLXArray) -> MLXArray {
    let h = MLXArray(0)
    h._updateInternal(a)
    return h
}

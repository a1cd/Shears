import Foundation
import MLX
import MLXNN

// MARK: - Settings

extension StreamedBlockHealer {
    /// Compile the per-layer forward and backward of a layer-window step (``windowedStep``) with
    /// `MLX.compile`, so the elementwise chains in a layer fuse into a few kernels instead of
    /// launching one by one. Rows are padded up to a length bucket (``sequenceBuckets``) so a compiled
    /// graph is reused by rows of different lengths; the padding changes nothing (attention is causal
    /// and the loss is masked to the row's true length), and the result equals the uncompiled step to
    /// float rounding (tested). A step that cannot be compiled (a custom loss, quantization-aware
    /// rounding, a layer that is not a `Module`) runs the uncompiled way whatever this says.
    nonisolated(unsafe) public static var compileLayerSteps = true

    /// The sequence lengths a compiled step pads a row up to, ascending: a row of `T` positions runs at
    /// the first bucket `>= T`, and every row that lands in the same bucket shares one compiled graph
    /// per layer. Past the last bucket a length is rounded up to a multiple of it. The default steps by at
    /// most a quarter (so padding adds at most 25% to a step's tokens, and its activations); each bucket a
    /// run's rows land in costs one trace of every compiled function (about a second for a window).
    nonisolated(unsafe) public static var sequenceBuckets: [Int] = [
        32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
        1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
    ]

    /// One sequence length for every compiled step (and, in ``BlockPipeline``, every teacher walk and frontier batch): a row
    /// of `T <= fixedSequenceLength` positions runs at exactly this length, so a run meets one shape per function instead of
    /// one per bucket. For a backend that compiles kernels at run time (MLX's CUDA backend). A row longer than it is
    /// refused by the pipeline up front. `nil` (the default) uses ``sequenceBuckets``.
    nonisolated(unsafe) public static var fixedSequenceLength: Int?

    /// The bucket a row of `length` positions runs at.
    public static func sequenceBucket(for length: Int) -> Int {
        if let fixed = fixedSequenceLength, length <= fixed { return fixed }
        let buckets = sequenceBuckets.sorted()
        if let b = buckets.first(where: { $0 >= length }) { return b }
        guard let last = buckets.last, last > 0 else { return length }
        return (length + last - 1) / last * last
    }
}

/// `a` right-padded with zeros along `axis` to `n` entries.
func zeroPadded(_ a: MLXArray, axis: Int, to n: Int) -> MLXArray {
    let current = a.dim(axis)
    if current == n { return a }
    precondition(n > current, "cannot pad \(current) down to \(n)")
    var shape = a.shape
    shape[axis] = n - current
    return concatenated([a, MLXArray.zeros(shape, dtype: a.dtype)], axis: axis)
}

// MARK: - Compiled layer steps

/// Compiled forward, top-window forward-and-backward and per-layer backward of a streamed block.
///
/// `MLX.compile` caches a graph by the function it was given and the shapes of its inputs, and takes
/// a function of arrays. A layer's weights change every step (a window is loaded again from the
/// checkpoint), and so do the adapters, so neither can be captured: both go in as explicit inputs, and
/// inside the traced body the layer's and the adapters' modules are pointed at those inputs (the way
/// ``checkpoint(_:_:)`` threads a module's parameters through) and put back afterwards. What the body
/// does capture is fixed for the life of this object: the expert sets, layer shapes and flags; the key
/// of a compiled function includes a signature of the kept experts, so a different set never reuses it.
///
/// One compiled function per layer (and one per top window), re-traced by MLX once per input shape: with
/// rows padded to buckets that is once per bucket. ``traces`` counts the traces (a trace is a compile), and
/// ``calls`` the compiled calls, so a test can see that rows of one bucket share a graph.
///
/// Compiled: the layer forward (attention, routing, the expert gathers, the combine), the layer's
/// vjp, and in the top window the loss and the per-layer checkpoint wrapper's recompute. Not compiled:
/// loading, the AdamW update (it has its own compiled kernel), and the `eval` between layers, which is
/// what keeps one layer's activations from being held across the walk.
final class CompiledLayerStep {
    /// Inputs of the traced body and everything it reads that is not an input, set before each call
    /// and cleared after it so a cached graph holds no layer weights.
    private final class Context {
        var layers: [any HealableMoELayer] = []
        var sets: [ExpertSet] = []
        var adapters: [LayerAdapters] = []
        var adapterKeys: [[String]] = []
        var layerKeys: [[String]] = []
        var sliced = false
        /// The first this-many layers of the window are recomputed in the backward (checkpointed); the rest keep theirs.
        var recomputed = 0
        var wantInputGrad = false
        func clear() {
            layers = []
            sets = []
            adapters = []
            adapterKeys = []
            layerKeys = []
        }
    }

    private final class Slot {
        let context = Context()
        var run: (([MLXArray]) -> [MLXArray])!
    }

    private var slots: [String: Slot] = [:]
    nonisolated(unsafe) static var live = 0
    init() { Self.live += 1 }
    deinit { Self.live -= 1 }
    /// Times a body was traced (compiled), across every function.
    private(set) var traces = 0
    /// Compiled calls made.
    private(set) var calls = 0
    /// Traces across every instance since launch (benchmarks and tests read it).
    nonisolated(unsafe) static var debugTraces = 0

    // MARK: parameter plumbing

    private struct Parameters {
        var keys: [String]
        var arrays: [MLXArray]
    }

    private static func parameters(of module: Module, trainable: Bool) -> Parameters {
        let flat = (trainable ? module.trainableParameters() : module.parameters()).flattened()
            .sorted { $0.0 < $1.0 }
        return Parameters(keys: flat.map(\.0), arrays: flat.map(\.1))
    }

    /// Points `module`'s parameters `keys` at `arrays` and returns the closure that puts back the arrays
    /// they held. (`Module.update` rewrites the module's array objects in place, so the old values are
    /// kept through separate handles.)
    private static func install(
        _ module: Module, keys: [String], arrays: some Sequence<MLXArray>, trainable: Bool
    ) -> () -> Void {
        let current = Dictionary(
            uniqueKeysWithValues: (trainable ? module.trainableParameters() : module.parameters()).flattened())
        let originals = keys.map { handle(to: current[$0]!) }
        module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, arrays))))
        return { module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, originals)))) }
    }

    private func slot(_ key: String, make: (Context) -> (([MLXArray]) -> [MLXArray])) -> Slot {
        if let s = slots[key] { return s }
        let s = Slot()
        s.run = make(s.context)
        slots[key] = s
        return s
    }

    /// Whether a window of layers can be run compiled.
    static func supports(layers: [any HealableMoELayer], adapters: [LayerAdapters]) -> Bool {
        for la in adapters {
            if !la.denseFormats.isEmpty || la.experts?.gateUpFormat != nil || la.experts?.downFormat != nil { return false }
        }
        return layers.allSatisfy { $0 is Module }
    }

    /// The part of a compiled function's key that its trace bakes in.
    static func signature(kept: [[Int]], sliced: Bool) -> String {
        var h = Hasher()
        h.combine(kept)
        return "\(sliced ? "s" : "m")\(h.finalize())|\(ExpertCompute.signature)"
    }

    /// The compiled forward of every layer in a body leaves the counter of built layer forwards alone, so
    /// it keeps counting what the step does, not what was traced: the caller adds the work of each call.
    private func traced<R>(_ body: () -> R) -> R {
        traces += 1
        Self.debugTraces += 1
        let saved = healingLayerForwardCount
        defer { healingLayerForwardCount = saved }
        return body()
    }

    private func layerArguments(_ layers: [any HealableMoELayer], _ ctx: Context) -> [MLXArray] {
        var out: [MLXArray] = []
        ctx.layerKeys = []
        for l in layers {
            let p = Self.parameters(of: l as! Module, trainable: false)
            ctx.layerKeys.append(p.keys)
            out += p.arrays
        }
        return out
    }

    private func adapterArguments(_ adapters: [LayerAdapters], _ ctx: Context) -> [MLXArray] {
        var out: [MLXArray] = []
        ctx.adapterKeys = []
        for a in adapters {
            let p = Self.parameters(of: a, trainable: true)
            ctx.adapterKeys.append(p.keys)
            out += p.arrays
        }
        return out
    }

    // MARK: forward of one layer (lower windows' forward walk)

    /// One layer's forward, compiled. `index` is the layer's index in the block.
    func forward(
        index: Int, layer: any HealableMoELayer, set: ExpertSet, adapters la: LayerAdapters, sliced: Bool,
        signature: String, x: MLXArray
    ) -> MLXArray {
        let s = slot("f|\(index)|\(signature)") { ctx in
            compile { [unowned self] (a: [MLXArray]) -> [MLXArray] in
                self.traced {
                    let nA = ctx.adapterKeys[0].count
                    var restores: [() -> Void] = []
                    restores.append(Self.install(
                        ctx.adapters[0], keys: ctx.adapterKeys[0], arrays: a[1 ..< 1 + nA], trainable: true))
                    restores.append(Self.install(
                        ctx.layers[0] as! Module, keys: ctx.layerKeys[0], arrays: a[(1 + nA)...], trainable: false))
                    defer { for r in restores.reversed() { r() } }
                    return [healingLayerForward(
                        ctx.layers[0], a[0], set: ctx.sets[0], adapters: ctx.adapters[0], sliced: ctx.sliced)]
                }
            }
        }
        let ctx = s.context
        ctx.layers = [layer]
        ctx.sets = [set]
        ctx.adapters = [la]
        ctx.sliced = sliced
        defer { ctx.clear() }
        let args = [x] + adapterArguments([la], ctx) + layerArguments([layer], ctx)
        calls += 1
        healingLayerForwardCount += 1
        return s.run(args)[0]
    }

    // MARK: top window: forward, loss and gradient in one pass

    struct TopResult {
        var loss: MLXArray
        /// The adapters' gradients scaled by `gradScale`, keyed `layers.<block index>.<name>` (the keys
        /// ``HealingAdamW/update(layers:indices:grads:learningRate:)`` reads).
        var grads: [String: MLXArray]
        /// The gradient of the window's input (unscaled), when asked for.
        var cotangent: MLXArray?
    }

    /// The top window's forward, pin loss and gradient, compiled as one function. The first `recomputed` layers are
    /// checkpointed (their forward is rebuilt in the backward); the rest keep their intermediates. The count is part of the key.
    /// `x` is the window's input
    /// (padded to the bucket), `target` and `weights` the padded teacher state and position weights, and
    /// `length` the true number of positions as a scalar array (so a new length is not a new graph).
    func top(
        first: Int, layers: [any HealableMoELayer], sets: [ExpertSet], adapters: [LayerAdapters], sliced: Bool,
        recomputed: Int, signature: String, x: MLXArray, wantInputGrad: Bool, target: MLXArray,
        weights: MLXArray, length: MLXArray, gradScale: MLXArray
    ) -> TopResult {
        let key = "t|\(first)-\(first + layers.count - 1)|r\(recomputed)|\(wantInputGrad ? "x" : "-")|\(signature)"
        let s = slot(key) { ctx in
            compile { [unowned self] (a: [MLXArray]) -> [MLXArray] in
                self.traced { Self.topBody(ctx, a) }
            }
        }
        let ctx = s.context
        ctx.layers = layers
        ctx.sets = sets
        ctx.adapters = adapters
        ctx.sliced = sliced
        ctx.recomputed = recomputed
        ctx.wantInputGrad = wantInputGrad
        defer { ctx.clear() }
        let args = [x, target, weights, length, gradScale] + adapterArguments(adapters, ctx) + layerArguments(layers, ctx)
        calls += 1
        healingLayerForwardCount += layers.count + recomputed
        let out = s.run(args)
        var k = 1
        var cot: MLXArray?
        if wantInputGrad {
            cot = out[1]
            k = 2
        }
        var grads: [String: MLXArray] = [:]
        for (j, keys) in ctx.adapterKeys.enumerated() {
            for name in keys {
                grads["layers.\(first + j).\(name)"] = out[k]
                k += 1
            }
        }
        return TopResult(loss: out[0], grads: grads, cotangent: cot)
    }

    private static func topBody(_ ctx: Context, _ a: [MLXArray]) -> [MLXArray] {
        let fixed = 5  // x, target, weights, length, gradScale
        let adapterCount = ctx.adapterKeys.reduce(0) { $0 + $1.count }
        var restores: [() -> Void] = []
        var offset = fixed + adapterCount
        for (j, layer) in ctx.layers.enumerated() {
            let k = ctx.layerKeys[j]
            restores.append(install(layer as! Module, keys: k, arrays: a[offset ..< offset + k.count], trainable: false))
            offset += k.count
        }
        // The adapters' real arrays, put back once the gradient (forward, then the checkpoint's recompute,
        // which runs while it is built) is done.
        var realAdapters: [() -> Void] = []
        for (j, la) in ctx.adapters.enumerated() {
            let current = Dictionary(uniqueKeysWithValues: la.trainableParameters().flattened())
            let handles = ctx.adapterKeys[j].map { handle(to: current[$0]!) }
            let keys = ctx.adapterKeys[j]
            realAdapters.append { la.update(parameters: ModuleParameters.unflattened(Array(zip(keys, handles)))) }
        }
        defer {
            for r in realAdapters { r() }
            for r in restores.reversed() { r() }
        }
        let differentiated = (ctx.wantInputGrad ? [0] : []) + Array(fixed ..< fixed + adapterCount)
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            var h = p[0]
            var o = fixed
            for (j, layer) in ctx.layers.enumerated() {
                let la = ctx.adapters[j]
                let k = ctx.adapterKeys[j]
                _ = install(la, keys: k, arrays: p[o ..< o + k.count], trainable: true)
                o += k.count
                let set = ctx.sets[j]
                let sliced = ctx.sliced
                if j < ctx.recomputed {
                    // The closure is kept in the cached graph: it reaches the layer through `ctx` (emptied after
                    // every call), not by capturing the layer, its adapters or the expert set.
                    h = checkpointApplied(la, h) { [unowned ctx] (m: LayerAdapters, x: MLXArray) in
                        healingLayerForward(ctx.layers[j], x, set: ctx.sets[j], adapters: m, sliced: ctx.sliced)
                    }
                } else {
                    h = healingLayerForward(layer, h, set: set, adapters: la, sliced: sliced)
                }
            }
            return [HealingLoss.pinBatchedMasked(
                [0: h], teacher: [0: p[1]], weights: p[2], length: p[3])]
        }, argumentNumbers: differentiated)
        let (value, grads) = vg(a)
        var out = [value[0]]
        var g = grads[...]
        if ctx.wantInputGrad {
            out.append(g.first!)
            g = g.dropFirst()
        }
        let scale = a[4]
        out += g.map { $0 * scale.asType($0.dtype) }
        return out
    }

    // MARK: backward of one lower layer

    struct BackResult {
        var grads: [String: MLXArray]
        var cotangent: MLXArray?
    }

    /// One lower layer's backward: the vjp of `layer` at `x` against the cotangent `cot` (its forward is
    /// built again inside, which is the recompute), compiled. The adapters' gradients come back scaled by
    /// `gradScale`; the input's gradient (unscaled) only when `wantInputGrad`.
    func back(
        index: Int, layer: any HealableMoELayer, set: ExpertSet, adapters la: LayerAdapters, sliced: Bool,
        signature: String, x: MLXArray, cotangent cot: MLXArray, wantInputGrad: Bool, gradScale: MLXArray
    ) -> BackResult {
        let s = slot("b|\(index)|\(wantInputGrad ? "x" : "-")|\(signature)") { ctx in
            compile { [unowned self] (a: [MLXArray]) -> [MLXArray] in
                self.traced {
                    let fixed = 3  // x, cotangent, gradScale
                    let nA = ctx.adapterKeys[0].count
                    let restoreLayer = Self.install(
                        ctx.layers[0] as! Module, keys: ctx.layerKeys[0], arrays: a[(fixed + nA)...], trainable: false)
                    let la = ctx.adapters[0]
                    let current = Dictionary(uniqueKeysWithValues: la.trainableParameters().flattened())
                    let keys = ctx.adapterKeys[0]
                    let handles = keys.map { handle(to: current[$0]!) }
                    defer {
                        la.update(parameters: ModuleParameters.unflattened(Array(zip(keys, handles))))
                        restoreLayer()
                    }
                    let differentiated = (ctx.wantInputGrad ? [0] : []) + Array(fixed ..< fixed + nA)
                    let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
                        _ = Self.install(la, keys: keys, arrays: p[fixed ..< fixed + nA], trainable: true)
                        let out = healingLayerForward(
                            ctx.layers[0], p[0], set: ctx.sets[0], adapters: la, sliced: ctx.sliced)
                        return [(out.asType(.float32) * p[1].asType(.float32)).sum()]
                    }, argumentNumbers: differentiated)
                    let (_, grads) = vg(a)
                    var out: [MLXArray] = []
                    var g = grads[...]
                    if ctx.wantInputGrad {
                        out.append(g.first!)
                        g = g.dropFirst()
                    }
                    let scale = a[2]
                    out += g.map { $0 * scale.asType($0.dtype) }
                    return out
                }
            }
        }
        let ctx = s.context
        ctx.layers = [layer]
        ctx.sets = [set]
        ctx.adapters = [la]
        ctx.sliced = sliced
        ctx.wantInputGrad = wantInputGrad
        defer { ctx.clear() }
        let args = [x, cot, gradScale] + adapterArguments([la], ctx) + layerArguments([layer], ctx)
        calls += 1
        healingLayerForwardCount += 1
        let out = s.run(args)
        var k = 0
        var cotOut: MLXArray?
        if wantInputGrad {
            cotOut = out[0]
            k = 1
        }
        var grads: [String: MLXArray] = [:]
        for name in ctx.adapterKeys[0] {
            grads["layers.\(index).\(name)"] = out[k]
            k += 1
        }
        return BackResult(grads: grads, cotangent: cotOut)
    }
}

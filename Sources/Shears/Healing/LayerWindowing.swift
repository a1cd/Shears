import Foundation
import MLX
import MLXNN

// MARK: - Memory model and budget

/// What a streamed block run needs, measured from the model and the run's shape: the inputs of
/// ``ResidencyPlanner``.
public struct ResidencyInputs: Sendable {
    /// Layers in the longest block.
    public var blockLayers: Int
    /// Bytes of one loaded (and, when the run slices, cut) layer; the largest of the first block's.
    public var layerWeightBytes: Int
    /// Bytes of one layer's adapters in float32 (the optimizer holds two more copies).
    public var adapterBytesPerLayer: Int
    /// Longest sequence the run trains on.
    public var sequence: Int
    public var hidden: Int
    /// Working memory of one layer's forward and backward, per token, in bytes, without the expert branch's
    /// (see ``expertWorkspaceBytesPerToken``).
    public var workspaceBytesPerToken: Int
    /// The expert branch's working memory per token, bytes: the routed gate, up and activation per picked expert, or,
    /// in the merged layout, the wide hidden layers over every kept expert.
    public var expertWorkspaceBytesPerToken: Int = 0
    /// Tokens a step processes at once (the longest sequence times the rows stacked in the step); 0 means `sequence`.
    public var tokenCount: Int = 0
    /// Whether the expert branch runs in the merged layout, whose token chunking (``ExpertCompute/tokenChunk``)
    /// lowers ``expertWorkspaceBytesPerToken`` times the tokens it holds at once.
    public var expertChunkable = false
    /// Attention score memory of one layer (`heads x min(T, chunk) x T`, float32, kept for the
    /// backward pass).
    public var attentionBytes: Int
    /// The most MLX memory a layer's load reaches above what the loaded layer then holds (the uncut
    /// layer, mapped or read, while it is being cut to its kept experts); measured, the largest of the first block's.
    public var loadTransientBytes: Int = 0
    /// Slack for what the model leaves out: rotary tables, masks, small kernels' scratch.
    public var slackBytes: Int = 8 << 20
    /// MLX memory already in use when the run starts (initial adapters, anything the caller holds).
    public var baselineBytes: Int = 0
}

/// The auto-residency choice for one block and why.
public struct ResidencyDecision: Sendable, Codable {
    public var residentLayers: Int
    /// The token chunk of the expert math the budget asked for (see ``ExpertCompute/tokenChunk``); `nil` is all tokens.
    public var tokenChunk: Int?
    /// What the run needs whatever training does: the larger of the other stages' peak and training's
    /// one-layer estimate.
    public var floorBytes: Int
    public var otherStagesPeak: Int
    public var trainingMinimum: Int
    public var estimate: ResidencyEstimate
    public var reason: String
    /// Layers of the top window that keep their forward intermediates instead of recomputing them (see
    /// ``ActivationKeep``); `nil` in a decision written before the setting existed (none were kept).
    public var keptLayers: Int?
    /// The ``ActivationKeep`` setting this decision was made under, so a resumed run under another one decides again.
    public var keepSetting: String?
}

/// The memory model's answer for one choice of resident layers.
public struct ResidencyEstimate: Sendable, Codable, CustomStringConvertible {
    public var residentLayers: Int
    public var weights: Int
    public var adapterState: Int
    public var gradients: Int
    public var activations: Int
    public var workspace: Int
    /// Sum of the parts times the safety margin.
    public var total: Int
    public var description: String {
        func g(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824) }
        return "\(residentLayers) resident layer(s): weights \(g(weights)), adapters + optimizer \(g(adapterState)), "
            + "gradients \(g(gradients)), activations \(g(activations)), workspace \(g(workspace)); total \(g(total))"
    }
}

/// Chooses how many layers of a block to hold at once so a run fits a memory budget, from a
/// model of the run's memory. The model is an estimate (calibrated against measured runs, with a
/// margin). MLX's memory limit is a guideline (it makes the allocator reclaim its cache; it never
/// refuses an allocation), so the budget is kept three ways: this estimate, which refuses a run it
/// says cannot fit; the measured peak, checked at every step, which stops the run with a checkpoint
/// when it passes the budget; and a report (`Report.peakOverLimit`) of any overshoot.
public enum ResidencyPlanner {
    /// Multiplier on the summed parts.
    public static let safetyMargin = 1.25

    /// Workspace of one layer whose forward intermediates are kept for the backward (not recomputed): the layer's
    /// working memory at the whole step's tokens (a token chunk bounds a layer's transient, not what is stored).
    public static func keptLayerBytes(_ i: ResidencyInputs) -> Int {
        let tokens = i.tokenCount > 0 ? i.tokenCount : i.sequence
        return i.workspaceBytesPerToken * i.sequence + i.expertWorkspaceBytesPerToken * tokens + i.attentionBytes
    }

    /// `keptLayers` of the top window keep their intermediates (clamped to the window); each adds
    /// ``keptLayerBytes(_:)`` to the workspace, on top of the one layer's transient the recompute needs.
    public static func estimate(
        _ i: ResidencyInputs, resident r: Int, tokenChunk: Int? = nil, keptLayers: Int = 0
    ) -> ResidencyEstimate {
        let lb = i.blockLayers
        let r = max(1, min(r, lb))
        let td = i.sequence * i.hidden * 4
        let weights = r * i.layerWeightBytes
        let adapterState = 3 * i.adapterBytesPerLayer * lb
        // The top window's gradients and the update's copies; a lower layer's are one layer's.
        let gradients = r == lb ? 3 * i.adapterBytesPerLayer * lb : 2 * i.adapterBytesPerLayer * r
        // A layer input for every layer (the lower ones kept for the backward walk, the top window's
        // checkpointed), the block's input and target, and the cotangent in and out.
        let activations = td * (lb + 4)
        let tokens = i.tokenCount > 0 ? i.tokenCount : i.sequence
        let held = i.expertChunkable ? min(tokens, tokenChunk ?? tokens) : tokens
        let workspace = i.workspaceBytesPerToken * i.sequence + i.expertWorkspaceBytesPerToken * held + i.attentionBytes
            + min(max(0, keptLayers), r) * keptLayerBytes(i)
        let parts = i.baselineBytes + i.slackBytes + weights + adapterState + gradients + activations + workspace + i.loadTransientBytes
        return ResidencyEstimate(
            residentLayers: r, weights: weights, adapterState: adapterState, gradients: gradients,
            activations: activations, workspace: workspace, total: Int(Double(parts) * safetyMargin))
    }

    /// The largest window whose estimate fits `budget` (a bigger window reloads less), or throws
    /// naming the smallest estimate when even one layer does not fit.
    public static func plan(budget: Int, _ i: ResidencyInputs) throws -> ResidencyEstimate {
        for r in stride(from: i.blockLayers, through: 1, by: -1) {
            let e = estimate(i, resident: r)
            if e.total <= budget { return e }
        }
        let one = estimate(i, resident: 1)
        func g(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824) }
        throw HealingError.budget(
            "a memory budget of \(g(budget)) cannot be met: even one resident layer needs about \(g(one.total)) "
                + "(\(one)). Shorten the sequences, use smaller adapter ranks, or raise the budget.")
    }

    /// The auto-residency rule (see ``ResidencyDecision``).
    ///
    /// **The whole block stays resident.** A pruned layer is small, and a window smaller than the block
    /// reloads its layers every step (per-layer overhead on top of the read), so the default holds every layer
    /// of the block and nothing is loaded between steps. Fewer layers are held only when a `ceiling` (the
    /// caller's memory budget) forces it: the window is then the largest whose estimate is within the ceiling.
    /// The floor (`max(otherStagesPeak, estimate(1 layer))`, what the run needs whatever training does) is
    /// still checked against the ceiling, and a run whose floor is above it is refused with the amount it needs.
    public static func chooseAuto(
        _ i: ResidencyInputs, otherStagesPeak: Int, ceiling: Int?, keep: ActivationKeep = .auto
    ) throws -> ResidencyDecision {
        // Smaller token chunks of the expert math are tried first (exact, and they cost no reloads), fewer resident layers
        // only after; the floor is training with one layer at the smallest chunk.
        let chunks: [Int?] = i.expertChunkable ? [nil] + [2048, 1024, 512, 256, 128].filter { $0 < max(i.tokenCount, i.sequence) } : [nil]
        let minimum = estimate(i, resident: 1, tokenChunk: chunks.last!).total
        let floor = max(otherStagesPeak, minimum)
        func g(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824) }
        if let ceiling, floor > ceiling {
            throw HealingError.budget(
                "a memory budget of \(g(ceiling)) is below what this run needs: \(g(floor)) "
                    + "(\(floor == minimum ? "training with one resident layer" : "its hungriest other stage, \(g(otherStagesPeak))")). "
                    + "Raise the budget, or shorten the sequences, the batch size or the adapter ranks.")
        }
        let lb = i.blockLayers
        // A requested number of kept layers (nil: auto). Under a ceiling it is honoured before the chunk is lowered, and
        // given up (layer by layer) before any layer is unloaded: kept layers cost time, windows cost reloads.
        let requested: Int? = {
            switch keep {
            case .auto: return nil
            case .fast: return lb
            case .lowMemory: return 0
            case .layers(let n): return min(max(0, n), lb)
            }
        }()
        var w = lb
        var chunk: Int?
        var k = requested ?? 0
        if let ceiling {
            var found = false
            search: for kk in stride(from: k, through: 0, by: -1) {
                for c in chunks where estimate(i, resident: lb, tokenChunk: c, keptLayers: kk).total <= ceiling {
                    chunk = c
                    k = kk
                    found = true
                    break search
                }
            }
            if !found {
                chunk = chunks.last!
                k = 0
                w = 1
                for r in stride(from: lb, through: 1, by: -1) where estimate(i, resident: r, tokenChunk: chunk).total <= ceiling {
                    w = r
                    break
                }
            }
            if requested == nil, w == lb {
                // auto: what the window and chunk left over buys kept layers.
                while k < lb, estimate(i, resident: lb, tokenChunk: chunk, keptLayers: k + 1).total <= ceiling { k += 1 }
            }
        }
        k = min(k, w)
        let e = estimate(i, resident: w, tokenChunk: chunk, keptLayers: k)
        var why = w == lb
            ? "the whole block fits (\(g(e.total)))\(ceiling.map { " under the memory budget of \(g($0))" } ?? ""), so nothing is reloaded between steps"
                + (chunk.map { "; the expert math runs \($0) tokens at a time" } ?? "")
            : "the memory budget of \(g(ceiling ?? 0)) fits \(w) of \(lb) layers (\(g(e.total)))"
        why += "; \(k) of \(w) layers keep their intermediates (\(keep))"
        return ResidencyDecision(
            residentLayers: w, tokenChunk: chunk, floorBytes: floor, otherStagesPeak: otherStagesPeak, trainingMinimum: minimum,
            estimate: e, reason: why, keptLayers: k, keepSetting: keep.description)
    }

    /// Measures the inputs from `source`: it loads the first block's layers one at a time.
    public static func inputs(
        source: LayerSource, keep: [[Int]], blockEnds: [Int], adapterConfiguration c: AdapterConfiguration,
        sliced: Bool, sequence: Int, block: Int = 0
    ) throws -> ResidencyInputs {
        let blocks = blockRanges(ends: blockEnds)
        let first = blocks[block]
        // A compiled step runs every row at its bucket's length (see ``StreamedBlockHealer/sequenceBuckets``),
        // so that is the length memory is spent at.
        let sequence = StreamedBlockHealer.compileLayerSteps ? StreamedBlockHealer.sequenceBucket(for: sequence) : sequence
        var transient = 0
        var weight = 0, adapter = 0, token = 0, attention = 0, hidden = 0, expertToken = 0
        var chunkable = true
        for i in first.start ... first.end {
            Memory.clearCache()
            let base = Memory.activeMemory
            Memory.peakMemory = base
            let layer = try source.loadLayers(i ... i, keeping: sliced ? [keep[i]] : nil)[0]
            let held = max(0, Memory.activeMemory - base)
            transient = max(transient, max(0, Memory.peakMemory - base - held))
            weight = max(weight, held)
            hidden = layer.hiddenSize
            let kept = keep[i].count
            let d = layer.hiddenSize, h = layer.expertHiddenSize
            var params = 0
            for (s, dims) in layer.denseSiteDimensions where c.denseRank > 0 && c.denseSites.contains(s) {
                params += c.denseRank * (dims.input + dims.output)
            }
            if c.expertRank > 0 {
                params += c.sharedExperts ? 3 * c.expertRank * (d + kept * h) : 3 * kept * c.expertRank * (d + h)
            }
            if c.routerRank > 0 { params += c.routerRank * (d + source.numExperts) }
            adapter = max(adapter, params * 4)
            let dense = layer.denseSiteDimensions[.mlpGate]?.output ?? 0
            let k = layer.expertRouter.topK
            var heads = max(1, d / 128), headDim = 128
            if let m = layer as? Gemma4MoEDecoderLayer { heads = m.selfAttn.numHeads; headDim = m.selfAttn.headDim }
            // 3 copies of q, k-v-o sized activations, 14 of the hidden width (norms, residuals, the
            // router), the dense MLP's three, and the experts' gate, up and activation per routed token.
            let mergedLayout = ExpertCompute.merged && kept <= Int(ExpertCompute.mergedMaxKeptRatio * Double(k))
            token = max(token, (14 * d + 4 * heads * headDim + 3 * dense) * 4)
            expertToken = max(expertToken, 4 * (mergedLayout ? kept : k) * h * 4)
            chunkable = chunkable && mergedLayout
            attention = max(attention, 2 * heads * min(sequence, Gemma4Attention.queryChunk > 0 ? Gemma4Attention.queryChunk : sequence) * sequence * 4)
        }
        return ResidencyInputs(
            blockLayers: blocks.map { $0.end - $0.start + 1 }.max() ?? 1, layerWeightBytes: weight,
            adapterBytesPerLayer: adapter, sequence: sequence, hidden: hidden,
            workspaceBytesPerToken: token, expertWorkspaceBytesPerToken: expertToken, tokenCount: sequence,
            expertChunkable: chunkable, attentionBytes: attention,
            loadTransientBytes: transient, baselineBytes: Memory.activeMemory)
    }
}

// MARK: - Process footprint sampling

/// Samples the process's `phys_footprint` from a background timer, so a peak that falls between
/// two points the caller thinks to look at is still seen. Per-step peaks are read with
/// ``takeStepPeak()``.
final class FootprintSampler {
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var runPeak = ProcessMemory.Snapshot(footprint: 0, fileBacked: 0, anonymous: 0)
    private var stepPeak = ProcessMemory.Snapshot(footprint: 0, fileBacked: 0, anonymous: 0)
    private var fileBackedStep = 0

    init(interval: TimeInterval = 0.01) {
        sample()
        let t = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "shears.footprint"))
        t.schedule(deadline: .now(), repeating: interval)
        t.setEventHandler { [weak self] in self?.sample() }
        t.resume()
        timer = t
    }
    deinit { timer?.cancel() }

    func sample() {
        let s = ProcessMemory.snapshot()
        lock.lock()
        if s.footprint > stepPeak.footprint { stepPeak = s }
        if s.footprint > runPeak.footprint { runPeak = s }
        fileBackedStep = max(fileBackedStep, s.fileBacked)
        lock.unlock()
    }

    /// The highest footprint since the last call (with the file-backed bytes seen), then starts a new window.
    func takeStepPeak() -> (footprint: Int, fileBacked: Int) {
        sample()
        lock.lock()
        defer { lock.unlock() }
        let r = (stepPeak.footprint, fileBackedStep)
        stepPeak = ProcessMemory.snapshot()
        fileBackedStep = stepPeak.fileBacked
        return r
    }

    var peak: Int { lock.lock(); defer { lock.unlock() }; return runPeak.footprint }
}

// MARK: - Layer-window step

/// Layers of the lowest window, kept from one step's backward walk to the next step's forward walk
/// so the step boundary does not load them again.
final class WindowCarry {
    var window: ClosedRange<Int>?
    var layers: [any HealableMoELayer]?
    /// The compiled layer steps of this block (see ``CompiledLayerStep``), kept with the carry so they live
    /// exactly as long as the block's expert sets they were traced with.
    var compiled: CompiledLayerStep?
}

/// A group of adapters (and optionally the activation entering it) as one module, so one
/// `valueAndGrad` returns the group's adapter gradients and the gradient of its input in one pass.
final class GroupVJPModel: Module {
    @ModuleInfo(key: "layers") var layers: [LayerAdapters]
    @ParameterInfo(key: "x") var x: MLXArray?

    init(layers: [LayerAdapters], input: MLXArray?) {
        _layers.wrappedValue = layers
        _x.wrappedValue = input
        super.init()
    }
}

extension StreamedBlockHealer {
    /// Apply and free each layer's gradients one layer at a time (see ``windowedStep``) instead of updating the whole
    /// top window at once. Bit-identical. Off by default: on a toy with activations dominating (whole block resident, one
    /// 2,048-token row, compiled) it raised MLX's peak by about a fifth and the step time by about 5%, rather than lowering
    /// the peak by the combined update's transient as intended. Turn it on only where a measurement of your own shows a gain.
    ///
    /// Why it raises the peak (measured, 6-layer toy, `lowMemory`: 3.1 GB off, 4.8 GB on; `fast` 7.4 GB off, 7.9 GB on): the
    /// default's single `eval` of every layer's update lets MLX hand each gradient to its update and free it as the backward
    /// produces it. This path evaluates layer 0 first, and layer 0's gradient is the last thing the backward produces, so
    /// the first eval runs the whole backward and every layer's recomputed activations and gradient stay alive until their
    /// own update is evaluated, as in `fast`. Updating the layers top-down instead keeps the peak (3.1 GB) but costs more
    /// syncs and is slower (0.50 vs 0.45 s/step), so it is not offered.
    nonisolated(unsafe) public static var freeGradientsEarly = false
    /// Layers updated through the early-freeing path since launch (a test reads it to see the path ran).
    nonisolated(unsafe) static var earlyFreedLayers = 0
    /// Queue the backward with the forward before the loss is read (see ``windowedStep``). Bit-identical. Off by default:
    /// on the same toy it raised MLX's peak by about a fifth and made the step no faster alone.
    ///
    /// Why it raises the peak: every gradient is requested as an output of the `asyncEval`, so all of them (and the
    /// activations their computation needs) are alive at once, where the default consumes each in its update as it is
    /// produced. It is the same jump as ``freeGradientsEarly`` and for the same reason, which is why either switch alone
    /// gives the same peak: the cost is "all gradients materialised before the update starts", not either switch's own code.
    nonisolated(unsafe) public static var overlapLossCheck = false
    /// Steps that took the overlapped path since launch (for tests).
    nonisolated(unsafe) static var overlappedSteps = 0

    /// Splits a block of `count` layers into windows of at most `resident` layers, cut from the top
    /// so the top window (which stays loaded from the forward walk into the backward walk) is full.
    static func windows(count: Int, resident: Int) -> [ClosedRange<Int>] {
        var out: [ClosedRange<Int>] = []
        var hi = count - 1
        while hi >= 0 {
            let lo = max(0, hi - resident + 1)
            out.append(lo ... hi)
            hi = lo - 1
        }
        return out.reversed()
    }

    /// One training step over a block that is only partly resident.
    ///
    /// The forward walk goes up the block one window at a time, keeping each window's input
    /// activation and dropping its weights. The top window runs the loss in the same pass as its
    /// gradient, so it is loaded once. The backward walk then goes down: each window is loaded
    /// again, its forward is recomputed under the same per-layer checkpointing the whole-block path
    /// uses (so the compute is the whole-block path's: one forward, one recompute, one backward per
    /// layer), the gradient of its adapters and of its input comes out of one `valueAndGrad`, the
    /// input's gradient goes down to the next window, and the window's adapters are updated at once.
    /// The cotangent is computed before the update, so the result equals the whole-block update.
    ///
    /// With ``StreamedBlockHealer/compileLayerSteps`` (the default) each layer's forward, the top window's
    /// forward, loss and gradient, and each lower layer's backward run as `MLX.compile`d functions
    /// (``CompiledLayerStep``), on rows padded to a length bucket; the loading, the `eval` between layers and
    /// the optimizer update are as before. A custom `loss`, or quantization-aware rounding, runs uncompiled.
    /// `preloaded` hands a single-window step the layers a whole-block caller already holds.
    ///
    /// `keptLayers` (with `checkpointed`): the top this-many layers of the top window keep their forward intermediates
    /// instead of recomputing them in the backward (see ``ActivationKeep``); the result is bit-identical, only what is
    /// stored differs. Lower windows always recompute.
    func windowedStep(
        block range: (start: Int, end: Int), windows: [ClosedRange<Int>], adapters: ModelAdapters,
        optimizer opt: HealingAdamW, sets: [ExpertSet], kept: [[Int]], input: MLXArray, target: MLXArray,
        weights w: MLXArray, learningRate lr: (Int) -> Float, blockCount nb: Float, checkpointed: Bool, keptLayers: Int = 0,
        carry: WindowCarry? = nil, loss lossFunction: ((MLXArray) -> MLXArray)? = nil,
        preloaded: [any HealableMoELayer]? = nil
    ) throws -> (loss: Float, applied: Bool, learningRate: Float) {
        let sliced = slicedStudent
        let c = configuration
        // Compiled when asked for, and when everything the step needs is expressible as arrays: no custom loss
        // (it may depend on the row's length in ways a padded row cannot honour), no CPU-rounded
        // quantization-aware formats (their rounded weights are state outside the layer's parameters).
        let compiled: CompiledLayerStep? = {
            guard Self.compileLayerSteps, lossFunction == nil,
                CompiledLayerStep.supports(layers: [], adapters: Array(adapters.layers))
            else { return nil }
            if let carry {
                if carry.compiled == nil { carry.compiled = CompiledLayerStep() }
                return carry.compiled
            }
            return CompiledLayerStep()
        }()
        let signature = CompiledLayerStep.signature(kept: kept, sliced: sliced)
        // Layers of the top window that recompute in the backward: all of them under `checkpointed`, minus the kept ones.
        let topCount = windows.last!.count
        let recomputed = checkpointed ? topCount - min(max(0, keptLayers), topCount) : 0
        let length = input.dim(1)
        // Rows run at their bucket's length: the input, the teacher state and the position weights are
        // right-padded with zeros. Attention is causal, so positions below `length` are exact, and the masked
        // loss (see ``HealingLoss/pinBatchedMasked``) scores only them, with zero cotangent beyond.
        let padTo = compiled == nil ? length : Self.sequenceBucket(for: length)
        let input = compiled == nil ? input : zeroPadded(input, axis: 1, to: padTo)
        let target = compiled == nil ? target : zeroPadded(target, axis: 1, to: padTo)
        let w = compiled == nil ? w : zeroPadded(w, axis: w.ndim - 1, to: padTo)
        let lengthArray = MLXArray(Float(length))
        let gradScale = MLXArray(1 / nb)
        func load(_ win: ClosedRange<Int>, reuse: Bool = false) throws -> [any HealableMoELayer] {
            let layers: [any HealableMoELayer]
            if let preloaded, windows.count == 1 { layers = preloaded }
            else if reuse, let c = carry?.layers, carry?.window == win { layers = c } else {
                layers = try source.loadLayers(
                    (range.start + win.lowerBound) ... (range.start + win.upperBound),
                    keeping: sliced ? Array(kept[win]) : nil)
            }
            carry?.layers = nil
            if compiled != nil, !layers.allSatisfy({ $0 is Module }) {
                throw HealingError.invalid(
                    "compiled layer steps need layers that are Modules; set StreamedBlockHealer.compileLayerSteps = false")
            }
            for (j, l) in layers.enumerated() {
                let i = win.lowerBound + j
                adapters.refreshRounding(layerAt: i, layer: l, set: sets[i], sliced: sliced)
            }
            return layers
        }
        func prefetch(_ win: ClosedRange<Int>) {
            source.prefetchLayers(
                (range.start + win.lowerBound) ... (range.start + win.upperBound), keeping: sliced ? Array(kept[win]) : nil)
        }
        func run(_ m: GroupVJPModel, _ layers: [any HealableMoELayer], _ win: ClosedRange<Int>, _ x0: MLXArray) -> MLXArray {
            var h = x0
            for (j, layer) in layers.enumerated() {
                let set = sets[win.lowerBound + j]
                let la = m.layers[j]
                if checkpointed, j < recomputed {
                    h = checkpoint(la) { (mm: LayerAdapters, x: MLXArray) in
                        healingLayerForward(layer, x, set: set, adapters: mm, sliced: sliced)
                    }(h)
                } else {
                    h = healingLayerForward(layer, h, set: set, adapters: la, sliced: sliced)
                }
            }
            return h
        }
        // Gradients keyed by the window's own layer names, renamed to the block's.
        func blockKeys(_ g: [String: MLXArray], _ win: ClosedRange<Int>) -> [String: MLXArray] {
            var out: [String: MLXArray] = [:]
            for (k, v) in g where k.hasPrefix("layers.") {
                let parts = k.split(separator: ".", maxSplits: 2).map(String.init)
                out["layers.\(Int(parts[1])! + win.lowerBound).\(parts[2])"] = v
            }
            return out
        }

        // Forward walk: everything below the top window.
        var h = input
        var layerInputs: [Int: MLXArray] = [:]
        let top = windows.last!
        for (n, win) in windows.dropLast().enumerated() {
            if n == 0, carry?.layers == nil { prefetch(win) }
            let layers = try load(win, reuse: n == 0)
            prefetch(windows[n + 1])
            for (j, l) in layers.enumerated() {
                let i = win.lowerBound + j
                layerInputs[i] = h
                if let compiled {
                    h = compiled.forward(
                        index: i, layer: l, set: sets[i], adapters: adapters.layers[i], sliced: sliced,
                        signature: signature, x: h)
                } else {
                    h = healingLayerForward(l, h, set: sets[i], adapters: adapters.layers[i], sliced: sliced)
                }
            }
            eval(h)
        }

        // Top window: forward, loss and gradient together.
        var cot: MLXArray?
        let loss: Float
        let rate = lr(opt.step)
        do {
            if windows.count == 1, preloaded == nil { prefetch(top) }
            let layers = try load(top)
            if windows.count > 1 { prefetch(windows[windows.count - 2]) }
            var pg: [String: MLXArray]
            if let compiled {
                let r = compiled.top(
                    first: top.lowerBound, layers: layers, sets: Array(sets[top]), adapters: Array(adapters.layers[top]),
                    sliced: sliced, recomputed: recomputed, signature: signature, x: h,
                    wantInputGrad: windows.count > 1, target: target, weights: w, length: lengthArray,
                    gradScale: gradScale)
                if Self.overlapLossCheck {
                    Self.overlappedSteps += 1
                    // Queue the backward with the forward: the loss is read (and a non-finite one skips the update)
                    // after both are on the GPU's queue, so reading it does not stall the queue between them.
                    asyncEval([r.loss] + (r.cotangent.map { [$0] } ?? []) + Array(r.grads.values))
                }
                loss = r.loss.item(Float.self)
                if !(loss.isFinite && loss <= c.maxLoss) { return (loss, false, rate) }
                cot = r.cotangent
                pg = r.grads
            } else {
                let model = GroupVJPModel(layers: Array(adapters.layers[top]), input: windows.count > 1 ? h : nil)
                let xConst = h
                let vg = valueAndGrad(model: model) { (m: GroupVJPModel, _: [MLXArray]) in
                    let out = run(m, layers, top, m.x ?? xConst)
                    return [lossFunction?(out) ?? HealingLoss.pinBatched([range.end: out], teacher: [range.end: target], weights: w)]
                }
                let (l, g) = vg(model, [])
                loss = l[0].item(Float.self)
                let flat = GradMath.flatten(g)
                let ok = loss.isFinite && loss <= c.maxLoss
                if !ok { return (loss, false, rate) }
                cot = flat["x"]
                pg = blockKeys(flat.filter { $0.key != "x" }, top)
                pg = GradMath.scale(pg, by: 1 / nb)
            }
            if let cot { eval(cot) }
            if Self.freeGradientsEarly {
                // One layer at a time: its update is evaluated, then its gradients are dropped before the next
                // layer's update builds its temporaries, so the gradients and the new moments are never all
                // alive together. The same arithmetic per tensor, so the result is bit-identical.
                for i in top {
                    let prefix = "layers.\(i)."
                    let mine = pg.filter { $0.key.hasPrefix(prefix) }
                    for k in mine.keys { pg[k] = nil }
                    Self.earlyFreedLayers += 1
                    eval(opt.update(layers: adapters, indices: [i], grads: mine, learningRate: rate))
                }
            } else {
                eval(opt.update(layers: adapters, indices: Array(top), grads: pg, learningRate: rate))
            }
        }

        // Backward walk: the windows below the top, from the top down, one layer at a time. A layer's
        // vjp builds its forward again (that is the recompute: the forward walk kept the layer's input
        // and dropped everything else) and goes back through it; no checkpoint wrapper is added, or the
        // layer would be run a third time.
        for (n, win) in windows.dropLast().enumerated().reversed() {
            let layers = try load(win)
            if n > 0 { prefetch(windows[n - 1]) }
            for (j, layer) in layers.enumerated().reversed() {
                let i = win.lowerBound + j
                let x = layerInputs[i]!
                let pg: [String: MLXArray]
                if let compiled {
                    let r = compiled.back(
                        index: i, layer: layer, set: sets[i], adapters: adapters.layers[i], sliced: sliced,
                        signature: signature, x: x, cotangent: cot!, wantInputGrad: i > 0, gradScale: gradScale)
                    cot = r.cotangent
                    pg = r.grads
                } else {
                    let model = GroupVJPModel(layers: [adapters.layers[i]], input: i > 0 ? x : nil)
                    let cot32 = cot!.asType(.float32)
                    let set = sets[i]
                    let vg = valueAndGrad(model: model) { (m: GroupVJPModel, _: [MLXArray]) in
                        let out = healingLayerForward(layer, m.x ?? x, set: set, adapters: m.layers[0], sliced: sliced)
                        return [(out.asType(.float32) * cot32).sum()]
                    }
                    let (_, g) = vg(model, [])
                    let flat = GradMath.flatten(g)
                    cot = flat["x"]
                    pg = GradMath.scale(blockKeys(flat.filter { $0.key != "x" }, i ... i), by: 1 / nb)
                }
                if let cot { eval(cot) }
                eval(opt.update(layers: adapters, indices: [i], grads: pg, learningRate: rate))
                layerInputs[i] = nil
            }
            if n == 0, let carry { carry.window = win; carry.layers = layers }
        }
        opt.completeStep()
        Memory.clearCache()
        return (loss, true, rate)
    }
}


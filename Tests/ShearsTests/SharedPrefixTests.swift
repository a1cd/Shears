import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// A layer that forwards everything to a Gemma 4 layer but does not offer the shared-prefix
/// split, so the shared forward has to fall back to running each row whole.
private final class WholeRowsOnlyLayer: HealableMoELayer {
    let inner: Gemma4MoEDecoderLayer
    init(_ inner: Gemma4MoEDecoderLayer) { self.inner = inner }
    var expertRouter: any ExpertRouter { inner.expertRouter }
    var expertBank: SwitchGLU { inner.expertBank }
    var stats: ExpertStats? {
        get { inner.stats }
        set { inner.stats = newValue }
    }
    var hiddenSize: Int { inner.hiddenSize }
    var expertHiddenSize: Int { inner.expertHiddenSize }
    var denseSiteDimensions: [DenseSite: (input: Int, output: Int)] { inner.denseSiteDimensions }
    func attentionPart(_ x: MLXArray, adapters: LayerAdapters?) -> MoELayerParts {
        inner.attentionPart(x, adapters: adapters)
    }
    func expertScale(_ indices: MLXArray) -> MLXArray { inner.expertScale(indices) }
    func combine(_ parts: MoELayerParts, expertBranch y: MLXArray) -> MLXArray {
        inner.combine(parts, expertBranch: y)
    }
    func normalizeExpertBranch(_ y: MLXArray) -> MLXArray { inner.normalizeExpertBranch(y) }
    var isGlobalAttention: Bool { inner.isGlobalAttention }
}

/// The shared-prefix forward equals running every row whole: outputs, and gradients of a loss summed
/// over the rows. The fixture's sliding window is 8, so prefixes of 1, 3, 5 (shorter than the
/// window), 7, 8, 9 (at it) and 13 (past it) are all covered.
final class SharedPrefixTests: ShearsTestCase {
    static let d = 64
    /// (prefix length, suffix length, rows)
    static let shapes: [(p: Int, s: Int, r: Int)] = [
        (13, 5, 3), (5, 6, 2), (3, 4, 1), (1, 6, 3), (7, 4, 2), (8, 4, 2), (9, 4, 2), (1, 1, 1),
    ]

    enum Mode: CaseIterable { case teacher, setOnly, adapters, sliced }

    struct Student {
        var layer: Gemma4MoEDecoderLayer
        var set: ExpertSet?
        var adapters: LayerAdapters?
        var sliced: Bool
    }

    func student(layer index: Int, mode: Mode) throws -> Student {
        let model = try HealingFixture.model()
        let layer = model.layers[index] as! Gemma4MoEDecoderLayer
        if mode == .teacher { return Student(layer: layer, set: nil, adapters: nil, sliced: false) }
        let set = ExpertSet(keep: HealingFixture.keepSets(count: 4)[index], numExperts: HealingFixture.experts)
        if mode == .setOnly { return Student(layer: layer, set: set, adapters: nil, sliced: false) }
        if mode == .sliced { sliceToKeptExperts(layer, set: set) }
        let adapters = ModelAdapters(
            layers: [layer], firstLayer: index, sets: [set], configuration: HealingFixture.smallAdapters)
        HealingFixture.randomizeAdapters(adapters, scale: 0.05)
        return Student(layer: layer, set: set, adapters: adapters.layers[0], sliced: mode == .sliced)
    }

    func hidden(_ shape: [Int]) -> MLXArray {
        let x = MLXRandom.normal(shape)
        eval(x)
        return x
    }

    /// Each row run whole through the existing forward, split into its prefix and suffix positions.
    func wholeRows(_ st: Student, prefix: MLXArray, suffix: MLXArray) -> [(prefix: MLXArray, suffix: MLXArray)] {
        let p = prefix.dim(1)
        return (0 ..< suffix.dim(0)).map { r in
            let out = healingLayerForward(
                st.layer, concatenated([prefix, suffix[r ..< (r + 1)]], axis: 1), set: st.set,
                adapters: st.adapters, sliced: st.sliced)
            return (out[0..., 0 ..< p, 0...], out[0..., p..., 0...])
        }
    }

    func assertShared(
        _ st: Student, p: Int, s: Int, r: Int, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
        let shared = healingLayerForwardShared(
            st.layer, prefix: prefix, suffix: suffix, set: st.set, adapters: st.adapters, sliced: st.sliced)
        XCTAssertEqual(shared.prefix.shape, [1, p, Self.d], label, file: file, line: line)
        XCTAssertEqual(shared.suffix.shape, [r, s, Self.d], label, file: file, line: line)
        let whole = wholeRows(st, prefix: prefix, suffix: suffix)
        for (i, w) in whole.enumerated() {
            assertClose(shared.prefix, w.prefix, atol: 1e-4, rtol: 1e-4, "prefix, row \(i), \(label)", file: file, line: line)
            assertClose(shared.suffix[i ..< (i + 1)], w.suffix, atol: 1e-4, rtol: 1e-4, "suffix, row \(i), \(label)", file: file, line: line)
        }
    }

    // MARK: Forward

    /// Sliding (0) and full-attention (1) layers, every mode, every shape.
    func testSharedForwardEqualsWholeRows() throws {
        for mode in Mode.allCases {
            for index in [0, 1] {
                let st = try student(layer: index, mode: mode)
                for (p, s, r) in Self.shapes {
                    assertShared(st, p: p, s: s, r: r, "layer \(index) \(mode) P=\(p) S=\(s) R=\(r)")
                }
            }
        }
    }

    /// The prefix keys a sliding layer drops, and the ones it keeps, are decided by the window alone:
    /// a suffix long enough to leave the window behind sees none of the prefix's early keys.
    func testSharedForwardWithSuffixLongerThanTheWindow() throws {
        for index in [0, 1] {
            let st = try student(layer: index, mode: .adapters)
            assertShared(st, p: 4, s: 20, r: 2, "layer \(index) long suffix")
            assertShared(st, p: 11, s: 12, r: 2, "layer \(index) long prefix and suffix")
        }
    }

    /// Chunked attention (queries scored a block at a time) with a block boundary inside the suffix
    /// and inside the prefix-plus-suffix span.
    func testSharedForwardWithChunkedAttention() throws {
        let saved = Gemma4Attention.queryChunk
        Gemma4Attention.queryChunk = 4
        defer { Gemma4Attention.queryChunk = saved }
        for index in [0, 1] {
            let st = try student(layer: index, mode: .adapters)
            assertShared(st, p: 13, s: 10, r: 2, "layer \(index) chunked")
            assertShared(st, p: 3, s: 9, r: 3, "layer \(index) chunked, short prefix")
            assertShared(st, p: 1, s: 5, r: 2, "layer \(index) chunked, P=1")
        }
    }

    /// The expert gather sorts rows once there are enough of them, and the prefix, the suffixes and
    /// the whole rows cross that threshold at different sizes; force both routes.
    func testSharedForwardOnBothExpertGatherRoutes() throws {
        for threshold in [1, 1 << 20] {
            let st = try student(layer: 0, mode: .adapters)
            st.layer.expertBank.sortThreshold = threshold
            assertShared(st, p: 9, s: 4, r: 3, "sort threshold \(threshold)")
        }
    }

    /// A layer without the split still gives the right answer, by running rows whole.
    func testLayerWithoutSharedSupportFallsBackToWholeRows() throws {
        let inner = try student(layer: 1, mode: .adapters)
        let wrapped = WholeRowsOnlyLayer(inner.layer)
        XCTAssertFalse(wrapped.supportsSharedPrefix)
        XCTAssertTrue(inner.layer.supportsSharedPrefix)
        let prefix = hidden([1, 9, Self.d]), suffix = hidden([2, 5, Self.d])
        let a = healingLayerForwardShared(
            wrapped, prefix: prefix, suffix: suffix, set: inner.set, adapters: inner.adapters)
        let b = healingLayerForwardShared(
            inner.layer, prefix: prefix, suffix: suffix, set: inner.set, adapters: inner.adapters)
        assertClose(a.prefix, b.prefix, atol: 1e-4, rtol: 1e-4)
        assertClose(a.suffix, b.suffix, atol: 1e-4, rtol: 1e-4)
    }

    /// A suffix is not independent of its prefix: the check above would pass vacuously if the
    /// suffix ignored the prefix and so did the reference.
    func testSuffixOutputDependsOnThePrefix() throws {
        let st = try student(layer: 1, mode: .adapters)
        let suffix = hidden([2, 5, Self.d])
        let a = healingLayerForwardShared(
            st.layer, prefix: hidden([1, 6, Self.d]), suffix: suffix, set: st.set, adapters: st.adapters)
        let b = healingLayerForwardShared(
            st.layer, prefix: hidden([1, 6, Self.d]), suffix: suffix, set: st.set, adapters: st.adapters)
        XCTAssertGreaterThan(abs(a.suffix - b.suffix).max().item(Float.self), 1e-2)
    }

    // MARK: Gradients

    /// Weights for a loss over a layer's outputs (a fixed random projection, so every output
    /// element and every row counts differently).
    func lossWeights(p: Int, s: Int, r: Int) -> (prefix: MLXArray, suffix: MLXArray) {
        (hidden([1, p, Self.d]), hidden([r, s, Self.d]))
    }

    func flat(_ g: ModuleParameters) -> [String: MLXArray] {
        let f = GradMath.flatten(g)
        eval(Array(f.values))
        return f
    }

    /// The gradient with respect to the layer's adapters of a loss summed over R whole rows equals the
    /// one through the shared form, in every mode, on sliding and full-attention layers.
    func testAdapterGradientsEqualWholeRows() throws {
        for mode in [Mode.adapters, .sliced] {
            for index in [0, 1] {
                let st = try student(layer: index, mode: mode)
                let (p, s, r) = (9, 5, 3)
                let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
                let w = lossWeights(p: p, s: s, r: r)
                let la = st.adapters!
                let wholeVG = valueAndGrad(model: la) { (m: LayerAdapters, _: [MLXArray]) in
                    var total = MLXArray(Float(0))
                    for i in 0 ..< r {
                        let out = healingLayerForward(
                            st.layer, concatenated([prefix, suffix[i ..< (i + 1)]], axis: 1), set: st.set,
                            adapters: m, sliced: st.sliced)
                        total = total + (out[0..., 0 ..< p, 0...] * w.prefix).sum()
                            + (out[0..., p..., 0...] * w.suffix[i ..< (i + 1)]).sum()
                    }
                    return [total]
                }
                let sharedVG = valueAndGrad(model: la) { (m: LayerAdapters, _: [MLXArray]) in
                    let out = healingLayerForwardShared(
                        st.layer, prefix: prefix, suffix: suffix, set: st.set, adapters: m, sliced: st.sliced)
                    // The prefix's positions are in every row, so they count R times.
                    return [Float(r) * (out.prefix * w.prefix).sum() + (out.suffix * w.suffix).sum()]
                }
                let (lw, gw) = wholeVG(la, [])
                let (ls, gs) = sharedVG(la, [])
                assertClose(ls[0], lw[0], atol: 1e-3, rtol: 1e-4, "loss layer \(index) \(mode)")
                let fw = flat(gw), fs = flat(gs)
                XCTAssertEqual(Set(fw.keys), Set(fs.keys))
                XCTAssertGreaterThan(fw.count, 10)
                var sawGradient = false
                for (k, g) in fw {
                    XCTAssertTrue(isFinite(g).all().item(Bool.self), k)
                    if abs(g).max().item(Float.self) > 1e-6 { sawGradient = true }
                    assertClose(fs[k]!, g, atol: 1e-3, rtol: 1e-3, "grad \(k) layer \(index) \(mode)")
                }
                XCTAssertTrue(sawGradient)
            }
        }
    }

    /// The gradient with respect to the layer's *inputs*: the prefix input collects every row's
    /// cotangent through the shared keys and values (and its own positions), the suffix input its own.
    func testInputGradientsEqualWholeRows() throws {
        for index in [0, 1] {
            let st = try student(layer: index, mode: .adapters)
            let (p, s, r) = (9, 5, 3)
            let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
            let w = lossWeights(p: p, s: s, r: r)
            let wholeGrad = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in
                var total = MLXArray(Float(0))
                for i in 0 ..< r {
                    let out = healingLayerForward(
                        st.layer, concatenated([a[0], a[1][i ..< (i + 1)]], axis: 1), set: st.set,
                        adapters: st.adapters, sliced: st.sliced)
                    total = total + (out[0..., 0 ..< p, 0...] * w.prefix).sum()
                        + (out[0..., p..., 0...] * w.suffix[i ..< (i + 1)]).sum()
                }
                return [total]
            }, argumentNumbers: [0, 1])
            let sharedGrad = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in
                let out = healingLayerForwardShared(
                    st.layer, prefix: a[0], suffix: a[1], set: st.set, adapters: st.adapters, sliced: st.sliced)
                return [Float(r) * (out.prefix * w.prefix).sum() + (out.suffix * w.suffix).sum()]
            }, argumentNumbers: [0, 1])
            let gw = wholeGrad([prefix, suffix]).1, gs = sharedGrad([prefix, suffix]).1
            XCTAssertGreaterThan(abs(gw[0]).max().item(Float.self), 1e-4)
            assertClose(gs[0], gw[0], atol: 1e-3, rtol: 1e-3, "d prefix, layer \(index)")
            assertClose(gs[1], gw[1], atol: 1e-3, rtol: 1e-3, "d suffix, layer \(index)")
        }
    }

    // MARK: Network

    func network(adapters: Bool, sliced: Bool = false) throws -> HealingNetwork {
        let model = try HealingFixture.model()
        let sets = HealingFixture.keepSets(count: 4).map { ExpertSet(keep: $0, numExperts: HealingFixture.experts) }
        let layers = model.healableLayers
        if sliced { for (l, s) in zip(layers, sets) { sliceToKeptExperts(l, set: s) } }
        var ad: ModelAdapters?
        if adapters {
            ad = ModelAdapters(layers: layers, sets: sets, configuration: HealingFixture.smallAdapters)
            HealingFixture.randomizeAdapters(ad!, scale: 0.05)
        }
        return HealingNetwork(layers: layers, sets: sets, adapters: ad, sliced: sliced)
    }

    func testNetworkSharedForwardEqualsWholeRows() throws {
        let (p, s, r) = (11, 5, 3)
        for (ad, sliced, student) in [(true, false, true), (true, true, true), (false, false, false), (false, false, true)] {
            let net = try network(adapters: ad, sliced: sliced)
            let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
            let shared = net.forwardShared(
                prefix: prefix, suffix: suffix, student: student, collect: [1, 3])
            for i in 0 ..< r {
                let whole = net.forward(
                    concatenated([prefix, suffix[i ..< (i + 1)]], axis: 1), student: student, collect: [1, 3])
                let label = "adapters \(ad) sliced \(sliced) student \(student) row \(i)"
                assertClose(shared.prefix, whole.0[0..., 0 ..< p, 0...], atol: 1e-3, rtol: 1e-4, label)
                assertClose(shared.suffix[i ..< (i + 1)], whole.0[0..., p..., 0...], atol: 1e-3, rtol: 1e-4, label)
                for L in [1, 3] {
                    assertClose(shared.collected[L]!.prefix, whole.1[L]![0..., 0 ..< p, 0...], atol: 1e-3, rtol: 1e-4, "L\(L) \(label)")
                    assertClose(shared.collected[L]!.suffix[i ..< (i + 1)], whole.1[L]![0..., p..., 0...], atol: 1e-3, rtol: 1e-4, "L\(L) \(label)")
                }
            }
        }
    }

    /// A sub-range of layers, from a state that is not the embedding.
    func testNetworkSharedForwardOverALayerRange() throws {
        let net = try network(adapters: true)
        let (p, s, r) = (6, 4, 2)
        let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
        let shared = net.forwardShared(prefix: prefix, suffix: suffix, from: 1, through: 2, student: true)
        for i in 0 ..< r {
            let whole = net.forward(
                concatenated([prefix, suffix[i ..< (i + 1)]], axis: 1), from: 1, through: 2, student: true)
            assertClose(shared.prefix, whole.0[0..., 0 ..< p, 0...], atol: 1e-3, rtol: 1e-4)
            assertClose(shared.suffix[i ..< (i + 1)], whole.0[0..., p..., 0...], atol: 1e-3, rtol: 1e-4)
        }
    }

    /// The gradient with respect to every layer's adapters, where an earlier layer's adapters reach the
    /// later layers' prefix keys and values: the recipe's loss over the final states, with and without
    /// checkpointing, and with the gradient cut before a layer.
    func testNetworkAdapterGradientsEqualWholeRows() throws {
        let (p, s, r) = (11, 5, 3)
        let w = lossWeights(p: p, s: s, r: r)
        for (checkpointed, detach, sliced) in [(false, nil, false), (true, nil, false), (true, 2, false), (false, nil, true)] as [(Bool, Int?, Bool)] {
            let net = try network(adapters: true, sliced: sliced)
            let ad = net.adapters!
            let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
            let wholeVG = valueAndGrad(model: ad) { (_: ModelAdapters, _: [MLXArray]) in
                var total = MLXArray(Float(0))
                for i in 0 ..< r {
                    let out = net.forward(
                        concatenated([prefix, suffix[i ..< (i + 1)]], axis: 1), student: true,
                        checkpointed: checkpointed, detachBefore: detach).0
                    total = total + (out[0..., 0 ..< p, 0...] * w.prefix).sum()
                        + (out[0..., p..., 0...] * w.suffix[i ..< (i + 1)]).sum()
                }
                return [total]
            }
            let sharedVG = valueAndGrad(model: ad) { (_: ModelAdapters, _: [MLXArray]) in
                let out = net.forwardShared(
                    prefix: prefix, suffix: suffix, student: true, checkpointed: checkpointed, detachBefore: detach)
                return [Float(r) * (out.prefix * w.prefix).sum() + (out.suffix * w.suffix).sum()]
            }
            let (lw, gw) = wholeVG(ad, [])
            let (ls, gs) = sharedVG(ad, [])
            let label = "checkpointed \(checkpointed) detach \(String(describing: detach)) sliced \(sliced)"
            assertClose(ls[0], lw[0], atol: 1e-2, rtol: 1e-4, "loss \(label)")
            let fw = flat(gw), fs = flat(gs)
            XCTAssertEqual(Set(fw.keys), Set(fs.keys), label)
            var reachedLayer0 = false
            for (k, g) in fw {
                XCTAssertTrue(isFinite(g).all().item(Bool.self), k)
                if k.hasPrefix("layers.0."), k.hasSuffix("b"), abs(g).max().item(Float.self) > 1e-6 { reachedLayer0 = true }
                // The gradients here are up to ~800 in size and pass through every layer, so float32 gives each
                // tensor an error proportional to its largest element, not to each element. On the CPU backend
                // (CI on Linux: AMD EPYC, 4 threads) that error is ~1e-5 of the tensor's max |g|: shared vs whole
                // rows differ by 0.0074 on layers.0.attn_out.b (max |g| 829), and nudging the inputs by a relative
                // 1e-6 moves the whole-row gradient by 0.038, five times more. So on the CPU the tolerance gets a
                // term of 1e-4 of the tensor's max |g|; a wrong gradient is off by far more. Metal keeps the exact one.
                let cpu = Device.defaultDevice().deviceType == .cpu
                let noise: Float = cpu ? 1e-4 * abs(g).max().item(Float.self) : 0
                assertClose(fs[k]!, g, atol: 2e-3 + noise, rtol: 2e-3, "grad \(k) \(label)")
            }
            // Layer 0's adapters sit below every shared K/V; with the gradient cut at layer 2 they get none.
            XCTAssertEqual(reachedLayer0, detach == nil, label)
        }
    }
}

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// TEMPORARY diagnostic (CI investigation of SharedPrefixTests.testNetworkAdapterGradientsEqualWholeRows on Linux CPU).
/// Not part of the suite once the investigation is over.
extension SharedPrefixTests {
    func testZDiagGradientConditioning() throws {
        let (p, s, r) = (11, 5, 3)
        let w = lossWeights(p: p, s: s, r: r)
        for (checkpointed, detach, sliced) in [(false, nil, false), (true, nil, false), (true, 2, false), (false, nil, true)] as [(Bool, Int?, Bool)] {
            let net = try network(adapters: true, sliced: sliced)
            let ad = net.adapters!
            let prefix = hidden([1, p, Self.d]), suffix = hidden([r, s, Self.d])
            func wholeGrad(_ pre: MLXArray, _ suf: MLXArray) -> [String: MLXArray] {
                let vg = valueAndGrad(model: ad) { (_: ModelAdapters, _: [MLXArray]) in
                    var total = MLXArray(Float(0))
                    for i in 0 ..< r {
                        let out = net.forward(
                            concatenated([pre, suf[i ..< (i + 1)]], axis: 1), student: true,
                            checkpointed: checkpointed, detachBefore: detach).0
                        total = total + (out[0..., 0 ..< p, 0...] * w.prefix).sum()
                            + (out[0..., p..., 0...] * w.suffix[i ..< (i + 1)]).sum()
                    }
                    return [total]
                }
                return flat(vg(ad, []).1)
            }
            func sharedGrad(_ pre: MLXArray, _ suf: MLXArray) -> [String: MLXArray] {
                let vg = valueAndGrad(model: ad) { (_: ModelAdapters, _: [MLXArray]) in
                    let out = net.forwardShared(prefix: pre, suffix: suf, student: true, checkpointed: checkpointed, detachBefore: detach)
                    return [Float(r) * (out.prefix * w.prefix).sum() + (out.suffix * w.suffix).sum()]
                }
                return flat(vg(ad, []).1)
            }
            let A = wholeGrad(prefix, suffix)
            let B = sharedGrad(prefix, suffix)
            // The same whole-row gradient with the inputs nudged by a relative 1e-6 (float32 rounding is 6e-8).
            let n1 = MLXRandom.normal(prefix.shape), n2 = MLXRandom.normal(suffix.shape)
            let C = wholeGrad(prefix * (1 + 1e-6 * n1), suffix * (1 + 1e-6 * n2))
            let label = "checkpointed \(checkpointed) detach \(String(describing: detach)) sliced \(sliced)"
            print("ZDIAG ==== \(label)")
            for k in A.keys.sorted() where k.hasPrefix("layers.0.") {
                guard let a = A[k], let b = B[k], let c = C[k] else { continue }
                let scale = abs(a).max().item(Float.self)
                let dab = abs(a - b), dac = abs(a - c)
                let tol = 2e-3 + 2e-3 * abs(a)
                let nab = (dab .> tol).sum().item(Int.self), nac = (dac .> tol).sum().item(Int.self)
                let flatDab = dab.reshaped(-1)
                let arg = flatDab.argMax().item(Int.self)
                print(String(format: "ZDIAG %@ n=%d max|g|=%.4g | shared-vs-whole max|d|=%.3g (%d over tol) rel=%.3g | nudged-vs-whole max|d|=%.3g (%d over tol) rel=%.3g | at argmax whole=%.5g shared=%.5g",
                             k, a.size, scale, dab.max().item(Float.self), nab, dab.max().item(Float.self) / max(scale, 1e-30),
                             dac.max().item(Float.self), nac, dac.max().item(Float.self) / max(scale, 1e-30),
                             a.reshaped(-1)[arg].item(Float.self), b.reshaped(-1)[arg].item(Float.self)))
            }
        }
    }
}

import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class CheckpointTests: ShearsTestCase {
    func testCheckpointGradientEqualsPlainGradient() {
        // Several primals, one of which the output doesn't depend on: the VJP must still
        // return one gradient per primal or MLX misattributes them.
        func f(_ a: [MLXArray]) -> [MLXArray] {
            let h = tanh(matmul(a[0], a[1]))
            return [(h * h).sum() + (a[0] * 0.5).sum()]
        }
        let x = MLXRandom.normal([4, 8])
        let w = MLXRandom.normal([8, 8])
        let unused = MLXRandom.normal([3])

        let plain = grad(f, argumentNumbers: [0, 1, 2])([x, w, unused])
        let cp = checkpoint(f)
        let viaCheckpoint = grad({ cp($0) }, argumentNumbers: [0, 1, 2])([x, w, unused])

        XCTAssertEqual(viaCheckpoint.count, 3)
        for (a, b) in zip(plain, viaCheckpoint) { assertClose(a, b) }
        XCTAssertEqual(viaCheckpoint[2].abs().sum().item(Float.self), 0)
    }

    func testCheckpointedModuleGradientEqualsPlain() {
        let model = Sequential(layers: Linear(16, 32), Tanh(), Linear(32, 16), Tanh(), Linear(16, 16))
        eval(model)
        let x = MLXRandom.normal([8, 16])
        let y = MLXRandom.normal([8, 16])

        let plain = valueAndGrad(model: model) { (m: Sequential, x, y) in
            mseLoss(predictions: m(x), targets: y)
        }
        let checkpointed = valueAndGrad(model: model) { (m: Sequential, x, y) in
            var h = x
            for layer in m.layers {
                h = checkpoint(layer as! Module) { mod, v in (mod as! UnaryLayer)(v) }(h)
            }
            return mseLoss(predictions: h, targets: y)
        }

        let (l1, g1) = plain(model, x, y)
        let (l2, g2) = checkpointed(model, x, y)
        assertClose(l1, l2)
        assertClose(g1, g2)
    }

    func testCheckpointThroughMoELayer() {
        let layer = Toy.layers(count: 1)[0]
        let x = Toy.batches(1)[0]
        let t = MLXRandom.normal(x.shape)

        let plain = valueAndGrad(model: layer) { (m: MoEResidualLayer, x, t) in
            mseLoss(predictions: m(x), targets: t)
        }
        let checkpointed = valueAndGrad(model: layer) { (m: MoEResidualLayer, x, t) in
            mseLoss(predictions: checkpoint(m) { $0($1) }(x), targets: t)
        }
        let (_, g1) = plain(layer, x, t)
        let (_, g2) = checkpointed(layer, x, t)
        assertClose(g1, g2)
    }
}

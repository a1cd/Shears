import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class SwitchGLUTests: ShearsTestCase {
    func testSortedAndUnsortedPathsAgree() {
        let glu = SwitchGLU(inputDims: 16, hiddenDims: 24, numExperts: 6)
        eval(glu)
        let x = MLXRandom.normal([3, 10, 16])
        let idx = MLXRandom.randInt(0 ..< 6, [3, 10, 2]).asType(.uint32)

        let unsorted = glu(x, idx, sort: false)
        let sorted = glu(x, idx, sort: true)
        XCTAssertEqual(unsorted.shape, [3, 10, 2, 16])
        assertClose(unsorted, sorted)
    }

    func testSortedAndUnsortedGradientsAgree() {
        // `sortedIndices` changes the backward pass (segmented matmul vs scatter-add), so
        // compare gradients for the weights and the input, not only the forward output.
        let glu = SwitchGLU(inputDims: 16, hiddenDims: 24, numExperts: 6)
        eval(glu)
        let x = MLXRandom.normal([4, 20, 16])
        let idx = MLXRandom.randInt(0 ..< 6, [4, 20, 2]).asType(.uint32)

        func grads(sort: Bool) -> (MLXArray, ModuleParameters) {
            let vg = valueAndGrad(model: glu) { (m: SwitchGLU, arrays: [MLXArray]) in
                [(m(arrays[0], idx, sort: sort) * m(arrays[0], idx, sort: sort)).sum()]
            }
            let (_, gw) = vg(glu, [x])
            let gx = grad({ (x: MLXArray) in (glu(x, idx, sort: sort) * glu(x, idx, sort: sort)).sum() })(x)
            return (gx, gw)
        }
        let (gxU, gwU) = grads(sort: false)
        let (gxS, gwS) = grads(sort: true)
        assertClose(gxU, gxS, atol: 1e-3, rtol: 1e-3)
        assertClose(gwU, gwS, atol: 1e-3, rtol: 1e-3)
    }

    func testTrainingModeAlwaysSorts() {
        // Consistency only: both paths give the same output, so this can't show which one
        // ran. It exercises the small-input training path (indices.size < sortThreshold).
        let glu = SwitchGLU(inputDims: 8, hiddenDims: 8, numExperts: 4)
        eval(glu)
        let x = MLXRandom.normal([1, 3, 8])
        let idx = MLXArray([0, 3, 1, 2, 3, 0] as [UInt32], [1, 3, 2])
        glu.train(true)
        let a = glu(x, idx)
        glu.train(false)
        let b = glu(x, idx)
        assertClose(a, b)
    }
}

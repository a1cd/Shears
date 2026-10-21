import MLX
import MLXNN
import MLXRandom
import MLXOptimizers
import XCTest

@testable import Shears

/// Caps MLX's memory and buffer cache so the suite can run alongside other GPU work
/// without competing for unified memory.
#if !canImport(ObjectiveC)
extension ShearsTestCase {
    /// corelibs XCTest has no `init()`. Some tests make an instance of another case only to call its
    /// helpers; this gives that instance a name and an empty body.
    convenience init() { self.init(name: "helper", testClosure: { _ in }) }
}
#endif

class ShearsTestCase: XCTestCase {
    override func setUp() {
        super.setUp()
        Memory.memoryLimit = 3 * 1024 * 1024 * 1024
        Memory.cacheLimit = 256 * 1024 * 1024
        MLXRandom.seed(0)
    }

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    func assertClose(
        _ a: MLXArray, _ b: MLXArray, atol: Float = 1e-4, rtol: Float = 1e-4,
        _ message: String = "", file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, "shape \(message)", file: file, line: line)
        let diff = abs(a - b).max().item(Float.self)
        let ok = allClose(a, b, rtol: Double(rtol), atol: Double(atol)).item(Bool.self)
        XCTAssertTrue(ok, "max |diff| = \(diff) \(message)", file: file, line: line)
    }

    func assertClose(
        _ a: ModuleParameters, _ b: ModuleParameters, atol: Float = 1e-4, rtol: Float = 1e-4,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let fa = Dictionary(uniqueKeysWithValues: a.flattened())
        let fb = Dictionary(uniqueKeysWithValues: b.flattened())
        XCTAssertEqual(Set(fa.keys), Set(fb.keys), file: file, line: line)
        for (k, va) in fa {
            guard let vb = fb[k] else { continue }
            assertClose(va, vb, atol: atol, rtol: rtol, k, file: file, line: line)
        }
    }
}

enum Toy {
    static func layers(
        count: Int, dims: Int = 32, hidden: Int = 64, experts: Int = 8, topK: Int = 2,
        seed: UInt64 = 1
    ) -> [MoEResidualLayer] {
        (0 ..< count).map { i in
            MLXRandom.seed(seed &+ UInt64(i))
            let l = MoEResidualLayer(dims: dims, hiddenDims: hidden, numExperts: experts, topK: topK)
            eval(l)
            return l
        }
    }

    static func clone(_ l: MoEResidualLayer) -> MoEResidualLayer {
        let c = MoEResidualLayer(
            dims: l.norm.weight.dim(0), hiddenDims: l.moe.experts.gateProj.outputDims,
            numExperts: l.moe.numExperts, topK: l.moe.router.topK)
        l.copyParameters(into: c)
        return c
    }

    static func batches(_ n: Int, tokens: Int = 64, dims: Int = 32, seed: UInt64 = 99) -> [MLXArray] {
        MLXRandom.seed(seed)
        let xs = (0 ..< n).map { _ in MLXRandom.normal([2, tokens / 2, dims]) }
        eval(xs)
        return xs
    }
}

/// Trains one layer against a frozen teacher with MSE and AdamW (a minimal local loop for
/// tests of the freezing helpers and quantized experts; not the healing recipe).
func mseHeal<B: Module & UnaryLayer>(
    student: B, teacher: B, inputs: [MLXArray], prepare: (B) -> Void,
    learningRate: Float = 3e-3, epochs: Int = 6
) -> (initial: Float, final: Float) {
    teacher.freeze()
    prepare(student)
    func mean() -> Float {
        inputs.map { mseLoss(predictions: student($0), targets: teacher($0)).item(Float.self) }
            .reduce(0, +) / Float(inputs.count)
    }
    let initial = mean()
    let opt = AdamW(learningRate: learningRate, weightDecay: 0)
    let vg = valueAndGrad(model: student) { (m: B, x, t) in mseLoss(predictions: m(x), targets: t) }
    for _ in 0 ..< epochs {
        for x in inputs {
            let (loss, g) = vg(student, x, stopGradient(teacher(x)))
            opt.update(model: student, gradients: g)
            eval(student, opt, loss)
        }
    }
    return (initial, mean())
}

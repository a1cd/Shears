import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class QuantizedTests: ShearsTestCase {
    func testQuantizedMatchesDequantizedFloat() {
        let float = SwitchLinear(inputDims: 64, outputDims: 32, numExperts: 4)
        eval(float)
        let q = QuantizedSwitchLinear(float, groupSize: 64, bits: 4)
        let deq = SwitchLinear(weight: q.dequantizedWeight())

        let x = expandedDimensions(MLXRandom.normal([2, 6, 64]), axes: [-2, -3])
        let idx = MLXRandom.randInt(0 ..< 4, [2, 6, 2]).asType(.uint32)
        assertClose(q(x, idx), deq(x, idx), atol: 1e-3, rtol: 1e-3)
    }

    func testGradientFlowsThroughFrozenQuantizedExperts() {
        let block = MoEBlock(dims: 64, hiddenDims: 128, numExperts: 4, topK: 2)
        eval(block)
        ExpertQuantizer.quantizeExperts(block, groupSize: 64, bits: 4)
        let reference = MoEBlock(
            router: TopKRouter(gateWeight: block.router.gateWeight, topK: 2),
            experts: SwitchGLU(
                gateProj: SwitchLinear(weight: (block.experts.gateProj as! QuantizedSwitchLinear).dequantizedWeight()),
                upProj: SwitchLinear(weight: (block.experts.upProj as! QuantizedSwitchLinear).dequantizedWeight()),
                downProj: SwitchLinear(weight: (block.experts.downProj as! QuantizedSwitchLinear).dequantizedWeight())))

        let x = MLXRandom.normal([2, 16, 64])
        let gq = grad({ (x: MLXArray) in block(x).square().sum() })(x)
        let gf = grad({ (x: MLXArray) in reference(x).square().sum() })(x)
        XCTAssertGreaterThan(gq.abs().sum().item(Float.self), 0)
        assertClose(gq, gf, atol: 2e-2, rtol: 2e-2)
    }

    func testLoRAStartsAsIdentityAndOnlyAdaptersTrain() {
        let block = MoEBlock(dims: 64, hiddenDims: 128, numExperts: 4, topK: 2)
        eval(block)
        ExpertQuantizer.quantizeExperts(block, groupSize: 64, bits: 4)
        let x = MLXRandom.normal([1, 8, 64])
        let before = block(x)

        let withLoRA = MoEBlock(dims: 64, hiddenDims: 128, numExperts: 4, topK: 2)
        eval(withLoRA)
        ExpertQuantizer.quantizeExperts(withLoRA, groupSize: 64, bits: 4, loraRank: 4)
        // Same base weights and router as `block`, LoRA B = 0.
        withLoRA.update(parameters: block.parameters())
        assertClose(withLoRA(x), before, atol: 1e-5)

        MoETraining.trainRouterAndLoRA(withLoRA)
        let keys = Set(withLoRA.trainableParameters().flattened().map(\.0))
        XCTAssertEqual(keys, Set(["gate.weight"] + ["gate_proj", "up_proj", "down_proj"].flatMap {
            ["switch_mlp.\($0).lora_a", "switch_mlp.\($0).lora_b"]
        }))
        // Router-and-experts on a quantized block must not unfreeze packed weights.
        MoETraining.trainRouterAndExperts(withLoRA)
        XCTAssertEqual(Set(withLoRA.trainableParameters().flattened().map(\.0)), keys)
    }

    func testLoRAHealingAroundQuantizedExpertsReducesLoss() {
        let teachers = Toy.layers(count: 2, dims: 64, hidden: 128, experts: 8)
        let inputs = Toy.batches(8, dims: 64)
        let students = teachers.map { t -> MoEResidualLayer in
            let s = Toy.clone(t)
            ExpertPruner.prune(s.moe, keep: [0, 1, 2, 3])
            ExpertQuantizer.quantizeExperts(s.moe, groupSize: 64, bits: 4, loraRank: 8)
            return s
        }
        let reports = zip(students, teachers).map {
            mseHeal(student: $0, teacher: $1, inputs: inputs, prepare: MoETraining.trainRouterAndLoRA)
        }
        for r in reports { XCTAssertLessThan(r.final, r.initial * 0.9, "\(r)") }
        // The adapters actually trained (router-only training also lowers the loss).
        for s in students {
            let b = (s.moe.experts.upProj as! QuantizedSwitchLinear).loraB!
            XCTAssertGreaterThan(b.abs().sum().item(Float.self), 0)
        }

        let routerOnly = teachers.map { t -> MoEResidualLayer in
            let s = Toy.clone(t)
            ExpertPruner.prune(s.moe, keep: [0, 1, 2, 3])
            ExpertQuantizer.quantizeExperts(s.moe, groupSize: 64, bits: 4)
            return s
        }
        let r2 = zip(routerOnly, teachers).map {
            mseHeal(student: $0, teacher: $1, inputs: inputs, prepare: MoETraining.trainRouterOnly)
        }
        for r in r2 { XCTAssertLessThan(r.final, r.initial, "\(r)") }
        for (lora, router) in zip(reports, r2) {
            XCTAssertLessThan(lora.final, router.final, "LoRA should beat router-only")
        }
    }

    func testLoRAIsAddedToAlreadyQuantizedExperts() {
        let block = MoEBlock(dims: 64, hiddenDims: 128, numExperts: 4, topK: 2)
        eval(block)
        ExpertQuantizer.quantizeExperts(block, groupSize: 64, bits: 4)
        let x = MLXRandom.normal([1, 8, 64])
        let before = block(x)
        ExpertQuantizer.quantizeExperts(block, loraRank: 4)
        XCTAssertTrue((block.experts.gateProj as! QuantizedSwitchLinear).hasLoRA)
        assertClose(block(x), before, atol: 1e-5)
        MoETraining.trainRouterAndLoRA(block)
        XCTAssertEqual(block.trainableParameters().flattened().count, 7)
    }

    func testPruningQuantizedBlock() {
        let block = MoEBlock(dims: 64, hiddenDims: 128, numExperts: 6, topK: 2)
        eval(block)
        ExpertQuantizer.quantizeExperts(block, groupSize: 64, bits: 4, loraRank: 2)
        let x = MLXRandom.normal([1, 64, 64])
        let before = block(x)
        ExpertPruner.prune(block, keep: Array(0 ..< 6))
        assertClose(block(x), before)

        ExpertPruner.prune(block, keep: [1, 4])
        let q = block.experts.gateProj as! QuantizedSwitchLinear
        XCTAssertEqual(q.weight.dim(0), 2)
        XCTAssertEqual(q.scales.dim(0), 2)
        XCTAssertEqual(q.loraA!.shape, [2, 2, 64])
        XCTAssertEqual(block(x).shape, [1, 64, 64])
    }
}

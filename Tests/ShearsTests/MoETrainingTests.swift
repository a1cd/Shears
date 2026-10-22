import MLX
import MLXNN
import XCTest

@testable import Shears

final class MoETrainingTests: ShearsTestCase {
    func testPrepareFreezesEverythingButRouterAndExperts() {
        let layer = Toy.layers(count: 1)[0]
        MoETraining.trainRouterAndExperts(layer)
        let keys = Set(layer.trainableParameters().flattened().map(\.0))
        XCTAssertEqual(keys, [
            "mlp.gate.weight", "mlp.switch_mlp.gate_proj.weight",
            "mlp.switch_mlp.up_proj.weight", "mlp.switch_mlp.down_proj.weight",
        ])
        MoETraining.trainRouterOnly(layer)
        XCTAssertEqual(Set(layer.trainableParameters().flattened().map(\.0)), ["mlp.gate.weight"])
    }
}

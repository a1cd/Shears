import Foundation
import MLX
import XCTest

@testable import Shears

/// Routers pick experts with mlx-lm's form, `argpartition(x, kth=-k)[..., -k:]`, so ties break as
/// they do in the Python reference. The expected indices below were produced by `mlx.core` itself
/// (mlx-lm 0.31.2's call, on the same Metal build family) for rows full of equal scores.
final class TopKFormTests: ShearsTestCase {
    func ids(_ a: MLXArray) -> [[Int]] {
        let k = a.dim(-1)
        let flat = a.asType(.int32).asArray(Int32.self).map(Int.init)
        return stride(from: 0, to: flat.count, by: k).map { Array(flat[$0 ..< $0 + k]) }
    }

    func testTiesBreakAsInMLXLM() {
        let x = MLXArray([
            1, 3, 3, 2, 3, 0,
            5, 5, 5, 5, 5, 5,
            0, 0, 1, 1, 0, 0,
            2, 1, 2, 1, 2, 1,
        ] as [Float], [4, 6])
        XCTAssertEqual(ids(topKIndices(x, 1)), [[4], [5], [3], [4]])
        XCTAssertEqual(ids(topKIndices(x, 2)), [[2, 4], [4, 5], [2, 3], [2, 4]])
        XCTAssertEqual(ids(topKIndices(x, 3)), [[1, 2, 4], [3, 4, 5], [5, 2, 3], [0, 2, 4]])
        let y = MLXArray(
            (0 ..< 16).map { Float($0 % 3) } + (0 ..< 16).map { Float(($0 * 7) % 4) }, [2, 16])
        XCTAssertEqual(ids(topKIndices(y, 2)), [[11, 14], [9, 13]])
        XCTAssertEqual(ids(topKIndices(y, 4)), [[5, 8, 11, 14], [1, 5, 9, 13]])
    }

    func testPicksTheLargestAndAllWhenKIsN() {
        let x = MLXArray([0.1, 0.9, 0.5, 0.7] as [Float], [1, 4])
        XCTAssertEqual(Set(ids(topKIndices(x, 2))[0]), [1, 3])
        XCTAssertEqual(Set(ids(topKIndices(x, 4))[0]), [0, 1, 2, 3])
    }
}

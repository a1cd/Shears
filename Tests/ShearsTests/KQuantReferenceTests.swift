import Foundation
import XCTest

@testable import Shears

/// The K-quant quantizers against bytes written by ggml itself, in the default suite (the full
/// comparison, `KQuantOracleTests`, needs the reference build and an environment variable). The
/// vectors are a few rows of the reference tool's output; see `KQuantReferenceVectors.swift`.
final class KQuantReferenceTests: XCTestCase {
    func floats(_ b64: String) -> [Float] {
        Data(base64Encoded: b64)!.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    func testQuantizersMatchGGMLBytes() {
        for e in KQuantReferenceVectors.entries {
            let x = floats(e.x)
            let im = e.importance ? floats(KQuantReferenceVectors.imatrices[e.kind]!) : []
            XCTAssertEqual(x.count, 512)
            let want = Data(base64Encoded: e.bytes)!
            let got: Data = x.withUnsafeBufferPointer { xp in
                im.withUnsafeBufferPointer { ip in
                    let i: UnsafeBufferPointer<Float>? = e.importance ? ip : nil
                    switch e.type {
                    case "q4_k": return GGUFQ4K.quantize(xp, perRow: 512, importance: i)
                    case "q5_k": return GGUFQ5K.quantize(xp, perRow: 512, importance: i)
                    case "q6_k": return GGUFQ6K.quantize(xp, perRow: 512, importance: i)
                    default: return GGUFQ5_0.quantize(xp, perRow: 512, importance: i)
                    }
                }
            }
            XCTAssertEqual(got, want, "\(e.kind) row \(e.row) \(e.type)\(e.importance ? " with importance" : "")")
        }
    }
}

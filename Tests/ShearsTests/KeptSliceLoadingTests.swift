import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Loading a layer straight to its kept experts (`loadDecoderLayer(_:keeping:)`) must equal loading the
/// whole layer and cutting it, for plain and 4-bit checkpoints, mapped or not.
final class KeptSliceLoadingTests: ShearsTestCase {
    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("kslice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func assertSameLayers(_ cp: Gemma4Checkpoint, dtype: DType?, mapped: Bool, file: StaticString = #filePath, line: UInt = #line) throws {
        // Non-contiguous, unsorted-input and single-run kept sets.
        for keep in [[1, 4, 5, 7], [6, 0, 3], [2, 3, 4], [0]] {
            for i in [0, 1, 3] {
                let direct = try cp.loadDecoderLayer(i, dtype: dtype, mapped: mapped, keeping: keep) as! Gemma4MoEDecoderLayer
                let cut = try cp.loadDecoderLayer(i, dtype: dtype, mapped: false) as! Gemma4MoEDecoderLayer
                sliceToKeptExperts(cut, set: ExpertSet(keep: keep, numExperts: cut.numExperts))
                XCTAssertEqual(direct.numExperts, cut.numExperts, file: file, line: line)
                XCTAssertEqual(direct.expertRouter.topK, cut.expertRouter.topK, file: file, line: line)
                let a = Dictionary(uniqueKeysWithValues: direct.parameters().flattened())
                let b = Dictionary(uniqueKeysWithValues: cut.parameters().flattened())
                XCTAssertEqual(Set(a.keys), Set(b.keys), file: file, line: line)
                for (k, v) in b {
                    guard let w = a[k] else { continue }
                    XCTAssertEqual(w.shape, v.shape, "\(k) layer \(i) keep \(keep)", file: file, line: line)
                    XCTAssertEqual(w.dtype, v.dtype, "\(k)", file: file, line: line)
                    XCTAssertTrue((w.shape == v.shape && (w .== v).all().item(Bool.self)), "\(k) layer \(i) keep \(keep) differs", file: file, line: line)
                }
            }
        }
    }

    func testFloatCheckpointKeptSliceEqualsLoadThenSlice() throws {
        let m = try HealingFixture.model()
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        try MergeAndSliceExport.write(
            model: m, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: 4), adapters: nil, to: dir, options: o)
        let cp = try Gemma4Checkpoint(directory: dir)
        try assertSameLayers(cp, dtype: .float32, mapped: false)
        try assertSameLayers(cp, dtype: .float32, mapped: true)
        try assertSameLayers(cp, dtype: .bfloat16, mapped: false)
    }

    func testQuantizedCheckpointKeptSliceEqualsLoadThenSlice() throws {
        let dir = try tempDir()
        _ = try QuantizedFixture.write(try HealingFixture.model(), to: dir)
        let cp = try Gemma4Checkpoint(directory: dir)
        try assertSameLayers(cp, dtype: nil, mapped: false)
        try assertSameLayers(cp, dtype: nil, mapped: true)
    }

    func testRowGatherReadsOnlyTheRequestedRows() throws {
        let dir = try tempDir()
        let t = MLXArray(Array(0 ..< 7 * 3 * 5).map(Float.init), [7, 3, 5])
        let url = dir.appendingPathComponent("t.safetensors")
        try save(arrays: ["w": t], url: url)
        let e = try MappedSafetensors.index(of: url)["w"]!
        let fd = open(url.path, O_RDONLY)
        defer { close(fd) }
        let g = try MappedSafetensors.gatherRows(e, rows: [5, 1, 2, 6], fd: fd, in: url)
        XCTAssertEqual(g.shape, [4, 3, 5])
        XCTAssertTrue((g .== t.take(MLXArray([5, 1, 2, 6] as [Int32]), axis: 0)).all().item(Bool.self))
    }
}

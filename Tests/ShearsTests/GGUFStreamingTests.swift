import Foundation
import MLX
import XCTest

@testable import Shears

/// Memory behaviour of the exporter on a checkpoint much larger than one tensor.
/// Skipped unless `TEST_RUNNER_SHEARS_BENCH` is set, like the repository's other benchmarks.
final class GGUFStreamingTests: ShearsTestCase {
    /// Writes a safetensors file of `count` zero-filled bfloat16 tensors of `elements` each,
    /// streaming so the test itself never holds them.
    static func writeBlankSafetensors(to url: URL, names: [String], shape: [Int]) throws {
        let elements = shape.reduce(1, *)
        let bytes = elements * 2
        var header: [String: Any] = [:]
        for (i, n) in names.enumerated() {
            header[n] = ["dtype": "BF16", "shape": shape, "data_offsets": [i * bytes, (i + 1) * bytes]]
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        var len = UInt64(json.count).littleEndian
        try h.write(contentsOf: Data(bytes: &len, count: 8))
        try h.write(contentsOf: json)
        let block = Data(repeating: 0x3F, count: 1 << 20)  // bf16 0x3F3F, about 0.74
        for _ in names {
            var left = bytes
            while left > 0 {
                let n = min(left, block.count)
                try h.write(contentsOf: block.prefix(n))
                left -= n
            }
        }
    }

    func testPeakMemoryStaysNearOneTensorNotTheModel() throws {
        guard ProcessInfo.processInfo.environment["SHEARS_BENCH"] != nil else {
            throw XCTSkip("set TEST_RUNNER_SHEARS_BENCH=1 to run")
        }
        let dir = try makeGGUFTempDirectory()
        // Four 96 MB tensors: 384 MB of weights, converted from bfloat16 to 8-bit.
        let names = (0 ..< 4).map { "w\($0)" }
        try Self.writeBlankSafetensors(
            to: dir.appendingPathComponent("model.safetensors"), names: names, shape: [6144, 8192])
        let index = try GGUFSafetensorsIndex(directory: dir)

        let writer = try GGUFWriter()
        for n in names {
            let entry = index.entries[n]!
            try writer.addTensor(name: n, shape: entry.shape, type: .q8_0, source: .array({ try entry.load() }))
        }
        Memory.cacheLimit = 0
        Memory.clearCache()
        let before = Memory.activeMemory
        let out = dir.appendingPathComponent("out.gguf")
        try writer.write(to: out)
        let peak = Memory.peakMemory
        let size = try FileManager.default.attributesOfItem(atPath: out.path)[.size] as? Int ?? 0
        print("streaming export: wrote \(size >> 20) MB, MLX peak \(peak >> 20) MB, active before \(before >> 20) MB")
        // One tensor in memory (96 MB), plus one 16M-element float32 slice and its output.
        XCTAssertLessThan(peak - before, 300 << 20, "peak should track one tensor, not four")
        XCTAssertGreaterThan(size, 4 * 6144 * 8192 / 32 * 34 - 1)
        let r = try GGUFReader(url: out)
        XCTAssertEqual(r.tensors.count, 4)
        let first = try r.floats("w3").prefix(64)
        XCTAssertEqual(first.first!, 0.7421875, accuracy: 0.01)
    }
}

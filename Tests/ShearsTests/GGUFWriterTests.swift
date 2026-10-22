import Foundation
import MLX
import MLXRandom
import XCTest

@testable import Shears

final class GGUFWriterTests: ShearsTestCase {
    var dir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("gguf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
        try super.tearDownWithError()
    }

    // MARK: Metadata

    func testMetadataRoundTripsEveryType() throws {
        let w = try GGUFWriter()
        let all: [(String, GGUFValue)] = [
            ("t.u8", .uint8(255)), ("t.i8", .int8(-128)),
            ("t.u16", .uint16(65535)), ("t.i16", .int16(-32768)),
            ("t.u32", .uint32(4_294_967_295)), ("t.i32", .int32(-2_147_483_648)),
            ("t.u64", .uint64(UInt64.max)), ("t.i64", .int64(Int64.min)),
            ("t.f32", .float32(-0.15625)), ("t.f64", .float64(1.0 / 3.0)),
            ("t.bool_true", .bool(true)), ("t.bool_false", .bool(false)),
            ("t.str", .string("héllo \u{1F600} 世界")), ("t.empty_str", .string("")),
            ("a.u8", .array(.uint8, [.uint8(0), .uint8(200)])),
            ("a.i8", .array(.int8, [.int8(-1), .int8(1)])),
            ("a.u16", .array(.uint16, [.uint16(1), .uint16(60000)])),
            ("a.i16", .array(.int16, [.int16(-300), .int16(300)])),
            ("a.u32", .uint32s([1, 2, 4_000_000_000])),
            ("a.i32", .int32s([-5, 0, 7])),
            ("a.u64", .array(.uint64, [.uint64(1 << 40)])),
            ("a.i64", .array(.int64, [.int64(-(1 << 40))])),
            ("a.f32", .float32s([0.5, -2.25, 1e30])),
            ("a.f64", .array(.float64, [.float64(2.718281828459045)])),
            ("a.bool", .bools([true, false, true])),
            ("a.str", .strings(["a", "", "ß"])),
            ("a.empty", .array(.int32, [])),
            ("a.nested", .array(.array, [.array(.uint8, [.uint8(1)]), .array(.uint8, [])])),
        ]
        for (k, v) in all { try w.set(k, v) }
        let url = dir.appendingPathComponent("meta.gguf")
        try w.write(to: url)

        let r = try GGUFReader(url: url)
        XCTAssertEqual(r.version, 3)
        XCTAssertEqual(r.tensors.count, 0)
        XCTAssertEqual(r.metadata.map { $0.key }, all.map { $0.0 }, "order is preserved")
        for (k, v) in all {
            XCTAssertEqual(r[k], v, k)
            XCTAssertEqual(r[k]?.type, v.type, k)
        }
    }

    func testSettingAKeyAgainReplacesItInPlace() throws {
        let w = try GGUFWriter()
        try w.set("a", .uint32(1))
        try w.set("b", .uint32(2))
        try w.set("a", .uint32(3))
        let url = dir.appendingPathComponent("replace.gguf")
        try w.write(to: url)
        let r = try GGUFReader(url: url)
        XCTAssertEqual(r.metadata.map { $0.key }, ["a", "b"])
        XCTAssertEqual(r["a"], .uint32(3))
    }

    func testHeaderBytesMatchTheSpec() throws {
        let w = try GGUFWriter()
        try w.set("k", .uint32(7))
        let h = try w.headerBytes()
        var expected = Data([0x47, 0x47, 0x55, 0x46])  // GGUF
        expected.append(contentsOf: [3, 0, 0, 0])  // version
        expected.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])  // tensors
        expected.append(contentsOf: [1, 0, 0, 0, 0, 0, 0, 0])  // kv
        expected.append(contentsOf: [1, 0, 0, 0, 0, 0, 0, 0, 0x6B])  // key "k"
        expected.append(contentsOf: [4, 0, 0, 0])  // UINT32
        expected.append(contentsOf: [7, 0, 0, 0])
        XCTAssertEqual(h, expected)
    }

    // MARK: Tensors

    func testFloatTypesRoundTrip() throws {
        MLXRandom.seed(3)
        let base = MLXRandom.normal([4, 64]) * 3
        let exactHalf = MLXArray((0 ..< 128).map { Float($0) / 8 - 8 }, [2, 64])  // exact in f16 and bf16
        let w = try GGUFWriter()
        for (name, a) in [("rand", base), ("exact", exactHalf)] {
            for t in [GGUFTensorType.f32, .f16, .bf16] {
                try w.addTensor(name: "\(name).\(t.name)", shape: a.shape, type: t, source: .array({ a }))
            }
        }
        let url = dir.appendingPathComponent("floats.gguf")
        try w.write(to: url)
        let r = try GGUFReader(url: url)

        let randValues = base.asArray(Float.self)
        let exactValues = exactHalf.asArray(Float.self)
        XCTAssertEqual(try r.floats("rand.F32"), randValues)
        XCTAssertEqual(try r.floats("exact.F16"), exactValues)
        XCTAssertEqual(try r.floats("exact.BF16"), exactValues)
        for (got, want) in zip(try r.floats("rand.F16"), randValues) {
            XCTAssertEqual(got, want, accuracy: abs(want) * 1e-3 + 1e-6)
        }
        for (got, want) in zip(try r.floats("rand.BF16"), randValues) {
            XCTAssertEqual(got, want, accuracy: abs(want) * 8e-3 + 1e-6)
        }
        // Stored sizes are what the type promises.
        XCTAssertEqual(r.tensorInfo("rand.F16")?.byteCount, 4 * 64 * 2)
        XCTAssertEqual(try r.tensorData("rand.BF16").count, 4 * 64 * 2)
    }

    func testQ8_0ExactBytes() throws {
        // amax is 127, so the scale is exactly 1 and quantization is the identity on integers.
        var block = [Float](repeating: 0, count: 32)
        block[0] = 127
        block[1] = -127
        block[2] = 2.5  // halves round away from zero
        block[3] = -2.5
        block[4] = 3.4
        block[5] = -3.6
        let w = try GGUFWriter()
        try w.addTensor(
            name: "q", shape: [1, 32], type: .q8_0, source: .array({ MLXArray(block, [1, 32]) }))
        let url = dir.appendingPathComponent("q.gguf")
        try w.write(to: url)
        let r = try GGUFReader(url: url)
        let raw = try r.tensorData("q")
        XCTAssertEqual(raw.count, 34)
        XCTAssertEqual(Array(raw[0 ..< 2]), [0x00, 0x3C], "scale 1.0 as little-endian half")
        let q = raw[2...].map { Int8(bitPattern: $0) }
        XCTAssertEqual(Array(q[0 ..< 6]), [127, -127, 3, -3, 3, -4])
        XCTAssertTrue(q[6...].allSatisfy { $0 == 0 })
    }

    func testQ8_0ErrorBoundOnRandomData() throws {
        MLXRandom.seed(5)
        let a = MLXRandom.normal([6, 96]) * 2
        let w = try GGUFWriter()
        try w.addTensor(name: "q", shape: a.shape, type: .q8_0, source: .array({ a }))
        let url = dir.appendingPathComponent("q2.gguf")
        try w.write(to: url)
        let r = try GGUFReader(url: url)
        XCTAssertEqual(r.tensorInfo("q")?.byteCount, 6 * 3 * 34)
        let got = try r.floats("q")
        let want = a.asArray(Float.self)
        for blockStart in stride(from: 0, to: want.count, by: 32) {
            let amax = want[blockStart ..< blockStart + 32].map(abs).max()!
            let step = amax / 127
            for i in blockStart ..< blockStart + 32 {
                // half a step, plus the scale being stored in half precision
                XCTAssertEqual(got[i], want[i], accuracy: step * 0.5 + amax * 1e-3 + 1e-7)
            }
        }
    }

    func testQ8_0AllZeroBlockAndNonFinite() throws {
        var v = [Float](repeating: 0, count: 64)
        v[40] = .nan
        v[41] = 1
        let w = try GGUFWriter()
        try w.addTensor(name: "z", shape: [64], type: .q8_0, source: .array({ MLXArray(v) }))
        let url = dir.appendingPathComponent("z.gguf")
        try w.write(to: url)
        let got = try GGUFReader(url: url).floats("z")
        XCTAssertTrue(got[..<32].allSatisfy { $0 == 0 })
        XCTAssertEqual(got[40], 0, "a non-finite input quantizes to zero")
        XCTAssertEqual(got[41], 1, accuracy: 0.01)
    }

    func testDimensionsAreStoredReversedAndDataIsRowMajor() throws {
        let values = (0 ..< 24).map { Float($0) }
        let w = try GGUFWriter()
        try w.addTensor(
            name: "cube", shape: [2, 3, 4], type: .f32, source: .array({ MLXArray(values, [2, 3, 4]) }))
        let url = dir.appendingPathComponent("rev.gguf")
        try w.write(to: url)

        // Parse the tensor info by hand, independently of GGUFReader.
        let bytes = try Data(contentsOf: url)
        let nameRange = try XCTUnwrap(bytes.range(of: Data("cube".utf8)))
        var p = nameRange.upperBound
        func u32() -> UInt32 {
            defer { p += 4 }
            return bytes.subdata(in: p ..< p + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        }
        func u64() -> UInt64 {
            defer { p += 8 }
            return bytes.subdata(in: p ..< p + 8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        }
        XCTAssertEqual(u32(), 3)
        XCTAssertEqual([u64(), u64(), u64()], [4, 3, 2], "ne[0] is the contiguous (last row-major) dimension")
        XCTAssertEqual(u32(), 0, "F32")

        let r = try GGUFReader(url: url)
        XCTAssertEqual(r.tensorInfo("cube")?.ne, [4, 3, 2])
        XCTAssertEqual(r.tensorInfo("cube")?.shape, [2, 3, 4])
        XCTAssertEqual(try r.floats("cube"), values)
    }

    func testAlignmentPaddingAndOffsets() throws {
        for alignment in [32, 64, 128] {
            let w = try GGUFWriter(alignment: alignment)
            try w.set("general.architecture", .string("x"))
            // Sizes that are not multiples of the alignment.
            try w.addTensor(name: "a", shape: [3], type: .f32, source: .array({ MLXArray([1, 2, 3] as [Float]) }))
            try w.addTensor(name: "b", shape: [5], type: .f16, source: .array({ MLXArray([1, 2, 3, 4, 5] as [Float]) }))
            try w.addTensor(name: "c", shape: [2, 2], type: .f32, source: .array({ MLXArray([9, 8, 7, 6] as [Float], [2, 2]) }))
            let url = dir.appendingPathComponent("align\(alignment).gguf")
            try w.write(to: url)

            let r = try GGUFReader(url: url)
            XCTAssertEqual(r.alignment, alignment)
            XCTAssertEqual(r.dataOffset % alignment, 0)
            for t in r.tensors { XCTAssertEqual(t.offset % alignment, 0, "\(t.name) @ \(alignment)") }
            XCTAssertEqual(r.tensors.map { $0.offset }, [0, alignment, 2 * alignment])
            XCTAssertEqual(try r.floats("a"), [1, 2, 3])
            XCTAssertEqual(try r.floats("b"), [1, 2, 3, 4, 5])
            XCTAssertEqual(try r.floats("c"), [9, 8, 7, 6])
            let size = try Data(contentsOf: url).count
            XCTAssertEqual(size % alignment, 0, "file ends on an alignment boundary")
            XCTAssertEqual(size, r.dataOffset + 2 * alignment + alignment)
            // The gap between header and data, and between tensors, is zero.
            let bytes = try Data(contentsOf: url)
            let gap = bytes[(r.dataOffset + 12) ..< (r.dataOffset + alignment)]
            XCTAssertTrue(gap.allSatisfy { $0 == 0 })
            // A non-default alignment is recorded; the default is not.
            XCTAssertEqual(r["general.alignment"], alignment == 32 ? nil : .uint32(UInt32(alignment)))
        }
    }

    func testTensorsAreProducedLazilyAndInOrder() throws {
        var calls: [String] = []
        let w = try GGUFWriter()
        for name in ["one", "two", "three"] {
            try w.addTensor(
                name: name, shape: [4], type: .f32,
                source: .array({
                    calls.append(name)
                    return MLXArray([1, 2, 3, 4] as [Float])
                }))
        }
        XCTAssertEqual(calls, [], "declaring tensors reads nothing")
        let url = dir.appendingPathComponent("lazy.gguf")
        try w.write(to: url)
        XCTAssertEqual(calls, ["one", "two", "three"])
    }

    func testChunkingDoesNotChangeTheBytes() throws {
        MLXRandom.seed(11)
        // 512 columns so the K-quants (256-wide blocks) can hold a row.
        let a = MLXRandom.normal([40, 512])
        let saved = GGUFTensorEncoder.chunkElements
        defer { GGUFTensorEncoder.chunkElements = saved }

        var outputs: [Int: [String: Data]] = [:]
        for chunk in [512, 1800, 1 << 20] {
            GGUFTensorEncoder.chunkElements = chunk
            let w = try GGUFWriter()
            for t in GGUFTensorType.allCases {
                try w.addTensor(name: t.name, shape: a.shape, type: t, source: .array({ a }))
            }
            let url = dir.appendingPathComponent("chunk\(chunk).gguf")
            try w.write(to: url)
            let r = try GGUFReader(url: url)
            outputs[chunk] = Dictionary(uniqueKeysWithValues: try GGUFTensorType.allCases.map { ($0.name, try r.tensorData($0.name)) })
        }
        XCTAssertEqual(outputs[512], outputs[1 << 20])
        XCTAssertEqual(outputs[1800], outputs[1 << 20])
    }

    func testEncodedAndFileSources() throws {
        let raw = Data((0 ..< 100).map { UInt8($0) })
        let src = dir.appendingPathComponent("raw.bin")
        try raw.write(to: src)
        let w = try GGUFWriter()
        try w.addTensor(name: "enc", shape: [4], type: .f32, source: .encoded({ Data(raw[0 ..< 16]) }))
        try w.addTensor(name: "file", shape: [5], type: .f32, source: .file(src, offset: 40, count: 20))
        let url = dir.appendingPathComponent("src.gguf")
        try w.write(to: url)
        let r = try GGUFReader(url: url)
        XCTAssertEqual(try r.tensorData("enc"), Data(raw[0 ..< 16]))
        XCTAssertEqual(try r.tensorData("file"), Data(raw[40 ..< 60]))
    }

    // MARK: Failure modes

    func testRejectsBadDeclarations() throws {
        let w = try GGUFWriter()
        try w.addTensor(name: "a", shape: [32], type: .f32, source: .encoded({ Data(count: 128) }))
        XCTAssertThrowsError(try w.addTensor(name: "a", shape: [32], type: .f32, source: .encoded({ Data() })))
        XCTAssertThrowsError(try w.addTensor(name: "q", shape: [4, 33], type: .q8_0, source: .encoded({ Data() })))
        XCTAssertThrowsError(try w.addTensor(name: "z", shape: [], type: .f32, source: .encoded({ Data() })))
        XCTAssertThrowsError(try w.addTensor(name: "z", shape: [2, 0], type: .f32, source: .encoded({ Data() })))
        XCTAssertThrowsError(try w.addTensor(name: "z", shape: [1, 1, 1, 1, 1], type: .f32, source: .encoded({ Data() })))
        XCTAssertThrowsError(try GGUFWriter(alignment: 48))
        XCTAssertThrowsError(try GGUFWriter(alignment: 0))
        XCTAssertThrowsError(try w.set("bad", .array(.uint8, [.uint16(1)])))
    }

    func testFailedWriteLeavesNothingBehind() throws {
        let url = dir.appendingPathComponent("fail.gguf")
        try Data("previous contents".utf8).write(to: url)
        let w = try GGUFWriter()
        try w.addTensor(
            name: "wrong", shape: [4], type: .f32, source: .array({ MLXArray([1, 2, 3] as [Float]) }))
        XCTAssertThrowsError(try w.write(to: url))
        XCTAssertEqual(try Data(contentsOf: url), Data("previous contents".utf8), "target untouched")
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.appendingPathExtension("partial").path))

        let w2 = try GGUFWriter()
        try w2.addTensor(name: "short", shape: [4], type: .f32, source: .encoded({ Data(count: 8) }))
        XCTAssertThrowsError(try w2.write(to: url)) { error in
            XCTAssertEqual(error as? GGUFError, .sizeMismatch(name: "short", expected: 16, got: 8))
        }
    }

    func testReaderRejectsGarbage() throws {
        let url = dir.appendingPathComponent("bad.gguf")
        try Data("not a gguf file at all".utf8).write(to: url)
        XCTAssertThrowsError(try GGUFReader(url: url)) { XCTAssertEqual($0 as? GGUFError, .invalidMagic) }

        // Truncate a valid file in the middle of its tensor table.
        let w = try GGUFWriter()
        try w.set("k", .string("v"))
        try w.addTensor(name: "t", shape: [4], type: .f32, source: .encoded({ Data(count: 16) }))
        let good = dir.appendingPathComponent("good.gguf")
        try w.write(to: good)
        let bytes = try Data(contentsOf: good)
        try bytes.prefix(40).write(to: url)
        XCTAssertThrowsError(try GGUFReader(url: url))

        var v2 = bytes
        v2[4] = 2
        try v2.write(to: url)
        XCTAssertThrowsError(try GGUFReader(url: url)) { XCTAssertEqual($0 as? GGUFError, .unsupportedVersion(2)) }
    }

    // MARK: Scalar conversions

    func testHalfAndBFloatBitsMatchMLX() throws {
        var values: [Float] = [
            0, -0.0, 1, -1, 0.1, 65504, 65520, 65536, 1e-8, 6.1035156e-5, 5.9604645e-8, 2.98e-8, 2.99e-8, 3e-8,
            1.0009765625, 1.00146484375, 1.0004882812, .infinity, -.infinity, 3.4e38,
        ]
        MLXRandom.seed(21)
        values += MLXRandom.normal([2000]).asArray(Float.self).map { $0 * 100 }
        values += MLXRandom.normal([500]).asArray(Float.self).map { $0 * 1e-5 }
        let a = MLXArray(values)
        let half = a.asType(.float16).view(dtype: .uint16).asArray(UInt16.self)
        let bf = a.asType(.bfloat16).view(dtype: .uint16).asArray(UInt16.self)
        for (i, v) in values.enumerated() {
            XCTAssertEqual(GGUFFloat.float16Bits(v), half[i], "half \(v)")
            XCTAssertEqual(GGUFFloat.bfloat16Bits(v), bf[i], "bf16 \(v)")
        }
        // Decoding is the exact inverse on representable values.
        for h in [UInt16](0 ..< 0xFFFF).filter({ ($0 & 0x7C00) != 0x7C00 }) {
            XCTAssertEqual(GGUFFloat.float16Bits(GGUFFloat.float(fromFloat16Bits: h)), h)
        }
        XCTAssertTrue(GGUFFloat.float(fromFloat16Bits: 0x7E00).isNaN)
        XCTAssertEqual(GGUFFloat.float(fromFloat16Bits: 0x7C00), .infinity)
    }
}

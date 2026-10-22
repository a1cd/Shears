import Foundation

/// A tensor entry from a GGUF file.
public struct GGUFTensorInfo: Equatable, Sendable {
    public var name: String
    /// Row-major shape (the reverse of the on-disk `ne` order).
    public var shape: [Int]
    public var type: GGUFTensorType
    /// Byte offset from the start of the data section.
    public var offset: Int

    /// The on-disk dimension order: `ne[0]` is the contiguous dimension.
    public var ne: [Int] { shape.reversed() }
    public var elementCount: Int { shape.reduce(1, *) }
    public var byteCount: Int { type.byteCount(shape: shape) }
}

/// A minimal GGUF v3 reader: header, metadata, tensor infos, and tensor bytes.
///
/// It exists to check what ``GGUFWriter`` produces, and to inspect exports. The file is
/// memory-mapped, so opening a large file does not load it.
public final class GGUFReader {
    public let url: URL
    public private(set) var version: UInt32 = 0
    public private(set) var metadata: [(key: String, value: GGUFValue)] = []
    public private(set) var tensors: [GGUFTensorInfo] = []
    public private(set) var alignment: Int = GGUFWriter.defaultAlignment
    /// Absolute file offset of the data section.
    public private(set) var dataOffset: Int = 0

    private let file: Data
    private var cursor = 0

    public init(url: URL) throws {
        self.url = url
        self.file = try Data(contentsOf: url, options: .alwaysMapped)
        guard file.count >= 4, file.prefix(4) == Data([0x47, 0x47, 0x55, 0x46]) else {
            throw GGUFError.invalidMagic
        }
        cursor = 4
        version = try readU32("version")
        guard version == 3 else { throw GGUFError.unsupportedVersion(version) }
        let tensorCount = Int(try readU64("tensor count"))
        let kvCount = Int(try readU64("metadata count"))
        var seen = Set<String>()
        for _ in 0 ..< kvCount {
            let key = try readString("metadata key")
            guard seen.insert(key).inserted else { throw GGUFError.duplicateKey(key) }
            let rawType = try readU32("metadata type")
            guard let type = GGUFValueType(rawValue: rawType) else {
                throw GGUFError.other("metadata \(key): unknown value type \(rawType)")
            }
            metadata.append((key, try readValue(type, "metadata \(key)")))
        }
        if let a = metadata.first(where: { $0.key == "general.alignment" })?.value.intValue {
            guard a > 0, a & (a - 1) == 0 else { throw GGUFError.invalidAlignment(a) }
            alignment = a
        }
        for _ in 0 ..< tensorCount {
            let name = try readString("tensor name")
            let nDims = Int(try readU32("tensor dims"))
            guard (1 ... GGUFWriter.maxDimensions).contains(nDims) else {
                throw GGUFError.invalidTensor("\(name) has \(nDims) dimensions")
            }
            var ne: [Int] = []
            for _ in 0 ..< nDims { ne.append(Int(try readU64("tensor dims"))) }
            let rawType = try readU32("tensor type")
            guard let type = GGUFTensorType(rawValue: rawType) else {
                throw GGUFError.unsupportedTensorType(rawType)
            }
            let offset = Int(try readU64("tensor offset"))
            tensors.append(GGUFTensorInfo(name: name, shape: ne.reversed(), type: type, offset: offset))
        }
        dataOffset = GGUFWriter.align(cursor, alignment)
    }

    public subscript(key: String) -> GGUFValue? {
        metadata.first(where: { $0.key == key })?.value
    }

    public func tensorInfo(_ name: String) -> GGUFTensorInfo? {
        tensors.first(where: { $0.name == name })
    }

    /// The tensor's raw bytes as stored.
    public func tensorData(_ name: String) throws -> Data {
        guard let info = tensorInfo(name) else { throw GGUFError.unknownTensor(name) }
        let start = dataOffset + info.offset
        guard start + info.byteCount <= file.count else { throw GGUFError.truncated("tensor \(name)") }
        return file.subdata(in: start ..< start + info.byteCount)
    }

    /// The tensor decoded to float32, row-major.
    public func floats(_ name: String) throws -> [Float] {
        guard let info = tensorInfo(name) else { throw GGUFError.unknownTensor(name) }
        let bytes = try tensorData(name)
        switch info.type {
        case .f32:
            return bytes.withUnsafeBytes { raw in
                (0 ..< info.elementCount).map {
                    Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self).littleEndian)
                }
            }
        case .f16:
            return bytes.withUnsafeBytes { raw in
                (0 ..< info.elementCount).map {
                    GGUFFloat.float(
                        fromFloat16Bits: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self).littleEndian)
                }
            }
        case .bf16:
            return bytes.withUnsafeBytes { raw in
                (0 ..< info.elementCount).map {
                    GGUFFloat.float(
                        fromBFloat16Bits: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self).littleEndian)
                }
            }
        case .q8_0:
            return GGUFQ8_0.dequantize(bytes)
        case .q5_0: return GGUFQ5_0.dequantize(bytes)
        case .q4_k: return GGUFQ4K.dequantize(bytes)
        case .q5_k: return GGUFQ5K.dequantize(bytes)
        case .q6_k: return GGUFQ6K.dequantize(bytes)
        }
    }

    // MARK: Parsing

    private func need(_ n: Int, _ what: String) throws {
        guard n >= 0, cursor + n <= file.count else { throw GGUFError.truncated(what) }
    }

    private func readU8(_ what: String) throws -> UInt8 {
        try need(1, what)
        defer { cursor += 1 }
        return file[file.startIndex + cursor]
    }

    private func readU16(_ what: String) throws -> UInt16 {
        try need(2, what)
        defer { cursor += 2 }
        return file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt16.self).littleEndian }
    }

    private func readU32(_ what: String) throws -> UInt32 {
        try need(4, what)
        defer { cursor += 4 }
        return file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self).littleEndian }
    }

    private func readU64(_ what: String) throws -> UInt64 {
        try need(8, what)
        defer { cursor += 8 }
        return file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt64.self).littleEndian }
    }

    private func readString(_ what: String) throws -> String {
        let n = Int(try readU64(what))
        try need(n, what)
        defer { cursor += n }
        let start = file.startIndex + cursor
        return String(decoding: file[start ..< start + n], as: UTF8.self)
    }

    private func readValue(_ type: GGUFValueType, _ what: String) throws -> GGUFValue {
        switch type {
        case .uint8: return .uint8(try readU8(what))
        case .int8: return .int8(Int8(bitPattern: try readU8(what)))
        case .uint16: return .uint16(try readU16(what))
        case .int16: return .int16(Int16(bitPattern: try readU16(what)))
        case .uint32: return .uint32(try readU32(what))
        case .int32: return .int32(Int32(bitPattern: try readU32(what)))
        case .uint64: return .uint64(try readU64(what))
        case .int64: return .int64(Int64(bitPattern: try readU64(what)))
        case .float32: return .float32(Float(bitPattern: try readU32(what)))
        case .float64: return .float64(Double(bitPattern: try readU64(what)))
        case .bool: return .bool(try readU8(what) != 0)
        case .string: return .string(try readString(what))
        case .array:
            let rawElement = try readU32(what)
            guard let element = GGUFValueType(rawValue: rawElement) else {
                throw GGUFError.other("\(what): unknown array element type \(rawElement)")
            }
            let count = Int(try readU64(what))
            var items: [GGUFValue] = []
            items.reserveCapacity(count)
            for _ in 0 ..< count { items.append(try readValue(element, what)) }
            return .array(element, items)
        }
    }
}

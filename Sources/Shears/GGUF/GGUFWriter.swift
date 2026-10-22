import Foundation
import MLX

/// Little-endian byte assembly for the GGUF header.
struct GGUFByteWriter {
    var data = Data()

    mutating func u8(_ v: UInt8) { data.append(v) }
    mutating func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }

    mutating func string(_ s: String) {
        let bytes = Array(s.utf8)
        u64(UInt64(bytes.count))
        data.append(contentsOf: bytes)
    }

    /// The value's payload, without its type tag.
    mutating func payload(_ v: GGUFValue) {
        switch v {
        case .uint8(let x): u8(x)
        case .int8(let x): u8(UInt8(bitPattern: x))
        case .uint16(let x): u16(x)
        case .int16(let x): u16(UInt16(bitPattern: x))
        case .uint32(let x): u32(x)
        case .int32(let x): u32(UInt32(bitPattern: x))
        case .uint64(let x): u64(x)
        case .int64(let x): u64(UInt64(bitPattern: x))
        case .float32(let x): u32(x.bitPattern)
        case .float64(let x): u64(x.bitPattern)
        case .bool(let x): u8(x ? 1 : 0)
        case .string(let s): string(s)
        case .array(let elementType, let items):
            u32(elementType.rawValue)
            u64(UInt64(items.count))
            for item in items { payload(item) }
        }
    }
}

/// Writes a GGUF v3 file.
///
/// Metadata and tensor descriptions are declared first; tensor data is produced lazily, one
/// tensor at a time, when ``write(to:)`` runs. Because a tensor's byte size follows from its
/// shape and type, every offset is known before any data exists, so the header can be written
/// up front and the data streamed behind it. Peak memory is one chunk of one tensor, not the
/// model.
///
/// Shapes are given in row-major order, as MLX and safetensors report them. GGUF stores the
/// dimensions reversed (`ne[0]` is the contiguous one); the writer does that reversal.
public final class GGUFWriter {
    /// Where a tensor's bytes come from.
    public enum TensorSource {
        /// An MLX array, converted to the declared type and written in slices along axis 0.
        case array(() throws -> MLXArray)
        /// Like `array`, with an importance matrix slice (one weight per column, per expert for a
        /// 3-D tensor) for the quantizers that use one.
        case weightedArray(() throws -> MLXArray, importance: [Float])
        /// Bytes already encoded in the declared type.
        case encoded(() throws -> Data)
        /// A byte range of a file already encoded in the declared type, copied in blocks.
        case file(URL, offset: UInt64, count: Int)
    }

    struct Entry {
        var name: String
        var shape: [Int]
        var type: GGUFTensorType
        var source: TensorSource
        var offset: Int
        var byteCount: Int
    }

    public static let defaultAlignment = 32
    public static let maxDimensions = 4

    public let alignment: Int
    private(set) var metadata: [(key: String, value: GGUFValue)] = []
    private(set) var entries: [Entry] = []
    private var nextOffset = 0
    private var tensorNames = Set<String>()

    /// - Parameter alignment: tensor data alignment; a power of two. Anything other than 32 is
    ///   also recorded as `general.alignment`.
    public init(alignment: Int = GGUFWriter.defaultAlignment) throws {
        guard alignment > 0, alignment & (alignment - 1) == 0 else {
            throw GGUFError.invalidAlignment(alignment)
        }
        self.alignment = alignment
    }

    public var tensorCount: Int { entries.count }

    /// Sets a metadata entry. Setting an existing key replaces its value and keeps its position.
    public func set(_ key: String, _ value: GGUFValue) throws {
        try Self.validate(value, key: key)
        if let i = metadata.firstIndex(where: { $0.key == key }) {
            metadata[i].value = value
        } else {
            metadata.append((key, value))
        }
    }

    public func metadataValue(_ key: String) -> GGUFValue? {
        metadata.first(where: { $0.key == key })?.value
    }

    static func validate(_ value: GGUFValue, key: String) throws {
        if case .array(let elementType, let items) = value {
            for item in items {
                guard item.type == elementType else {
                    throw GGUFError.other(
                        "metadata \(key): array declared \(elementType) holds a \(item.type) element")
                }
                try validate(item, key: key)
            }
        }
    }

    /// Declares a tensor. `shape` is row-major. Offsets are assigned in declaration order.
    public func addTensor(
        name: String, shape: [Int], type: GGUFTensorType, source: TensorSource
    ) throws {
        guard !tensorNames.contains(name) else { throw GGUFError.duplicateTensor(name) }
        guard (1 ... Self.maxDimensions).contains(shape.count), shape.allSatisfy({ $0 > 0 }) else {
            throw GGUFError.invalidTensor("\(name) has shape \(shape); need 1-4 positive dimensions")
        }
        guard shape[shape.count - 1] % type.blockSize == 0 else {
            throw GGUFError.invalidTensor(
                "\(name): row length \(shape[shape.count - 1]) is not a multiple of the \(type.name) block size \(type.blockSize)"
            )
        }
        let bytes = type.byteCount(shape: shape)
        let offset = Self.align(nextOffset, alignment)
        entries.append(
            Entry(name: name, shape: shape, type: type, source: source, offset: offset, byteCount: bytes))
        tensorNames.insert(name)
        nextOffset = offset + bytes
    }

    static func align(_ x: Int, _ a: Int) -> Int { (x + a - 1) / a * a }

    func headerBytes() throws -> Data {
        var kv = metadata
        if alignment != Self.defaultAlignment {
            let a = GGUFValue.uint32(UInt32(alignment))
            if let existing = kv.first(where: { $0.key == "general.alignment" })?.value, existing != a {
                throw GGUFError.other("general.alignment is set to a value other than the writer's alignment")
            }
            if !kv.contains(where: { $0.key == "general.alignment" }) {
                kv.append(("general.alignment", a))
            }
        }
        var w = GGUFByteWriter()
        w.data.append(contentsOf: [0x47, 0x47, 0x55, 0x46])  // "GGUF"
        w.u32(3)
        w.u64(UInt64(entries.count))
        w.u64(UInt64(kv.count))
        for (key, value) in kv {
            w.string(key)
            w.u32(value.type.rawValue)
            w.payload(value)
        }
        for e in entries {
            w.string(e.name)
            w.u32(UInt32(e.shape.count))
            for d in e.shape.reversed() { w.u64(UInt64(d)) }
            w.u32(e.type.rawValue)
            w.u64(UInt64(e.offset))
        }
        return w.data
    }

    /// Writes the file. The data goes to `<url>.partial` first and is moved into place on
    /// success, so a failed export never leaves a file that looks complete.
    public func write(to url: URL) throws {
        let fm = FileManager.default
        let partial = url.appendingPathExtension("partial")
        try? fm.removeItem(at: partial)
        guard fm.createFile(atPath: partial.path, contents: nil) else {
            throw GGUFError.other("cannot create \(partial.path)")
        }
        let handle = try FileHandle(forWritingTo: partial)
        var closed = false
        defer {
            if !closed { try? handle.close() }
        }
        do {
            let header = try headerBytes()
            try handle.write(contentsOf: header)
            var position = header.count
            let dataStart = Self.align(position, alignment)
            try Self.pad(handle, from: position, to: dataStart)
            position = dataStart

            for e in entries {
                let target = dataStart + e.offset
                try Self.pad(handle, from: position, to: target)
                position = target
                var written = 0
                try autoreleasepool {
                    try stream(e) { chunk in
                        try handle.write(contentsOf: chunk)
                        written += chunk.count
                    }
                }
                guard written == e.byteCount else {
                    throw GGUFError.sizeMismatch(name: e.name, expected: e.byteCount, got: written)
                }
                position += written
                Memory.clearCache()
            }
            // Like the reference writer, pad after the last tensor too.
            try Self.pad(handle, from: position, to: Self.align(position, alignment))
            try handle.close()
            closed = true
            try? fm.removeItem(at: url)
            try fm.moveItem(at: partial, to: url)
        } catch {
            try? handle.close()
            closed = true
            try? fm.removeItem(at: partial)
            throw error
        }
    }

    private static func pad(_ handle: FileHandle, from: Int, to: Int) throws {
        if to > from { try handle.write(contentsOf: Data(count: to - from)) }
    }

    private func stream(_ e: Entry, sink: (Data) throws -> Void) throws {
        switch e.source {
        case .array(let produce):
            let array = try produce()
            try GGUFTensorEncoder.encode(array, name: e.name, shape: e.shape, as: e.type, sink: sink)
        case .weightedArray(let produce, let importance):
            let array = try produce()
            try GGUFTensorEncoder.encode(array, name: e.name, shape: e.shape, as: e.type, importance: importance, sink: sink)
        case .encoded(let produce):
            try sink(try produce())
        case .file(let url, let offset, let count):
            let reader = try FileHandle(forReadingFrom: url)
            defer { try? reader.close() }
            try reader.seek(toOffset: offset)
            var remaining = count
            while remaining > 0 {
                let n = min(remaining, 8 << 20)
                guard let block = try reader.read(upToCount: n), block.count == n else {
                    throw GGUFError.truncated("tensor \(e.name) in \(url.lastPathComponent)")
                }
                try sink(block)
                remaining -= n
            }
        }
    }
}

/// Converts MLX arrays to GGUF tensor bytes in slices along axis 0.
public enum GGUFTensorEncoder {
    /// Elements converted at a time. A float32 slice of this size is 64 MB.
    public static var chunkElements = 16 << 20

    public static func encode(
        _ array: MLXArray, name: String, shape: [Int], as type: GGUFTensorType,
        importance: [Float]? = nil, sink: (Data) throws -> Void
    ) throws {
        guard array.shape == shape else {
            throw GGUFError.shapeMismatch(name: name, expected: shape, got: array.shape)
        }
        // A 3-D tensor is a stack of expert matrices, each with its own slice of the importance data.
        if shape.count == 3 {
            let perExpert = shape[2]
            for e in 0 ..< shape[0] {
                let slice = importance.map { Array($0[e * perExpert ..< (e + 1) * perExpert]) }
                try encode(array[e], name: name, shape: Array(shape.dropFirst()), as: type, importance: slice, sink: sink)
            }
            return
        }
        let rows = shape[0]
        let perRow = shape.dropFirst().reduce(1, *)
        let rowsPerChunk = max(1, chunkElements / perRow)
        var start = 0
        while start < rows {
            let stop = min(rows, start + rowsPerChunk)
            let slice = (start == 0 && stop == rows) ? array : array[start ..< stop]
            try sink(try encodeChunk(slice, as: type, importance: importance))
            start = stop
            Memory.clearCache()
        }
    }

    /// Encodes one array (or slice) as `type`, row-major.
    public static func encodeChunk(_ chunk: MLXArray, as type: GGUFTensorType, importance: [Float]? = nil) throws -> Data {
        switch type {
        case .f32: return chunk.asType(.float32).asData(access: .copy).data
        case .f16: return chunk.asType(.float16).asData(access: .copy).data
        case .bf16: return chunk.asType(.bfloat16).asData(access: .copy).data
        case .q8_0:
            let floats = chunk.asType(.float32).asArray(Float.self)
            return floats.withUnsafeBufferPointer { GGUFQ8_0.quantize($0) }
        case .q5_0, .q4_k, .q5_k, .q6_k:
            let floats = chunk.asType(.float32).asArray(Float.self)
            let perRow = chunk.shape.last!
            func run(_ im: UnsafeBufferPointer<Float>?) -> Data {
                floats.withUnsafeBufferPointer { v in
                    switch type {
                    case .q5_0: GGUFQ5_0.quantize(v, perRow: perRow, importance: im)
                    case .q4_k: GGUFQ4K.quantize(v, perRow: perRow, importance: im)
                    case .q5_k: GGUFQ5K.quantize(v, perRow: perRow, importance: im)
                    default: GGUFQ6K.quantize(v, perRow: perRow, importance: im)
                    }
                }
            }
            if let importance {
                precondition(importance.count == perRow, "one importance weight per column")
                return importance.withUnsafeBufferPointer { run($0) }
            }
            return run(nil)
        }
    }
}

import Foundation

/// Value type tags of a GGUF metadata entry (GGUF v3).
public enum GGUFValueType: UInt32, Sendable {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12
}

/// A typed GGUF metadata value. Arrays carry their element type explicitly so that an empty
/// array still has a well-defined type on disk.
public indirect enum GGUFValue: Equatable, Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case uint64(UInt64)
    case int64(Int64)
    case float32(Float)
    case float64(Double)
    case bool(Bool)
    case string(String)
    case array(GGUFValueType, [GGUFValue])

    public var type: GGUFValueType {
        switch self {
        case .uint8: .uint8
        case .int8: .int8
        case .uint16: .uint16
        case .int16: .int16
        case .uint32: .uint32
        case .int32: .int32
        case .uint64: .uint64
        case .int64: .int64
        case .float32: .float32
        case .float64: .float64
        case .bool: .bool
        case .string: .string
        case .array: .array
        }
    }

    // Convenience constructors for the array shapes metadata actually uses.
    public static func strings(_ v: [String]) -> GGUFValue { .array(.string, v.map { .string($0) }) }
    public static func uint32s(_ v: [Int]) -> GGUFValue { .array(.uint32, v.map { .uint32(UInt32($0)) }) }
    public static func int32s(_ v: [Int]) -> GGUFValue { .array(.int32, v.map { .int32(Int32($0)) }) }
    public static func float32s(_ v: [Float]) -> GGUFValue { .array(.float32, v.map { .float32($0) }) }
    public static func bools(_ v: [Bool]) -> GGUFValue { .array(.bool, v.map { .bool($0) }) }

    /// The value as an integer, for any integer case.
    public var intValue: Int? {
        switch self {
        case .uint8(let v): Int(v)
        case .int8(let v): Int(v)
        case .uint16(let v): Int(v)
        case .int16(let v): Int(v)
        case .uint32(let v): Int(v)
        case .int32(let v): Int(v)
        case .uint64(let v): Int(exactly: v)
        case .int64(let v): Int(v)
        default: nil
        }
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    public var floatValue: Double? {
        switch self {
        case .float32(let v): Double(v)
        case .float64(let v): v
        default: nil
        }
    }

    public var arrayValue: [GGUFValue]? {
        if case .array(_, let a) = self { return a }
        return nil
    }
}

/// Tensor storage types the writer and reader handle. Raw values are the `ggml_type` ids.
public enum GGUFTensorType: UInt32, Sendable, CaseIterable {
    case f32 = 0
    case f16 = 1
    case q5_0 = 6
    case q8_0 = 8
    case q4_k = 12
    case q5_k = 13
    case q6_k = 14
    case bf16 = 30

    /// Elements per quantization block (1 for plain float types).
    public var blockSize: Int {
        switch self {
        case .q8_0, .q5_0: 32
        case .q4_k, .q5_k, .q6_k: 256
        default: 1
        }
    }

    /// Bytes per block.
    public var typeSize: Int {
        switch self {
        case .f32: 4
        case .f16, .bf16: 2
        case .q8_0: 2 + 32
        case .q5_0: 22
        case .q4_k: 144
        case .q5_k: 176
        case .q6_k: 210
        }
    }

    /// Bytes in one row of `rowLength` elements (`ne[0]`).
    public func rowSize(_ rowLength: Int) -> Int { rowLength / blockSize * typeSize }

    /// Total bytes for a tensor of the given row-major shape.
    public func byteCount(shape: [Int]) -> Int {
        guard let last = shape.last else { return typeSize }  // 0-d: a single element
        let rows = shape.dropLast().reduce(1, *)
        return rows * rowSize(last)
    }

    public var name: String {
        switch self {
        case .f32: "F32"
        case .f16: "F16"
        case .q8_0: "Q8_0"
        case .q5_0: "Q5_0"
        case .q4_k: "Q4_K"
        case .q5_k: "Q5_K"
        case .q6_k: "Q6_K"
        case .bf16: "BF16"
        }
    }
}

/// `general.file_type` values (`LlamaFileType`) for the tensor types this package writes.
public enum GGUFFileType {
    public static func value(for type: GGUFTensorType) -> UInt32 {
        switch type {
        case .f32: 0
        case .f16: 1
        case .q8_0: 7
        case .q5_0: 8
        // Q4_K is only produced as part of the Q4_K_M mix, whose value this is.
        case .q4_k: 15
        case .q5_k: 17
        case .q6_k: 18
        case .bf16: 32
        }
    }
}

/// `tokenizer.ggml.token_type` values.
public enum GGUFTokenType: Int32, Sendable {
    case normal = 1
    case unknown = 2
    case control = 3
    case userDefined = 4
    case unused = 5
    case byte = 6
}

public enum GGUFError: Error, CustomStringConvertible, Equatable {
    case invalidMagic
    case unsupportedVersion(UInt32)
    case truncated(String)
    case duplicateKey(String)
    case duplicateTensor(String)
    case invalidAlignment(Int)
    case invalidTensor(String)
    case unsupportedTensorType(UInt32)
    case shapeMismatch(name: String, expected: [Int], got: [Int])
    case sizeMismatch(name: String, expected: Int, got: Int)
    case unknownTensor(String)
    case other(String)

    public var description: String {
        switch self {
        case .invalidMagic: "not a GGUF file (bad magic)"
        case .unsupportedVersion(let v): "unsupported GGUF version \(v)"
        case .truncated(let s): "file ends early while reading \(s)"
        case .duplicateKey(let k): "duplicate metadata key \(k)"
        case .duplicateTensor(let n): "duplicate tensor \(n)"
        case .invalidAlignment(let a): "alignment must be a power of two, got \(a)"
        case .invalidTensor(let s): "invalid tensor: \(s)"
        case .unsupportedTensorType(let t): "unsupported ggml tensor type id \(t)"
        case .shapeMismatch(let n, let e, let g): "tensor \(n): expected shape \(e), got \(g)"
        case .sizeMismatch(let n, let e, let g): "tensor \(n): expected \(e) bytes, produced \(g)"
        case .unknownTensor(let n): "no tensor named \(n)"
        case .other(let s): s
        }
    }
}

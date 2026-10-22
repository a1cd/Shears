import Foundation
import MLX

/// One tensor in a safetensors file, located without reading its data.
public struct GGUFSafetensorsEntry: Sendable {
    public var name: String
    public var dtype: String
    /// Row-major shape.
    public var shape: [Int]
    public var file: URL
    /// Absolute byte offset of the tensor data within `file`.
    public var offset: UInt64
    public var byteCount: Int

    public var elementCount: Int { shape.reduce(1, *) }

    /// The MLX dtype for the safetensors dtype string, for the float types a checkpoint uses.
    public var mlxDType: DType? {
        switch dtype {
        case "F32": .float32
        case "F16": .float16
        case "BF16": .bfloat16
        default: nil
        }
    }

    /// The GGUF tensor type with identical byte layout, if there is one.
    public var nativeGGUFType: GGUFTensorType? {
        switch dtype {
        case "F32": .f32
        case "F16": .f16
        case "BF16": .bf16
        default: nil
        }
    }

    /// Reads the tensor into an MLX array. With `firstRows`, reads only that many leading rows
    /// (slices of axis 0, which are contiguous in row-major storage).
    public func load(firstRows: Int? = nil) throws -> MLXArray {
        guard let dt = mlxDType else {
            throw GGUFError.other("tensor \(name): unsupported safetensors dtype \(dtype)")
        }
        var shape = self.shape
        var count = byteCount
        if let rows = firstRows, let first = shape.first {
            guard rows >= 0, rows <= first else {
                throw GGUFError.other("tensor \(name): asked for \(rows) of \(first) rows")
            }
            count = byteCount / max(first, 1) * rows
            shape[0] = rows
        }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        try handle.seek(toOffset: offset)
        guard let data = try handle.read(upToCount: count), data.count == count else {
            throw GGUFError.truncated("tensor \(name) in \(file.lastPathComponent)")
        }
        return MLXArray(data, shape, dtype: dt)
    }
}

/// The tensors of a safetensors checkpoint (a single file, or shards listed by an index),
/// read from the JSON headers only.
public struct GGUFSafetensorsIndex: Sendable {
    public private(set) var entries: [String: GGUFSafetensorsEntry] = [:]
    public private(set) var files: [URL] = []

    /// Opens `directory`: `model.safetensors.index.json` when present, else every `*.safetensors`.
    public init(directory: URL) throws {
        let fm = FileManager.default
        var shardNames: [String]
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path) {
            let json = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
            guard let map = json?["weight_map"] as? [String: String] else {
                throw GGUFError.other("model.safetensors.index.json has no weight_map")
            }
            shardNames = Array(Set(map.values)).sorted()
        } else {
            shardNames = try fm.contentsOfDirectory(atPath: directory.path)
                .filter { $0.hasSuffix(".safetensors") }.sorted()
        }
        guard !shardNames.isEmpty else {
            throw GGUFError.other("no safetensors files in \(directory.path)")
        }
        for shard in shardNames {
            let url = directory.appendingPathComponent(shard)
            files.append(url)
            for entry in try Self.parseHeader(url) {
                if entries[entry.name] != nil {
                    throw GGUFError.other("tensor \(entry.name) appears in more than one shard")
                }
                entries[entry.name] = entry
            }
        }
    }

    static func parseHeader(_ url: URL) throws -> [GGUFSafetensorsEntry] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lenBytes = try handle.read(upToCount: 8), lenBytes.count == 8 else {
            throw GGUFError.truncated("safetensors header of \(url.lastPathComponent)")
        }
        let headerLength = lenBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard headerLength < (1 << 30),
            let headerBytes = try handle.read(upToCount: Int(headerLength)),
            headerBytes.count == Int(headerLength)
        else { throw GGUFError.truncated("safetensors header of \(url.lastPathComponent)") }
        guard let json = try JSONSerialization.jsonObject(with: headerBytes) as? [String: Any] else {
            throw GGUFError.other("\(url.lastPathComponent): header is not a JSON object")
        }
        let dataStart = UInt64(8) + headerLength
        var out: [GGUFSafetensorsEntry] = []
        for (name, raw) in json where name != "__metadata__" {
            guard let info = raw as? [String: Any],
                let dtype = info["dtype"] as? String,
                let shape = (info["shape"] as? [NSNumber])?.map({ $0.intValue }),
                let offsets = (info["data_offsets"] as? [NSNumber])?.map({ $0.uint64Value }),
                offsets.count == 2, offsets[1] >= offsets[0]
            else { throw GGUFError.other("\(url.lastPathComponent): bad header entry for \(name)") }
            out.append(
                GGUFSafetensorsEntry(
                    name: name, dtype: dtype, shape: shape, file: url,
                    offset: dataStart + offsets[0], byteCount: Int(offsets[1] - offsets[0])))
        }
        return out
    }
}

import Foundation
import MLX

/// What a healed-model export promises about how it must be shipped, written next to its
/// safetensors as `shears-export.json` and checked by ``GGUFExport``.
///
/// A model healed against a weight format (``QATConfiguration``) is only the model that ships if
/// the converter quantizes the exported tensors with that same format, from the same dtype, and
/// without anything of its own on top (an importance matrix changes the codes a K-quant picks;
/// training never used one). The record lets the converter refuse a mismatch instead of silently
/// writing a different model.
struct ExportRecord: Codable, Equatable {
    static let fileName = "shears-export.json"

    var version = 1
    /// Safetensors dtype name of the exported floating-point weights (`BF16`, `F16`, `F32`).
    var dtype: String
    /// Rounding formats by GGUF tensor name, for the tensors healing rounded. Empty for a model
    /// healed without QAT.
    var formats: [String: String]

    init(dtype: DType, formats: [String: QuantFormat]) {
        self.dtype = Self.safetensorsName(dtype)
        self.formats = formats.mapValues { $0.recordName }
    }

    static func safetensorsName(_ d: DType) -> String {
        switch d {
        case .bfloat16: "BF16"
        case .float16: "F16"
        default: "F32"
        }
    }

    /// The covered tensors of `adapters`' layers under their GGUF names.
    static func formats(of adapters: ModelAdapters) -> [String: QuantFormat] {
        var out: [String: QuantFormat] = [:]
        for (i, la) in adapters.layers.enumerated() { out.merge(formats(of: la, layerIndex: adapters.firstLayer + i)) { $1 } }
        return out
    }

    /// The covered tensors of one layer's adapters (model layer `layerIndex`) under their GGUF names.
    static func formats(of la: LayerAdapters, layerIndex: Int) -> [String: QuantFormat] {
        var out: [String: QuantFormat] = [:]
        let p = "blk.\(layerIndex)."
        for (site, f) in la.denseFormats { out[p + site.ggufName + ".weight"] = f }
        if let e = la.experts {
            if let f = e.gateUpFormat { out[p + "ffn_gate_up_exps.weight"] = f }
            if let f = e.downFormat { out[p + "ffn_down_exps.weight"] = f }
        }
        return out
    }

    func write(to directory: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .prettyPrinted]
        try enc.encode(self).write(to: directory.appendingPathComponent(Self.fileName))
    }

    /// The record in `directory`, or nil if it holds none (a checkpoint not written by
    /// ``MergeAndSliceExport``). A record that cannot be read is an error, not "none".
    static func read(from directory: URL) throws -> ExportRecord? {
        let url = directory.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(ExportRecord.self, from: Data(contentsOf: url))
    }
}

extension QuantFormat {
    /// Stable name used in ``ExportRecord``: a GGUF tensor type name, or `affine:<bits>:<group>`.
    var recordName: String {
        switch self {
        case .gguf(let t): t.name
        case .q8_0: GGUFTensorType.q8_0.name
        case .affine(let bits, let group): "affine:\(bits):\(group)"
        }
    }
}

import Foundation

/// Which type each tensor gets under `llama-quantize ... Q4_K_M`, reproduced from
/// `llama_tensor_get_type` and `tensor_type_fallback` in `src/llama-quant.cpp` (llama.cpp commit
/// a3a1c47) for the `gemma4` architecture.
///
/// The mix depends on the tensors it is given: how many `attn_v` tensors there are and where a
/// tensor falls among them, the layer number in its name, whether the model has exactly eight
/// experts, whether the embeddings are tied to the output, and whether a row length is a
/// multiple of 256 (K-quants fall back to `Q5_0` or `Q8_0` otherwise).
public enum GGUFQuantMix {
    /// Types for every tensor in `tensors` (name and row-major shape; the last dimension is the
    /// row length `ne[0]`).
    ///
    /// - Parameters:
    ///   - layerCount: `n_layer` of the model.
    ///   - expertCount: experts per layer, `nil` for a dense model.
    ///   - half: the 16-bit type tensors that are not quantized keep (what the converter wrote).
    public static func q4KM(
        _ tensors: [(name: String, shape: [Int])], layerCount: Int, expertCount: Int?,
        half: GGUFTensorType
    ) -> [String: GGUFTensorType] {
        let nExpert = max(1, expertCount ?? 0)
        func category(_ n: String) -> String {
            if n == "output.weight" { return "output" }
            if n == "token_embd.weight" || n == "per_layer_token_embd.weight" { return "token_embd" }
            if n.contains("attn_qkv.weight") { return "qkv" }
            if n.contains("attn_kv_b.weight") { return "kv_b" }
            if n.contains("attn_v.weight") { return "v" }
            if n.contains("attn_k.weight") { return "k" }
            if n.contains("attn_q.weight") { return "q" }
            if n.contains("attn_output.weight") { return "attn_output" }
            if n.contains("ffn_up") { return "ffn_up" }
            if n.contains("ffn_gate") { return "ffn_gate" }
            if n.contains("ffn_down") { return "ffn_down" }
            return "other"
        }
        func isAttnV(_ c: String) -> Bool { c == "v" || c == "qkv" || c == "kv_b" }
        func allows(_ n: String, _ shape: [Int]) -> Bool {
            guard shape.count >= 2, n.hasSuffix("weight") else { return false }
            for banned in ["_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "altup", "laurel", "per_layer_model_proj"]
            where n.contains(banned) { return false }
            return true
        }
        func useMoreBits(_ i: Int, _ n: Int) -> Bool { i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2 }

        // The running counters follow the tensors' order in the file: by layer, then name.
        let ordered = tensors.sorted {
            let a = Self.layer(of: $0.name) ?? -1, b = Self.layer(of: $1.name) ?? -1
            return a != b ? a < b : $0.name < $1.name
        }
        let hasOutput = ordered.contains { $0.name == "output.weight" }
        let nAttentionV = ordered.filter { isAttnV(category($0.name)) }.count
        var iAttentionV = 0, iFfnDown = 0

        var out: [String: GGUFTensorType] = [:]
        for (name, shape) in ordered {
            guard allows(name, shape) else {
                out[name] = GGUFGemma4Precision.type(ggufName: name, shape: shape, requested: half)
                continue
            }
            let cat = category(name)
            var type = GGUFTensorType.q4_k
            let nx = shape[shape.count - 1]
            if cat == "output" || (!hasOutput && cat == "token_embd") {
                type = nx % 256 != 0 ? .q8_0 : .q6_k
            } else if isAttnV(cat) {
                if useMoreBits(iAttentionV, nAttentionV) { type = .q6_k }
                if expertCount == 8 { type = .q8_0 }
                iAttentionV += 1
            } else if cat == "k" {
                if expertCount == 8 { type = .q8_0 }
            } else if cat == "ffn_down" {
                var layer = iFfnDown
                if nExpert > 1, let l = Self.layer(of: name) { layer = l }
                if useMoreBits(layer, layerCount) { type = .q6_k }
                iFfnDown += 1
            } else if cat == "attn_output" {
                if expertCount == 8 { type = .q5_k }
            }
            // tensor_type_fallback: a row that is not a whole number of super-blocks.
            if nx % type.blockSize != 0 {
                switch type {
                case .q4_k: type = .q5_0
                case .q6_k: type = .q8_0
                default: preconditionFailure("\(name): Q5_K falls back to Q5_1, which this package does not write")
                }
                if nx % type.blockSize != 0 { type = .f16 }
            }
            out[name] = type
        }
        return out
    }

    static func layer(of name: String) -> Int? {
        guard name.hasPrefix("blk.") else { return nil }
        return Int(name.dropFirst(4).prefix { $0 != "." })
    }
}

/// A llama.cpp importance matrix in the legacy binary layout (`llama-imatrix`'s `.dat`).
///
/// Per tensor, one weight for each column (and, for the stacked expert tensors, each expert),
/// already divided by the call count the way `llama-quantize` does on load.
public struct GGUFImatrix: Sendable {
    public var entries: [String: [Float]]
    public init(entries: [String: [Float]]) { self.entries = entries }

    public static func load(_ url: URL) throws -> GGUFImatrix {
        let data = try Data(contentsOf: url)
        var cursor = 0
        func i32() throws -> Int {
            guard cursor + 4 <= data.count else { throw GGUFError.truncated("importance matrix") }
            defer { cursor += 4 }
            return Int(data.withUnsafeBytes { Int32(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: Int32.self)) })
        }
        let n = try i32()
        guard n > 0 && n < 1_000_000 else { throw GGUFError.other("not a legacy importance matrix file") }
        var entries: [String: [Float]] = [:]
        for _ in 0 ..< n {
            let len = try i32()
            guard len > 0, len < 4096, cursor + len <= data.count else { throw GGUFError.truncated("importance matrix name") }
            let name = String(decoding: data[cursor ..< cursor + len], as: UTF8.self)
            cursor += len
            let ncall = try i32()
            let nval = try i32()
            guard nval > 0, cursor + nval * 4 <= data.count else { throw GGUFError.truncated("importance matrix values") }
            var values = [Float](repeating: 0, count: nval)
            data.withUnsafeBytes { raw in
                for k in 0 ..< nval {
                    let sum = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: cursor + 4 * k, as: UInt32.self)))
                    values[k] = ncall > 0 ? sum / Float(ncall) : sum
                }
            }
            cursor += nval * 4
            entries[name] = values
        }
        return GGUFImatrix(entries: entries)
    }

    /// Writes the legacy layout (for tests and tools).
    public func write(to url: URL, callCount: Int32 = 1) throws {
        var d = Data()
        func put(_ v: Int32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        put(Int32(entries.count))
        for (name, values) in entries.sorted(by: { $0.key < $1.key }) {
            put(Int32(name.utf8.count))
            d.append(contentsOf: Array(name.utf8))
            put(callCount)
            put(Int32(values.count))
            for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
        }
        put(callCount)
        put(0)
        try d.write(to: url)
    }
}

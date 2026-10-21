import Foundation

/// The sampling defaults the reference converter copies from `generation_config.json` into the
/// `general.sampling.*` keys (`Metadata.load` and `Metadata.set_gguf_meta_model` in llama.cpp's
/// `gguf-py`, commit a3a1c47).
///
/// Mapping, in the order the converter writes the keys: `sequence` (string), `top_k` (int32),
/// `top_p`, `min_p`, `xtc_probability`, `xtc_threshold`, `temperature` (as `temp`), `penalty_last_n`
/// (int32), `penalty_repeat` (falling back to `repetition_penalty`), `mirostat` (int32),
/// `mirostat_tau`, `mirostat_eta`. Anything else in the file (`do_sample`, `eos_token_id`, ...) is
/// not copied. A missing or unparseable file, a key that is absent or `null`, or a value of the
/// wrong kind (a fractional `top_k`, a string `top_p`) writes nothing for that key.
public enum GGUFGenerationConfig {
    /// The metadata entries for `directory`'s `generation_config.json`; empty when there is none.
    public static func samplingEntries(directory: URL) -> [(String, GGUFValue)] {
        let url = directory.appendingPathComponent("generation_config.json")
        guard let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data),
            let json = object as? [String: Any]
        else { return [] }

        func number(_ key: String) -> Double? {
            guard let n = json[key] as? NSNumber, !n.isBoolean else { return nil }
            return n.doubleValue
        }
        func integer(_ key: String) -> Int32? {
            guard let d = number(key), d == d.rounded(), abs(d) < Double(Int32.max) else { return nil }
            return Int32(d)
        }
        var out: [(String, GGUFValue)] = []
        if let s = json["sequence"] as? String { out.append(("general.sampling.sequence", .string(s))) }
        if let v = integer("top_k") { out.append(("general.sampling.top_k", .int32(v))) }
        func float(_ gguf: String, _ key: String) {
            if let v = number(key) { out.append(("general.sampling.\(gguf)", .float32(Float(v)))) }
        }
        float("top_p", "top_p")
        float("min_p", "min_p")
        float("xtc_probability", "xtc_probability")
        float("xtc_threshold", "xtc_threshold")
        float("temp", "temperature")
        if let v = integer("penalty_last_n") { out.append(("general.sampling.penalty_last_n", .int32(v))) }
        float("penalty_repeat", json["penalty_repeat"] != nil ? "penalty_repeat" : "repetition_penalty")
        if let v = integer("mirostat") { out.append(("general.sampling.mirostat", .int32(v))) }
        float("mirostat_tau", "mirostat_tau")
        float("mirostat_eta", "mirostat_eta")
        return out
    }

    /// `tokenizer.ggml.suppress_tokens`, which the converter writes for the unified Gemma 4
    /// architecture from `suppress_tokens` in the same file.
    public static func suppressTokens(directory: URL) -> GGUFValue? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")),
            let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let ids = json["suppress_tokens"] as? [Any]
        else { return nil }
        let values = ids.compactMap { ($0 as? NSNumber).flatMap { !$0.isBoolean ? Int32(exactly: $0.int64Value) : nil } }
        guard values.count == ids.count else { return nil }
        return .array(.int32, values.map { .int32($0) })
    }
}

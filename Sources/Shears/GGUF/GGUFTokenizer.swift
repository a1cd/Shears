import Foundation

/// The `tokenizer.*` metadata for a GGUF file, plus the vocabulary size it implies.
public struct GGUFTokenizerMetadata {
    public var entries: [(key: String, value: GGUFValue)]
    /// Number of entries in `tokenizer.ggml.tokens`. Embedding tables must have this many rows.
    public var tokenCount: Int

    public init(entries: [(key: String, value: GGUFValue)], tokenCount: Int) {
        self.entries = entries
        self.tokenCount = tokenCount
    }

    public subscript(key: String) -> GGUFValue? {
        entries.first(where: { $0.key == key })?.value
    }
}

/// Carries a Hugging Face byte-fallback BPE tokenizer (`tokenizer.json`, `tokenizer_config.json`,
/// optional `chat_template.jinja`) into GGUF metadata.
///
/// This follows llama.cpp's converter: the token list is the model vocabulary followed by added
/// tokens, ordered by id; every score is -1000; token types come from the special-token names
/// in `tokenizer_config.json`; merges are joined with a space.
///
/// Not handled: `TemplateProcessing` post-processors (BOS/EOS insertion flags are read only from
/// `tokenizer_config.json`), `normalizer` types beyond lowercase / NFC / strip-accents, and
/// tokenizer models other than BPE.
public enum GGUFTokenizer {
    /// The special-token kinds the converter looks for, in its order.
    static let specialKinds = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]
    /// Kinds that have a `tokenizer.ggml.*_token_id` key.
    static let idKeys: [String: String] = [
        "bos": "tokenizer.ggml.bos_token_id",
        "eos": "tokenizer.ggml.eos_token_id",
        "unk": "tokenizer.ggml.unknown_token_id",
        "sep": "tokenizer.ggml.seperator_token_id",
        "pad": "tokenizer.ggml.padding_token_id",
        "mask": "tokenizer.ggml.mask_token_id",
    ]
    static let addKeys: [String: String] = [
        "bos": "tokenizer.ggml.add_bos_token",
        "eos": "tokenizer.ggml.add_eos_token",
        "sep": "tokenizer.ggml.add_sep_token",
    ]

    public static func load(
        directory: URL,
        model: String,
        userDefinedTokens: Set<String> = []
    ) throws -> GGUFTokenizerMetadata {
        guard let tokenizer = try GGUFJSON.load(directory.appendingPathComponent("tokenizer.json")) else {
            throw GGUFError.other("no tokenizer.json in \(directory.path)")
        }
        let config = try GGUFJSON.load(directory.appendingPathComponent("tokenizer_config.json"))

        guard let modelJSON = tokenizer["model"], modelJSON["type"]?.string == "BPE" else {
            throw GGUFError.other("tokenizer.json: only BPE tokenizers are supported")
        }
        guard let vocab = modelJSON["vocab"]?.object else {
            throw GGUFError.other("tokenizer.json: model.vocab is missing")
        }

        struct Added {
            var id: Int
            var content: String
            var special: Bool
        }
        var added: [Added] = []
        for item in tokenizer["added_tokens"]?.array ?? [] {
            guard let id = item["id"]?.int, let content = item["content"]?.string else {
                throw GGUFError.other("tokenizer.json: malformed added token")
            }
            added.append(Added(id: id, content: content, special: item["special"]?.bool ?? false))
        }

        // Token text by id. Vocabulary entries are kept as ordered pairs, never as a Swift
        // dictionary, so tokens that differ only in Unicode normalization stay distinct.
        var maxID = -1
        for (_, value) in vocab {
            guard let id = value.int, id >= 0 else { throw GGUFError.other("tokenizer.json: bad token id") }
            maxID = max(maxID, id)
        }
        for a in added { maxID = max(maxID, a.id) }
        guard maxID >= 0 else { throw GGUFError.other("tokenizer.json: empty vocabulary") }
        let count = maxID + 1
        var text = [String?](repeating: nil, count: count)
        var baseIDs = Set<Int>()
        for (token, value) in vocab {
            let id = value.int!
            if let existing = text[id] {
                throw GGUFError.other("tokenizer.json: tokens \(existing) and \(token) share id \(id)")
            }
            text[id] = token
            baseIDs.insert(id)
        }
        for a in added {
            guard a.id >= 0 else { throw GGUFError.other("tokenizer.json: negative added token id") }
            if let existing = text[a.id], Array(existing.utf8) != Array(a.content.utf8) {
                throw GGUFError.other("tokenizer.json: added token \(a.content) collides with \(existing) at id \(a.id)")
            }
            text[a.id] = a.content
        }

        // Which tokens are "special": named in tokenizer_config.json, else flagged in tokenizer.json.
        var specials = Set<String>()
        if let config {
            func content(_ v: GGUFJSON?) -> String? { v?.string ?? v?["content"]?.string }
            for (key, value) in config.object ?? [] {
                if key.hasSuffix("_token"), let s = content(value) { specials.insert(s) }
            }
            for key in ["extra_special_tokens", "additional_special_tokens"] {
                if let list = config[key]?.array {
                    for v in list { if let s = content(v) { specials.insert(s) } }
                } else if let pairs = config[key]?.object {
                    for (_, v) in pairs { if let s = content(v) { specials.insert(s) } }
                }
            }
        } else {
            for a in added where a.special { specials.insert(a.content) }
        }

        var tokens: [String] = []
        var types: [Int32] = []
        tokens.reserveCapacity(count)
        types.reserveCapacity(count)
        for id in 0 ..< count {
            guard let t = text[id] else {
                tokens.append("[PAD\(id)]")
                types.append(GGUFTokenType.unused.rawValue)
                continue
            }
            var type: GGUFTokenType
            if baseIDs.contains(id) {
                type = isByteToken(t) ? .byte : (specials.contains(t) ? .control : .normal)
            } else {
                type = specials.contains(t) ? .control : .userDefined
            }
            if userDefinedTokens.contains(t) { type = .userDefined }
            tokens.append(t)
            types.append(type.rawValue)
        }

        var entries: [(String, GGUFValue)] = []
        entries.append(("tokenizer.ggml.model", .string(model)))
        entries.append(("tokenizer.ggml.tokens", .strings(tokens)))
        entries.append(("tokenizer.ggml.scores", .float32s([Float](repeating: -1000, count: count))))
        entries.append(("tokenizer.ggml.token_type", .int32s(types.map { Int($0) })))

        if let merges = try mergeStrings(modelJSON["merges"]), !merges.isEmpty {
            entries.append(("tokenizer.ggml.merges", .strings(merges)))
        }

        // Special token ids: names from tokenizer_config.json resolved against added tokens,
        // then config.json ids for kinds still unset.
        var ids: [String: Int] = [:]
        var addFlags: [String: Bool] = [:]
        var order: [String] = []
        if let config {
            for kind in specialKinds {
                if let flag = config["add_\(kind)_token"]?.bool { addFlags[kind] = flag }
                let entry = config["\(kind)_token"]
                if let name = entry?.string ?? entry?["content"]?.string,
                    let match = added.first(where: { $0.content == name }), ids[kind] == nil, match.id >= 0
                {
                    ids[kind] = match.id
                    order.append(kind)
                }
            }
        }
        if let modelConfig = try GGUFJSON.load(directory.appendingPathComponent("config.json")) {
            for kind in specialKinds where ids[kind] == nil {
                let v = modelConfig["\(kind)_token_id"] ?? modelConfig["text_config"]?["\(kind)_token_id"]
                if let id = v?.int, id >= 0 {
                    ids[kind] = id
                    order.append(kind)
                }
            }
        }
        for kind in order {
            if let key = idKeys[kind], let id = ids[kind] { entries.append((key, .uint32(UInt32(id)))) }
        }
        for kind in specialKinds {
            if let key = addKeys[kind], let flag = addFlags[kind] { entries.append((key, .bool(flag))) }
        }

        if let config, let template = try chatTemplate(config: config, directory: directory) {
            entries.append(contentsOf: template)
        }

        if let normalizer = tokenizer["normalizer"] {
            var flags = Normalizer()
            flags.parse(normalizer)
            if let v = flags.nfc { entries.append(("tokenizer.ggml.normalizer.nfc", .bool(v))) }
            if let v = flags.lowercase { entries.append(("tokenizer.ggml.normalizer.lowercase", .bool(v))) }
            if let v = flags.stripAccents { entries.append(("tokenizer.ggml.normalizer.strip_accents", .bool(v))) }
        }

        return GGUFTokenizerMetadata(
            entries: entries.map { (key: $0.0, value: $0.1) }, tokenCount: count)
    }

    // MARK: Pieces

    /// `<0xHH>` byte-fallback tokens.
    static func isByteToken(_ t: String) -> Bool {
        let u = Array(t.utf8)
        guard u.count == 6, u[0] == 0x3C, u[1] == 0x30, u[2] == 0x78, u[5] == 0x3E else { return false }
        func hex(_ c: UInt8) -> Bool {
            (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x46) || (c >= 0x61 && c <= 0x66)
        }
        return hex(u[3]) && hex(u[4])
    }

    /// Merges as `"a b"` strings. The newer pair format joins with a space, and a space inside a
    /// part is written as U+0120 so the separator stays unambiguous.
    static func mergeStrings(_ raw: GGUFJSON?) throws -> [String]? {
        guard let list = raw?.array, !list.isEmpty else { return nil }
        var out: [String] = []
        out.reserveCapacity(list.count)
        for item in list {
            if let s = item.string {
                out.append(s)
            } else if let pair = item.array, pair.count == 2, let a = pair[0].string, let b = pair[1].string {
                out.append(escapeSpaces(a) + " " + escapeSpaces(b))
            } else {
                throw GGUFError.other("tokenizer.json: unknown merges format")
            }
        }
        return out
    }

    /// Replaces each space code point with U+0120, scalar by scalar (Foundation's string search
    /// would also see through combining marks).
    static func escapeSpaces(_ s: String) -> String {
        var scalars = String.UnicodeScalarView()
        for u in s.unicodeScalars { scalars.append(u == " " ? "\u{0120}" : u) }
        return String(scalars)
    }

    static func chatTemplate(config: GGUFJSON, directory: URL) throws -> [(String, GGUFValue)]? {
        let fm = FileManager.default
        var alt: GGUFJSON?
        let jinja = directory.appendingPathComponent("chat_template.jinja")
        let json = directory.appendingPathComponent("chat_template.json")
        if fm.fileExists(atPath: jinja.path) {
            let main = try String(contentsOf: jinja, encoding: .utf8)
            let extraDir = directory.appendingPathComponent("additional_chat_templates")
            let extras = ((try? fm.contentsOfDirectory(atPath: extraDir.path)) ?? [])
                .filter { $0.hasSuffix(".jinja") }.sorted()
            if extras.isEmpty {
                alt = .string(main)
            } else {
                var list: [GGUFJSON] = [.object([("name", .string("default")), ("template", .string(main))])]
                for f in extras {
                    let body = try String(contentsOf: extraDir.appendingPathComponent(f), encoding: .utf8)
                    list.append(
                        .object([("name", .string((f as NSString).deletingPathExtension)), ("template", .string(body))]))
                }
                alt = .array(list)
            }
        } else if fm.fileExists(atPath: json.path) {
            alt = try GGUFJSON.load(json)?["chat_template"]
        }
        guard let value = config["chat_template"] ?? alt else { return nil }
        if let s = value.string { return [("tokenizer.chat_template", .string(s))] }
        guard let choices = value.array else { return nil }
        var out: [(String, GGUFValue)] = []
        var names: [String] = []
        var defaultTemplate: String?
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for choice in choices {
            let rawName = choice["name"]?.string ?? ""
            let name = String(rawName.map { allowed.contains($0) ? $0 : "_" })
            guard !name.isEmpty, let template = choice["template"]?.string else { continue }
            if name == "default" {
                defaultTemplate = template
            } else {
                if !names.contains(name) { names.append(name) }
                out.append(("tokenizer.chat_template.\(name)", .string(template)))
            }
        }
        if !names.isEmpty { out.append(("tokenizer.chat_templates", .strings(names))) }
        if let d = defaultTemplate { out.append(("tokenizer.chat_template", .string(d))) }
        return out
    }

    struct Normalizer {
        var lowercase: Bool?
        var nfc: Bool?
        var stripAccents: Bool?

        mutating func parse(_ n: GGUFJSON) {
            switch n["type"]?.string {
            case "Lowercase": lowercase = true
            case "NFC": nfc = true
            case "StripAccents": stripAccents = true
            case "BertNormalizer":
                if let v = n["lowercase"]?.bool { lowercase = v }
                if let v = n["strip_accents"]?.bool { stripAccents = v }
            case "Sequence":
                for child in n["normalizers"]?.array ?? [] { parse(child) }
            default: break
            }
        }
    }
}

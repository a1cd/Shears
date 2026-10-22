import Foundation

/// A small JSON reader that keeps strings exactly as written.
///
/// Foundation's `JSONSerialization` drops a U+FEFF at the start of a string, and a Swift
/// `[String: _]` merges keys that differ only in Unicode normalization. Both lose real tokens
/// from tokenizer vocabularies, so tokenizer files are read with this instead. Objects keep
/// their entries in order, as pairs.
indirect enum GGUFJSON {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([GGUFJSON])
    case object([(key: String, value: GGUFJSON)])

    struct ParseError: Error, CustomStringConvertible {
        var offset: Int
        var reason: String
        var description: String { "JSON error at byte \(offset): \(reason)" }
    }

    static func parse(_ text: String) throws -> GGUFJSON { try parse(Data(text.utf8)) }

    static func parse(_ data: Data) throws -> GGUFJSON {
        var p = Parser(bytes: [UInt8](data))
        p.skipSpace()
        let v = try p.value()
        p.skipSpace()
        guard p.i == p.bytes.count else { throw ParseError(offset: p.i, reason: "trailing characters") }
        return v
    }

    static func load(_ url: URL) throws -> GGUFJSON? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do { return try parse(Data(contentsOf: url)) } catch let e as ParseError {
            throw GGUFError.other("\(url.lastPathComponent): \(e)")
        }
    }

    // MARK: Access

    /// The first entry named `key`, for an object.
    subscript(key: String) -> GGUFJSON? {
        guard case .object(let pairs) = self else { return nil }
        for (k, v) in pairs where k == key { return v }
        return nil
    }

    var string: String? { if case .string(let s) = self { s } else { nil } }
    var bool: Bool? { if case .bool(let b) = self { b } else { nil } }
    var array: [GGUFJSON]? { if case .array(let a) = self { a } else { nil } }
    var object: [(key: String, value: GGUFJSON)]? { if case .object(let o) = self { o } else { nil } }
    var int: Int? {
        switch self {
        case .int(let i): i
        case .double(let d): d == d.rounded() && abs(d) < 1e15 ? Int(d) : nil
        default: nil
        }
    }
    var double: Double? {
        switch self {
        case .int(let i): Double(i)
        case .double(let d): d
        default: nil
        }
    }

    // MARK: Parser

    private struct Parser {
        var bytes: [UInt8]
        var i = 0

        init(bytes: [UInt8]) {
            self.bytes = bytes
            // A leading byte-order mark belongs to the file, not to any string in it.
            if bytes.starts(with: [0xEF, 0xBB, 0xBF]) { i = 3 }
        }

        func fail(_ reason: String) -> ParseError { ParseError(offset: i, reason: reason) }

        mutating func skipSpace() {
            while i < bytes.count, bytes[i] == 0x20 || bytes[i] == 0x0A || bytes[i] == 0x0D || bytes[i] == 0x09 {
                i += 1
            }
        }

        mutating func value() throws -> GGUFJSON {
            guard i < bytes.count else { throw fail("unexpected end") }
            switch bytes[i] {
            case UInt8(ascii: "{"): return try object()
            case UInt8(ascii: "["): return try array()
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try literal("true"); return .bool(true)
            case UInt8(ascii: "f"): try literal("false"); return .bool(false)
            case UInt8(ascii: "n"): try literal("null"); return .null
            default: return try number()
            }
        }

        mutating func literal(_ word: String) throws {
            let w = Array(word.utf8)
            guard bytes.count - i >= w.count, Array(bytes[i ..< i + w.count]) == w else { throw fail("bad literal") }
            i += w.count
        }

        mutating func number() throws -> GGUFJSON {
            let start = i
            var isInt = true
            if i < bytes.count, bytes[i] == UInt8(ascii: "-") { i += 1 }
            while i < bytes.count {
                let c = bytes[i]
                if c >= 0x30 && c <= 0x39 { i += 1 } else if c == UInt8(ascii: ".") || c == UInt8(ascii: "e") || c == UInt8(ascii: "E") || c == UInt8(ascii: "+") || c == UInt8(ascii: "-") {
                    isInt = false
                    i += 1
                } else { break }
            }
            guard i > start else { throw fail("unexpected character") }
            let text = String(decoding: bytes[start ..< i], as: UTF8.self)
            if isInt, let v = Int(text) { return .int(v) }
            guard let d = Double(text) else { throw fail("bad number") }
            return .double(d)
        }

        mutating func array() throws -> GGUFJSON {
            i += 1
            var items: [GGUFJSON] = []
            skipSpace()
            if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
            while true {
                skipSpace()
                items.append(try value())
                skipSpace()
                guard i < bytes.count else { throw fail("unterminated array") }
                if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                if bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
                throw fail("expected , or ]")
            }
        }

        mutating func object() throws -> GGUFJSON {
            i += 1
            var pairs: [(key: String, value: GGUFJSON)] = []
            skipSpace()
            if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
            while true {
                skipSpace()
                guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else { throw fail("expected a key") }
                let key = try string()
                skipSpace()
                guard i < bytes.count, bytes[i] == UInt8(ascii: ":") else { throw fail("expected :") }
                i += 1
                skipSpace()
                pairs.append((key, try value()))
                skipSpace()
                guard i < bytes.count else { throw fail("unterminated object") }
                if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                if bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
                throw fail("expected , or }")
            }
        }

        mutating func hex4() throws -> UInt32 {
            guard bytes.count - i >= 4 else { throw fail("short \\u escape") }
            var v: UInt32 = 0
            for _ in 0 ..< 4 {
                let c = bytes[i]
                let d: UInt32
                switch c {
                case 0x30 ... 0x39: d = UInt32(c - 0x30)
                case 0x41 ... 0x46: d = UInt32(c - 0x41 + 10)
                case 0x61 ... 0x66: d = UInt32(c - 0x61 + 10)
                default: throw fail("bad \\u escape")
                }
                v = v << 4 | d
                i += 1
            }
            return v
        }

        mutating func string() throws -> String {
            i += 1  // opening quote
            let start = i
            // Fast path: no escapes.
            while i < bytes.count, bytes[i] != UInt8(ascii: "\""), bytes[i] != UInt8(ascii: "\\") { i += 1 }
            guard i < bytes.count else { throw fail("unterminated string") }
            if bytes[i] == UInt8(ascii: "\"") {
                defer { i += 1 }
                return String(decoding: bytes[start ..< i], as: UTF8.self)
            }
            var out = Array(bytes[start ..< i])
            while i < bytes.count {
                let c = bytes[i]
                if c == UInt8(ascii: "\"") { i += 1; return String(decoding: out, as: UTF8.self) }
                if c != UInt8(ascii: "\\") { out.append(c); i += 1; continue }
                i += 1
                guard i < bytes.count else { break }
                let e = bytes[i]
                i += 1
                switch e {
                case UInt8(ascii: "\""): out.append(0x22)
                case UInt8(ascii: "\\"): out.append(0x5C)
                case UInt8(ascii: "/"): out.append(0x2F)
                case UInt8(ascii: "b"): out.append(0x08)
                case UInt8(ascii: "f"): out.append(0x0C)
                case UInt8(ascii: "n"): out.append(0x0A)
                case UInt8(ascii: "r"): out.append(0x0D)
                case UInt8(ascii: "t"): out.append(0x09)
                case UInt8(ascii: "u"):
                    var code = try hex4()
                    if (0xD800 ..< 0xDC00).contains(code), bytes.count - i >= 6, bytes[i] == 0x5C, bytes[i + 1] == UInt8(ascii: "u") {
                        let save = i
                        i += 2
                        let low = try hex4()
                        if (0xDC00 ..< 0xE000).contains(low) {
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            i = save
                        }
                    }
                    // A lone surrogate has no UTF-8 form; it becomes U+FFFD.
                    let scalar = Unicode.Scalar(code) ?? "\u{FFFD}"
                    out.append(contentsOf: Array(String(Character(scalar)).utf8))
                default: throw fail("bad escape")
                }
            }
            throw fail("unterminated string")
        }
    }
}

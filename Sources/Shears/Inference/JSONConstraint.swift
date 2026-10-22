import Foundation
import MLX

/// A character-level recognizer for one JSON object (RFC 8259), the language `response_format:
/// {"type": "json_object"}` asks for: whitespace, then `{`, then members, then the matching `}`.
public struct JSONObjectAutomaton: Equatable {
    enum Mode: Equatable {
        case start
        case value // a value is expected
        case arrayValueOrEnd // just after `[`
        case keyOrEnd // just after `{`
        case key // after `,` in an object
        case colon
        case afterValue
        case string(key: Bool)
        case escape(key: Bool)
        case unicode(key: Bool, remaining: Int)
        case number(NumberState)
        case literal([UInt8])
        case done
    }
    enum NumberState: Equatable { case minus, zero, integer, dot, fraction, exponent, exponentSign, exponentDigits }

    var mode = Mode.start
    /// `{` or `[` per open container.
    var stack: [UInt8] = []

    public init() {}

    /// The object is complete: only whitespace may follow.
    public var isComplete: Bool { mode == .done }

    private static func isSpace(_ c: UInt8) -> Bool { c == 0x20 || c == 0x0A || c == 0x0D || c == 0x09 }
    private static func isDigit(_ c: UInt8) -> Bool { c >= 0x30 && c <= 0x39 }
    private static func isHex(_ c: UInt8) -> Bool {
        isDigit(c) || (c >= 0x41 && c <= 0x46) || (c >= 0x61 && c <= 0x66)
    }

    /// Feeds one byte; false (and no change guaranteed) if no valid JSON object continues with it.
    public mutating func accept(_ c: UInt8) -> Bool {
        switch mode {
        case .done:
            return Self.isSpace(c)
        case .start:
            if Self.isSpace(c) { return true }
            guard c == UInt8(ascii: "{") else { return false }
            stack = [c]
            mode = .keyOrEnd
            return true
        case .value, .arrayValueOrEnd:
            if Self.isSpace(c) { return true }
            if mode == .arrayValueOrEnd, c == UInt8(ascii: "]") { return close(c) }
            return beginValue(c)
        case .keyOrEnd:
            if Self.isSpace(c) { return true }
            if c == UInt8(ascii: "}") { return close(c) }
            guard c == UInt8(ascii: "\"") else { return false }
            mode = .string(key: true)
            return true
        case .key:
            if Self.isSpace(c) { return true }
            guard c == UInt8(ascii: "\"") else { return false }
            mode = .string(key: true)
            return true
        case .colon:
            if Self.isSpace(c) { return true }
            guard c == UInt8(ascii: ":") else { return false }
            mode = .value
            return true
        case .afterValue:
            if Self.isSpace(c) { return true }
            if c == UInt8(ascii: ",") {
                mode = stack.last == UInt8(ascii: "{") ? .key : .value
                return true
            }
            return close(c)
        case .string(let key):
            if c == UInt8(ascii: "\"") { mode = key ? .colon : .afterValue; return true }
            if c == UInt8(ascii: "\\") { mode = .escape(key: key); return true }
            return c >= 0x20
        case .escape(let key):
            switch c {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"), UInt8(ascii: "b"), UInt8(ascii: "f"),
                UInt8(ascii: "n"), UInt8(ascii: "r"), UInt8(ascii: "t"):
                mode = .string(key: key)
                return true
            case UInt8(ascii: "u"):
                mode = .unicode(key: key, remaining: 4)
                return true
            default: return false
            }
        case .unicode(let key, let remaining):
            guard Self.isHex(c) else { return false }
            mode = remaining == 1 ? .string(key: key) : .unicode(key: key, remaining: remaining - 1)
            return true
        case .literal(let rest):
            guard c == rest[0] else { return false }
            if rest.count == 1 { mode = .afterValue } else { mode = .literal(Array(rest.dropFirst())) }
            return true
        case .number(let n):
            return number(n, c)
        }
    }

    private mutating func close(_ c: UInt8) -> Bool {
        guard let open = stack.last else { return false }
        guard (open == UInt8(ascii: "{") && c == UInt8(ascii: "}")) || (open == UInt8(ascii: "[") && c == UInt8(ascii: "]")) else {
            return false
        }
        stack.removeLast()
        mode = stack.isEmpty ? .done : .afterValue
        return true
    }

    private mutating func beginValue(_ c: UInt8) -> Bool {
        switch c {
        case UInt8(ascii: "{"): stack.append(c); mode = .keyOrEnd
        case UInt8(ascii: "["): stack.append(c); mode = .arrayValueOrEnd
        case UInt8(ascii: "\""): mode = .string(key: false)
        case UInt8(ascii: "-"): mode = .number(.minus)
        case UInt8(ascii: "0"): mode = .number(.zero)
        case UInt8(ascii: "1") ... UInt8(ascii: "9"): mode = .number(.integer)
        case UInt8(ascii: "t"): mode = .literal(Array("rue".utf8))
        case UInt8(ascii: "f"): mode = .literal(Array("alse".utf8))
        case UInt8(ascii: "n"): mode = .literal(Array("ull".utf8))
        default: return false
        }
        return true
    }

    private mutating func number(_ n: NumberState, _ c: UInt8) -> Bool {
        let e = c == UInt8(ascii: "e") || c == UInt8(ascii: "E")
        switch n {
        case .minus:
            if c == UInt8(ascii: "0") { mode = .number(.zero); return true }
            if Self.isDigit(c) { mode = .number(.integer); return true }
            return false
        case .zero, .integer:
            if n == .integer, Self.isDigit(c) { return true }
            if c == UInt8(ascii: ".") { mode = .number(.dot); return true }
            if e { mode = .number(.exponent); return true }
        case .dot:
            if Self.isDigit(c) { mode = .number(.fraction); return true }
            return false
        case .fraction:
            if Self.isDigit(c) { return true }
            if e { mode = .number(.exponent); return true }
        case .exponent:
            if c == UInt8(ascii: "+") || c == UInt8(ascii: "-") { mode = .number(.exponentSign); return true }
            if Self.isDigit(c) { mode = .number(.exponentDigits); return true }
            return false
        case .exponentSign:
            if Self.isDigit(c) { mode = .number(.exponentDigits); return true }
            return false
        case .exponentDigits:
            if Self.isDigit(c) { return true }
        }
        // A finished number: the byte belongs to what follows.
        mode = .afterValue
        return accept(c)
    }

    /// Whether the whole of `bytes` can follow, leaving `self` unchanged.
    public func accepts(_ bytes: [UInt8]) -> Bool {
        var copy = self
        for b in bytes where !copy.accept(b) { return false }
        return true
    }
}

/// `response_format: json_object`: only tokens that keep the output a prefix of one JSON object, and,
/// once the object is closed, only the end-of-generation tokens.
///
/// Each step tests the most likely tokens first, which is nearly always enough, and scans the whole
/// vocabulary only when none of them fits.
public final class JSONObjectConstraint: GenerationConstraint {
    private var automaton = JSONObjectAutomaton()
    private let tokenBytes: [[UInt8]?]
    private let stopTokens: [Int]
    private let vocabularySize: Int
    /// How many of the most likely tokens are tested before the full scan.
    public var candidateCount = 256

    public init(tokenizer: BPETokenizer, stopTokenIDs: [Int], logitsVocabularySize: Int) {
        vocabularySize = logitsVocabularySize
        stopTokens = stopTokenIDs
        tokenBytes = tokenizer.wholeUTF8TokenBytes(count: logitsVocabularySize)
    }

    public var isComplete: Bool { automaton.isComplete }

    public func allowedMask(logits: MLXArray) -> MLXArray? {
        var allowed = [Bool](repeating: false, count: vocabularySize)
        if automaton.isComplete {
            for s in stopTokens where s < vocabularySize { allowed[s] = true }
            return MLXArray(allowed)
        }
        let n = logits.dim(-1)
        let k = min(candidateCount, n)
        let top = argPartition(logits, kth: n - k, axis: -1)[(n - k)...]
        eval(top)
        var any = false
        for id in top.asArray(Int32.self) {
            let i = Int(id)
            if i < tokenBytes.count, let b = tokenBytes[i], automaton.accepts(b) { allowed[i] = true; any = true }
        }
        if !any {
            for i in 0 ..< tokenBytes.count {
                if let b = tokenBytes[i], automaton.accepts(b) { allowed[i] = true; any = true }
            }
        }
        if !any { for s in stopTokens where s < vocabularySize { allowed[s] = true } }
        return MLXArray(allowed)
    }

    public func accept(_ token: Int) {
        guard token < tokenBytes.count, let b = tokenBytes[token] else { return }
        for byte in b { _ = automaton.accept(byte) }
    }
}

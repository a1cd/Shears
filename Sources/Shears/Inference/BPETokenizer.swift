import Foundation

/// A Hugging Face `tokenizer.json` byte-fallback BPE tokenizer: encode, decode, and decode one token at a
/// time.
///
/// Why this is not swift-transformers: that package brings its hub client, networking and crypto
/// dependencies into a library meant to ship in an app, and a Swift `[String: Int]` vocabulary
/// silently merges tokens that differ only in Unicode normalization. This type keys everything by
/// UTF-8 bytes and reads only what Gemma-style tokenizers use:
///
/// - normalizer: `Replace` with a string pattern (and a `Sequence` of them), `Prepend`
/// - pre-tokenizer: none, or `Split` on a string pattern (every behavior), or a `Sequence` of them
/// - model: `BPE` with `byte_fallback`, optional `unk_token`
/// - added tokens, matched in the raw text before anything else (`special` is honored when decoding)
/// - post-processor: none, or a `TemplateProcessing` single template
/// - decoder: any `Sequence` of `Replace`, `ByteFallback`, `Fuse`
///
/// Anything else makes `init` throw rather than tokenize differently from the reference.
public final class BPETokenizer: @unchecked Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public var description: String
        init(_ d: String) { description = d }
    }

    private enum Normalizer {
        case replace(pattern: [Unicode.Scalar], content: [Unicode.Scalar])
        case prepend([Unicode.Scalar])
    }

    private struct Split {
        enum Behavior { case removed, isolated, mergedWithPrevious, mergedWithNext, contiguous }
        var pattern: [Unicode.Scalar]
        var behavior: Behavior
        var invert: Bool
    }

    private struct AddedToken {
        var id: Int
        var scalars: [Unicode.Scalar]
        var special: Bool
    }

    private enum DecoderStep {
        case replace(pattern: String, content: String)
        case byteFallback
        case fuse
    }

    /// Token id by its UTF-8 bytes.
    private var vocabulary: [[UInt8]: Int] = [:]
    /// Token text by id (UTF-8 bytes), nil for ids with no token.
    private var tokenBytes: [[UInt8]?] = []
    private var normalizers: [Normalizer] = []
    private var splits: [Split] = []
    private var added: [AddedToken] = []
    private var addedByFirst: [Unicode.Scalar: [AddedToken]] = [:]
    private var specialIDs: Set<Int> = []
    private var decoderSteps: [DecoderStep] = []
    private var byteFallback = false
    private var unknownID: Int?
    /// `(left id, right id)` to `(rank, merged id)`.
    private var merges: [UInt64: (rank: Int32, merged: Int32)] = [:]
    /// Ids of `<0x00>` ... `<0xFF>`, when the vocabulary has them.
    private var byteTokenIDs: [Int?] = Array(repeating: nil, count: 256)
    /// Byte value for ids that are `<0xXX>` tokens.
    private var byteValueOfID: [Int: UInt8] = [:]
    private var templateSingle: [Int] = []

    public private(set) var bosToken: String?
    public private(set) var eosToken: String?
    public private(set) var unknownToken: String?
    public private(set) var padToken: String?
    public var bosTokenID: Int? { bosToken.flatMap(tokenID(for:)) }
    public var eosTokenID: Int? { eosToken.flatMap(tokenID(for:)) }

    /// Number of ids, one more than the largest token id.
    public var vocabularySize: Int { tokenBytes.count }

    /// Loads `tokenizer.json` (and `tokenizer_config.json`, for the special token names) from `directory`.
    public convenience init(directory: URL) throws {
        let tj = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tj.path) else {
            throw Failure("no tokenizer.json in \(directory.lastPathComponent)")
        }
        let cfg = directory.appendingPathComponent("tokenizer_config.json")
        try self.init(
            tokenizerJSON: try Data(contentsOf: tj),
            configuration: FileManager.default.fileExists(atPath: cfg.path) ? try Data(contentsOf: cfg) : nil)
    }

    public init(tokenizerJSON: Data, configuration: Data? = nil) throws {
        let root = try GGUFJSON.parse(tokenizerJSON)
        guard let model = root["model"], model["type"]?.string == "BPE" else {
            throw Failure("tokenizer.json: only BPE tokenizers are supported")
        }
        guard let vocab = model["vocab"]?.object else { throw Failure("tokenizer.json: model.vocab is missing") }
        byteFallback = model["byte_fallback"]?.bool ?? false
        if model["dropout"].map({ if case .null = $0 { false } else { true } }) ?? false {
            throw Failure("tokenizer.json: BPE dropout is not supported")
        }
        if let prefix = model["continuing_subword_prefix"]?.string, !prefix.isEmpty {
            throw Failure("tokenizer.json: continuing_subword_prefix is not supported")
        }
        if let suffix = model["end_of_word_suffix"]?.string, !suffix.isEmpty {
            throw Failure("tokenizer.json: end_of_word_suffix is not supported")
        }

        var maxID = -1
        for (_, v) in vocab {
            guard let id = v.int, id >= 0 else { throw Failure("tokenizer.json: bad token id") }
            maxID = max(maxID, id)
        }
        var addedList: [AddedToken] = []
        for item in root["added_tokens"]?.array ?? [] {
            guard let id = item["id"]?.int, let content = item["content"]?.string else {
                throw Failure("tokenizer.json: malformed added token")
            }
            if item["lstrip"]?.bool == true || item["rstrip"]?.bool == true || item["single_word"]?.bool == true {
                throw Failure("tokenizer.json: added token \(content) uses lstrip/rstrip/single_word, which are not supported")
            }
            addedList.append(AddedToken(id: id, scalars: Array(content.unicodeScalars), special: item["special"]?.bool ?? false))
            maxID = max(maxID, id)
        }
        guard maxID >= 0 else { throw Failure("tokenizer.json: empty vocabulary") }
        tokenBytes = Array(repeating: nil, count: maxID + 1)
        vocabulary.reserveCapacity(vocab.count + addedList.count)
        for (token, v) in vocab {
            let bytes = Array(token.utf8)
            vocabulary[bytes] = v.int!
            tokenBytes[v.int!] = bytes
        }
        for a in addedList {
            var s = String.UnicodeScalarView()
            s.append(contentsOf: a.scalars)
            let bytes = Array(String(s).utf8)
            vocabulary[bytes] = a.id
            tokenBytes[a.id] = bytes
            if a.special { specialIDs.insert(a.id) }
        }
        added = addedList
        for a in addedList.sorted(by: { $0.scalars.count > $1.scalars.count }) {
            if let f = a.scalars.first { addedByFirst[f, default: []].append(a) }
        }
        for b in 0 ..< 256 {
            let name = Array(String(format: "<0x%02X>", b).utf8)
            if let id = vocabulary[name] {
                byteTokenIDs[b] = id
                byteValueOfID[id] = UInt8(b)
            }
        }
        if byteFallback, byteTokenIDs.contains(where: { $0 == nil }) {
            throw Failure("tokenizer.json: byte_fallback is set but the vocabulary lacks some <0xXX> tokens")
        }

        // Merges, as ids.
        var rank: Int32 = 0
        for item in model["merges"]?.array ?? [] {
            var a: String?, b: String?
            if let pair = item.array, pair.count == 2 {
                a = pair[0].string
                b = pair[1].string
            } else if let s = item.string {
                // "left right": the first space separates the two (a token never contains one in these vocabularies).
                let parts = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 { a = String(parts[0]); b = String(parts[1]) }
            }
            guard let a, let b else { throw Failure("tokenizer.json: unknown merges format") }
            defer { rank += 1 }
            let ab = Array(a.utf8)
            let bb = Array(b.utf8)
            guard let ia = vocabulary[ab], let ib = vocabulary[bb], let im = vocabulary[ab + bb] else { continue }
            let key = UInt64(UInt32(ia)) << 32 | UInt64(UInt32(ib))
            if merges[key] == nil { merges[key] = (rank, Int32(im)) }
        }

        if let unk = model["unk_token"]?.string { unknownToken = unk; unknownID = vocabulary[Array(unk.utf8)] }

        try parseNormalizer(root["normalizer"])
        try parsePreTokenizer(root["pre_tokenizer"])
        try parsePostProcessor(root["post_processor"])
        try parseDecoder(root["decoder"])

        if let configuration {
            let c = try GGUFJSON.parse(configuration)
            func name(_ key: String) -> String? { c[key]?.string ?? c[key]?["content"]?.string }
            bosToken = name("bos_token")
            eosToken = name("eos_token")
            padToken = name("pad_token")
            if unknownToken == nil { unknownToken = name("unk_token") }
        }
    }

    // MARK: Parsing the pipeline

    private func scalars(of pattern: GGUFJSON?) throws -> [Unicode.Scalar] {
        guard let s = pattern?["String"]?.string else {
            throw Failure("tokenizer.json: only string patterns are supported")
        }
        return Array(s.unicodeScalars)
    }

    private func parseNormalizer(_ n: GGUFJSON?) throws {
        guard let n, n.object != nil else { return }
        switch n["type"]?.string {
        case "Replace":
            normalizers.append(.replace(pattern: try scalars(of: n["pattern"]), content: Array((n["content"]?.string ?? "").unicodeScalars)))
        case "Prepend":
            normalizers.append(.prepend(Array((n["prepend"]?.string ?? "").unicodeScalars)))
        case "Sequence":
            for item in n["normalizers"]?.array ?? [] { try parseNormalizer(item) }
        default:
            throw Failure("tokenizer.json: normalizer \(n["type"]?.string ?? "?") is not supported")
        }
    }

    private func parsePreTokenizer(_ p: GGUFJSON?) throws {
        guard let p, p.object != nil else { return }
        switch p["type"]?.string {
        case "Split":
            let behavior: Split.Behavior
            switch p["behavior"]?.string {
            case "Removed": behavior = .removed
            case "Isolated": behavior = .isolated
            case "MergedWithPrevious": behavior = .mergedWithPrevious
            case "MergedWithNext": behavior = .mergedWithNext
            case "Contiguous": behavior = .contiguous
            default: throw Failure("tokenizer.json: unknown Split behavior")
            }
            splits.append(Split(pattern: try scalars(of: p["pattern"]), behavior: behavior, invert: p["invert"]?.bool ?? false))
        case "Sequence":
            for item in p["pretokenizers"]?.array ?? [] { try parsePreTokenizer(item) }
        default:
            throw Failure("tokenizer.json: pre-tokenizer \(p["type"]?.string ?? "?") is not supported")
        }
    }

    private func parsePostProcessor(_ p: GGUFJSON?) throws {
        guard let p, p.object != nil else { return }
        guard p["type"]?.string == "TemplateProcessing" else {
            throw Failure("tokenizer.json: post-processor \(p["type"]?.string ?? "?") is not supported")
        }
        let specials = p["special_tokens"]
        for item in p["single"]?.array ?? [] {
            if let sp = item["SpecialToken"]?["id"]?.string {
                guard let ids = specials?[sp]?["ids"]?.array?.compactMap({ $0.int }), !ids.isEmpty else {
                    throw Failure("tokenizer.json: unknown special token \(sp) in post-processor")
                }
                templateSingle.append(contentsOf: ids.map { $0 })
            } else {
                templateSingle.append(-1) // the sequence itself
            }
        }
    }

    private func parseDecoder(_ d: GGUFJSON?) throws {
        guard let d, d.object != nil else { return }
        switch d["type"]?.string {
        case "Sequence":
            for item in d["decoders"]?.array ?? [] { try parseDecoder(item) }
        case "Replace":
            decoderSteps.append(.replace(pattern: d["pattern"]?["String"]?.string ?? "", content: d["content"]?.string ?? ""))
        case "ByteFallback": decoderSteps.append(.byteFallback)
        case "Fuse": decoderSteps.append(.fuse)
        default:
            throw Failure("tokenizer.json: decoder \(d["type"]?.string ?? "?") is not supported")
        }
    }

    // MARK: Lookup

    public func tokenID(for token: String) -> Int? { vocabulary[Array(token.utf8)] }

    /// The token's text as stored (before the decoder), or nil for an unused id.
    public func token(for id: Int) -> String? {
        guard id >= 0, id < tokenBytes.count, let b = tokenBytes[id] else { return nil }
        return String(decoding: b, as: UTF8.self)
    }

    public func isSpecial(_ id: Int) -> Bool { specialIDs.contains(id) }

    /// Ids of the added tokens marked special (`<bos>`, `<turn|>`, ...), ascending.
    public var specialTokenIDs: [Int] { specialIDs.sorted() }

    /// Ids of `<0x00>` ... `<0xFF>` in byte order (empty when the vocabulary has no byte-fallback tokens).
    public var byteFallbackTokenIDs: [Int] { byteTokenIDs.compactMap { $0 } }

    /// A SHA-256 over every token's bytes in id order (and the vocabulary size), so a file made for one
    /// vocabulary can tell it is being used with another.
    public func vocabularyFingerprint() -> String {
        var fp = Fingerprint()
        fp.add("vocabulary \(tokenBytes.count)")
        for b in tokenBytes { fp.add(Data(b ?? [])) }
        return fp.hex
    }

    // MARK: Encode

    /// Marks a special token's text as plain text for ``encode(templateOutput:)``. A Unicode noncharacter, so it
    /// never occurs in real text.
    static let escapeMarker = Unicode.Scalar(0xFDD0)!

    /// `text` with every special token's string (`<turn|>`, `<|channel>`, `<bos>`, ...) broken by an invisible
    /// marker after its first character, so that ``encode(templateOutput:)`` encodes it as ordinary characters
    /// and not as the control token, and so a chat template that searches message content for a control
    /// string (Gemma's thinking stripper) does not find it either.
    ///
    /// This is for text that came from a user, an assistant or a tool, never for the chat template's own
    /// markup. Any marker already in `text` is dropped first, so the input cannot forge an escape.
    public func escapingSpecialTokens(in text: String) -> String {
        let all = Array(text.unicodeScalars.filter { $0 != Self.escapeMarker })
        var out = String.UnicodeScalarView()
        var i = 0
        while i < all.count {
            if let candidates = addedByFirst[all[i]],
                let hit = candidates.first(where: { c in
                    c.special && i + c.scalars.count <= all.count && all[i ..< i + c.scalars.count].elementsEqual(c.scalars)
                })
            {
                out.append(all[i])
                out.append(Self.escapeMarker)
                out.append(contentsOf: all[i + 1 ..< i + hit.scalars.count])
                i += hit.scalars.count
            } else {
                out.append(all[i])
                i += 1
            }
        }
        return String(out)
    }

    /// Token ids for a chat template's rendering whose message text went through
    /// ``escapingSpecialTokens(in:)``: the template's own control tokens become single ids, an escaped one
    /// is encoded as the plain characters it spells.
    public func encode(templateOutput text: String) -> [Int] {
        encode(text, addSpecialTokens: false, honorEscapes: true)
    }

    /// Token ids for `text`. Added tokens (`<bos>`, `<|turn>`, ...) written in the text become their single
    /// ids. With `addSpecialTokens`, the post-processor's template is applied (Gemma's has none).
    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        encode(text, addSpecialTokens: addSpecialTokens, honorEscapes: false)
    }

    private func encode(_ text: String, addSpecialTokens: Bool, honorEscapes: Bool) -> [Int] {
        var ids: [Int] = []
        let all = Array(text.unicodeScalars)
        var segment: [Unicode.Scalar] = []
        func flush() {
            if !segment.isEmpty { ids.append(contentsOf: encodePlain(segment)); segment.removeAll(keepingCapacity: true) }
        }
        var i = 0
        while i < all.count {
            if let candidates = addedByFirst[all[i]],
                let hit = candidates.first(where: { c in
                    i + c.scalars.count <= all.count && all[i ..< i + c.scalars.count].elementsEqual(c.scalars)
                })
            {
                flush()
                ids.append(hit.id)
                i += hit.scalars.count
            } else {
                if !(honorEscapes && all[i] == Self.escapeMarker) { segment.append(all[i]) }
                i += 1
            }
        }
        flush()
        if addSpecialTokens, !templateSingle.isEmpty {
            var out: [Int] = []
            for t in templateSingle { if t == -1 { out.append(contentsOf: ids) } else { out.append(t) } }
            return out
        }
        return ids
    }

    private func normalize(_ input: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var s = input
        for n in normalizers {
            switch n {
            case .prepend(let p): s = p + s
            case .replace(let pattern, let content):
                guard !pattern.isEmpty else { continue }
                var out: [Unicode.Scalar] = []
                out.reserveCapacity(s.count)
                var i = 0
                while i < s.count {
                    if i + pattern.count <= s.count, s[i ..< i + pattern.count].elementsEqual(pattern) {
                        out.append(contentsOf: content)
                        i += pattern.count
                    } else {
                        out.append(s[i])
                        i += 1
                    }
                }
                s = out
            }
        }
        return s
    }

    private func preTokenize(_ input: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var pieces: [[Unicode.Scalar]] = [input]
        for sp in splits {
            var next: [[Unicode.Scalar]] = []
            for piece in pieces { next.append(contentsOf: split(piece, sp)) }
            pieces = next
        }
        return pieces
    }

    /// `Split` with a string pattern, per Hugging Face's `SplitDelimiterBehavior`.
    private func split(_ s: [Unicode.Scalar], _ sp: Split) -> [[Unicode.Scalar]] {
        guard !sp.pattern.isEmpty, !s.isEmpty else { return s.isEmpty ? [] : [s] }
        // (range, isMatch) pieces covering s.
        var parts: [(r: Range<Int>, match: Bool)] = []
        var last = 0
        var i = 0
        while i < s.count {
            if i + sp.pattern.count <= s.count, s[i ..< i + sp.pattern.count].elementsEqual(sp.pattern) {
                if last < i { parts.append((last ..< i, false)) }
                parts.append((i ..< i + sp.pattern.count, true))
                i += sp.pattern.count
                last = i
            } else {
                i += 1
            }
        }
        if last < s.count { parts.append((last ..< s.count, false)) }
        if sp.invert { parts = parts.map { ($0.r, !$0.match) } }
        var out: [[Unicode.Scalar]] = []
        switch sp.behavior {
        case .removed:
            for p in parts where !p.match { out.append(Array(s[p.r])) }
        case .isolated:
            for p in parts { out.append(Array(s[p.r])) }
        case .contiguous:
            var run: [Unicode.Scalar] = []
            var runIsMatch: Bool?
            for p in parts {
                if runIsMatch == p.match { run += s[p.r] } else {
                    if !run.isEmpty { out.append(run) }
                    run = Array(s[p.r])
                    runIsMatch = p.match
                }
            }
            if !run.isEmpty { out.append(run) }
        case .mergedWithPrevious:
            for p in parts {
                if p.match, !out.isEmpty { out[out.count - 1] += s[p.r] } else { out.append(Array(s[p.r])) }
            }
        case .mergedWithNext:
            var carry: [Unicode.Scalar] = []
            for p in parts {
                if p.match { carry += s[p.r] } else { out.append(carry + s[p.r]); carry = [] }
            }
            if !carry.isEmpty { out.append(carry) }
        }
        return out
    }

    private func encodePlain(_ raw: [Unicode.Scalar]) -> [Int] {
        var ids: [Int] = []
        for word in preTokenize(normalize(raw)) where !word.isEmpty {
            ids.append(contentsOf: bpe(word))
        }
        return ids
    }

    /// Byte-pair merging of one pre-token: a doubly linked list of symbols and a heap of candidate merges
    /// by rank, so a long segment (a whole prompt, when the pre-tokenizer splits nothing) costs
    /// `n log n`.
    private func bpe(_ word: [Unicode.Scalar]) -> [Int] {
        var symbols: [Int32] = []
        symbols.reserveCapacity(word.count)
        var unknownRun = false
        for sc in word {
            var buf = String.UnicodeScalarView()
            buf.append(sc)
            let bytes = Array(String(buf).utf8)
            if let id = vocabulary[bytes] {
                symbols.append(Int32(id))
                unknownRun = false
            } else if byteFallback {
                for b in bytes { symbols.append(Int32(byteTokenIDs[Int(b)]!)) }
                unknownRun = false
            } else if let unk = unknownID {
                // fuse_unk: a run of unknown characters becomes one unknown token.
                if !unknownRun { symbols.append(Int32(unk)) }
                unknownRun = true
            }
        }
        let n = symbols.count
        if n < 2 { return symbols.map(Int.init) }
        var prev = (0 ..< n).map { Int32($0 - 1) }
        var next = (0 ..< n).map { Int32($0 + 1) }
        next[n - 1] = -1
        var alive = [Bool](repeating: true, count: n)

        struct Candidate: Comparable {
            var rank: Int32
            var left: Int32
            var leftID: Int32
            var rightID: Int32
            static func < (a: Candidate, b: Candidate) -> Bool {
                a.rank != b.rank ? a.rank < b.rank : a.left < b.left
            }
        }
        var heap: [Candidate] = []
        func push(_ c: Candidate) {
            heap.append(c)
            var i = heap.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                if heap[i] < heap[p] { heap.swapAt(i, p); i = p } else { break }
            }
        }
        func pop() -> Candidate? {
            guard !heap.isEmpty else { return nil }
            let top = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var i = 0
                while true {
                    let l = 2 * i + 1, r = l + 1
                    var m = i
                    if l < heap.count, heap[l] < heap[m] { m = l }
                    if r < heap.count, heap[r] < heap[m] { m = r }
                    if m == i { break }
                    heap.swapAt(i, m)
                    i = m
                }
            }
            return top
        }
        func consider(_ l: Int) {
            let r = Int(next[l])
            guard r >= 0 else { return }
            let key = UInt64(UInt32(symbols[l])) << 32 | UInt64(UInt32(symbols[r]))
            if let m = merges[key] { push(Candidate(rank: m.rank, left: Int32(l), leftID: symbols[l], rightID: symbols[r])) }
        }
        for i in 0 ..< n - 1 { consider(i) }
        while let c = pop() {
            let l = Int(c.left)
            // Stale: either side changed since this candidate was pushed.
            guard alive[l], symbols[l] == c.leftID else { continue }
            let r = Int(next[l])
            guard r >= 0, alive[r], symbols[r] == c.rightID else { continue }
            let key = UInt64(UInt32(c.leftID)) << 32 | UInt64(UInt32(c.rightID))
            guard let m = merges[key] else { continue }
            symbols[l] = m.merged
            alive[r] = false
            let rn = next[r]
            next[l] = rn
            if rn >= 0 { prev[Int(rn)] = Int32(l) }
            if prev[l] >= 0 { consider(Int(prev[l])) }
            consider(l)
        }
        var out: [Int] = []
        var i = 0
        while i >= 0 && i < n {
            out.append(Int(symbols[i]))
            i = Int(next[i])
        }
        return out
    }

    private var tokenBytesTable: (count: Int, table: [[UInt8]?])?
    private let tableLock = NSLock()

    /// For each id below `count`, the bytes the token contributes to the output, or nil for a special
    /// token, an unused id, or a token that is not whole UTF-8 (a lone byte-fallback byte). Built once and
    /// kept: it walks the whole vocabulary.
    func wholeUTF8TokenBytes(count: Int) -> [[UInt8]?] {
        tableLock.lock()
        defer { tableLock.unlock() }
        if let cached = tokenBytesTable, cached.count == count { return cached.table }
        var t = [[UInt8]?](repeating: nil, count: count)
        for id in 0 ..< min(vocabularySize, count) where !isSpecial(id) {
            let b = decodedBytes(of: id)
            if !b.isEmpty, String(validatingUTF8: b.map { CChar(bitPattern: $0) } + [0]) != nil { t[id] = b }
        }
        tokenBytesTable = (count, t)
        return t
    }

    // MARK: Decode

    /// The bytes one token contributes to the output, after the decoder's per-token steps.
    func decodedBytes(of id: Int) -> [UInt8] {
        guard id >= 0, id < tokenBytes.count, let raw = tokenBytes[id] else { return [] }
        var steps = decoderSteps
        if steps.isEmpty { steps = [] }
        if let b = byteValueOfID[id], steps.contains(where: { if case .byteFallback = $0 { true } else { false } }) {
            return [b]
        }
        var text = String(decoding: raw, as: UTF8.self)
        for step in steps {
            switch step {
            case .replace(let pattern, let content):
                if !pattern.isEmpty { text = replacingScalars(text, pattern, content) }
            default: break
            }
        }
        return Array(text.utf8)
    }

    private func replacingScalars(_ text: String, _ pattern: String, _ content: String) -> String {
        let p = Array(pattern.unicodeScalars)
        let s = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        while i < s.count {
            if i + p.count <= s.count, s[i ..< i + p.count].elementsEqual(p) {
                out.append(contentsOf: content.unicodeScalars)
                i += p.count
            } else {
                out.append(s[i])
                i += 1
            }
        }
        return String(out)
    }

    /// Text for `ids`. Special tokens are dropped when `skipSpecialTokens` is set. Bytes of byte-fallback
    /// tokens are fused and decoded as UTF-8 (an incomplete sequence becomes U+FFFD).
    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var bytes: [UInt8] = []
        for id in ids where !(skipSpecialTokens && specialIDs.contains(id)) {
            bytes.append(contentsOf: decodedBytes(of: id))
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// A decoder that takes one token at a time and returns the text that is now complete: bytes of an
    /// unfinished UTF-8 character are held back until the rest arrives.
    public func makeStreamingDecoder(skipSpecialTokens: Bool = false) -> StreamingDecoder {
        StreamingDecoder(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
    }

    public struct StreamingDecoder {
        let tokenizer: BPETokenizer
        let skipSpecialTokens: Bool
        private var pending: [UInt8] = []

        init(tokenizer: BPETokenizer, skipSpecialTokens: Bool) {
            self.tokenizer = tokenizer
            self.skipSpecialTokens = skipSpecialTokens
        }

        /// Text completed by `id` (possibly empty).
        public mutating func append(_ id: Int) -> String {
            if skipSpecialTokens, tokenizer.isSpecial(id) { return "" }
            pending.append(contentsOf: tokenizer.decodedBytes(of: id))
            let cut = Self.completePrefixLength(pending)
            guard cut > 0 else { return "" }
            let text = String(decoding: pending[0 ..< cut], as: UTF8.self)
            pending.removeFirst(cut)
            return text
        }

        /// Whatever is still held, with U+FFFD for an incomplete character.
        public mutating func finish() -> String {
            defer { pending.removeAll() }
            return String(decoding: pending, as: UTF8.self)
        }

        /// Length of the longest prefix of `b` that does not end inside a multi-byte character.
        static func completePrefixLength(_ b: [UInt8]) -> Int {
            var i = b.count
            // Look back at most 3 bytes for a lead byte whose sequence is not finished.
            var back = 0
            while i > 0, back < 4 {
                i -= 1
                back += 1
                let c = b[i]
                if c & 0xC0 == 0x80 { continue } // continuation
                let need: Int
                if c & 0x80 == 0 { need = 1 } else if c & 0xE0 == 0xC0 { need = 2 } else if c & 0xF0 == 0xE0 { need = 3 } else if c & 0xF8 == 0xF0 { need = 4 } else { need = 1 }
                return b.count - i >= need ? b.count : i
            }
            return b.count
        }
    }
}

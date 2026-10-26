import Foundation

/// The fixed part of a model's allowed tokens: a list of token ids made for one tokenizer, kept in a JSON file.
///
/// A ``VocabularyWindow`` loads these rows once and adds, for every request, the rows of the tokens in that
/// request's context. The list is built by ``StaticVocabularyBuilder`` (`shears-cli vocab build`) from
///
/// - the **byte-fallback** tokens (`<0x00>` ... `<0xFF>`), so any text can be spelled;
/// - the **special** tokens (`<eos>`, `<turn|>`, ...);
/// - **JSON, verdict and punctuation** tokens: the pieces of a classifier's answer, the digits, and every
///   printable ASCII character alone and after a space;
/// - a **broad English and code base**: the most frequent tokens of a text corpus, then the pieces of a
///   dictionary's words, up to a chosen size.
public struct StaticVocabulary: Codable, Equatable {
    public static let formatName = "shears-vocab-subset-v1"

    public var format: String
    public var vocabularySize: Int
    /// ``BPETokenizer/vocabularyFingerprint()`` of the tokenizer the ids belong to.
    public var tokenizerFingerprint: String
    /// Token ids, ascending, without duplicates.
    public var ids: [Int]
    /// How many ids each always-included section holds (`bytes`, `special`, `json`, `ascii`), and `corpus` and
    /// `english` and `corpus` for what the ranking added.
    public var sections: [String: Int]
    public var build: BuildSummary?

    public struct BuildSummary: Codable, Equatable {
        public var requestedSize: Int
        public var corpusFiles: Int
        public var corpusBytes: Int
        public var corpusTokens: Int
        public var dictionaryWords: Int
        /// Share of a held-out part of the corpus (every tenth file) whose tokens are in the list, 0 to 1.
        public var heldOutCoverage: Double?
        /// Free-text note: where the corpus came from.
        public var note: String?
    }

    public init(
        vocabularySize: Int, tokenizerFingerprint: String, ids: [Int], sections: [String: Int], build: BuildSummary? = nil
    ) {
        format = Self.formatName
        self.vocabularySize = vocabularySize
        self.tokenizerFingerprint = tokenizerFingerprint
        self.ids = Array(Set(ids)).sorted()
        self.sections = sections
        self.build = build
    }

    public static func load(_ url: URL) throws -> StaticVocabulary {
        let v = try JSONDecoder().decode(StaticVocabulary.self, from: Data(contentsOf: url))
        guard v.format == formatName else {
            throw InferenceError.invalid("\(url.lastPathComponent) is not a \(formatName) file (format \(v.format))")
        }
        return v
    }

    public func write(to url: URL) throws {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .prettyPrinted]
        try e.encode(self).write(to: url, options: .atomic)
    }

    /// Throws unless the list was made for `tokenizer`'s vocabulary.
    public func validate(for tokenizer: BPETokenizer) throws {
        guard vocabularySize == tokenizer.vocabularySize, tokenizerFingerprint == tokenizer.vocabularyFingerprint() else {
            throw InferenceError.invalid("this static vocabulary was made for a different tokenizer")
        }
    }
}

/// Builds a ``StaticVocabulary``.
public final class StaticVocabularyBuilder {
    public let tokenizer: BPETokenizer
    /// Token counts over the training part of the corpus.
    public private(set) var counts: [Int]
    /// Token counts over the held-out part (every tenth file).
    public private(set) var heldOutCounts: [Int]
    /// How many dictionary-word forms contain each token.
    public private(set) var dictionaryCredit: [Int]
    /// How many dictionary-word forms are exactly this one token (a whole English word, or an inflection of one).
    public private(set) var wholeWordCredit: [Int]
    public private(set) var corpusFiles = 0
    public private(set) var corpusBytes = 0
    public private(set) var dictionaryWords = 0

    public init(tokenizer: BPETokenizer) {
        self.tokenizer = tokenizer
        let n = tokenizer.vocabularySize
        counts = Array(repeating: 0, count: n)
        heldOutCounts = Array(repeating: 0, count: n)
        dictionaryCredit = Array(repeating: 0, count: n)
        wholeWordCredit = Array(repeating: 0, count: n)
    }

    // MARK: Always-included tokens

    /// Texts whose tokens are always in the list: the shape of a classifier's JSON answer, its four verdicts, a
    /// reply's usual openers. (More texts can be added with ``mandatoryIDs(extraTexts:)``.)
    public static let defaultJSONTexts: [String] = [
        #"{"state": "ON_TASK", "confidence": 0.9, "reason": "You are working on the declared task."}"#,
        #"{"state": "NOT_ALIGNED", "confidence": 0.85, "reason": "You are reading unrelated pages."}"#,
        #"{"state": "IDLE", "confidence": 1.0, "reason": "No input for a while."}"#,
        #"{"state": "NO_TASK", "confidence": 0.75, "reason": ""}"#,
        "{\"state\":\"ON_TASK\",\"confidence\":0.95,\"reason\":\"\"}\n{\n  \"state\": \"NOT_ALIGNED\",\n  \"confidence\": 0.5,\n  \"reason\": \"\"\n}",
        "```json\n{\"state\": \"IDLE\"}\n```\n```",
        "ON_TASK NOT_ALIGNED IDLE NO_TASK ON TASK NOT ALIGNED NO ID LE _ \"state\" \"confidence\" \"reason\"",
        " ON_TASK NOT_ALIGNED IDLE NO_TASK",
        "true false null [] {} , : \" \\ \\n \\\" \\\\ \\u00e9 0123456789 0.0 1.0 0.5 .5 1 0 e E - + .",
    ]

    /// The ids that are always in the list, by section.
    public func mandatorySections(extraTexts: [String] = []) -> [(name: String, ids: Set<Int>)] {
        var out: [(String, Set<Int>)] = []
        out.append(("bytes", Set(tokenizer.byteFallbackTokenIDs)))
        out.append(("special", Set(tokenizer.specialTokenIDs)))
        var json = Set<Int>()
        for t in Self.defaultJSONTexts + extraTexts { json.formUnion(tokenizer.encode(t)) }
        out.append(("json", json))
        // Every printable ASCII character alone and after a space (and the same inside a word, via bytes):
        // anything ASCII can be spelled without a rare piece.
        var ascii = Set<Int>()
        for u in 0x20 ..< 0x7F {
            let c = String(UnicodeScalar(UInt8(u)))
            ascii.formUnion(tokenizer.encode(c))
            ascii.formUnion(tokenizer.encode(" " + c))
        }
        ascii.formUnion(tokenizer.encode("\n\t\n\n  "))
        out.append(("ascii", ascii))
        return out
    }

    // MARK: Counting

    /// Whether a file is read as corpus text: by extension (the usual text and source formats), skipping what holds NUL bytes.
    public static let textExtensions: Set<String> = [
        "txt", "md", "markdown", "rst", "text", "json", "jsonl", "csv", "tsv", "html", "htm", "xml", "yaml", "yml", "toml",
        "swift", "py", "js", "ts", "tsx", "jsx", "c", "h", "cc", "cpp", "hpp", "m", "mm", "java", "kt", "go", "rs", "rb",
        "sh", "zsh", "sql", "css", "tex", "cfg", "ini", "log",
    ]

    /// Lists the corpus files under `paths` (a file is taken as it is; a directory is walked), at most `maxFileBytes` each.
    public static func corpusFiles(under paths: [URL], maxFileBytes: Int = 4 << 20) -> [URL] {
        var out: [URL] = []
        let fm = FileManager.default
        for p in paths {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: p.path, isDirectory: &isDir) else { continue }
            if !isDir.boolValue { out.append(p); continue }
            guard let e = fm.enumerator(at: p, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else { continue }
            for case let u as URL in e {
                guard textExtensions.contains(u.pathExtension.lowercased()) else { continue }
                let v = try? u.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                guard v?.isRegularFile == true, let size = v?.fileSize, size > 0, size <= maxFileBytes else { continue }
                out.append(u)
            }
        }
        return out.sorted { $0.path < $1.path }
    }

    /// Counts the tokens of `files`, in parallel; a tenth of the files (every tenth, by name order) is kept apart as
    /// held-out text for ``coverage(sizes:selection:)``. Stops after about `maxTotalBytes`.
    public func addCorpus(files: [URL], maxTotalBytes: Int = .max, log: ((String) -> Void)? = nil) {
        let lock = NSLock()
        var budget = maxTotalBytes
        var taken: [(Int, URL)] = []
        for (i, f) in files.enumerated() {
            let size = (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if budget <= 0 { break }
            budget -= size
            taken.append((i, f))
        }
        let n = tokenizer.vocabularySize
        var done = 0
        DispatchQueue.concurrentPerform(iterations: taken.count) { k in
            let (i, url) = taken[k]
            guard let data = try? Data(contentsOf: url), !data.contains(0) else { return }
            let text = String(decoding: data, as: UTF8.self)
            var local = [Int32](repeating: 0, count: n)
            for chunk in Self.chunks(of: text, limit: 6000) {
                for id in tokenizer.encode(chunk) where id < n { local[id] += 1 }
            }
            lock.lock()
            let held = i % 10 == 9
            for id in 0 ..< n where local[id] != 0 {
                if held { heldOutCounts[id] += Int(local[id]) } else { counts[id] += Int(local[id]) }
            }
            corpusFiles += 1
            corpusBytes += data.count
            done += 1
            if done % 500 == 0 { log?("counted \(done) of \(taken.count) files") }
            lock.unlock()
        }
    }

    /// Splits text into pieces of at most `limit` characters, at line ends where it can.
    static func chunks(of text: String, limit: Int) -> [String] {
        var out: [String] = []
        var current = ""
        current.reserveCapacity(limit)
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if current.count + line.count + 1 > limit, !current.isEmpty {
                out.append(current)
                current = ""
            }
            // A single very long line is cut into pieces.
            var rest = Substring(line)
            while rest.count > limit {
                out.append(String(rest.prefix(limit)))
                rest = rest.dropFirst(limit)
            }
            current += rest
            current += "\n"
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    /// Endings tried on every dictionary word, so the tokens of inflected forms ("aligns", "aligned", "aligning") get
    /// credit though a dictionary lists only "align". Made-up forms cost nothing: they tokenize into pieces already
    /// credited.
    public static let inflections = ["", "s", "es", "ed", "d", "ing", "er", "ers", "ly", "est", "ment", "ness"]

    /// Credits the tokens that spell each dictionary word (after a space, lower case and capitalised, with the
    /// ``inflections``), and separately the tokens that are a whole such form. Run in parallel.
    public func addDictionary(words: [String], inflect: Bool = true) {
        let lock = NSLock()
        let n = tokenizer.vocabularySize
        let endings = inflect ? Self.inflections : [""]
        let batch = 2000
        let batches = (words.count + batch - 1) / batch
        DispatchQueue.concurrentPerform(iterations: batches) { b in
            var pieces = [Int32](repeating: 0, count: n)
            var whole = [Int32](repeating: 0, count: n)
            var counted = 0
            for w in words[(b * batch) ..< min(words.count, (b + 1) * batch)] where !w.isEmpty {
                counted += 1
                var stems = [w]
                if w.hasSuffix("e") { stems.append(String(w.dropLast())) }
                for stem in stems {
                    for ending in endings {
                        if stem != w, ending != "ing", ending != "ed", ending != "er", ending != "ers", ending != "est" { continue }
                        let word = stem + ending
                        var forms = [word, " " + word]
                        let capital = " " + word.prefix(1).uppercased() + word.dropFirst()
                        if capital != forms[1] { forms.append(capital) }
                        for f in forms {
                            let ids = tokenizer.encode(f)
                            for id in Set(ids) where id < n { pieces[id] += 1 }
                            if ids.count == 1, ids[0] < n { whole[ids[0]] += 1 }
                        }
                    }
                }
            }
            lock.lock()
            for id in 0 ..< n {
                dictionaryCredit[id] += Int(pieces[id])
                wholeWordCredit[id] += Int(whole[id])
            }
            dictionaryWords += counted
            lock.unlock()
        }
    }

    // MARK: Selecting

    /// Candidate ids best first: seen in the corpus (most frequent first), then only in the dictionary (most credited first).
    public func ranking(excluding excluded: Set<Int>) -> [Int] {
        var ids: [Int] = []
        for id in 0 ..< counts.count where !excluded.contains(id) && (counts[id] > 0 || dictionaryCredit[id] > 0) {
            ids.append(id)
        }
        ids.sort { a, b in
            if counts[a] != counts[b] { return counts[a] > counts[b] }
            if dictionaryCredit[a] != dictionaryCredit[b] { return dictionaryCredit[a] > dictionaryCredit[b] }
            return a < b
        }
        return ids
    }

    /// Whole English words (a token that is a dictionary word or an inflection of one), seen in the corpus first by
    /// frequency, then the rest by how many forms spell them.
    public func englishRanking(excluding excluded: Set<Int>) -> [Int] {
        var ids: [Int] = []
        for id in 0 ..< counts.count where !excluded.contains(id) && wholeWordCredit[id] > 0 { ids.append(id) }
        ids.sort { a, b in
            if counts[a] != counts[b] { return counts[a] > counts[b] }
            if wholeWordCredit[a] != wholeWordCredit[b] { return wholeWordCredit[a] > wholeWordCredit[b] }
            return a < b
        }
        return ids
    }

    /// Fills `base` up to `size` ids: `englishShare` of the room from ``englishRanking(excluding:)`` and the rest from
    /// ``ranking(excluding:)`` (the most frequent tokens of the corpus, whatever they are: code, punctuation, names),
    /// alternating so the list is balanced at every size. Returns the number taken from each.
    func fill(_ base: Set<Int>, to size: Int, englishShare: Double) -> (ids: Set<Int>, english: Int, general: Int) {
        var chosen = base
        let english = englishRanking(excluding: base)
        let general = ranking(excluding: base)
        var ie = 0, ig = 0, takenE = 0, takenG = 0
        while chosen.count < size {
            let wantEnglish = Double(takenE) < englishShare * Double(takenE + takenG + 1)
            var progressed = false
            if wantEnglish {
                while ie < english.count, chosen.contains(english[ie]) { ie += 1 }
                if ie < english.count { chosen.insert(english[ie]); takenE += 1; progressed = true }
            }
            if !progressed {
                while ig < general.count, chosen.contains(general[ig]) { ig += 1 }
                if ig < general.count { chosen.insert(general[ig]); takenG += 1; progressed = true }
            }
            if !progressed {
                // The preferred list is empty: take from the other one.
                while ie < english.count, chosen.contains(english[ie]) { ie += 1 }
                if ie < english.count { chosen.insert(english[ie]); takenE += 1 } else { break }
            }
        }
        return (chosen, takenE, takenG)
    }

    public struct Selection {
        public var vocabulary: StaticVocabulary
        public var mandatoryCount: Int
        /// Ranked ids that were not taken (the list is shorter than asked when the corpus and dictionary run out).
        public var shortfall: Int
    }

    /// The list of `size` ids: the always-included tokens, then the best-ranked others.
    public func select(size: Int, extraTexts: [String] = [], englishShare: Double = 0.5, note: String? = nil) throws -> Selection {
        let sections = mandatorySections(extraTexts: extraTexts)
        var chosen = Set<Int>()
        var sectionCounts: [String: Int] = [:]
        for (name, ids) in sections {
            let fresh = ids.subtracting(chosen)
            sectionCounts[name] = fresh.count
            chosen.formUnion(ids)
        }
        guard size >= chosen.count else {
            throw InferenceError.invalid("--size \(size) is below the \(chosen.count) tokens that are always included")
        }
        let mandatory = chosen.count
        let filled = fill(chosen, to: size, englishShare: englishShare)
        chosen = filled.ids
        sectionCounts["english"] = filled.english
        sectionCounts["corpus"] = filled.general
        let corpusTokens = counts.reduce(0, +) + heldOutCounts.reduce(0, +)
        let held = heldOutCoverage(of: chosen)
        let summary = StaticVocabulary.BuildSummary(
            requestedSize: size, corpusFiles: corpusFiles, corpusBytes: corpusBytes, corpusTokens: corpusTokens,
            dictionaryWords: dictionaryWords, heldOutCoverage: held, note: note)
        let v = StaticVocabulary(
            vocabularySize: tokenizer.vocabularySize, tokenizerFingerprint: tokenizer.vocabularyFingerprint(),
            ids: Array(chosen), sections: sectionCounts, build: summary)
        return Selection(vocabulary: v, mandatoryCount: mandatory, shortfall: max(0, size - chosen.count))
    }

    /// Share of the held-out corpus tokens that `set` holds.
    public func heldOutCoverage(of set: Set<Int>) -> Double? {
        let total = heldOutCounts.reduce(0, +)
        guard total > 0 else { return nil }
        var inside = 0
        for id in set where id < heldOutCounts.count { inside += heldOutCounts[id] }
        return Double(inside) / Double(total)
    }

    /// For each size, the share of held-out (and of training) corpus tokens its list would hold, so a size can be chosen.
    public func coverage(sizes: [Int], extraTexts: [String] = [], englishShare: Double = 0.5) -> [(size: Int, rows: Int, heldOut: Double?, training: Double)] {
        var base = Set<Int>()
        for (_, ids) in mandatorySections(extraTexts: extraTexts) { base.formUnion(ids) }
        let trainTotal = max(1, counts.reduce(0, +))
        return sizes.sorted().map { size in
            let set = fill(base, to: size, englishShare: englishShare).ids
            var inside = 0
            for id in set { inside += counts[id] }
            return (size, set.count, heldOutCoverage(of: set), Double(inside) / Double(trainTotal))
        }
    }
}

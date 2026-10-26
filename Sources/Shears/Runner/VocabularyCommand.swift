import Foundation

/// `shears-cli vocab`: builds and inspects the static token list a vocabulary window starts from.
///
/// ```
/// shears-cli vocab build --tokenizer DIR --corpus PATH [--corpus PATH ...] --out FILE [--size N]
///                        [--dictionary FILE ...] [--max-mb N] [--mandatory-text FILE] [--sizes 4096,8192,...] [--note TEXT]
/// shears-cli vocab info FILE [--tokenizer DIR]
/// ```
public enum VocabularyCommand {
    public static let usage = """
        usage:
          shears-cli vocab build --tokenizer DIR --corpus PATH [--corpus PATH ...] --out FILE [--size N]
                                 [--dictionary FILE ...] [--max-mb N] [--mandatory-text FILE] [--sizes N,N,...] [--note TEXT]
                                 [--emit SIZE:FILE ...] [--english-share F] [--no-punctuation true] [--max-file-mb N]
          shears-cli vocab coverage --tokenizer DIR --prompts FILE --vocab FILE [--vocab FILE ...] [--limit N] [--show N]
          shears-cli vocab info FILE [--tokenizer DIR]

        build counts the tokens of a text corpus (files, or directories walked for text and source files) and writes the
        --size most useful token ids (default 32768) to FILE: the always-included tokens (the 256 byte-fallback tokens, the
        special tokens, JSON/verdict pieces, every printable ASCII character) first, then the corpus's most frequent
        tokens, then the pieces of --dictionary words (one per line) for broad coverage. Every tenth corpus file is held
        out and the share of its tokens the list holds is reported, also for the --sizes given, to help choose a size.
        --max-mb limits the corpus read (default 512); files over --max-file-mb (default 32) are skipped. --mandatory-text adds the tokens of a text file to the always-included set.
        --english-share F (default 0.5) is the part of the room beyond the always-included tokens given to whole English
        words (a token that is a --dictionary word or an inflection of one, most frequent in the corpus first); the rest
        goes to the corpus's most frequent tokens of any kind (code, punctuation, names).
        --no-punctuation true leaves out the ~4,400 tokens with no letter or digit in them that are otherwise always included (for a
        list that only has to score a few verdict tokens).
        --emit SIZE:FILE also writes a list of another size from the same counting pass (lists of different sizes are nested).

        coverage reads a JSONL file of {"messages": [...], "reference": "text"} and says, for each --vocab list, how many of
        the reference text's tokens are outside the list alone and outside the list plus the tokens of the messages (the
        context a request would add). It only tokenizes: no model runs.
        """

    struct UsageError: Error { var message: String }

    public static func main(_ arguments: [String]) -> Int32 {
        do {
            guard let sub = arguments.first else { throw UsageError(message: "vocab needs build or info") }
            var multi: [String: [String]] = [:]
            var rest: [String] = []
            var i = 1
            while i < arguments.count {
                let a = arguments[i]
                if a.hasPrefix("--") {
                    guard i + 1 < arguments.count else { throw UsageError(message: "\(a) needs a value") }
                    multi[String(a.dropFirst(2)), default: []].append(arguments[i + 1])
                    i += 2
                } else {
                    rest.append(a)
                    i += 1
                }
            }
            func one(_ k: String) -> String? { multi[k]?.last }
            switch sub {
            case "build": return try build(multi: multi, one: one)
            case "coverage": return try coverage(multi: multi, one: one)
            case "info":
                guard let file = rest.first else { throw UsageError(message: "vocab info needs a file") }
                return try info(file, tokenizer: one("tokenizer"))
            default: throw UsageError(message: "unknown vocab command \(sub)")
            }
        } catch let e as UsageError {
            FileHandle.standardError.write(Data("shears-cli vocab: \(e.message)\n\(usage)\n".utf8))
            return 2
        } catch {
            FileHandle.standardError.write(Data("shears-cli vocab: \(error)\n".utf8))
            return 1
        }
    }

    private static func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

    private static func build(multi: [String: [String]], one: (String) -> String?) throws -> Int32 {
        guard let t = one("tokenizer"), let corpus = multi["corpus"], let out = one("out") else {
            throw UsageError(message: "--tokenizer, --corpus and --out are required")
        }
        let size = try one("size").map { s -> Int in
            guard let n = Int(s), n > 0 else { throw UsageError(message: "bad --size") }
            return n
        } ?? 32768
        let maxMB = try one("max-mb").map { s -> Int in
            guard let n = Int(s), n > 0 else { throw UsageError(message: "bad --max-mb") }
            return n
        } ?? 512
        let sizes = try (one("sizes") ?? "4096,8192,16384,32768,65536").split(separator: ",").map { s -> Int in
            guard let n = Int(s) else { throw UsageError(message: "bad --sizes") }
            return n
        }
        let tokenizer = try BPETokenizer(directory: URL(fileURLWithPath: t))
        let builder = StaticVocabularyBuilder(tokenizer: tokenizer)
        let maxFileMB = one("max-file-mb").flatMap { Int($0) } ?? 32
        let files = StaticVocabularyBuilder.corpusFiles(under: corpus.map { URL(fileURLWithPath: $0) }, maxFileBytes: maxFileMB << 20)
        log("corpus: \(files.count) files")
        let started = Date()
        builder.addCorpus(files: files, maxTotalBytes: maxMB << 20, log: log)
        log(String(format: "counted %d files, %.1f MB in %.0f s", builder.corpusFiles, Double(builder.corpusBytes) / 1e6, Date().timeIntervalSince(started)))
        for d in multi["dictionary"] ?? [] {
            let words = try String(contentsOfFile: d, encoding: .utf8).split(separator: "\n").map(String.init)
            builder.addDictionary(words: words)
            log("dictionary \(d): \(words.count) words")
        }
        var extra: [String] = []
        if let m = one("mandatory-text") { extra.append(try String(contentsOfFile: m, encoding: .utf8)) }
        let share = try one("english-share").map { s -> Double in
            guard let f = Double(s), f >= 0, f <= 1 else { throw UsageError(message: "bad --english-share") }
            return f
        } ?? 0.5
        let punctuation = one("no-punctuation") != "true"
        let selection = try builder.select(size: size, extraTexts: extra, englishShare: share, punctuation: punctuation, note: one("note"))
        try selection.vocabulary.write(to: URL(fileURLWithPath: out))
        for e in multi["emit"] ?? [] {
            let parts = e.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2, let n = Int(parts[0]) else { throw UsageError(message: "bad --emit \(e)") }
            let other = try builder.select(size: n, extraTexts: extra, englishShare: share, punctuation: punctuation, note: one("note"))
            try other.vocabulary.write(to: URL(fileURLWithPath: parts[1]))
            print("wrote \(parts[1]): \(other.vocabulary.ids.count) tokens")
        }
        print("wrote \(out): \(selection.vocabulary.ids.count) tokens (\(selection.mandatoryCount) always included)")
        if selection.shortfall > 0 { print("note: the corpus and dictionary held \(selection.shortfall) fewer candidates than --size asked for") }
        print("sections: " + selection.vocabulary.sections.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
        print("size      rows    training tokens held   held-out tokens held")
        for c in builder.coverage(sizes: sizes, extraTexts: extra, englishShare: share, punctuation: punctuation) {
            let held = c.heldOut.map { String(format: "%.4f", $0) } ?? "-"
            print(String(format: "%-8d %-7d %-22.4f", c.size, c.rows, c.training) + held)
        }
        return 0
    }

    private static func coverage(multi: [String: [String]], one: (String) -> String?) throws -> Int32 {
        guard let t = one("tokenizer"), let p = one("prompts"), let lists = multi["vocab"], !lists.isEmpty else {
            throw UsageError(message: "--tokenizer, --prompts and --vocab are required")
        }
        let tokenizer = try BPETokenizer(directory: URL(fileURLWithPath: t))
        var prompts = try VocabularyEvaluation.loadPrompts(URL(fileURLWithPath: p)).filter { $0.reference != nil }
        if let l = one("limit"), let n = Int(l), n < prompts.count { prompts = Array(prompts.prefix(n)) }
        let vocabularies = try lists.map { try StaticVocabulary.load(URL(fileURLWithPath: $0)) }
        for v in vocabularies { try v.validate(for: tokenizer) }
        let sets = vocabularies.map { Set($0.ids) }
        var referenceTokens = 0
        let variants = tokenizer.spaceVariantMap()
        var outsideStatic = [Int](repeating: 0, count: sets.count)
        var outsideContext = [Int](repeating: 0, count: sets.count)
        var outsideAll = [Int](repeating: 0, count: sets.count)
        var promptsWithAny = [Int](repeating: 0, count: sets.count)
        var outsideCounts = [[String: Int]](repeating: [:], count: sets.count)
        var contextOutside = 0, contextDistinct = 0
        for prompt in prompts {
            var plain = Set<Int>()
            for m in prompt.messages {
                if let c = m["content"] as? String { plain.formUnion(tokenizer.encode(tokenizer.escapingSpecialTokens(in: c))) }
            }
            let context = VocabularyWindow.contextSet(plain, variants: variants)
            let reference = tokenizer.encode(prompt.reference ?? "")
            referenceTokens += reference.count
            contextDistinct += context.count
            for (i, set) in sets.enumerated() {
                if i == 0 { contextOutside += context.filter { !set.contains($0) }.count }
                var any = false
                for id in reference where !set.contains(id) {
                    outsideStatic[i] += 1
                    if !plain.contains(id) { outsideContext[i] += 1 }
                    if !context.contains(id) {
                        outsideAll[i] += 1
                        any = true
                        outsideCounts[i][tokenizer.token(for: id) ?? "#\(id)", default: 0] += 1
                    }
                }
                if any { promptsWithAny[i] += 1 }
            }
        }
        print("\(prompts.count) prompts, \(referenceTokens) reference tokens; \(contextDistinct / max(1, prompts.count)) distinct context tokens per prompt on average")
        print("list                                   rows    outside the list   + context tokens   + their space variants   prompts with any")
        for (i, v) in vocabularies.enumerated() {
            let name = URL(fileURLWithPath: lists[i]).lastPathComponent.padding(toLength: 36, withPad: " ", startingAt: 0)
            let a = Double(outsideStatic[i]) / Double(max(1, referenceTokens)), b = Double(outsideAll[i]) / Double(max(1, referenceTokens))
            let c = Double(outsideContext[i]) / Double(max(1, referenceTokens))
            print(name + "  " + String(format: "%-7d %-18.5f %-18.5f %-25.5f", v.ids.count, a, c, b) + "\(promptsWithAny[i]) of \(prompts.count)")
            if let n = one("show").flatMap({ Int($0) }), n > 0 {
                let top = outsideCounts[i].sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.prefix(n)
                print("    most frequent tokens outside list + context: " + top.map { "\($0.key.debugDescription) x\($0.value)" }.joined(separator: " "))
            }
        }
        return 0
    }

    private static func info(_ file: String, tokenizer: String?) throws -> Int32 {
        let v = try StaticVocabulary.load(URL(fileURLWithPath: file))
        print("\(file): \(v.ids.count) of \(v.vocabularySize) tokens")
        print("sections: " + v.sections.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
        if let b = v.build {
            print("built from \(b.corpusFiles) files, \(b.corpusBytes / 1_000_000) MB, \(b.corpusTokens) tokens, \(b.dictionaryWords) dictionary words; requested size \(b.requestedSize)")
            if let h = b.heldOutCoverage { print(String(format: "held-out corpus tokens held: %.4f", h)) }
            if let n = b.note { print("note: \(n)") }
        }
        if let t = tokenizer {
            let tok = try BPETokenizer(directory: URL(fileURLWithPath: t))
            do { try v.validate(for: tok); print("matches the tokenizer") } catch { print("does NOT match the tokenizer: \(error)") }
        }
        return 0
    }
}

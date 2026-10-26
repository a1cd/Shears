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
          shears-cli vocab info FILE [--tokenizer DIR]

        build counts the tokens of a text corpus (files, or directories walked for text and source files) and writes the
        --size most useful token ids (default 32768) to FILE: the always-included tokens (the 256 byte-fallback tokens, the
        special tokens, JSON/verdict pieces, every printable ASCII character) first, then the corpus's most frequent
        tokens, then the pieces of --dictionary words (one per line) for broad coverage. Every tenth corpus file is held
        out and the share of its tokens the list holds is reported, also for the --sizes given, to help choose a size.
        --max-mb limits the corpus read (default 512). --mandatory-text adds the tokens of a text file to the always-included set.
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
        let files = StaticVocabularyBuilder.corpusFiles(under: corpus.map { URL(fileURLWithPath: $0) })
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
        let selection = try builder.select(size: size, extraTexts: extra, note: one("note"))
        try selection.vocabulary.write(to: URL(fileURLWithPath: out))
        print("wrote \(out): \(selection.vocabulary.ids.count) tokens (\(selection.mandatoryCount) always included)")
        if selection.shortfall > 0 { print("note: the corpus and dictionary held \(selection.shortfall) fewer candidates than --size asked for") }
        print("sections: " + selection.vocabulary.sections.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
        print("size      rows    training tokens held   held-out tokens held")
        for c in builder.coverage(sizes: sizes, extraTexts: extra) {
            print(String(format: "%-8d %-7d %-22.4f %@", c.size, c.rows, c.training, c.heldOut.map { String(format: "%.4f", $0) } ?? "-"))
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

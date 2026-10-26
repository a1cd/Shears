import Foundation
import MLX

/// Measures what a vocabulary window gives up: how often a model with its whole vocabulary would choose, or
/// have a real chance of choosing, a token the window lacks; whether generation over the window equals
/// generation over everything; and whether scoring just a few candidate tokens (a verdict) picks what the
/// full model picks.
///
/// It runs in one process on a model that holds its whole embedding table (the reference), and attaches a
/// window built from the files for the constrained runs. Nothing here is specific to a model's task; the prompts
/// come from a JSONL file of chat messages.
public enum VocabularyEvaluation {
    /// One prompt of the evaluation.
    public struct Prompt {
        public var id: String
        public var messages: [[String: Any]]
        /// Text a good answer might contain (for example a logged answer), tokenized and checked against the
        /// window without running the model. Optional.
        public var reference: String?
    }

    public static func loadPrompts(_ url: URL) throws -> [Prompt] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var out: [Prompt] = []
        for (n, line) in text.split(separator: "\n").enumerated() where !line.isEmpty {
            guard let obj = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                let messages = obj["messages"] as? [[String: Any]]
            else { throw InferenceError.invalid("\(url.lastPathComponent) line \(n + 1): expected {\"messages\": [...]}") }
            out.append(Prompt(id: obj["id"] as? String ?? "\(n)", messages: messages, reference: obj["reference"] as? String))
        }
        return out
    }

    public struct Options {
        public var maxNewTokens = 120
        /// The verdict words and the JSON text that precedes them, for the candidate-scoring check.
        public var verdicts = ["ON_TASK", "NOT_ALIGNED", "IDLE", "NO_TASK"]
        public var verdictPrefix = "{\"state\": \""
        public var topK = 64
        public var topP: Float = 0.95
        /// Generate under the JSON-object constraint (`response_format: json_object`), as the classifier does.
        public var jsonObject = true
        public init() {}
    }

    public struct PromptReport: Codable {
        public var id: String
        public var promptTokens: Int
        /// Distinct tokens in the prompt, and how many of those the static base lacks (rows read for the request).
        public var contextDistinct: Int
        public var contextOutsideStatic: Int
        public var generatedTokens: Int
        /// Steps where the unconstrained greedy token was not in static + context (so far), and not in static alone.
        public var stepsOutsideSet: Int
        public var stepsOutsideStatic: Int
        public var outsideTokens: [String]
        /// Per step, the probability the full softmax puts outside the set (averaged / maximum over steps).
        public var meanMassOutsideSet: Double
        public var maxMassOutsideSet: Double
        public var meanMassOutsideStatic: Double
        /// The same inside the top-k / top-p sampling distribution (what a sampled reply could pick).
        public var meanNucleusMassOutsideSet: Double
        /// Chance that a sampled reply holds at least one token outside the set: 1 - prod(1 - nucleus mass).
        public var chanceReplyLeavesSet: Double
        public var chanceReplyLeavesStatic: Double
        public var unconstrainedText: String
        public var constrainedText: String
        public var constrainedEqualsUnconstrained: Bool
        /// Index of the first differing token (nil when equal).
        public var firstDivergence: Int?
        public var unconstrainedVerdict: String?
        public var constrainedVerdict: String?
        // Verdict scoring
        public var verdictForcedFullChoice: String?
        public var verdictCandidatesChoice: String?
        /// Probability the full softmax puts on the candidate tokens at the verdict position.
        public var verdictCandidateMass: Double?
        /// The same scoring after the model's own words up to the verdict (its code fence included), when the free
        /// generation has a verdict at a candidate token: what the candidates choose, and the mass they hold there.
        public var verdictNaturalCandidatesChoice: String?
        public var verdictNaturalCandidateMass: Double?
        // Reference text
        public var referenceTokens: Int?
        public var referenceOutsideSet: Int?
        public var referenceOutsideStatic: Int?
    }

    public struct Summary: Codable {
        public var prompts: Int
        public var windowStaticRows: Int
        public var meanContextOutsideStatic: Double
        public var generatedTokens: Int
        public var greedyStepsOutsideSet: Int
        public var greedyStepsOutsideStatic: Int
        public var promptsWithAnyGreedyStepOutsideSet: Int
        public var meanMassOutsideSet: Double
        public var meanMassOutsideStatic: Double
        public var meanNucleusMassOutsideSet: Double
        public var meanChanceReplyLeavesSet: Double
        public var meanChanceReplyLeavesStatic: Double
        public var constrainedEqualsUnconstrained: Int
        public var verdictAgreementFreeVsConstrained: Int
        public var verdictsParsed: Int
        public var verdictForcedAgreement: Int
        public var verdictForcedPrompts: Int
        /// Candidate scoring after the forced start, against the verdict the unconstrained free generation wrote.
        public var verdictForcedCandidatesAgreeWithFree: Int
        public var verdictForcedCandidatesCompared: Int
        /// Candidate scoring after the model's own start, against the verdict its free generation wrote.
        public var verdictNaturalAgreeWithFree: Int
        public var verdictNaturalCompared: Int
        public var meanVerdictCandidateMass: Double
        public var meanVerdictNaturalCandidateMass: Double
        /// What was forced and scored: the text before the verdict, and the candidate tokens (one per verdict).
        public var verdictStartText: String
        public var verdictCandidateTokens: [String]
        public var referenceTokens: Int
        public var referenceOutsideSet: Int
        public var referenceOutsideStatic: Int
        public var perPrompt: [PromptReport]
    }

    /// Runs the evaluation. `runtime` must hold the whole table (the reference); the window is built from `directory`.
    public static func run(
        runtime: InferenceRuntime, directory: URL, vocabulary: StaticVocabulary, prompts: [Prompt],
        options: Options = .init(), log: (String) -> Void = { _ in }
    ) throws -> Summary {
        let model = runtime.model
        let tokenizer = runtime.tokenizer
        guard model.vocabularyWindow == nil else { throw InferenceError.invalid("the reference model must hold its whole table") }
        try vocabulary.validate(for: tokenizer)
        let checkpoint = try Gemma4Checkpoint(directory: directory)
        let source = try checkpoint.embeddingRowSource()
        let dtype = model.model.model.norm.weight.dtype
        let window = try VocabularyWindow(source: source, staticIDs: vocabulary.ids, dtype: dtype)
        let variants = tokenizer.spaceVariantMap()
        window.contextVariants = variants
        let staticSet = Set(vocabulary.ids)
        let vocab = tokenizer.vocabularySize
        let stop = Set(runtime.defaults.eosTokenIDs)
        var reports: [PromptReport] = []

        func verdict(_ text: String) -> String? {
            guard let r = text.range(of: #""state"\s*:\s*"([A-Z_]+)""#, options: .regularExpression) else { return nil }
            let m = String(text[r])
            return m.split(separator: "\"").dropLast().last.map(String.init)
        }

        // Candidate tokens for the verdict position: the first token after the common start of each verdict's tokens.
        let fulls = options.verdicts.map { tokenizer.encode(options.verdictPrefix + $0) }
        var common = 0
        while let first = fulls.first, common < first.count, fulls.allSatisfy({ common < $0.count && $0[common] == first[common] }) {
            common += 1
        }
        let candidates = fulls.map { common < $0.count ? $0[common] : -1 }
        let candidatesDistinct = !candidates.contains(-1) && Set(candidates).count == candidates.count
        let verdictStart = fulls.first.map { Array($0[..<common]) } ?? []
        let verdictStartText = tokenizer.decode(verdictStart)
        let candidateTokenText = candidates.map { tokenizer.token(for: $0) ?? "#\($0)" }

        for (n, prompt) in prompts.enumerated() {
            let tokens = try runtime.promptTokens(messages: prompt.messages)
            let promptSet = VocabularyWindow.contextSet(tokens, variants: variants)
            var context = promptSet
            let contextOutsideStatic = promptSet.filter { !staticSet.contains($0) }.count

            // 1. Unconstrained greedy over the whole vocabulary.
            model.model.vocabularyWindow = nil
            let cache = model.newCache()
            var logits = try runtime.generator.prefill(tokens, cache: cache)
            var outsideF = [Float](repeating: 1, count: vocab)
            var outsideStaticF = [Float](repeating: 1, count: vocab)
            for id in staticSet { outsideF[id] = 0; outsideStaticF[id] = 0 }
            for id in promptSet { outsideF[id] = 0 }
            let jsonConstraint: JSONObjectConstraint? = options.jsonObject
                ? JSONObjectConstraint(tokenizer: tokenizer, stopTokenIDs: Array(stop), logitsVocabularySize: vocab) : nil
            var generated: [Int] = []
            var stepsOutSet = 0, stepsOutStatic = 0
            var outsideTokens: [String] = []
            var mass: [Double] = [], massStatic: [Double] = [], nucleusMass: [Double] = [], nucleusMassStatic: [Double] = []
            for _ in 0 ..< options.maxNewTokens {
                var row = logits[0].asType(.float32)
                if let mask = jsonConstraint?.allowedMask(logits: row) { row = MLX.where(mask, row, MLXArray(-Float.infinity)) }
                let probs = softmax(row, axis: -1)
                let m = (probs * MLXArray(outsideF)).sum()
                let ms = (probs * MLXArray(outsideStaticF)).sum()
                let k = min(options.topK, vocab)
                let idx = argPartition(row, kth: vocab - k, axis: -1)[(vocab - k)...]
                let topLogits = row[idx]
                eval(m, ms, idx, topLogits)
                let t = argMax(row, axis: -1).item(Int.self)
                mass.append(Double(m.item(Float.self)))
                massStatic.append(Double(ms.item(Float.self)))
                // top-k, then softmax, then top-p: the share of the nucleus outside the set.
                let ids = idx.asArray(Int32.self).map(Int.init)
                let ls = topLogits.asArray(Float.self)
                let order = ls.indices.sorted { ls[$0] > ls[$1] }
                let mx = ls[order[0]]
                let ex = order.map { Double(exp(ls[$0] - mx)) }
                let total = ex.reduce(0, +)
                var cum = 0.0, nucleus: [Int] = []
                for (r, i) in order.enumerated() {
                    nucleus.append(ids[i])
                    cum += ex[r] / total
                    if cum >= Double(options.topP) { break }
                }
                let nucleusWeight = nucleus.indices.map { ex[$0] }
                let nucleusTotal = nucleusWeight.reduce(0, +)
                var out = 0.0, outS = 0.0
                for (j, id) in nucleus.enumerated() {
                    if !context.contains(id) && !staticSet.contains(id) { out += nucleusWeight[j] }
                    if !staticSet.contains(id) { outS += nucleusWeight[j] }
                }
                nucleusMass.append(out / nucleusTotal)
                nucleusMassStatic.append(outS / nucleusTotal)

                jsonConstraint?.accept(t)
                if stop.contains(t) { break }
                if !staticSet.contains(t) {
                    stepsOutStatic += 1
                    if !context.contains(t) {
                        stepsOutSet += 1
                        outsideTokens.append(tokenizer.token(for: t) ?? "#\(t)")
                    }
                }
                generated.append(t)
                context.insert(t)
                outsideF[t] = 0
                logits = model.nextLogits(MLXArray([Int32(t)])[.newAxis], cache: cache)
                eval(logits)
            }
            let freeText = tokenizer.decode(generated, skipSpecialTokens: true)

            // 2. The same greedy decode over the window.
            model.model.vocabularyWindow = window
            let g = runtime.generator
            let savedReuse = g.reusesPromptCache
            g.reusesPromptCache = false
            g.resetCache()
            let windowJSON: JSONObjectConstraint? = options.jsonObject
                ? JSONObjectConstraint(tokenizer: tokenizer, stopTokenIDs: Array(stop), logitsVocabularySize: vocab) : nil
            let constrained = try g.generate(
                GenerationRequest(promptTokens: tokens, maxTokens: options.maxNewTokens, stopTokenIDs: stop, constraint: windowJSON))
            g.reusesPromptCache = savedReuse
            var divergence: Int?
            let shared = min(generated.count, constrained.tokenIDs.count)
            for i in 0 ..< shared where generated[i] != constrained.tokenIDs[i] { divergence = i; break }
            if divergence == nil, generated.count != constrained.tokenIDs.count { divergence = shared }

            // 3. Verdict scoring: the forced start of the answer, the full vocabulary against the candidates alone.
            var forcedFull: String?, candidatesChoice: String?, candidateMass: Double?
            if candidatesDistinct {
                let forced = tokens + verdictStart
                model.model.vocabularyWindow = nil
                let c2 = model.newCache()
                let l2 = try g.prefill(forced, cache: c2)[0].asType(.float32)
                let p2 = softmax(l2, axis: -1)
                let candMass = p2[MLXArray(candidates.map(Int32.init))].sum()
                let top = argMax(l2, axis: -1)
                eval(candMass, top)
                candidateMass = Double(candMass.item(Float.self))
                let topID = top.item(Int.self)
                forcedFull = candidates.firstIndex(of: topID).map { options.verdicts[$0] } ?? "(other: \(tokenizer.token(for: topID) ?? "?"))"
                model.model.vocabularyWindow = window
                let scores = try g.candidateScores(promptTokens: forced, candidates: candidates)
                if let best = scores.indices.max(by: { scores[$0] < scores[$1] }) { candidatesChoice = options.verdicts[best] }
            }

            // 3b. The same, after the model's own start of its answer (a code fence, whitespace, key order) rather than a forced one.
            var naturalChoice: String?, naturalMass: Double?
            if candidatesDistinct {
                for k in generated.indices where candidates.contains(generated[k]) {
                    let before = tokenizer.decode(Array(generated[..<k]), skipSpecialTokens: true)
                    guard before.range(of: #""state"\s*:\s*"$"#, options: .regularExpression) != nil else { continue }
                    let forced = tokens + Array(generated[..<k])
                    model.model.vocabularyWindow = nil
                    let c3 = model.newCache()
                    let l3 = try g.prefill(forced, cache: c3)[0].asType(.float32)
                    let mass3 = softmax(l3, axis: -1)[MLXArray(candidates.map(Int32.init))].sum()
                    eval(mass3)
                    naturalMass = Double(mass3.item(Float.self))
                    model.model.vocabularyWindow = window
                    let scores = try g.candidateScores(promptTokens: forced, candidates: candidates)
                    if let best = scores.indices.max(by: { scores[$0] < scores[$1] }) { naturalChoice = options.verdicts[best] }
                    break
                }
            }

            var rep = PromptReport(
                id: prompt.id, promptTokens: tokens.count, contextDistinct: promptSet.count,
                contextOutsideStatic: contextOutsideStatic, generatedTokens: generated.count,
                stepsOutsideSet: stepsOutSet, stepsOutsideStatic: stepsOutStatic, outsideTokens: outsideTokens,
                meanMassOutsideSet: mass.reduce(0, +) / Double(max(1, mass.count)),
                maxMassOutsideSet: mass.max() ?? 0,
                meanMassOutsideStatic: massStatic.reduce(0, +) / Double(max(1, massStatic.count)),
                meanNucleusMassOutsideSet: nucleusMass.reduce(0, +) / Double(max(1, nucleusMass.count)),
                chanceReplyLeavesSet: 1 - nucleusMass.reduce(1) { $0 * (1 - $1) },
                chanceReplyLeavesStatic: 1 - nucleusMassStatic.reduce(1) { $0 * (1 - $1) },
                unconstrainedText: freeText, constrainedText: tokenizer.decode(constrained.tokenIDs, skipSpecialTokens: true),
                constrainedEqualsUnconstrained: divergence == nil, firstDivergence: divergence,
                unconstrainedVerdict: verdict(freeText), constrainedVerdict: verdict(constrained.text),
                verdictForcedFullChoice: forcedFull, verdictCandidatesChoice: candidatesChoice,
                verdictCandidateMass: candidateMass, verdictNaturalCandidatesChoice: naturalChoice,
                verdictNaturalCandidateMass: naturalMass)
            if let ref = prompt.reference {
                let r = tokenizer.encode(ref)
                rep.referenceTokens = r.count
                rep.referenceOutsideStatic = r.filter { !staticSet.contains($0) }.count
                rep.referenceOutsideSet = r.filter { !staticSet.contains($0) && !promptSet.contains($0) }.count
            }
            reports.append(rep)
            model.model.vocabularyWindow = nil
            log(
                "\(n + 1)/\(prompts.count) \(prompt.id): \(generated.count) tokens, outside set \(stepsOutSet), "
                    + "outside static \(stepsOutStatic), equal \(divergence == nil), verdict \(rep.unconstrainedVerdict ?? "-") / \(rep.constrainedVerdict ?? "-")")
        }

        func mean(_ f: (PromptReport) -> Double) -> Double { reports.isEmpty ? 0 : reports.map(f).reduce(0, +) / Double(reports.count) }
        let parsed = reports.filter { $0.unconstrainedVerdict != nil }
        let forced = reports.filter { $0.verdictForcedFullChoice != nil }
        let natural = reports.filter { $0.verdictNaturalCandidatesChoice != nil }
        return Summary(
            prompts: reports.count, windowStaticRows: vocabulary.ids.count,
            meanContextOutsideStatic: mean { Double($0.contextOutsideStatic) },
            generatedTokens: reports.map(\.generatedTokens).reduce(0, +),
            greedyStepsOutsideSet: reports.map(\.stepsOutsideSet).reduce(0, +),
            greedyStepsOutsideStatic: reports.map(\.stepsOutsideStatic).reduce(0, +),
            promptsWithAnyGreedyStepOutsideSet: reports.filter { $0.stepsOutsideSet > 0 }.count,
            meanMassOutsideSet: mean(\.meanMassOutsideSet), meanMassOutsideStatic: mean(\.meanMassOutsideStatic),
            meanNucleusMassOutsideSet: mean(\.meanNucleusMassOutsideSet),
            meanChanceReplyLeavesSet: mean(\.chanceReplyLeavesSet), meanChanceReplyLeavesStatic: mean(\.chanceReplyLeavesStatic),
            constrainedEqualsUnconstrained: reports.filter(\.constrainedEqualsUnconstrained).count,
            verdictAgreementFreeVsConstrained: parsed.filter { $0.unconstrainedVerdict == $0.constrainedVerdict }.count,
            verdictsParsed: parsed.count,
            verdictForcedAgreement: forced.filter { $0.verdictForcedFullChoice == $0.verdictCandidatesChoice }.count,
            verdictForcedPrompts: forced.count,
            verdictForcedCandidatesAgreeWithFree: reports.filter {
                $0.unconstrainedVerdict != nil && $0.unconstrainedVerdict == $0.verdictCandidatesChoice
            }.count,
            verdictForcedCandidatesCompared: reports.filter { $0.unconstrainedVerdict != nil && $0.verdictCandidatesChoice != nil }.count,
            verdictNaturalAgreeWithFree: natural.filter { $0.unconstrainedVerdict == $0.verdictNaturalCandidatesChoice }.count,
            verdictNaturalCompared: natural.filter { $0.unconstrainedVerdict != nil }.count,
            meanVerdictCandidateMass: forced.isEmpty ? 0 : forced.compactMap(\.verdictCandidateMass).reduce(0, +) / Double(forced.count),
            meanVerdictNaturalCandidateMass: natural.isEmpty ? 0 : natural.compactMap(\.verdictNaturalCandidateMass).reduce(0, +) / Double(natural.count),
            verdictStartText: verdictStartText, verdictCandidateTokens: candidateTokenText,
            referenceTokens: reports.compactMap(\.referenceTokens).reduce(0, +),
            referenceOutsideSet: reports.compactMap(\.referenceOutsideSet).reduce(0, +),
            referenceOutsideStatic: reports.compactMap(\.referenceOutsideStatic).reduce(0, +),
            perPrompt: reports)
    }
}

extension Generator {
    /// The scores (logits, softcapped) of just `candidates` as the next token after `promptTokens`, computed from
    /// the final hidden state with only those rows of the table: a classification, not a generation. Uses its own
    /// cache and leaves the generator's alone. With a vocabulary window, the candidates must be in its static base.
    public func candidateScores(promptTokens: [Int], candidates: [Int]) throws -> [Float] {
        guard !promptTokens.isEmpty, !candidates.isEmpty else { throw InferenceError.invalid("empty prompt or candidates") }
        guard promptTokens.count + 1 <= contextLength else {
            throw InferenceError.invalid("prompt of \(promptTokens.count) tokens does not fit the \(contextLength)-token context")
        }
        try model.prepareVocabulary(promptTokens: promptTokens)
        let cache = model.newCache()
        var hidden: MLXArray?
        var i = 0
        while i < promptTokens.count {
            let end = min(promptTokens.count, i + max(1, prefillChunk))
            let chunk = MLXArray(promptTokens[i ..< end].map(Int32.init))[.newAxis]
            let h = model.lastHidden(chunk, cache: cache)
            if end < promptTokens.count {
                eval(cache.compactMap { $0.keys } + cache.compactMap { $0.values })
            } else {
                hidden = h
            }
            i = end
        }
        let scores = try model.model.candidateLogits(fromHidden: hidden!, candidates: candidates)
        eval(scores)
        return scores.asType(.float32).reshaped(-1).asArray(Float.self)
    }
}

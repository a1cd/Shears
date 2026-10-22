import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class InferenceGenerationTests: ShearsTestCase {
    func makeGenerator(layers: Int = 4) throws -> (Generator, InferenceModel) {
        let model = try InferenceModel(model: InferenceFixture.model(layers: layers))
        return (Generator(model: model, tokenizer: try InferenceFixture.tokenizer()), model)
    }

    /// Logits of the next token after each prefix of `tokens`, by repeated whole-sequence forward passes.
    func referenceNextLogits(_ model: InferenceModel, _ tokens: [Int]) -> MLXArray {
        let l = model.fullLogits(MLXArray(tokens.map(Int32.init))[.newAxis])
        return l[0, tokens.count - 1]
    }

    /// Greedy generation with the KV cache equals the argmax of repeated full forward passes, and the
    /// cached logits equal the full ones. The prompt and the generated text both run past the sliding
    /// window (8), so sliding layers are exercised both in prefill and in decode.
    func testGreedyDecodeEqualsRepeatedFullForward() throws {
        let (gen, model) = try makeGenerator()
        let prompt = InferenceFixture.prompt(14)
        XCTAssertGreaterThan(prompt.count, model.configuration.slidingWindow)
        let result = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 24))
        XCTAssertEqual(result.tokenIDs.count, 24)
        XCTAssertEqual(result.finishReason, .length)
        var seq = prompt
        for (i, t) in result.tokenIDs.enumerated() {
            let ref = referenceNextLogits(model, seq)
            let want = argMax(ref).item(Int.self)
            XCTAssertEqual(t, want, "step \(i): cached \(t) vs full-forward argmax \(want), gap \(gap(ref, to: t))")
            seq.append(t)
        }
    }

    /// Decode that starts inside the window and runs far past it.
    func testDecodePastTheWindowFromAShortPrompt() throws {
        let (gen, model) = try makeGenerator()
        let prompt = InferenceFixture.prompt(3, seed: 9)
        let result = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 40))
        XCTAssertEqual(result.tokenIDs.count, 40)
        XCTAssertGreaterThan(prompt.count + 40, 4 * model.configuration.slidingWindow)
        var seq = prompt
        for t in result.tokenIDs {
            let ref = referenceNextLogits(model, seq)
            XCTAssertLessThan(gap(ref, to: t), 2e-3)
            seq.append(t)
        }
    }

    /// Logits, not only tokens: each cached step matches the full forward within float32 noise.
    func testCachedLogitsMatchFullLogits() throws {
        let (_, model) = try makeGenerator()
        let tokens = InferenceFixture.prompt(30)
        let full = model.fullLogits(MLXArray(tokens.map(Int32.init))[.newAxis])
        let cache = model.newCache()
        // 12 tokens at once, then one at a time.
        var out = model.nextLogits(MLXArray(tokens[0 ..< 12].map(Int32.init))[.newAxis], cache: cache)
        assertClose(out[0], full[0, 11], atol: 2e-3, rtol: 2e-3, "after 12")
        for i in 12 ..< tokens.count {
            out = model.nextLogits(MLXArray([Int32(tokens[i])])[.newAxis], cache: cache)
            assertClose(out[0], full[0, i], atol: 2e-3, rtol: 2e-3, "position \(i)")
        }
    }

    func testChunkedPrefillEqualsWholePrefill() throws {
        let (gen, _) = try makeGenerator()
        let prompt = InferenceFixture.prompt(37)
        gen.prefillChunk = 512
        gen.reusesPromptCache = false
        let whole = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 10))
        gen.prefillChunk = 5
        let chunked = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 10))
        XCTAssertEqual(whole.tokenIDs, chunked.tokenIDs)
    }

    func testPromptCacheReuse() throws {
        let (gen, _) = try makeGenerator()
        let base = InferenceFixture.prompt(20)
        let first = try gen.generate(GenerationRequest(promptTokens: base, maxTokens: 6))
        XCTAssertEqual(first.cachedPromptTokenCount, 0)
        // Same prompt again: everything but the last token is reused.
        let again = try gen.generate(GenerationRequest(promptTokens: base, maxTokens: 6))
        XCTAssertEqual(again.cachedPromptTokenCount, 19)
        XCTAssertEqual(again.tokenIDs, first.tokenIDs)
        // The previous prompt plus its answer plus a new turn: prompt and generated tokens are reused.
        let extended = base + first.tokenIDs.dropLast() + [10, 11, 12]
        let cont = try gen.generate(GenerationRequest(promptTokens: extended, maxTokens: 8, logprobs: 0))
        XCTAssertEqual(cont.cachedPromptTokenCount, base.count + first.tokenIDs.count - 1)
        // Reusing gives the same tokens as starting cold.
        gen.reusesPromptCache = false
        let cold = try gen.generate(GenerationRequest(promptTokens: extended, maxTokens: 8, logprobs: 0))
        XCTAssertEqual(cont.tokenIDs, cold.tokenIDs)
        // Same distributions, not only the same winners.
        for (a, b) in zip(cont.events.filter { !$0.isTail }, cold.events.filter { !$0.isTail }) { XCTAssertEqual(a.logprob!.logprob, b.logprob!.logprob, accuracy: 1e-3) }
        // A different start reuses nothing.
        gen.reusesPromptCache = true
        gen.resetCache()
        _ = try gen.generate(GenerationRequest(promptTokens: base, maxTokens: 2))
        let other = try gen.generate(GenerationRequest(promptTokens: Array(InferenceFixture.prompt(20, seed: 99)), maxTokens: 2))
        XCTAssertEqual(other.cachedPromptTokenCount, 0)
    }

    func testStopConditions() throws {
        let (gen, _) = try makeGenerator()
        let prompt = InferenceFixture.prompt(10)
        let free = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 12))
        XCTAssertEqual(free.finishReason, .length)
        // A stop token ends generation and is not part of the output.
        let stopToken = free.tokenIDs[4]
        let firstOccurrence = free.tokenIDs.firstIndex(of: stopToken)!
        let stopped = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 12, stopTokenIDs: [stopToken]))
        XCTAssertEqual(stopped.finishReason, .stop)
        XCTAssertEqual(stopped.tokenIDs, Array(free.tokenIDs[..<firstOccurrence]))
        // A stop string ends generation before it; text before it is released.
        let tok = try InferenceFixture.tokenizer()
        let pieces = free.tokenIDs.map { tok.decode([$0]) }
        let target = pieces[6...8].joined()
        guard !target.isEmpty else { return XCTFail("fixture produced empty text") }
        let stringStop = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 12, stopStrings: [target]))
        XCTAssertEqual(stringStop.finishReason, .stop)
        XCTAssertEqual(stringStop.text, tok.decode(free.tokenIDs).components(separatedBy: target).first!)
        XCTAssertEqual(stringStop.events.map(\.text).joined(), stringStop.text)
        // maxTokens 0 generates nothing.
        XCTAssertEqual(try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 0)).tokenIDs, [])
        // A prompt that does not fit is refused.
        XCTAssertThrowsError(try gen.generate(GenerationRequest(promptTokens: Array(repeating: 7, count: 600), maxTokens: 1)))
        XCTAssertThrowsError(try gen.generate(GenerationRequest(promptTokens: [], maxTokens: 1)))
    }

    func testCancellationStopsAtTheHandlersWord() throws {
        let (gen, _) = try makeGenerator()
        var seen = 0
        let r = try gen.generate(GenerationRequest(promptTokens: InferenceFixture.prompt(8), maxTokens: 30)) { _ in
            seen += 1
            return seen < 5
        }
        XCTAssertEqual(r.finishReason, .cancelled)
        XCTAssertEqual(r.tokenIDs.count, 5)
    }

    func testStopStringHoldback() {
        XCTAssertEqual(Generator.heldSuffixLength(of: "hello wor", stops: ["world"]), 3)
        XCTAssertEqual(Generator.heldSuffixLength(of: "hello", stops: ["world"]), 0)
        XCTAssertEqual(Generator.heldSuffixLength(of: "xx<", stops: ["<|end|>", "zz"]), 1)
        XCTAssertNotNil(Generator.earliestStop(in: "abcdef", stops: ["ef", "cd"]))
        let text = "abcdef"
        XCTAssertEqual(text[..<Generator.earliestStop(in: text, stops: ["ef", "cd"])!], "ab")
        XCTAssertNil(Generator.earliestStop(in: "abc", stops: ["x"]))
    }

    func testLogprobs() throws {
        let (gen, model) = try makeGenerator()
        let prompt = InferenceFixture.prompt(10)
        let r = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 6, logprobs: 3))
        var seq = prompt
        for e in r.events {
            let ref = referenceNextLogits(model, seq).asType(.float32)
            let lse = logSumExp(ref).item(Float.self)
            let lp = try XCTUnwrap(e.logprob)
            XCTAssertEqual(lp.id, e.tokenID)
            XCTAssertEqual(lp.logprob, ref[e.tokenID].item(Float.self) - lse, accuracy: 2e-3)
            XCTAssertEqual(e.topLogprobs.count, 3)
            // Greedy: the chosen token is the most likely one, and the alternatives are ordered.
            XCTAssertEqual(e.topLogprobs.first?.id, e.tokenID)
            XCTAssertEqual(e.topLogprobs.map(\.logprob), e.topLogprobs.map(\.logprob).sorted(by: >))
            XCTAssertLessThanOrEqual(lp.logprob, 0)
            seq.append(e.tokenID)
        }
        // None requested, none computed.
        let plain = try gen.generate(GenerationRequest(promptTokens: prompt, maxTokens: 2))
        XCTAssertNil(plain.events.first?.logprob)
    }

    // MARK: Sampling

    func testSamplingChain() throws {
        let logits = MLXArray([Float(0.0), 1.0, 5.0, 4.9, -2.0, 0.5])
        // Greedy.
        XCTAssertEqual(Sampler(.greedy).sample(logits), 2)
        // top-k 1 and a vanishing top-p are greedy at any temperature.
        for seed in 0 ..< 8 {
            XCTAssertEqual(Sampler(SamplingParameters(temperature: 1, topK: 1, seed: UInt64(seed))).sample(logits), 2)
            XCTAssertEqual(Sampler(SamplingParameters(temperature: 1, topP: 0.01, seed: UInt64(seed))).sample(logits), 2)
        }
        // top-k 2 only ever returns one of the two best; both appear over many draws.
        var seen = Set<Int>()
        let s = Sampler(SamplingParameters(temperature: 1, topK: 2, seed: 4))
        for _ in 0 ..< 200 { seen.insert(s.sample(logits)) }
        XCTAssertEqual(seen, [2, 3])
        // top-p 0.6: the best token holds ~52% and the next ~47%, so both are kept; nothing else is.
        var nucleus = Set<Int>()
        let p = Sampler(SamplingParameters(temperature: 1, topP: 0.6, seed: 5))
        for _ in 0 ..< 200 { nucleus.insert(p.sample(logits)) }
        XCTAssertEqual(nucleus, [2, 3])
        // A mask removes tokens before anything else.
        let allowed = MLXArray([true, true, false, false, true, true])
        XCTAssertEqual(Sampler(.greedy).sample(logits, allowed: allowed), 1)
        // The same seed repeats; a high temperature spreads over many tokens.
        let a = Sampler(SamplingParameters(temperature: 5, seed: 1))
        let b = Sampler(SamplingParameters(temperature: 5, seed: 1))
        let draws = (0 ..< 30).map { _ in a.sample(logits) }
        XCTAssertEqual(draws, (0 ..< 30).map { _ in b.sample(logits) })
        XCTAssertGreaterThan(Set(draws).count, 3)
    }

    func testSampledGenerationIsSeedDeterministic() throws {
        let (gen, _) = try makeGenerator()
        let prompt = InferenceFixture.prompt(10)
        func run(_ seed: UInt64) throws -> [Int] {
            try gen.generate(GenerationRequest(
                promptTokens: prompt, maxTokens: 16, sampling: SamplingParameters(temperature: 1.5, topK: 40, topP: 0.95, seed: seed))).tokenIDs
        }
        XCTAssertEqual(try run(1), try run(1))
        XCTAssertNotEqual(try run(1), try run(2))
    }

    func testGenerationDefaultsFromConfigFile() throws {
        let dir = try InferenceFixture.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(try GenerationDefaults.load(directory: dir), GenerationDefaults())
        try InferenceFixture.writeTokenizerFiles(to: dir, sampling: false)
        var d = try GenerationDefaults.load(directory: dir)
        XCTAssertEqual(d.sampling, .greedy)
        XCTAssertEqual(d.eosTokenIDs, [1, 351])
        XCTAssertEqual(d.bosTokenID, 2)
        try InferenceFixture.writeTokenizerFiles(to: dir, sampling: true)
        d = try GenerationDefaults.load(directory: dir)
        XCTAssertEqual(d.sampling, SamplingParameters(temperature: 0.7, topK: 20, topP: 0.9))
        try Data(#"{"do_sample": true, "eos_token_id": 7}"#.utf8).write(to: dir.appendingPathComponent("generation_config.json"))
        d = try GenerationDefaults.load(directory: dir)
        XCTAssertEqual(d.sampling, SamplingParameters(temperature: 1, topK: 50, topP: 1))
        XCTAssertEqual(d.eosTokenIDs, [7])
    }

    // MARK: Loading

    func testLoadsAnExportedDirectoryAndChats() throws {
        let tokDir = try InferenceFixture.tempDirectory("tok")
        let dir = try InferenceFixture.tempDirectory("exp")
        defer { try? FileManager.default.removeItem(at: tokDir); try? FileManager.default.removeItem(at: dir) }
        try InferenceFixture.writeTokenizerFiles(to: tokDir)
        let model = try InferenceFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let healer = try HealingFixture.healer(model: model, keep: keep)
        try InferenceFixture.exportDirectory(model: model, keep: keep, adapters: healer.adapters, tokenizerDirectory: tokDir, in: dir)

        let rt = try InferenceRuntime.load(directory: dir, dtype: .float32)
        XCTAssertEqual(rt.defaults.eosTokenIDs, [1, 351])
        XCTAssertEqual(rt.model.configuration.numExperts, 4)
        let tokens = try rt.promptTokens(messages: [["role": "user", "content": "the cat"]])
        XCTAssertEqual(tokens.first, rt.tokenizer.bosTokenID)
        XCTAssertEqual(tokens.last.flatMap { rt.tokenizer.token(for: $0) }, "\n")
        XCTAssertEqual(tokens.filter { $0 == rt.tokenizer.tokenID(for: "<|turn>") }.count, 2)
        let r = try rt.generator.generate(rt.request(promptTokens: tokens, maxTokens: 8))
        XCTAssertFalse(r.tokenIDs.isEmpty)
        // A directory with no chat template says so.
        try FileManager.default.removeItem(at: dir.appendingPathComponent("chat_template.jinja"))
        let bare = try InferenceRuntime.load(directory: dir, dtype: .float32)
        XCTAssertThrowsError(try bare.promptTokens(messages: [["role": "user", "content": "x"]]))
    }

    /// A 4-bit quantized checkpoint (mlx-community layout, mixed 4- and 8-bit modules) loads and decodes over
    /// the cache exactly as it runs without one.
    func testQuantizedCheckpointGeneratesOverTheCache() throws {
        let dir = try InferenceFixture.tempDirectory("q4")
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try QuantizedFixture.write(try HealingFixture.model(), to: dir)
        try InferenceFixture.writeTokenizerFiles(to: dir)
        let rt = try InferenceRuntime.load(directory: dir)
        XCTAssertTrue((rt.model.model.layers[1] as! Gemma4MoEDecoderLayer).selfAttn.qProj is QuantizedLinear)
        let prompt = InferenceFixture.prompt(13)
        let r = try rt.generator.generate(GenerationRequest(promptTokens: prompt, maxTokens: 20))
        XCTAssertEqual(r.tokenIDs.count, 20)
        var seq = prompt
        for t in r.tokenIDs {
            XCTAssertLessThan(gap(referenceNextLogits(rt.model, seq), to: t), 5e-3)
            seq.append(t)
        }
    }

    func testLoadsInBFloat16() throws {
        let tokDir = try InferenceFixture.tempDirectory("tok")
        let dir = try InferenceFixture.tempDirectory("bf")
        defer { try? FileManager.default.removeItem(at: tokDir); try? FileManager.default.removeItem(at: dir) }
        try InferenceFixture.writeTokenizerFiles(to: tokDir)
        let model = try InferenceFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let healer = try HealingFixture.healer(model: model, keep: keep)
        try InferenceFixture.exportDirectory(model: model, keep: keep, adapters: healer.adapters, tokenizerDirectory: tokDir, in: dir)
        let rt = try InferenceRuntime.load(directory: dir, dtype: .bfloat16)
        XCTAssertEqual((rt.model.model.layers[0] as! Gemma4MoEDecoderLayer).router.perExpertScale.dtype, .bfloat16)
        let r = try rt.generator.generate(GenerationRequest(promptTokens: InferenceFixture.prompt(10), maxTokens: 10, logprobs: 0))
        XCTAssertEqual(r.tokenIDs.count, 10)
        XCTAssertTrue(r.events.allSatisfy { ($0.logprob?.logprob.isFinite ?? false) })
    }
}

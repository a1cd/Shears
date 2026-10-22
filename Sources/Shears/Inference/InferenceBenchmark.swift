import Foundation
import MLX

/// Throughput and memory of the inference path for one model and prompt length.
public struct BenchmarkResult: Codable, Equatable {
    public var promptTokens: Int
    public var decodeTokens: Int
    public var repeats: Int
    /// Prompt tokens per second during prefill (mean over repeats).
    public var prefillTokensPerSecond: Double
    /// Generated tokens per second after the first (mean over repeats).
    public var decodeTokensPerSecond: Double
    /// Seconds from the request to its first token (mean over repeats).
    public var timeToFirstTokenSeconds: Double
    /// Peak MLX allocation during the runs, bytes.
    public var peakMLXMemoryBytes: Int
    /// Peak process physical footprint seen at the sampling points, bytes.
    public var peakPhysFootprintBytes: Int
    /// Physical footprint when the runs finished, bytes.
    public var finalPhysFootprintBytes: Int
}

public enum InferenceBenchmark {
    /// A deterministic prompt of `length` ordinary tokens (neither special nor a byte-fallback token).
    public static func syntheticPrompt(length: Int, tokenizer: BPETokenizer, vocabularySize: Int) -> [Int] {
        var pool: [Int] = []
        var id = 0
        let limit = min(vocabularySize, tokenizer.vocabularySize)
        while id < limit, pool.count < 4096 {
            if let t = tokenizer.token(for: id), !tokenizer.isSpecial(id), !(t.hasPrefix("<0x") && t.hasSuffix(">")) {
                pool.append(id)
            }
            id += 1
        }
        precondition(!pool.isEmpty, "the tokenizer has no ordinary tokens")
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        return (0 ..< length).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return pool[Int((state >> 33) % UInt64(pool.count))]
        }
    }

    /// Runs `repeats` greedy generations of `decodeTokens` tokens after a prompt of `promptTokens` tokens,
    /// each from an empty cache (after one warm-up run at the same prompt length).
    public static func run(
        runtime: InferenceRuntime, promptTokens: Int, decodeTokens: Int, repeats: Int = 3
    ) throws -> BenchmarkResult {
        let prompt = syntheticPrompt(
            length: promptTokens, tokenizer: runtime.tokenizer, vocabularySize: runtime.model.configuration.vocabSize)
        let generator = runtime.generator
        let savedReuse = generator.reusesPromptCache
        generator.reusesPromptCache = false
        defer { generator.reusesPromptCache = savedReuse; generator.resetCache() }

        _ = try generator.generate(GenerationRequest(promptTokens: prompt, maxTokens: 4)) // same shapes, compiles the kernels
        Memory.peakMemory = 0 // resets the peak
        var peakPhys = ProcessMemory.snapshot().footprint
        var prefill = 0.0, decode = 0.0, first = 0.0
        for _ in 0 ..< repeats {
            generator.resetCache()
            let start = Date()
            var firstAt: Date?
            var lastAt = start
            var count = 0
            let r = try generator.generate(GenerationRequest(promptTokens: prompt, maxTokens: decodeTokens)) { event in
                let now = Date()
                if firstAt == nil {
                    firstAt = now
                    peakPhys = max(peakPhys, ProcessMemory.snapshot().footprint)
                }
                if !event.isTail { count += 1; lastAt = now }
                return true
            }
            peakPhys = max(peakPhys, ProcessMemory.snapshot().footprint)
            let ttft = (firstAt ?? start).timeIntervalSince(start)
            first += ttft
            prefill += r.promptSeconds > 0 ? Double(promptTokens) / r.promptSeconds : 0
            let span = lastAt.timeIntervalSince(firstAt ?? start)
            decode += span > 0 && count > 1 ? Double(count - 1) / span : 0
        }
        let n = Double(max(repeats, 1))
        return BenchmarkResult(
            promptTokens: promptTokens, decodeTokens: decodeTokens, repeats: repeats,
            prefillTokensPerSecond: prefill / n, decodeTokensPerSecond: decode / n, timeToFirstTokenSeconds: first / n,
            peakMLXMemoryBytes: Memory.peakMemory, peakPhysFootprintBytes: peakPhys,
            finalPhysFootprintBytes: ProcessMemory.snapshot().footprint)
    }
}

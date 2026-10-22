import Foundation
import MLX
import XCTest

@testable import Shears

/// The `shears` runner end to end on a tiny random model: select, prepass, heal (two block stages
/// and a plan stage), export and gguf, driven by the checked-in sample config.
final class RunnerTests: ShearsTestCase {
    func tempDir(_ name: String = "runner") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    static var sampleConfig: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Examples/tiny-config.json")
    }

    /// A working directory with the sample config, a tiny checkpoint and a row file at the paths
    /// the config names.
    func workspace(sameModelAs other: URL? = nil, configEdit: ((inout [String: Any]) -> Void)? = nil) throws -> URL {
        let dir = try tempDir()
        if let other {  // a fresh random model differs from every other (init is unseeded), so copy it
            for f in ["model", "rows.jsonl"] {
                try FileManager.default.copyItem(at: other.appendingPathComponent(f), to: dir.appendingPathComponent(f))
            }
            var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.sampleConfig)) as! [String: Any]
            configEdit?(&cfg)
            try JSONSerialization.data(withJSONObject: cfg, options: [.sortedKeys, .prettyPrinted])
                .write(to: dir.appendingPathComponent("config.json"))
            return dir
        }
        let tok = try tempDir("tok")
        try GGUFTinyGemma4.writeTokenizer(to: tok)
        let model = try HealingFixture.model()
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        o.sourceDirectory = tok
        try MergeAndSliceExport.write(
            model: model, keep: Array(repeating: Array(0 ..< HealingFixture.experts), count: 4), adapters: nil,
            to: dir.appendingPathComponent("model"), options: o)
        var lines: [String] = []
        for i in 0 ..< 4 {
            let r = HealingFixture.row(length: 20 + 2 * (i % 2), seed: UInt64(10 + i), key: "row-\(i)")
            let rec: [String: Any] = [
                "key": r.key, "tokens": r.tokens.map { Int($0) }, "answerStart": r.length - 6,
                "klAnchors": [["position": r.length - 7, "weight": 5]], "klSampleRange": [r.length - 6, r.length - 1],
                "validation": i == 3,
            ]
            lines.append(String(data: try JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]), encoding: .utf8)!)
        }
        try (lines.joined(separator: "\n") + "\n").write(to: dir.appendingPathComponent("rows.jsonl"), atomically: true, encoding: .utf8)
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.sampleConfig)) as! [String: Any]
        configEdit?(&cfg)
        try JSONSerialization.data(withJSONObject: cfg, options: [.sortedKeys, .prettyPrinted])
            .write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func config(_ dir: URL) throws -> RunnerConfig { try RunnerConfig.load(from: dir.appendingPathComponent("config.json")) }

    func stageAdapters(_ dir: URL, _ stage: String) throws -> [String: MLXArray] {
        try loadArrays(url: dir.appendingPathComponent("out/stage-\(stage)/adapters.safetensors"))
    }

    func testSampleConfigRunsEveryCommandEndToEnd() throws {
        let dir = try workspace()
        let cfg = try config(dir)
        var lines: [String] = []
        var env = RunEnvironment()
        env.log = { lines.append($0) }

        XCTAssertEqual(try ShearsRunner.execute(.select, config: cfg, environment: env), .finished)
        let keep = try KeepSets.load(from: dir.appendingPathComponent("out/keep.json"), layerCount: 4)
        XCTAssertEqual(keep.map(\.count), [4, 4, 4, 4])

        XCTAssertEqual(try ShearsRunner.execute(.prepass, config: cfg, environment: env), .finished)
        // A rerun finds every stage done.
        XCTAssertEqual(try ShearsRunner.execute(.prepass, config: cfg, environment: env), .finished)
        XCTAssertTrue(lines.contains { $0.contains("stages already done") && !$0.contains(" 0 stages") })

        XCTAssertEqual(try ShearsRunner.execute(.heal, config: cfg, environment: env), .finished)
        for stage in ["A", "B", "C"] {
            let t = try stageAdapters(dir, stage)
            XCTAssertFalse(t.isEmpty)
            XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("out/stage-\(stage)/done.json").path))
        }
        // Stage B started from A's adapters with its own schedule: its first rate is on B's warmup (0).
        let bSteps = try String(contentsOf: dir.appendingPathComponent("out/stage-B/steps.jsonl"), encoding: .utf8)
            .split(separator: "\n").map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        XCTAssertEqual(bSteps.count, 12)  // 2 blocks x 6 steps
        XCTAssertEqual(bSteps[0]["lr"] as! Double, 0, accuracy: 1e-12)
        // Progress lines carry step, loss, seconds per step and peak memory.
        let progress = lines.first { $0.hasPrefix("[A block 1/2] step 2/6") }
        XCTAssertNotNil(progress)
        for part in ["loss", "lr", "s/step", "peak"] { XCTAssertTrue(progress?.contains(part) ?? false, part) }
        // A finished stage is not run again.
        lines.removeAll()
        XCTAssertEqual(try ShearsRunner.execute(.heal, config: cfg, environment: env), .finished)
        XCTAssertEqual(lines.filter { $0.contains("already done") }.count, 3)

        XCTAssertEqual(try ShearsRunner.execute(.export, config: cfg, environment: env), .finished)
        let exported = try Gemma4Checkpoint(directory: dir.appendingPathComponent("out/export"))
        XCTAssertEqual(exported.configuration.numExperts, 4)

        XCTAssertEqual(try ShearsRunner.execute(.gguf, config: cfg, environment: env), .finished)
        let gguf = dir.appendingPathComponent("out/model.gguf")
        let reader = try GGUFReader(url: gguf)
        XCTAssertNotNil(try? reader.floats("blk.0.attn_norm.weight"))

        // The healed export is the stage-C model: merged adapters move it away from the unhealed prune.
        var unhealed = cfg
        unhealed.export = RunnerConfig.Export(directory: "out/export-plain", adapters: "none", dtype: "float32", layersPerShard: nil)
        _ = try ShearsRunner.execute(.export, config: unhealed, environment: env)
        let a = try Gemma4Checkpoint(directory: dir.appendingPathComponent("out/export")).loadModel(dtype: .float32)
        let b = try Gemma4Checkpoint(directory: dir.appendingPathComponent("out/export-plain")).loadModel(dtype: .float32)
        let t = MLXArray(HealingFixture.row().tokens)[.newAxis]
        XCTAssertGreaterThan(abs(a(t) - b(t)).max().item(Float.self), 1e-3)
    }

    /// A stop file mid-stage saves a checkpoint and a rerun finishes the stage with the adapters an
    /// uninterrupted run produces.
    func testHealStopsOnTheStopFileAndResumes() throws {
        let straight = try workspace { $0["stages"] = [($0["stages"] as! [[String: Any]])[0]] }
        let interrupted = try workspace(sameModelAs: straight) { $0["stages"] = [($0["stages"] as! [[String: Any]])[0]] }
        var quiet = RunEnvironment()
        quiet.log = { _ in }
        // Both runs use one keep file (selection on two different random models would pick different sets).
        _ = try ShearsRunner.execute(.select, config: try config(straight), environment: quiet)
        try FileManager.default.createDirectory(at: interrupted.appendingPathComponent("out"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: straight.appendingPathComponent("out/keep.json"), to: interrupted.appendingPathComponent("out/keep.json"))
        for dir in [straight, interrupted] {
            _ = try ShearsRunner.execute(.prepass, config: try config(dir), environment: quiet)
        }
        _ = try ShearsRunner.execute(.heal, config: try config(straight), environment: quiet)

        let cfg = try config(interrupted)
        let stopFile = interrupted.appendingPathComponent("STOP")
        var env = RunEnvironment()
        env.log = { line in
            if line.hasPrefix("[A block 2/2] step 4/6") { FileManager.default.createFile(atPath: stopFile.path, contents: Data()) }
        }
        let first = try ShearsRunner.execute(.heal, config: cfg, environment: env)
        guard case .stopped = first else { return XCTFail("expected a stop, got \(first)") }
        XCTAssertFalse(FileManager.default.fileExists(atPath: stopFile.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: interrupted.appendingPathComponent("out/stage-A/done.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: interrupted.appendingPathComponent("checkpoints/stage-A/block-1/ckpt/meta.json").path))

        env.log = { _ in }
        XCTAssertEqual(try ShearsRunner.execute(.heal, config: cfg, environment: env), .finished)
        let want = try stageAdapters(straight, "A"), got = try stageAdapters(interrupted, "A")
        XCTAssertEqual(Set(want.keys), Set(got.keys))
        var worst: Float = 0
        for (k, v) in want { worst = max(worst, abs(v - got[k]!).max().item(Float.self)) }
        XCTAssertLessThan(worst, 2e-3)
    }

    func testResidentLayersAndMemoryBudgetFromTheCommandLine() throws {
        let stageA: (inout [String: Any]) -> Void = { $0["stages"] = [($0["stages"] as! [[String: Any]])[0]] }
        let straight = try workspace(configEdit: stageA)
        let windowed = try workspace(sameModelAs: straight, configEdit: stageA)
        var quiet = RunEnvironment()
        quiet.log = { _ in }
        _ = try ShearsRunner.execute(.select, config: try config(straight), environment: quiet)
        try FileManager.default.createDirectory(at: windowed.appendingPathComponent("out"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: straight.appendingPathComponent("out/keep.json"), to: windowed.appendingPathComponent("out/keep.json"))
        for dir in [straight, windowed] { _ = try ShearsRunner.execute(.prepass, config: try config(dir), environment: quiet) }
        _ = try ShearsRunner.execute(.heal, config: try config(straight), environment: quiet)

        // A budget too small for even one layer is refused (exit 2) before any step runs.
        let cfgPath = windowed.appendingPathComponent("config.json").path
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears", "heal", "--config", cfgPath, "--memory-budget", "1KB"], installSignalHandlers: false), 2)
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears", "heal", "--config", cfgPath, "--memory-budget", "0"], installSignalHandlers: false), 2)
        let stepsFile = windowed.appendingPathComponent("out/stage-A/steps.jsonl")
        let before = (try? String(contentsOf: stepsFile)) ?? ""
        XCTAssertFalse(before.contains("\"step\""), "a refused budget must not train")

        // A generous budget is only a ceiling: the stage trains like the unbudgeted one (whole block).
        XCTAssertEqual(
            ShearsRunner.main(arguments: ["shears", "heal", "--config", cfgPath, "--memory-budget", "1GB"], installSignalHandlers: false), 0)
        // The resident-layers knob is gone from the command line and the config.
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears", "heal", "--config", cfgPath, "--resident-layers", "1"], installSignalHandlers: false), 2)
        let want = try stageAdapters(straight, "A"), got = try stageAdapters(windowed, "A")
        var worst: Float = 0
        for (k, v) in want { worst = max(worst, abs(v - got[k]!).max().item(Float.self)) }
        XCTAssertLessThan(worst, 2e-3)
        let log = try String(contentsOf: windowed.appendingPathComponent("out/stage-A/steps.jsonl"))
        XCTAssertTrue(log.contains("footprintPeakBytes") && log.contains("mlxPeakBytes") && log.contains("fileBackedBytes"))
        XCTAssertEqual(parseSizeGB("3GB"), 3)
        XCTAssertEqual(parseSizeGB("512MB"), 0.5)
        XCTAssertNil(parseSizeGB("lots"))
    }

    func testRunCommandSelectsTrainsAndExportsBlockSequentially() throws {
        let dir = try workspace(configEdit: { $0["export"] = ["adapters": "out/stage-pipeline/adapters.safetensors", "dtype": "bfloat16"] })
        var quiet = RunEnvironment()
        var lines: [String] = []
        quiet.log = { lines.append($0) }
        _ = try ShearsRunner.execute(.run, config: try config(dir), environment: quiet)
        XCTAssertTrue(lines.contains { $0.contains("layers resident") }, "the chosen window and why are reported")
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("out/keep.json").path))
        let adapters = try stageAdapters(dir, "pipeline")
        XCTAssertTrue(adapters.keys.contains { $0.hasPrefix("layers.0.") } && adapters.keys.contains { $0.hasPrefix("layers.3.") })
        // Rerunning resumes (selection records, batches, blocks): nothing is redone and the result is unchanged.
        let keepBefore = try Data(contentsOf: dir.appendingPathComponent("out/keep.json"))
        lines.removeAll()
        _ = try ShearsRunner.execute(.run, config: try config(dir), environment: quiet)
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("out/keep.json")), keepBefore)
        let again = try stageAdapters(dir, "pipeline")
        XCTAssertEqual(Set(again.keys), Set(adapters.keys))
        for (k, v) in adapters { assertClose(again[k]!, v, atol: 0, rtol: 0, k) }
        XCTAssertFalse(lines.contains { $0.contains("selected layer") }, "no layer is selected twice")
        // The pipeline's adapters export like any stage's.
        _ = try ShearsRunner.execute(.export, config: try config(dir), environment: quiet)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("out/export/config.json").path))
    }

    func testSweepRunsEndToEndAndWritesATableAndJSON() throws {
        let dir = try workspace()
        var quiet = RunEnvironment()
        quiet.log = { _ in }
        _ = try ShearsRunner.execute(.sweep, config: try config(dir), environment: quiet)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("out/sweep.json"))) as! [String: Any]
        let results = json["results"] as! [[String: Any]]
        XCTAssertEqual(results.count, 16, "4 variants x 2 learning rates x 2 step counts")
        XCTAssertEqual(Set(results.map { $0["variant"] as! String }), ["base", "rows2", "plateau", "bf16"])
        XCTAssertTrue(results.allSatisfy { ($0["heldOutLoss"] as! Double).isFinite })
        XCTAssertEqual(json["heldOutRows"] as? Int, 1)
        // Same initial adapters and optimizer, so at the same steps a different peak lr gives a different score.
        let byLR = Dictionary(grouping: results.filter { $0["steps"] as? Int == 4 && $0["variant"] as? String == "base" }, by: { $0["learningRate"] as! Double })
        XCTAssertEqual(byLR.count, 2)
        let scores = byLR.values.map { ($0[0]["heldOutLoss"] as! Double) }
        XCTAssertNotEqual(scores[0], scores[1])
        XCTAssertGreaterThan(json["unhealedLoss"] as! Double, 0)
        // The variants are really different training: plateau annealing fired in its runs (and nowhere else), and
        // stacking two rows per step changes the result.
        func score(_ v: String, _ lr: Double) -> Double {
            results.first { $0["variant"] as? String == v && $0["learningRate"] as? Double == lr && $0["steps"] as? Int == 4 }!["heldOutLoss"] as! Double
        }
        XCTAssertTrue(results.filter { $0["variant"] as? String == "plateau" && $0["steps"] as? Int == 4 }.allSatisfy { $0["annealedAt"] != nil })
        XCTAssertTrue(results.filter { $0["variant"] as? String != "plateau" }.allSatisfy { $0["annealedAt"] == nil })
        XCTAssertGreaterThan(abs(score("rows2", 0.01) - score("base", 0.01)), 1e-5)
        XCTAssertGreaterThan(abs(score("bf16", 0.01) - score("base", 0.01)), 1e-6)
        // A second sweep reuses the prepared block-0 data.
        _ = try ShearsRunner.execute(.sweep, config: try config(dir), environment: quiet)
    }

    func testConfigRejectsUnknownKeysAndBadRows() throws {
        let dir = try workspace { $0["stages"] = [["name": "A", "mode": "block", "steps": 2, "learnigRate": 0.1]] }
        XCTAssertThrowsError(try config(dir)) { XCTAssertTrue("\($0)".contains("learnigRate")) }

        let ok = try workspace()
        try "{\"key\":\"a\",\"tokens\":[1,2]}\n{\"key\":\"a\",\"tokens\":[3]}\n".write(
            to: ok.appendingPathComponent("rows.jsonl"), atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try RowFile.load(from: ok.appendingPathComponent("rows.jsonl"))) {
            XCTAssertTrue("\($0)".contains("duplicate key"))
        }
    }

    func testMainParsesArgumentsAndReportsStatus() throws {
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears"], installSignalHandlers: false), 2)
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears", "bogus"], installSignalHandlers: false), 2)
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears", "heal"], installSignalHandlers: false), 2)
        XCTAssertEqual(ShearsRunner.main(arguments: ["shears", "help"], installSignalHandlers: false), 0)
        let dir = try workspace()
        // heal before select/prepass: a config error (exit 2), not a crash.
        XCTAssertEqual(
            ShearsRunner.main(arguments: ["shears", "heal", "--config", dir.appendingPathComponent("config.json").path], installSignalHandlers: false), 2)
    }
}

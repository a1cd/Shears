import CShearsDiag
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Subcommands of `shears`.
public enum ShearsCommand: String, CaseIterable {
    case select, prepass, heal, export, gguf, run, sweep
}

public enum RunOutcome: Equatable {
    case finished
    /// A stop was honoured (signal, stop file, memory limit); everything needed to resume is saved.
    case stopped(String)
}

/// What a run needs from its caller.
public struct RunEnvironment {
    /// Progress lines. The default writes to standard error, one flushed line each.
    public var log: (String) -> Void = { line in
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
    /// Polled between steps (and between pre-pass batches).
    public var shouldStop: (() -> Bool)?
    public init() {}
}

/// Runs the subcommands against a config. The same entry point serves the `shears-cli` executable and
/// the tests.
public enum ShearsRunner {
    public static let usage = """
        usage: shears-cli <command> --config <file.json> [--stage <name>]

        commands (each reads the same config file and is safe to rerun: it resumes):
          select    choose the kept experts per layer; writes the keep file
          prepass   run the teacher once over the rows into the boundary store
          run       the block-sequential pipeline: per block, teacher pass + expert selection, then training on
                    clean and dirty inputs, then the healed block's outputs for the next block (config: pipeline)
          sweep     short healing runs on block 0 over a grid of learning rates and step counts, scored on
                    held-out rows (config: sweep); prints a table and writes JSON
          heal      run the config's stages in order (or just --stage <name>)
          export    merge the adapters and slice to the kept experts (Hugging Face layout)
          gguf      convert the exported directory to a GGUF file
          sample    shears-cli sample <dir>: write a tiny random Gemma 4 MoE (model/) and rows.jsonl into <dir>, to try
                    the commands on with Examples/tiny-config.json (copy it to <dir>/config.json). Needs no --config
          device    shears-cli device: run one small matrix product on MLX's default device and print which device that
                    was ("mlx device: gpu" or "mlx device: cpu"). Needs no --config. Every other command prints the same
                    "mlx device:" line first

        options of heal:
          --memory-budget <size>    a ceiling on MLX memory, e.g. 3GB or 512MB. It is only checked: the run needs what
                                    its hungriest stage needs (training holds as many layers as fit under that,
                                    automatically) and is refused up front, with the amount, if that is more

        options of run and heal:
          --keep-intermediates <v>  auto (default), fast, lowMemory or a layer count: how many layers of a block keep their
                                    forward intermediates for the backward pass instead of recomputing them. Same results
                                    for every value; fast is quickest and uses the most memory, lowMemory the least. auto
                                    is lowMemory, plus as many kept layers as a --memory-budget leaves room for

        A stop file, SIGINT or SIGTERM makes a running command save a checkpoint and exit with status 3;
        run it again to continue. Progress goes to standard error.
        """

    public static func main(arguments: [String], installSignalHandlers: Bool = true) -> Int32 {
        shears_install_terminate_handler()
        var args = Array(arguments.dropFirst())
        guard let name = args.first else {
            FileHandle.standardError.write(Data((usage + "\n").utf8))
            return 2
        }
        if name == "help" || name == "--help" || name == "-h" {
            print(usage)
            return 0
        }
        args.removeFirst()
        if name == "device" {
            // One real op on the default device, so the answer is "what ran", not "what is installed": a build whose
            // GPU backend cannot start (no driver, a library of the wrong version) fails here, not later.
            // Every stage goes to standard error, flushed, before it starts, so a crash shows how far it got.
            func stage(_ text: String) {
                FileHandle.standardError.write(Data(("device: " + text + "\n").utf8))
            }
            stage("choosing MLX's default device")
            let device = Device.defaultDevice().deviceType.rawValue
            stage("default device is \(device)")
            stage("creating a 4x4 array")
            let x = MLXArray((0 ..< 16).map { Float($0) }, [4, 4])
            stage("evaluating x + x")
            let doubled = x + x
            eval(doubled)
            stage("evaluating x @ x")
            let product = matmul(x, x)
            eval(product)
            stage("reading the sum back")
            let sum = product.sum().item(Float.self)
            print("mlx device: \(device)")
            print("probe: 4x4 matrix product on \(device), sum \(sum) (expected 3920)")
            fflush(stdout)
            stage("done, exiting")
            return sum == 3920 ? 0 : 1
        }
        if name == "sample" {
            guard args.count == 1, !args[0].hasPrefix("-") else {
                FileHandle.standardError.write(Data("usage: shears-cli sample <dir>\n".utf8))
                return 2
            }
            do {
                try SampleWorkspace.write(to: URL(fileURLWithPath: args[0]))
                print("wrote \(args[0])/model and \(args[0])/rows.jsonl")
                return 0
            } catch {
                FileHandle.standardError.write(Data("shears-cli: \(error)\n".utf8))
                return 1
            }
        }
        do {
            guard let command = ShearsCommand(rawValue: name) else {
                throw RunnerError.usage("unknown command '\(name)'\n\(usage)")
            }
            var configPath: String?
            var stage: String?
            var budget: Double?
            var keep: ActivationKeep?
            var i = 0
            while i < args.count {
                switch args[i] {
                case "--config", "-c":
                    i += 1
                    guard i < args.count else { throw RunnerError.usage("--config needs a file") }
                    configPath = args[i]
                case "--stage":
                    i += 1
                    guard i < args.count else { throw RunnerError.usage("--stage needs a name") }
                    stage = args[i]
                case "--memory-budget":
                    i += 1
                    guard i < args.count, let gb = parseSizeGB(args[i]) else {
                        throw RunnerError.usage("--memory-budget needs a size such as 3GB or 512MB")
                    }
                    budget = gb
                case "--keep-intermediates":
                    i += 1
                    guard i < args.count, let k = try? ActivationKeep(parsing: args[i]) else {
                        throw RunnerError.usage("--keep-intermediates needs auto, fast, lowMemory or a layer count")
                    }
                    keep = k
                default: throw RunnerError.usage("unexpected argument '\(args[i])'\n\(usage)")
                }
                i += 1
            }
            guard let configPath else { throw RunnerError.usage("--config <file.json> is required\n\(usage)") }
            var config = try RunnerConfig.load(from: URL(fileURLWithPath: configPath))
            if let budget {
                var m = config.memory ?? RunnerConfig.Memory()
                m.budgetGB = budget
                config.memory = m
            }
            if let keep {
                config.pipeline?.keepIntermediates = keep
                config.stages = config.stages?.map { var s = $0; s.keepIntermediates = keep; return s }
            }
            var env = RunEnvironment()
            env.log("mlx device: \(Device.defaultDevice().deviceType.rawValue)")
            if installSignalHandlers {
                let flag = StopSignal.install()
                env.shouldStop = { flag.requested }
                setvbuf(stdout, nil, _IOLBF, 0)
            }
            let outcome = try execute(command, config: config, stage: stage, environment: env)
            if case .stopped(let why) = outcome {
                env.log("stopped: \(why). Run the same command again to resume.")
                return 3
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data("shears-cli: \(error)\n".utf8))
            if case HealingError.budget = error { return 2 }
            return error is RunnerError ? 2 : 1
        }
    }

    public static func execute(
        _ command: ShearsCommand, config: RunnerConfig, stage: String? = nil, environment: RunEnvironment = .init()
    ) throws -> RunOutcome {
        let run = Run(config: config, env: environment)
        switch command {
        case .select: return try run.select()
        case .prepass: return try run.prepass()
        case .heal: return try run.heal(only: stage)
        case .export: return try run.export()
        case .gguf: return try run.gguf()
        case .run: return try run.pipeline()
        case .sweep: return try run.sweep()
        }
    }
}

/// SIGINT and SIGTERM ask the current command to stop at its next step; a second signal kills it.
public final class StopSignal {
    private let lock = NSLock()
    private var flag = false
    private var sources: [DispatchSourceSignal] = []
    public var requested: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func raise() { lock.lock(); flag = true; lock.unlock() }

    public static func install() -> StopSignal {
        let s = StopSignal()
        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .global())
            src.setEventHandler {
                if s.requested { exit(130) }
                s.raise()
            }
            src.resume()
            s.sources.append(src)
        }
        return s
    }
}

// MARK: - The commands

final class Run {
    let c: RunnerConfig
    let env: RunEnvironment
    let fm = FileManager.default
    init(config: RunnerConfig, env: RunEnvironment) {
        self.c = config
        self.env = env
    }

    func log(_ s: String) { env.log(s) }

    // paths
    func required(_ value: String?, _ what: String) throws -> URL {
        guard let value else { throw RunnerError.config("this command needs '\(what)' in the config") }
        return c.resolve(value)
    }
    var outputDir: URL { c.resolve(c.outputDir ?? "out") }
    var keepURL: URL { c.resolve(c.keep ?? outputDir.appendingPathComponent("keep.json").path) }
    func stageDir(_ name: String) -> URL { outputDir.appendingPathComponent("stage-\(name)") }
    func checkpointRoot() -> URL? { c.checkpointDir.map { c.resolve($0) } }

    var memory: MemoryConfiguration {
        let gb = 1024.0 * 1024 * 1024
        var m = MemoryConfiguration()
        m.memoryLimitBytes = c.memory?.memoryLimitGB.map { Int($0 * gb) }
        m.cacheLimitBytes = c.memory?.cacheLimitGB.map { Int($0 * gb) }
        m.peakStopBytes = c.memory?.peakStopGB.map { Int($0 * gb) }
        return m
    }

    func checkpoint() throws -> Gemma4Checkpoint { try Gemma4Checkpoint(directory: c.resolve(c.model)) }
    var loadDType: DType? { get throws { try parseDType(c.dtype) } }

    func records() throws -> [RowRecord] { try RowFile.load(from: try required(c.rows, "rows")) }

    func keepSets(_ cp: Gemma4Checkpoint) throws -> [[Int]] {
        guard fm.fileExists(atPath: keepURL.path) else {
            throw RunnerError.config("no keep file at \(keepURL.path); run 'shears-cli select' or point 'keep' at one")
        }
        return try KeepSets.load(from: keepURL, layerCount: cp.configuration.numHiddenLayers)
    }

    func blockEnds(_ cp: Gemma4Checkpoint) throws -> [Int] {
        if let e = c.blockEnds {
            guard let l = e.last, l == cp.configuration.numHiddenLayers - 1, zip(e, e.dropFirst()).allSatisfy({ $0 < $1 }) else {
                throw RunnerError.config("blockEnds must increase and end at layer \(cp.configuration.numHiddenLayers - 1)")
            }
            return e
        }
        let n = cp.configuration.numHiddenLayers
        var ends = cp.configuration.layerTypes.indices.filter { cp.configuration.layerTypes[$0] == "full_attention" }
        if ends.last != n - 1 { ends.append(n - 1) }
        return ends
    }

    func adapterConfiguration(model: Gemma4TextModel? = nil, cp: Gemma4Checkpoint, keptExperts: Int) throws -> AdapterConfiguration {
        var a = AdapterConfiguration()
        if let x = c.adapters?.expertRank { a.expertRank = x }
        if let x = c.adapters?.denseRank { a.denseRank = x }
        if let x = c.adapters?.routerRank { a.routerRank = x }
        if let x = c.adapters?.routerLRMultiplier { a.routerLRMultiplier = x }
        if let x = c.adapters?.sharedExperts { a.sharedExperts = x }
        if let x = c.adapters?.scale { a.scale = x }
        if let q = c.adapters?.qat {
            guard q == "q4_k_m" else { throw RunnerError.config("adapters.qat: only 'q4_k_m' is supported") }
            // Only shapes are read, so a model that was never evaluated does.
            let shape = model ?? Gemma4TextModel(cp.configuration)
            a.qat = QATConfiguration.q4KM(
                for: shape, keptExperts: keptExperts, exportDType: try parseDType(c.export?.dtype) ?? .bfloat16)
        }
        return a
    }

    func atomicWrite(_ tensors: [String: MLXArray], to url: URL) throws {
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + ".partial")
        try MLX.save(arrays: tensors, url: tmp.deletingLastPathComponent().appendingPathComponent(tmp.lastPathComponent + ".safetensors"))
        let written = tmp.deletingLastPathComponent().appendingPathComponent(tmp.lastPathComponent + ".safetensors")
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        try fm.moveItem(at: written, to: url)
    }

    // MARK: select

    func select() throws -> RunOutcome {
        guard let sc = c.select else { throw RunnerError.config("select needs a 'select' section with 'keep'") }
        memory.apply()
        MLXRandom.seed(c.seed ?? 0)
        let cp = try checkpoint()
        let items = try records().map(\.selectionItem)
        let dir = outputDir.appendingPathComponent("select")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        var existing: [LayerSelection] = []
        for f in try fm.contentsOfDirectory(atPath: dir.path).sorted() where f.hasPrefix("layer-") && f.hasSuffix(".json") {
            existing.append(try JSONDecoder().decode(LayerSelection.self, from: Data(contentsOf: dir.appendingPathComponent(f))))
        }
        if !existing.isEmpty { log("select: \(existing.count) layers already searched") }
        var s = SelectionConfiguration(keep: sc.keep)
        if let x = sc.importanceTokens { s.importanceTokens = x }
        if let x = sc.poolByMass { s.poolByMass = x }
        if let x = sc.poolByImportance { s.poolByImportance = x }
        if let x = sc.poolSize { s.poolSize = x }
        if let x = sc.swapRounds { s.swapRounds = x }
        if let x = sc.timeLimitPerLayer { s.timeLimitPerLayer = x == 0 ? nil : x }
        if let x = sc.candidateBatch { s.candidateBatch = x }
        if let x = sc.seed { s.seed = x }
        if let x = sc.importanceCandidates { s.importanceCandidates = x }
        let model = try cp.loadModel(dtype: try loadDType)
        model.train(false)
        let started = Date()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let layers = cp.configuration.numHiddenLayers
        let result = ExpertSelector(model: model, configuration: s).select(items: items, existing: existing) { rec in
            let url = dir.appendingPathComponent(String(format: "layer-%03d.json", rec.layer))
            if let data = try? encoder.encode(rec) { try? data.write(to: url, options: .atomic) }
            self.log("select: layer \(rec.layer + 1)/\(layers) kept \(rec.chosen.count)  error \(rec.chosenError)  \(Int(Date().timeIntervalSince(started))) s")
        }
        try fm.createDirectory(at: keepURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try KeepSets.save(KeepSets.keep(from: result), to: keepURL)
        log("select: wrote \(keepURL.path)")
        return .finished
    }

    // MARK: prepass

    func prepass() throws -> RunOutcome {
        memory.apply()
        let cp = try checkpoint()
        let rows = try records().map(\.healingRow)
        let storeURL = try required(c.store, "store")
        var o = TeacherPrepass.Options()
        o.storageDType = try parseDType(c.prepass?.storageDType)
        if let x = c.prepass?.maxBatchTokens { o.maxBatchTokens = x }
        if let x = c.prepass?.maxBatchRows { o.maxBatchRows = x }
        let ends = try blockEnds(cp)
        let store = try DiskBoundaryStore(directory: storeURL)
        let started = Date()
        let prepass = TeacherPrepass(source: Gemma4LayerSource(checkpoint: cp, dtype: try loadDType), blockEnds: ends, options: o)
        let report = try prepass.run(rows: rows, store: store, shouldStop: env.shouldStop, onStageFinished: { stage in
            self.log("prepass: stage \(stage)/\(ends.count) done  \(Int(Date().timeIntervalSince(started))) s  peak \(gb(Memory.peakMemory))")
        })
        if report.stopped { return .stopped("stop requested during the pre-pass") }
        log("prepass: \(report.batchesRun) batches, \(report.skippedStages.count) stages already done")
        return .finished
    }

    // MARK: heal

    struct StageDone: Codable { var steps: Int; var finished: String }

    func heal(only: String?) throws -> RunOutcome {
        guard let stages = c.stages, !stages.isEmpty else { throw RunnerError.config("heal needs 'stages'") }
        guard Set(stages.map(\.name)).count == stages.count else { throw RunnerError.config("stage names must be unique") }
        if let only, !stages.contains(where: { $0.name == only }) { throw RunnerError.config("no stage named '\(only)'") }
        memory.apply()
        for (i, st) in stages.enumerated() {
            if let only, st.name != only { continue }
            let dir = stageDir(st.name)
            if fm.fileExists(atPath: dir.appendingPathComponent("done.json").path) {
                log("stage \(st.name): already done")
                continue
            }
            let previous = i > 0 ? stageDir(stages[i - 1].name).appendingPathComponent("adapters.safetensors") : nil
            let outcome = try runStage(st, previous: previous)
            if case .stopped = outcome { return outcome }
        }
        return .finished
    }

    func configuration(_ st: RunnerConfig.Stage) -> HealingConfiguration {
        var h = HealingConfiguration()
        if let x = st.learningRate { h.learningRate = x }
        if let x = st.learningRateFloor { h.learningRateFloor = x }
        if let x = st.warmup { h.warmupSteps = x }
        h.scheduleLength = st.scheduleLength
        if let x = st.scheduleOffset { h.scheduleOffset = x }
        if let x = st.scheduleLengthChanges { h.scheduleLengthChanges = x.map { .init(from: $0.from, total: $0.total) } }
        if let x = st.weightDecay { h.weightDecay = x }
        if let x = st.clipNorm { h.clipNorm = x }
        if let x = st.maxLoss { h.maxLoss = x }
        if let x = st.checkpointBlocks { h.checkpointBlocks = x }
        if let x = st.keepIntermediates { h.keepIntermediates = x }
        if let x = st.dirtyLambda { h.dirtyLambda = x }
        if let x = st.pinWeight { h.pinWeight = x }
        if let x = st.klWeight { h.klWeight = x }
        if let x = st.endSamplePositions { h.endSamplePositions = x }
        if let x = st.trainFromLayer { h.trainFromLayer = x }
        if let x = st.answerPinWeight { h.answerPinWeight = x }
        if let x = st.checkpointEvery { h.checkpointEvery = x }
        h.milestoneEvery = st.milestoneEvery ?? 0
        h.seed = st.seed ?? c.seed ?? 0
        h.evaluateEvery = Int.max / 2
        return h
    }

    func initialAdapters(_ st: RunnerConfig.Stage, previous: URL?) throws -> [String: MLXArray]? {
        guard let spec = st.initialAdapters else { return nil }
        let url: URL
        if spec == "previous" {
            guard let previous, fm.fileExists(atPath: previous.path) else {
                throw RunnerError.config("stage \(st.name): 'previous' has no earlier stage with adapters")
            }
            url = previous
        } else {
            url = c.resolve(spec)
        }
        return try loadArrays(url: url)
    }

    func runStage(_ st: RunnerConfig.Stage, previous: URL?) throws -> RunOutcome {
        let cp = try checkpoint()
        let keep = try keepSets(cp)
        let all = try records()
        let rows = all.map(\.healingRow)
        let base = try c.rowSource(rows: rows)
        let source: HealingRowSource = OffsetRows(base: base, offset: st.rowOffset ?? 0)
        let ends = try blockEnds(cp)
        let storeURL = try required(c.store, "store")
        let store = try DiskBoundaryStore(directory: storeURL)
        let config = configuration(st)
        let dir = stageDir(st.name)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let ckptRoot = checkpointRoot()?.appendingPathComponent("stage-\(st.name)")
        let initial = try initialAdapters(st, previous: previous)
        let layerSource = Gemma4LayerSource(checkpoint: cp, dtype: try loadDType, mapWeights: c.memory?.mapWeights ?? true)
        let keptExperts = keep.first?.count ?? 0
        let adapterConfig = try adapterConfiguration(cp: cp, keptExperts: keptExperts)

        var callbacks = HealingCallbacks()
        callbacks.shouldStop = env.shouldStop
        callbacks.stopFile = c.stopFile.map { c.resolve($0) }
        callbacks.pauseFile = c.pauseFile.map { c.resolve($0) }
        let progress = Progress(stage: st.name, total: st.steps, every: c.progressEvery ?? 10, log: log)
        let stepLog = try StepLog(url: dir.appendingPathComponent("steps.jsonl"))
        log("stage \(st.name): \(st.mode) \(st.steps) steps, schedule length \(config.scheduleLength ?? st.steps) offset \(config.scheduleOffset)\(ckptRoot == nil ? ", no checkpoints" : "")")

        let result: [String: MLXArray]
        switch st.mode {
        case "block":
            guard config.clipNorm == 0 else { throw RunnerError.config("stage \(st.name): block mode can't clip by global norm (clipNorm must be 0)") }
            var memory = self.memory
            if let budget = c.memory?.budgetGB.map({ Int($0 * 1024 * 1024 * 1024) }) {
                // A ceiling, never a reason to hold fewer layers: the whole block must fit under it.
                let longest = rows.map(\.length).max() ?? 1
                let inputs = try ResidencyPlanner.inputs(
                    source: layerSource, keep: keep, blockEnds: ends, adapterConfiguration: adapterConfig, sliced: true, sequence: longest)
                let need = ResidencyPlanner.estimate(inputs, resident: inputs.blockLayers).total
                guard need <= budget else {
                    throw HealingError.budget("a memory budget of \(gb(budget)) is below what stage \(st.name) needs: about \(gb(need)). Raise the budget, or shorten the sequences or the adapter ranks.")
                }
                memory.memoryLimitBytes = min(memory.memoryLimitBytes ?? budget, budget)
                memory.peakStopBytes = min(memory.peakStopBytes ?? budget, budget)
            }
            let healer = StreamedBlockHealer(
                source: layerSource, keep: keep, blockEnds: ends, adapterConfiguration: adapterConfig,
                configuration: config, initialAdapters: initial)
            let blocks = blockRanges(ends: ends).count
            let report = try healer.train(
                rows: source, steps: st.steps, store: store, outputDirectory: dir.appendingPathComponent("blocks"),
                checkpointDirectory: ckptRoot, memory: memory, callbacks: callbacks,
                onStep: { b, r in
                    progress.step(label: "block \(b + 1)/\(blocks)", r)
                    stepLog.write(block: b, r)
                },
                onBlockFinished: { b, _ in self.log("stage \(st.name): block \(b + 1)/\(blocks) done") })
            log("stage \(st.name): MLX peak \(gb(report.peakMemoryBytes)), process peak \(gb(report.peakFootprintBytes)) (MLX counts memory-mapped weights; the process figure, Activity Monitor's \"Memory\", does not)"
                + (report.estimate.map { "; model estimate \(gb($0.total))" } ?? ""))
            if let over = report.peakOverLimit {
                log("stage \(st.name): WARNING the MLX peak reached \(gb(over)), over the memory limit this run was given; MLX's own limit does not refuse allocations, so the run stopped at its next check")
            }
            if case .stopped(let step, let why) = report.outcome {
                return .stopped("stage \(st.name): \(why) (block step \(step))")
            }
            result = report.adapters
        case "plan":
            guard let planText = st.plan else { throw RunnerError.config("stage \(st.name): mode 'plan' needs 'plan'") }
            let plan = try PhasePlan(parsing: planText, total: st.steps)
            let model: Gemma4TextModel
            let healer: PruneHealer
            if (st.student ?? "sliced") == "masked" {
                model = try cp.loadModel(dtype: try loadDType)
                model.train(false)
                healer = try PruneHealer(
                    model: model, keep: keep, adapterConfiguration: adapterConfig, configuration: config,
                    blockEnds: ends, initialAdapters: initial)
            } else {
                model = try cp.loadModel(keeping: keep, dtype: try loadDType)
                model.train(false)
                healer = try PruneHealer(
                    model: model, keep: keep, adapterConfiguration: adapterConfig, configuration: config,
                    blockEnds: ends, initialAdapters: initial,
                    student: .preSliced(numExperts: cp.configuration.numExperts ?? 0), teacher: store,
                    teacherFingerprint: try layerSource.teacherFingerprint())
            }
            var cbs = callbacks
            cbs.onStep = { r in
                progress.step(label: r.phase.rawValue, r)
                stepLog.write(block: nil, r)
            }
            let summary = try healer.run(
                rows: source, plan: plan, checkpointDirectory: ckptRoot, memory: memory, callbacks: cbs)
            if case .stopped(let step, let why) = summary.outcome { return .stopped("stage \(st.name): \(why) (step \(step))") }
            result = healer.adapters.tensors()
        default:
            throw RunnerError.config("stage \(st.name): unknown mode '\(st.mode)' (block or plan)")
        }
        try atomicWrite(result, to: dir.appendingPathComponent("adapters.safetensors"))
        let done = StageDone(steps: st.steps, finished: ISO8601DateFormatter().string(from: Date()))
        try JSONEncoder().encode(done).write(to: dir.appendingPathComponent("done.json"), options: .atomic)
        log("stage \(st.name): done, adapters at \(dir.appendingPathComponent("adapters.safetensors").path)")
        return .finished
    }

    // MARK: export and gguf

    var exportDirectory: URL { c.resolve(c.export?.directory ?? outputDir.appendingPathComponent("export").path) }

    func export() throws -> RunOutcome {
        memory.apply()
        let cp = try checkpoint()
        let keep = try keepSets(cp)
        let dtype = try parseDType(c.export?.dtype) ?? .bfloat16
        var adapterConfig: AdapterConfiguration?
        var layerAdapters: MergeAndSliceExport.LayerAdaptersProvider?
        let spec = c.export?.adapters
        if spec != "none" {
            let url: URL
            if let spec { url = c.resolve(spec) } else if let last = c.stages?.last {
                url = stageDir(last.name).appendingPathComponent("adapters.safetensors")
            } else if c.pipeline != nil {
                url = stageDir("pipeline").appendingPathComponent("adapters.safetensors")
            } else {
                throw RunnerError.config("export needs 'export.adapters' (a file, or 'none') or at least one stage")
            }
            guard fm.fileExists(atPath: url.path) else { throw RunnerError.config("no adapters at \(url.path); run heal first") }
            let n = cp.configuration.numExperts ?? 0
            var ac = try adapterConfiguration(cp: cp, keptExperts: keep.first?.count ?? 0)
            if ac.qat != nil { ac.qat?.exportDType = dtype }
            // The file's tensors are lazy handles; each layer's adapters are built from its own tensors when that
            // layer is merged, so no model is loaded to get their shapes and the others' are never read.
            let tensors = try loadArrays(url: url)
            let layers = Set(tensors.keys.compactMap { k -> Int? in
                let parts = k.split(separator: ".", maxSplits: 2)
                return parts.count == 3 && parts[0] == "layers" ? Int(parts[1]) : nil
            })
            if let extra = layers.max(), extra >= cp.configuration.numHiddenLayers {
                throw HealingError.adapterMismatch("\(url.path) has adapters for layer \(extra) but the model has \(cp.configuration.numHiddenLayers) layers")
            }
            adapterConfig = ac
            let configuration = ac
            layerAdapters = { i, layer, _ in
                let ma = ModelAdapters(
                    layers: [layer], firstLayer: i, sets: [ExpertSet(keep: keep[i], numExperts: n)], configuration: configuration)
                try ma.load(subsetOf: tensors)
                return ma.layers[0]
            }
        }
        var o = MergeAndSliceExport.Options()
        o.dtype = dtype
        if let x = c.export?.layersPerShard { o.layersPerShard = x }
        o.textOnly = c.export?.textOnly ?? true
        o.sourceDirectory = c.resolve(c.model)
        o.log = { self.log("export: \($0)") }
        let target = exportDirectory
        let tmp = target.deletingLastPathComponent().appendingPathComponent(target.lastPathComponent + ".partial")
        try? fm.removeItem(at: tmp)
        try fm.createDirectory(at: tmp.deletingLastPathComponent(), withIntermediateDirectories: true)
        log("export: merging \(adapterConfig == nil ? "no adapters (pruned model)" : "adapters") into \(keep.first?.count ?? 0) experts per layer, \(dtype), one layer at a time")
        let loadDType = try loadDType
        try MergeAndSliceExport.write(
            source: Gemma4LayerSource(checkpoint: cp, dtype: loadDType, mapWeights: c.memory?.mapWeights ?? true),
            globals: { try cp.loadGlobalTensors(dtype: loadDType) }, configuration: cp.configuration, keep: keep,
            adapterConfiguration: adapterConfig, layerAdapters: layerAdapters, to: tmp, options: o)
        try? fm.removeItem(at: target)
        try fm.moveItem(at: tmp, to: target)
        log("export: wrote \(target.path)")
        return .finished
    }

    func gguf() throws -> RunOutcome {
        let input = c.gguf?.input.map { c.resolve($0) } ?? exportDirectory
        let output = try required(c.gguf?.output ?? outputDir.appendingPathComponent("model.gguf").path, "gguf.output")
        let type: GGUFOutputType
        switch c.gguf?.type ?? "auto" {
        case "auto": type = .auto
        case "f32": type = .f32
        case "f16": type = .f16
        case "bf16": type = .bf16
        case "q8_0": type = .q8_0
        case "q4_k_m": type = .q4_k_m
        case let other: throw RunnerError.config("gguf.type '\(other)': auto, f32, f16, bf16, q8_0 or q4_k_m")
        }
        guard fm.fileExists(atPath: input.appendingPathComponent("config.json").path) else {
            throw RunnerError.config("no exported model at \(input.path); run export first")
        }
        try fm.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = output.deletingLastPathComponent().appendingPathComponent(output.lastPathComponent + ".partial")
        try? fm.removeItem(at: tmp)
        let summary = try GGUFExport.gemma4(hfDirectory: input, output: tmp, type: type)
        for w in summary.warnings { log("gguf: warning: \(w)") }
        try? fm.removeItem(at: output)
        try fm.moveItem(at: tmp, to: output)
        log("gguf: wrote \(output.path) (\(summary.tensorCount) tensors, \(summary.bytes) bytes)")
        return .finished
    }
}

/// `3GB`, `1.5GiB`, `512MB`, `3` (gigabytes) to gigabytes; `nil` if it is not a positive size.
func parseSizeGB(_ text: String) -> Double? {
    var t = text.trimmingCharacters(in: .whitespaces).lowercased()
    var scale = 1.0
    for (suffix, f) in [("gib", 1.0), ("gb", 1.0), ("g", 1.0), ("mib", 1.0 / 1024), ("mb", 1.0 / 1024), ("m", 1.0 / 1024)]
    where t.hasSuffix(suffix) {
        t.removeLast(suffix.count)
        scale = f
        break
    }
    guard let v = Double(t.trimmingCharacters(in: .whitespaces)), v > 0 else { return nil }
    return v * scale
}

func gb(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1_073_741_824) }

/// Prints a progress line every `every` steps: step, loss, learning rate, seconds per step (mean of
/// the last ten) and the allocator's peak.
final class Progress {
    let stage: String, total: Int, every: Int, log: (String) -> Void
    private var recent: [Double] = []
    init(stage: String, total: Int, every: Int, log: @escaping (String) -> Void) {
        self.stage = stage
        self.total = total
        self.every = max(1, every)
        self.log = log
    }
    func step(label: String, _ r: StepRecord) {
        recent.append(r.seconds)
        if recent.count > 10 { recent.removeFirst() }
        let mean = recent.reduce(0, +) / Double(recent.count)
        guard r.step == 1 || r.step % every == 0 || r.step == total else { return }
        let process = r.peakFootprintBytes > 0
            ? "  process peak \(gb(r.peakFootprintBytes)) (file-backed pages resident in the process: \(gb(r.fileBackedBytes)))" : ""
        log(String(
            format: "[%@ %@] step %d/%d  loss %.5f  lr %.3e  this step %.2f s  (mean of the last %d: %.2f s/step)  mlx peak %@%@%@", stage,
            label, r.step, total, r.loss, r.learningRate, r.seconds, recent.count, mean, gb(Memory.peakMemory), process,
            r.applied ? "" : "  (update skipped)"))
    }
}

/// One JSON line per step in `steps.jsonl`.
final class StepLog {
    private let handle: FileHandle?
    init(url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        handle = try FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }
    func write(block: Int?, _ r: StepRecord) {
        var d: [String: Any] = [
            "step": r.step, "phase": r.phase.rawValue, "loss": r.loss.isFinite ? Double(r.loss) : -1,
            "lr": Double(r.learningRate), "seconds": r.seconds, "applied": r.applied,
        ]
        // Only what was measured: plan stages do not sample the process footprint.
        if r.peakMemoryBytes > 0 { d["mlxPeakBytes"] = r.peakMemoryBytes }
        if r.peakFootprintBytes > 0 {
            d["footprintPeakBytes"] = r.peakFootprintBytes
            d["fileBackedBytes"] = r.fileBackedBytes
        }
        if let block { d["block"] = block }
        if let data = try? JSONSerialization.data(withJSONObject: d, options: [.sortedKeys]) {
            handle?.write(data + Data("\n".utf8))
        }
    }
    deinit { try? handle?.close() }
}

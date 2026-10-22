import Foundation
import MLX
import MLXNN
import MLXRandom

extension Run {
    var gibibyte: Double { 1024.0 * 1024 * 1024 }

    /// Rows for `run` and `sweep`: records not marked `validation` train; those that are held out.
    func splitRecords() throws -> (train: [RowRecord], held: [RowRecord]) {
        let all = try records()
        return (all.filter { !($0.validation ?? false) }, all.filter { $0.validation ?? false })
    }

    func pipelineConfiguration(_ p: RunnerConfig.Pipeline) -> HealingConfiguration {
        var h = HealingConfiguration()
        if let x = p.learningRate { h.learningRate = x }
        if let x = p.learningRateFloor { h.learningRateFloor = x }
        if let x = p.warmup { h.warmupSteps = x }
        h.scheduleLength = p.scheduleLength
        if let x = p.weightDecay { h.weightDecay = x }
        if let x = p.maxLoss { h.maxLoss = x }
        if let x = p.checkpointBlocks { h.checkpointBlocks = x }
        if let x = p.keepIntermediates { h.keepIntermediates = x }
        if let x = p.checkpointEvery { h.checkpointEvery = x }
        h.seed = c.seed ?? 0
        h.evaluateEvery = Int.max / 2
        return h
    }

    func selectionConfiguration() -> SelectionConfiguration? {
        guard let sc = c.select else { return nil }
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
        return s
    }

    func pipelineOptions(_ p: RunnerConfig.Pipeline?, records: [RowRecord], cp: Gemma4Checkpoint) throws -> BlockPipelineOptions {
        var o = BlockPipelineOptions()
        if let x = p?.batchSteps { o.batchSteps = x }
        if let m = p?.mix {
            guard let mix = InputMix(rawValue: m) else { throw RunnerError.config("pipeline.mix '\(m)': clean, dirty or both") }
            o.mix = mix
        }
        o.storageDType = try parseDType(p?.storageDType)
        if let x = p?.maxBatchTokens { o.maxBatchTokens = x }
        if let x = p?.maxBatchRows { o.maxBatchRows = x }
        if let t = p?.teacherPass {
            guard let pass = TeacherPass(rawValue: t) else { throw RunnerError.config("pipeline.teacherPass '\(t)': block or batch") }
            o.teacherPass = pass
        }
        if let x = p?.teacherStateBudgetGB { o.teacherStateBudgetBytes = Int(x * gibibyte) }
        if let x = p?.padWalks { o.padWalks = x }
        o.maxBlocks = p?.maxBlocks
        o.tokenChunk = c.memory?.tokenChunk
        if let r = c.memory?.mergedMaxKeptRatio { ExpertCompute.mergedMaxKeptRatio = r }
        if let x = p?.rowsPerStep { o.rowsPerStep = x }
        switch p?.sharedPrefix ?? "auto" {
        case "auto": o.sharedPrefix = .auto
        case "off": o.sharedPrefix = .off
        case let n:
            guard let k = Int(n), k > 0 else { throw RunnerError.config("pipeline.sharedPrefix '\(n)': auto, off or a token count") }
            o.sharedPrefix = .length(k)
        }
        if let pl = p?.plateau { o.plateau = pl.value }
        o.memoryBudget = c.memory?.budgetGB.map { Int($0 * gibibyte) }
        if fm.fileExists(atPath: keepURL.path), c.select == nil {
            o.fixedKeep = try KeepSets.load(from: keepURL, layerCount: cp.configuration.numHiddenLayers)
        } else {
            guard let s = selectionConfiguration() else {
                throw RunnerError.config("run needs a 'select' section (to choose experts as it goes) or a keep file with no 'select'")
            }
            o.selection = s
            o.items = Dictionary(uniqueKeysWithValues: records.map { ($0.key, $0.selectionItem) })
        }
        return o
    }

    // MARK: run

    func pipeline() throws -> RunOutcome {
        guard let p = c.pipeline else { throw RunnerError.config("run needs a 'pipeline' section with 'steps'") }
        let cp = try checkpoint()
        let (train, held) = try splitRecords()
        guard !train.isEmpty else { throw RunnerError.config("no training rows (every row is marked validation)") }
        let rows = train.map(\.healingRow)
        let source = try c.rowSource(rows: rows)
        let dir = c.resolve(p.directory ?? outputDir.appendingPathComponent("pipeline").path)
        let layerSource = Gemma4LayerSource(checkpoint: cp, dtype: try loadDType, mapWeights: c.memory?.mapWeights ?? true)
        var options = try pipelineOptions(p, records: train, cp: cp)
        if options.plateau != nil {
            guard !held.isEmpty else { throw RunnerError.config("pipeline.plateau needs held-out rows (mark some rows \"validation\": true)") }
            options.evalRows = held.map(\.healingRow)
        }
        let ends = try blockEnds(cp)
        let keptCount = options.fixedKeep?.first?.count ?? options.selection?.keep ?? 0
        let pipeline = try BlockPipeline(
            source: layerSource, blockEnds: ends, adapterConfiguration: try adapterConfiguration(cp: cp, keptExperts: keptCount),
            configuration: pipelineConfiguration(p), options: options, directory: dir)
        var callbacks = HealingCallbacks()
        callbacks.shouldStop = env.shouldStop
        callbacks.stopFile = c.stopFile.map { c.resolve($0) }
        callbacks.pauseFile = c.pauseFile.map { c.resolve($0) }
        let blocks = ends.count
        let progress = Progress(stage: "pipeline", total: p.steps, every: c.progressEvery ?? 10, log: log)
        let stepLog = try StepLog(url: dir.appendingPathComponent("steps.jsonl"))
        log("run: \(p.steps) steps per block over \(blocks) blocks, \(options.batchSteps) steps per data batch, mix \(options.mix.rawValue); frontier in \(dir.path)")
        let report = try pipeline.run(
            rows: source, steps: p.steps, memory: memory, callbacks: callbacks, log: { self.log("run: \($0)") },
            onStep: { b, r in
                progress.step(label: "block \(b + 1)/\(blocks)", r)
                stepLog.write(block: b, r)
            },
            onBlockFinished: { b, _ in self.log("run: block \(b + 1)/\(blocks) trained") })
        for b in report.blocks {
            log("run: block \(b.block + 1): \(b.residentLayers) layers resident, \(String(format: "%.3f", b.secondsPerStep)) s/step (median-free mean after the first), training peak \(gb(b.trainingPeakBytes))")
        }
        log("run: other stages peaked at \(gb(report.otherStagesPeakBytes)), training at \(gb(report.trainingPeakBytes)); frontier on disk \(gb(report.frontierBytes))")
        if case .stopped(let step, let why) = report.outcome { return .stopped("pipeline: \(why) (step \(step))") }
        if options.fixedKeep == nil {
            try fm.createDirectory(at: keepURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try KeepSets.save(report.keep, to: keepURL)
            log("run: wrote \(keepURL.path)")
        }
        let out = stageDir("pipeline")
        try fm.createDirectory(at: out, withIntermediateDirectories: true)
        try atomicWrite(report.adapters, to: out.appendingPathComponent("adapters.safetensors"))
        let done = StageDone(steps: p.steps, finished: ISO8601DateFormatter().string(from: Date()))
        try JSONEncoder().encode(done).write(to: out.appendingPathComponent("done.json"), options: .atomic)
        log("run: adapters at \(out.appendingPathComponent("adapters.safetensors").path)")
        return .finished
    }

    // MARK: sweep

    func sweep() throws -> RunOutcome {
        guard let sw = c.sweep else { throw RunnerError.config("sweep needs a 'sweep' section with 'learningRates' and 'steps'") }
        let cp = try checkpoint()
        let (train, held) = try splitRecords()
        var trainRecords = train, heldRecords = held
        if let n = sw.heldOutRows {
            let all = try records()
            guard n > 0, n < all.count else { throw RunnerError.config("sweep.heldOutRows must leave rows to train on") }
            heldRecords = Array(all.suffix(n))
            trainRecords = Array(all.dropLast(n))
        }
        if let n = sw.trainRows { trainRecords = Array(trainRecords.prefix(n)) }
        guard !trainRecords.isEmpty, !heldRecords.isEmpty else {
            throw RunnerError.config("sweep needs training rows and held-out rows (mark some rows \"validation\": true, or set sweep.heldOutRows)")
        }
        let ends = try blockEnds(cp)
        let dir = c.resolve(sw.directory ?? outputDir.appendingPathComponent("sweep").path)
        let mapWeights = c.memory?.mapWeights ?? true
        let defaultDType = try loadDType
        let options = try pipelineOptions(c.pipeline, records: trainRecords, cp: cp)
        var base = c.pipeline.map(pipelineConfiguration) ?? HealingConfiguration()
        if let w = sw.warmup { base.warmupSteps = w }
        base.checkpointEvery = Int.max / 2
        let keptCount = options.fixedKeep?.first?.count ?? options.selection?.keep ?? 0
        memory.apply()
        let sweep = LearningRateSweep(
            sourceFactory: { Gemma4LayerSource(checkpoint: cp, dtype: $0 ?? defaultDType, mapWeights: mapWeights) }, blockEnds: ends, adapterConfiguration: try adapterConfiguration(cp: cp, keptExperts: keptCount),
            base: base, directory: dir, options: options)
        let report = try sweep.run(
            trainRows: trainRecords.map(\.healingRow), heldOut: heldRecords.map(\.healingRow), learningRates: sw.learningRates,
            steps: sw.steps,
            variants: sw.variants?.map {
                SweepVariant(
                    name: $0.name, rowsPerStep: $0.rowsPerStep ?? 1, plateau: $0.plateau?.value, computeDType: $0.dtype,
                    routerRank: $0.routerRank, routerLRMultiplier: $0.routerLRMultiplier,
                    expertRank: $0.expertRank, sharedExperts: $0.sharedExperts)
            } ?? [SweepVariant(name: "base")],
            log: { self.log($0) })
        print(report.table)
        let out = c.resolve(sw.output ?? outputDir.appendingPathComponent("sweep.json").path)
        try fm.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        try report.json().write(to: out, options: .atomic)
        log("sweep: wrote \(out.path)")
        return .finished
    }
}

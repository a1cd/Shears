import Foundation
import MLX
import MLXNN

/// One way of training to compare in a sweep, besides the learning rate and the step count.
public struct SweepVariant: Codable, Sendable {
    public var name: String
    /// Rows stacked in one update.
    public var rowsPerStep = 1
    /// Plateau-accelerated annealing (needs the held-out rows, which the sweep always has).
    public var plateau: PlateauAnnealing?
    /// Load the model in this dtype (`float32`, `bfloat16`, `float16`) while the adapters and optimizer stay
    /// float32; `nil` is the source's own.
    public var computeDType: String?
    /// Router correction rank for this variant (0 turns it off); `nil` keeps the sweep's adapter configuration.
    public var routerRank: Int?
    /// Learning-rate multiplier of the router correction; `nil` keeps the sweep's.
    public var routerLRMultiplier: Float?
    /// Expert LoRA rank for this variant; `nil` keeps the sweep's.
    public var expertRank: Int?
    /// A shared expert LoRA (see ``AdapterConfiguration/sharedExperts``) for this variant; `nil` keeps the sweep's.
    public var sharedExperts: Bool?
    public init(
        name: String, rowsPerStep: Int = 1, plateau: PlateauAnnealing? = nil, computeDType: String? = nil,
        routerRank: Int? = nil, routerLRMultiplier: Float? = nil, expertRank: Int? = nil, sharedExperts: Bool? = nil
    ) {
        self.expertRank = expertRank
        self.sharedExperts = sharedExperts
        self.name = name
        self.rowsPerStep = rowsPerStep
        self.plateau = plateau
        self.computeDType = computeDType
        self.routerRank = routerRank
        self.routerLRMultiplier = routerLRMultiplier
    }
}

/// One point of a sweep.
public struct SweepResult: Codable, Sendable {
    public var variant: String
    public var learningRate: Float
    public var steps: Int
    /// Mean pin loss on the held-out rows after training (lower is better).
    public var heldOutLoss: Float
    /// Mean training loss over the last quarter of the updates.
    public var trainLoss: Float
    public var seconds: Double
    public var skippedUpdates: Int
    /// The update at which plateau annealing triggered, if it did.
    public var annealedAt: Int?
    /// How closely the router correction's routing follows the original router's on the held-out rows
    /// (`nil` when the variant has no router correction).
    public var routing: RoutingAgreement?
}

public struct SweepReport: Codable, Sendable {
    /// Held-out pin loss of the pruned, unhealed block (fresh adapters: the reference to beat), measured with the first
    /// variant's dtype.
    public var unhealedLoss: Float
    public var heldOutRows: Int
    public var trainRows: Int
    public var results: [SweepResult]

    /// A plain-text table with the best point marked.
    public var table: String {
        let best = results.min { $0.heldOutLoss < $1.heldOutLoss }
        var out = String(format: "unhealed held-out loss %.5f over %d held-out rows (%d training rows)\n", unhealedLoss, heldOutRows, trainRows)
        out += "  variant         peak lr    steps   held-out   train(last 1/4)   seconds   annealed   route KL    top-k overlap\n"
        for r in results {
            let isBest = best.map { $0.variant == r.variant && $0.learningRate == r.learningRate && $0.steps == r.steps } == true
            out += String(
                format: "  %-14@  %-9.2e  %5d   %.5f    %.5f          %6.1f   %-8@   %@  %@%@\n", r.variant as NSString, r.learningRate, r.steps,
                r.heldOutLoss, r.trainLoss, r.seconds, (r.annealedAt.map { "at \($0)" } ?? "-") as NSString,
                (r.routing.map { String(format: "%.3e", $0.kl) } ?? "-") as NSString,
                (r.routing.map { String(format: "%.4f", $0.topKOverlap) } ?? "-") as NSString,
                (isBest ? "   <- best" : "") as NSString)
        }
        return out
    }

    public func json() throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try e.encode(self)
    }
}

/// Short healing runs on block 0 over a grid of peak learning rates, step counts and training variants (rows per
/// step, plateau annealing, compute dtype), each scored by the pin loss on held-out rows. Every point starts from
/// the same seeded adapters with a fresh optimizer and a warmup-and-cosine schedule exactly as long as its steps,
/// and trains through the same code as the pipeline's blocks. The block's data (the teacher's clean input and
/// target per row, and the selected experts) is prepared once by the pipeline (resumable) and reused by every
/// point and every later sweep over the same rows.
public struct LearningRateSweep {
    public let sourceFactory: (DType?) throws -> LayerSource
    public let blockEnds: [Int]
    public let adapterConfiguration: AdapterConfiguration
    public let base: HealingConfiguration
    public let directory: URL
    public var pipelineOptions: BlockPipelineOptions

    public init(
        sourceFactory: @escaping (DType?) throws -> LayerSource, blockEnds: [Int], adapterConfiguration: AdapterConfiguration,
        base: HealingConfiguration, directory: URL, options: BlockPipelineOptions
    ) {
        self.sourceFactory = sourceFactory
        self.blockEnds = blockEnds
        self.adapterConfiguration = adapterConfiguration
        self.base = base
        self.directory = directory
        self.pipelineOptions = options
    }

    public init(
        source: LayerSource, blockEnds: [Int], adapterConfiguration: AdapterConfiguration, base: HealingConfiguration,
        directory: URL, options: BlockPipelineOptions
    ) {
        self.init(
            sourceFactory: { _ in source }, blockEnds: blockEnds, adapterConfiguration: adapterConfiguration, base: base,
            directory: directory, options: options)
    }

    public func run(
        trainRows: [HealingRow], heldOut: [HealingRow], learningRates: [Float], steps: [Int],
        variants: [SweepVariant] = [SweepVariant(name: "base")], log: ((String) -> Void)? = nil
    ) throws -> SweepReport {
        precondition(!trainRows.isEmpty && !heldOut.isEmpty && !learningRates.isEmpty && !steps.isEmpty && !variants.isEmpty)
        let all = trainRows + heldOut
        var results: [SweepResult] = []
        var unhealed: Float = 0
        var prepared: [String: (keep: [[Int]], dir: URL, source: LayerSource)] = [:]
        for v in variants {
            let dtypeName = v.computeDType ?? "default"
            if prepared[dtypeName] == nil {
                let source = try sourceFactory(try parseDType(v.computeDType))
                let dir = directory.appendingPathComponent("prepared-\(dtypeName)")
                var o = pipelineOptions
                o.prepareOnly = true
                o.maxBlocks = 1
                o.batchSteps = max(1, all.count)
                o.rowsPerStep = 1
                let pipeline = try BlockPipeline(
                    source: source, blockEnds: blockEnds, adapterConfiguration: adapterConfiguration, configuration: base,
                    options: o, directory: dir)
                let r = try pipeline.run(rows: SequentialRowSource(rows: all), steps: all.count, log: log)
                prepared[dtypeName] = (r.keep, dir, source)
            }
            let (keep, dir, source) = prepared[dtypeName]!
            func point(lr: Float, n: Int, name: String) throws -> SweepResult {
                var c = base
                c.learningRate = lr
                c.learningRateFloor = min(base.learningRateFloor, lr / 10)
                c.warmupSteps = min(base.warmupSteps, max(0, n / 10))
                c.scheduleLength = n
                c.scheduleOffset = 0
                c.scheduleLengthChanges = []
                c.checkpointEvery = Int.max / 2
                var ac = adapterConfiguration
                if let x = v.routerRank { ac.routerRank = x }
                if let x = v.routerLRMultiplier { ac.routerLRMultiplier = x }
                if let x = v.expertRank { ac.expertRank = x }
                if let x = v.sharedExperts { ac.sharedExperts = x }
                var o = pipelineOptions
                o.fixedKeep = keep
                o.selection = nil
                o.trainOnly = true
                o.maxBlocks = 1
                o.batchSteps = n
                o.rowsPerStep = v.rowsPerStep
                o.mix = .clean
                o.evalRows = heldOut
                o.plateau = v.plateau
                let tag = "\(name)-\(Int(lr * 1e9))-\(n)"
                // Every point trains afresh; only the prepared frontier is kept between points and sweeps.
                for leftover in ["pipeline-\(tag).json", "blocks-\(tag)", "checkpoints-\(tag)"] {
                    try? FileManager.default.removeItem(at: dir.appendingPathComponent(leftover))
                }
                let pipeline = try BlockPipeline(
                    source: source, blockEnds: blockEnds, adapterConfiguration: ac, configuration: c,
                    options: o, directory: dir, runName: tag)
                let t0 = Date()
                let r = try pipeline.run(rows: ShuffledRows(rows: trainRows, seed: base.seed), steps: n)
                let info = r.blocks[0]
                let tail = info.losses.suffix(max(1, info.losses.count / 4))
                return SweepResult(
                    variant: v.name, learningRate: lr, steps: n, heldOutLoss: info.finalHeldOut ?? .nan,
                    trainLoss: tail.reduce(0, +) / Float(max(1, tail.count)), seconds: Date().timeIntervalSince(t0),
                    skippedUpdates: info.skippedUpdates, annealedAt: info.annealedAt, routing: info.finalRouting)
            }
            if results.isEmpty { unhealed = try point(lr: 0, n: 1, name: "unhealed").heldOutLoss }
            for lr in learningRates {
                for n in steps {
                    let res = try point(lr: lr, n: n, name: v.name)
                    results.append(res)
                    log?(String(format: "sweep %@ lr %.2e steps %d: held-out %.5f (unhealed %.5f)", v.name, lr, n, res.heldOutLoss, unhealed))
                }
            }
        }
        return SweepReport(unhealedLoss: unhealed, heldOutRows: heldOut.count, trainRows: trainRows.count, results: results)
    }
}

/// Row `i % count` at step `i`.
struct SequentialRowSource: HealingRowSource {
    var rows: [HealingRow]
    func row(forStep step: Int) -> HealingRow { rows[step % rows.count] }
}

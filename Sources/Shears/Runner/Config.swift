import Foundation
import MLX

/// The JSON file that drives every `shears-cli` subcommand. Relative paths are relative to the
/// directory holding the file. Unknown keys are an error: a misspelt knob must not silently run
/// with its default.
public struct RunnerConfig: Decodable {
    /// A Hugging Face Gemma 4 checkpoint directory (safetensors, `config.json`, optional tokenizer files).
    public var model: String
    /// Load dtype for the model's floating-point tensors (`float32`, `bfloat16`, `float16`); default: as stored.
    public var dtype: String?
    /// JSON Lines file of rows (see ``RowFile``).
    public var rows: String?
    /// Kept experts per layer: written by `select`, read by everything after it.
    public var keep: String?
    public var order: Order?
    /// Directory of the teacher boundary store (`prepass` fills it, `heal` reads it).
    public var store: String?
    public var checkpointDir: String?
    /// Adapters, step logs, selection records and stage results go here.
    public var outputDir: String?
    /// While this file exists the run stops at the next step, saves a checkpoint and deletes the file.
    public var stopFile: String?
    /// While this file exists the run idles between steps.
    public var pauseFile: String?
    public var memory: Memory?
    /// Seeds MLX's generator for adapter initialisation (block `b` uses seed + its first layer).
    public var seed: UInt64?
    public var adapters: Adapters?
    /// Layer indices that end each block; default: the global-attention layers plus the last layer.
    public var blockEnds: [Int]?
    public var select: Select?
    public var prepass: Prepass?
    public var stages: [Stage]?
    public var pipeline: Pipeline?
    public var sweep: Sweep?
    public var export: Export?
    public var gguf: GGUF?
    /// Print every N steps (and always the first and last of a block). Default 10.
    public var progressEvery: Int?

    public struct Order: Decodable {
        /// `shuffled` (a fresh seeded order each epoch), `sequential`, or `file`.
        public var kind: String
        public var seed: UInt64?
        /// For `file`: a JSON array of row indices; step `i` uses entry `i % count`.
        public var path: String?
    }
    public struct Memory: Decodable {
        public var memoryLimitGB: Double?
        public var cacheLimitGB: Double?
        /// If the allocator's peak passes this between steps, the run checkpoints and stops.
        public var peakStopGB: Double?
        /// A ceiling on MLX memory. The whole block stays resident unless this forces fewer layers (smaller expert
        /// token chunks are tried first); a run that needs more than this even with one layer is refused up front with
        /// the amount it needs. `--memory-budget 3GB` sets it too.
        public var budgetGB: Double?
        /// Read layer weights as views of the checkpoint files (default true); `false` copies them.
        public var mapWeights: Bool?
        /// Tokens the expert math processes at once (identical results, less working memory); default all, or what `budgetGB` needs.
        public var tokenChunk: Int?
        /// The merged expert layout is used when the kept count is at most this multiple of the top-k (default 1.5; 0 turns it off).
        public var mergedMaxKeptRatio: Double?
    }
    /// The block-sequential pipeline (`run`): selection, teacher pass and training one block at a time.
    public struct Pipeline: Decodable {
        /// Training steps per block (one row each, in `order`).
        public var steps: Int
        /// Steps in one rolling data batch: its rows are prepared, trained on, then the next batch is prepared.
        public var batchSteps: Int?
        /// `both` (default), `clean` or `dirty`: which inputs a block trains on.
        public var mix: String?
        public var learningRate: Float?
        public var learningRateFloor: Float?
        public var warmup: Int?
        public var scheduleLength: Int?
        public var weightDecay: Float?
        public var maxLoss: Float?
        public var checkpointBlocks: Bool?
        /// `auto` (default), `fast`, `lowMemory` or a layer count: how many layers keep their forward intermediates
        /// instead of recomputing them (see ``ActivationKeep``). `--keep-intermediates` on the command line sets it.
        public var keepIntermediates: ActivationKeep?
        public var checkpointEvery: Int?
        public var storageDType: String?
        public var maxBatchTokens: Int?
        public var maxBatchRows: Int?
        /// Where the frontier (the states entering the current block, clean and dirty, for every row), the
        /// manifest, block adapters and checkpoints go. It can be large: about 2 x rows x tokens x hidden x
        /// bytes. Default: `pipeline` under the output directory.
        public var directory: String?
        /// Stop after this many blocks.
        public var maxBlocks: Int?
        /// Rows stacked in one update (their losses are averaged). Default 1.
        public var rowsPerStep: Int?
        /// `auto` (default: the longest token prefix every row shares, if it is long enough), `off`, or a token count.
        public var sharedPrefix: String?
        /// Anneal sooner when the held-out loss (the `validation` rows) plateaus.
        public var plateau: Plateau?
    }
    public struct Plateau: Decodable {
        public var evalEvery: Int?
        public var patience: Int?
        public var threshold: Float?
        public var annealSteps: Int?
        var value: PlateauAnnealing {
            var p = PlateauAnnealing()
            if let x = evalEvery { p.evalEvery = x }
            if let x = patience { p.patience = x }
            if let x = threshold { p.threshold = x }
            if let x = annealSteps { p.annealSteps = x }
            return p
        }
    }
    public struct SweepVariantConfig: Decodable {
        public var name: String
        public var rowsPerStep: Int?
        public var plateau: Plateau?
        /// Compute dtype (`float32`, `bfloat16`, `float16`); adapters and optimizer stay float32.
        public var dtype: String?
        /// Router correction rank for this variant (0 turns the router correction off); default the `adapters` section's.
        public var routerRank: Int?
        /// Learning-rate multiplier of the router correction for this variant.
        public var routerLRMultiplier: Float?
        /// Expert LoRA rank for this variant; default the `adapters` section's.
        public var expertRank: Int?
        /// A shared expert LoRA (one on the merged expert MLP) for this variant; default the `adapters` section's.
        public var sharedExperts: Bool?
    }
    /// `sweep`: short healing runs on block 0 over a grid, scored on held-out rows.
    public struct Sweep: Decodable {
        public var learningRates: [Float]
        public var steps: [Int]
        /// Rows used for training the grid points (default: all non-validation rows).
        public var trainRows: Int?
        /// Held-out rows: those marked `validation` in the row file (default), or the last N rows.
        public var heldOutRows: Int?
        public var warmup: Int?
        public var output: String?
        public var directory: String?
        /// Training variants to compare at every grid point (rows per step, plateau annealing, compute dtype).
        public var variants: [SweepVariantConfig]?
    }
    public struct Adapters: Decodable {
        public var expertRank: Int?
        public var denseRank: Int?
        public var routerRank: Int?
        /// Multiplies the router correction's learning rate (and its weight decay with it). Default 1.
        public var routerLRMultiplier: Float?
        /// One LoRA of `expertRank` on the merged expert MLP instead of one per kept expert (see `AdapterConfiguration.sharedExperts`).
        public var sharedExperts: Bool?
        public var scale: Float?
        /// `q4_k_m` heals against the weights Q4_K_M will write (see `QATConfiguration.q4KM`).
        public var qat: String?
    }
    public struct Select: Decodable {
        public var keep: Int
        public var importanceTokens: Int?
        public var poolByMass: Int?
        public var poolByImportance: Int?
        public var poolSize: Int?
        public var swapRounds: Int?
        /// Seconds per layer; omit for 180, 0 for unlimited (deterministic).
        public var timeLimitPerLayer: Double?
        public var candidateBatch: Int?
        public var seed: UInt64?
        /// Test remove-one-expert importance for this many experts (the highest routing mass) instead of all.
        public var importanceCandidates: Int?
    }
    public struct Prepass: Decodable {
        public var storageDType: String?
        public var maxBatchTokens: Int?
        public var maxBatchRows: Int?
    }
    public struct LengthChange: Decodable {
        public var from: Int
        public var total: Int
    }
    /// One run of the trainer over the adapters, with its own optimizer and schedule.
    public struct Stage: Decodable {
        public var name: String
        /// `block` trains block by block against the stored teacher (one block resident). `plan`
        /// runs phases (`plan`) on a student holding only its kept experts.
        public var mode: String
        /// For `plan`: `block@0,freerun@600,end@1400`.
        public var plan: String?
        public var steps: Int
        /// For `plan`: `sliced` (default, needs the store) or `masked` (whole model, live teacher).
        public var student: String?
        public var learningRate: Float?
        public var learningRateFloor: Float?
        public var warmup: Int?
        /// Length of the schedule; default the stage's `steps`.
        public var scheduleLength: Int?
        /// Position on the schedule of the stage's first update.
        public var scheduleOffset: Int?
        public var scheduleLengthChanges: [LengthChange]?
        public var weightDecay: Float?
        public var clipNorm: Float?
        public var maxLoss: Float?
        public var checkpointBlocks: Bool?
        /// See ``ActivationKeep``: `auto`, `fast`, `lowMemory` or a layer count.
        public var keepIntermediates: ActivationKeep?
        public var dirtyLambda: Float?
        public var pinWeight: Float?
        public var klWeight: Float?
        public var endSamplePositions: Int?
        public var trainFromLayer: Int?
        public var answerPinWeight: Float?
        public var checkpointEvery: Int?
        public var milestoneEvery: Int?
        public var seed: UInt64?
        /// Position in the row order of the stage's first step.
        public var rowOffset: Int?
        /// Start from these adapters (a path, or `previous` for the previous stage's result) with a
        /// fresh optimizer. Ignored when the stage has a checkpoint.
        public var initialAdapters: String?
    }
    public struct Export: Decodable {
        /// Output directory of the merged, sliced model.
        public var directory: String?
        /// Adapters to merge; default: the last stage's. `none` exports the pruned model unhealed.
        public var adapters: String?
        public var dtype: String?
        public var layersPerShard: Int?
        public init(directory: String? = nil, adapters: String? = nil, dtype: String? = nil, layersPerShard: Int? = nil) {
            self.directory = directory
            self.adapters = adapters
            self.dtype = dtype
            self.layersPerShard = layersPerShard
        }
    }
    public struct GGUF: Decodable {
        /// `auto`, `f32`, `f16`, `bf16`, `q8_0`, `q4_k_m`.
        public var type: String?
        public var output: String?
        /// The exported directory to convert; default the `export` directory.
        public var input: String?
    }

    /// Directory of the config file; relative paths resolve against it.
    public var baseDirectory = URL(fileURLWithPath: ".")

    enum CodingKeys: String, CodingKey {
        case model, dtype, rows, keep, order, store, checkpointDir, outputDir, stopFile, pauseFile, memory
        case seed, adapters, blockEnds, select, prepass, stages, export, gguf, progressEvery, pipeline, sweep
    }

    public static func load(from url: URL) throws -> RunnerConfig {
        let data = try Data(contentsOf: url)
        let tree = try JSONSerialization.jsonObject(with: data)
        try Schema.check(tree, against: Schema.root, path: "config")
        var c = try JSONDecoder().decode(RunnerConfig.self, from: data)
        c.baseDirectory = url.deletingLastPathComponent()
        return c
    }

    public func resolve(_ path: String) -> URL {
        path.hasPrefix("/") ? URL(fileURLWithPath: path) : URL(fileURLWithPath: path, relativeTo: baseDirectory).standardizedFileURL
    }
}

/// The key tree the strict check walks.
enum Schema {
    indirect enum Node {
        case leaf
        case object([String: Node])
        case array(Node)
        func withChild(_ key: String, _ child: Node) -> Node {
            guard case .object(var d) = self else { return self }
            d[key] = child
            return .object(d)
        }
    }
    static let stage: Node = .object(Dictionary(uniqueKeysWithValues: [
        "name", "mode", "plan", "steps", "student", "learningRate", "learningRateFloor", "warmup", "scheduleLength",
        "scheduleOffset", "weightDecay", "clipNorm", "maxLoss", "checkpointBlocks", "keepIntermediates", "dirtyLambda", "pinWeight", "klWeight",
        "endSamplePositions", "trainFromLayer", "answerPinWeight", "checkpointEvery", "milestoneEvery", "seed", "rowOffset",
        "initialAdapters",
    ].map { ($0, Node.leaf) } + [("scheduleLengthChanges", Node.array(.object(["from": .leaf, "total": .leaf])))]))

    static func leaves(_ keys: [String]) -> Node { .object(Dictionary(uniqueKeysWithValues: keys.map { ($0, Node.leaf) })) }

    static let root: Node = .object([
        "model": .leaf, "dtype": .leaf, "rows": .leaf, "keep": .leaf, "store": .leaf, "checkpointDir": .leaf,
        "outputDir": .leaf, "stopFile": .leaf, "pauseFile": .leaf, "seed": .leaf, "blockEnds": .leaf,
        "progressEvery": .leaf,
        "order": leaves(["kind", "seed", "path"]),
        "memory": leaves(["memoryLimitGB", "cacheLimitGB", "peakStopGB", "budgetGB", "mapWeights", "tokenChunk", "mergedMaxKeptRatio"]),
        "pipeline": leaves([
            "steps", "batchSteps", "mix", "learningRate", "learningRateFloor", "warmup", "scheduleLength", "weightDecay",
            "maxLoss", "checkpointBlocks", "keepIntermediates", "checkpointEvery", "storageDType", "maxBatchTokens", "maxBatchRows", "directory",
            "maxBlocks", "rowsPerStep", "sharedPrefix",
        ]).withChild("plateau", leaves(["evalEvery", "patience", "threshold", "annealSteps"])),
        "sweep": leaves(["learningRates", "steps", "trainRows", "heldOutRows", "warmup", "output", "directory"])
            .withChild("variants", .array(leaves(["name", "rowsPerStep", "dtype", "routerRank", "routerLRMultiplier", "expertRank", "sharedExperts"]).withChild("plateau", leaves(["evalEvery", "patience", "threshold", "annealSteps"])))),
        "adapters": leaves(["expertRank", "denseRank", "routerRank", "routerLRMultiplier", "sharedExperts", "scale", "qat"]),
        "select": leaves([
            "keep", "importanceTokens", "poolByMass", "poolByImportance", "poolSize", "swapRounds",
            "timeLimitPerLayer", "candidateBatch", "seed", "importanceCandidates",
        ]),
        "prepass": leaves(["storageDType", "maxBatchTokens", "maxBatchRows"]),
        "stages": .array(stage),
        "export": leaves(["directory", "adapters", "dtype", "layersPerShard"]),
        "gguf": leaves(["type", "output", "input"]),
    ])

    static func check(_ value: Any, against node: Node, path: String) throws {
        switch node {
        case .leaf: return
        case .array(let inner):
            guard let a = value as? [Any] else { return }
            for (i, v) in a.enumerated() { try check(v, against: inner, path: "\(path)[\(i)]") }
        case .object(let keys):
            guard let d = value as? [String: Any] else { return }
            for (k, v) in d {
                guard let n = keys[k] else {
                    throw RunnerError.config("unknown key '\(k)' in \(path) (known: \(keys.keys.sorted().joined(separator: ", ")))")
                }
                try check(v, against: n, path: "\(path).\(k)")
            }
        }
    }
}

public enum RunnerError: Error, CustomStringConvertible {
    case config(String)
    case usage(String)
    public var description: String {
        switch self {
        case .config(let m): "config: \(m)"
        case .usage(let m): m
        }
    }
}

func parseDType(_ s: String?) throws -> DType? {
    switch s {
    case nil: return nil
    case "float32"?: return .float32
    case "bfloat16"?: return .bfloat16
    case "float16"?: return .float16
    default: throw RunnerError.config("unknown dtype '\(s!)' (float32, bfloat16 or float16)")
    }
}

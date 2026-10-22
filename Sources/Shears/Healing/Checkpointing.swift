import Foundation
import MLX

/// Resumable checkpoints: adapters, optimizer moments, the optimizer's update count and the
/// loop step, written to a temporary directory and renamed into place so an interrupted save
/// never leaves a half-written checkpoint.
enum Checkpointing {
    struct Meta: Codable {
        var step: Int
        var optimizerStep: Int
        var phase: String
        var plan: String
        var total: Int
        var final: Bool
        var saved: String
    }

    /// Writes a checkpoint directory by building it under `ckpt.tmp` and renaming it into place.
    /// Between its two renames only `ckpt.old` exists; ``existing(in:)`` knows to read that.
    static func commit(to root: URL, build: (URL) throws -> Void) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let ck = root.appendingPathComponent("ckpt")
        let tmp = root.appendingPathComponent("ckpt.tmp")
        let old = root.appendingPathComponent("ckpt.old")
        try? fm.removeItem(at: tmp)
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        try build(tmp)
        if fm.fileExists(atPath: ck.path) {
            try? fm.removeItem(at: old)
            try fm.moveItem(at: ck, to: old)
        }
        try fm.moveItem(at: tmp, to: ck)
        try? fm.removeItem(at: old)
    }

    /// The newest complete checkpoint directory under `root`: `ckpt`, or `ckpt.old` when a save was
    /// interrupted between its two renames. `nil` when there is none.
    static func existing(in root: URL) -> URL? {
        let fm = FileManager.default
        for name in ["ckpt", "ckpt.old"] {
            let d = root.appendingPathComponent(name)
            if fm.fileExists(atPath: d.appendingPathComponent("meta.json").path) { return d }
        }
        return nil
    }

    static func save(
        to root: URL, adapters: ModelAdapters, optimizer: HealingAdamW, step: Int,
        phase: HealingPhase, plan: PhasePlan, final: Bool
    ) throws {
        try commit(to: root) { tmp in
            try MLX.save(arrays: adapters.tensors(), url: tmp.appendingPathComponent("adapter.safetensors"))
            let opt = optimizer.stateTensors()
            if !opt.isEmpty {
                try MLX.save(arrays: opt, url: tmp.appendingPathComponent("opt.safetensors"))
            }
            let meta = Meta(
                step: step, optimizerStep: optimizer.step, phase: phase.rawValue, plan: plan.text,
                total: plan.total, final: final, saved: ISO8601DateFormatter().string(from: Date()))
            try JSONEncoder().encode(meta).write(to: tmp.appendingPathComponent("meta.json"))
        }
    }

    /// Loads a checkpoint if one exists, verifying adapter keys and shapes.
    static func resume(from root: URL, adapters: ModelAdapters, optimizer: HealingAdamW) throws -> Meta? {
        guard let ck = existing(in: root) else { return nil }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: ck.appendingPathComponent("meta.json")))
        try adapters.load(tensors: try loadArrays(url: ck.appendingPathComponent("adapter.safetensors")))
        let optURL = ck.appendingPathComponent("opt.safetensors")
        let state = FileManager.default.fileExists(atPath: optURL.path) ? try loadArrays(url: optURL) : [:]
        optimizer.restore(tensors: state, step: meta.optimizerStep)
        eval(Array(state.values))
        return meta
    }

    /// A checkpoint inside one block of a streamed run.
    struct BlockMeta: Codable {
        var block: Int
        /// Steps of this block already done; the next step to run is this one. The row for a step is
        /// a pure function of the step, so this is also the row-order position.
        var step: Int
        var optimizerStep: Int
        var losses: [Float]
        var skipped: Int
        var saved: String
    }

    static func saveBlock(
        to root: URL, adapters: ModelAdapters, optimizer: HealingAdamW, meta: BlockMeta
    ) throws {
        try commit(to: root) { tmp in
            try MLX.save(arrays: adapters.tensors(), url: tmp.appendingPathComponent("adapter.safetensors"))
            let opt = optimizer.stateTensors()
            if !opt.isEmpty { try MLX.save(arrays: opt, url: tmp.appendingPathComponent("opt.safetensors")) }
            try JSONEncoder().encode(meta).write(to: tmp.appendingPathComponent("meta.json"))
        }
    }

    static func resumeBlock(from root: URL, block: Int, adapters: ModelAdapters, optimizer: HealingAdamW) throws -> BlockMeta? {
        guard let ck = existing(in: root) else { return nil }
        let meta = try JSONDecoder().decode(BlockMeta.self, from: Data(contentsOf: ck.appendingPathComponent("meta.json")))
        guard meta.block == block else { return nil }
        try adapters.load(tensors: try loadArrays(url: ck.appendingPathComponent("adapter.safetensors")))
        eval(adapters)
        let optURL = ck.appendingPathComponent("opt.safetensors")
        let state = FileManager.default.fileExists(atPath: optURL.path) ? try loadArrays(url: optURL) : [:]
        optimizer.restore(tensors: state, step: meta.optimizerStep)
        eval(Array(state.values))
        return meta
    }

    static func saveMilestone(to root: URL, adapters: ModelAdapters, step: Int) throws {
        let dir = root.appendingPathComponent("milestones")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try MLX.save(
            arrays: adapters.tensors(),
            url: dir.appendingPathComponent(String(format: "adapter-step%05d.safetensors", step)))
    }
}

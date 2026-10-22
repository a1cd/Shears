import Foundation
import MLX
import MLXNN

/// A calibration sequence for expert selection.
public struct SelectionItem {
    public var key: String
    public var tokens: [Int32]
    /// Positions the search scores (a subset of the sequence, for cost) and their weights.
    public var searchPositions: [Int]
    public var searchWeights: [Float]
    /// Validation items take no part in the search; they only feed the reported errors.
    public var isValidation: Bool

    public init(
        key: String = "", tokens: [Int32], searchPositions: [Int], searchWeights: [Float]? = nil,
        isValidation: Bool = false
    ) {
        precondition(searchWeights == nil || searchWeights!.count == searchPositions.count)
        self.key = key
        self.tokens = tokens
        self.searchPositions = searchPositions
        self.searchWeights = searchWeights ?? searchPositions.map { _ in 1 }
        self.isValidation = isValidation
    }
}

public struct SelectionConfiguration {
    /// Experts to keep per layer.
    public var keep: Int
    /// Tokens used to measure each expert's importance by single removal.
    public var importanceTokens = 400
    /// The candidate pool: this many by routing mass, this many by importance, topped up by
    /// mass to `poolSize` (all clamped to the expert count).
    public var poolByMass = 40
    public var poolByImportance = 24
    public var poolSize = 64
    public var swapRounds = 3
    /// Remove-one-expert importance is measured for this many experts only, the ones the routing statistics
    /// (the same mass that seeds the pool, gathered in the same pass) rank highest; the rest cannot enter the
    /// pool by importance and are only reachable by mass. `nil` is the exhaustive recipe: every expert is
    /// tested. The pruned search picks the same set whenever the experts it skips are ones the swap search
    /// would not have used (tested on a toy); it costs `candidates / experts` of the removal scoring.
    public var importanceCandidates: Int?
    /// Wall-clock cap, in seconds from the start of each layer's set search, on the swap search
    /// (the recipe's 90 s cap, applied as twice that: 180 s). `nil` is unlimited, which makes
    /// selection deterministic.
    public var timeLimitPerLayer: TimeInterval? = 180
    /// Candidate sets scored per batch. The recipe's 4: 16 peaked about 12 GB above the model
    /// at roughly 4k search tokens on the 26B.
    public var candidateBatch = 4
    public var expertChunk = 512
    public var seed: UInt64 = 0
    /// Routing-mass prior per layer (`[layers][experts]`) that seeds the candidate pool and the
    /// popularity start. The recipe used a full-model probe over representative data, with
    /// prompt-position mass and generated-position mass each normalised per layer and then
    /// summed: build that with ``RoutingProbe/mass(model:sequences:)`` and
    /// ``RoutingProbe/normalizedSum(_:)``. Default: the teacher's routing weight mass on the
    /// search positions (weighted by the search weights), which needs no extra pass.
    public var prior: [[Float]]?
    public init(keep: Int) { self.keep = keep }
}

/// What the search found for one layer.
public struct LayerSelection: Codable, Sendable {
    public var layer: Int
    public var chosen: [Int]
    public var chosenError: Float
    /// `"greedy"` or `"popularity"`: which start the best swap search came from.
    public var start: String
    public var greedyError: Float
    public var greedySwapError: Float
    public var popularitySwapError: Float
    public var popularity: [Int]
    public var popularityError: Float
    public var allExpertsError: Float
    public var poolSize: Int
    public var outsideMassTopK: Int
    public var searchTokens: Int
    /// Mean relative layer-output drift after the layer, by role (`selection`, `validation`).
    public var layerOutputRelError: [String: Float]
    /// Mean relative expert-branch error of the chosen set and of popularity, on validation items.
    public var validationBranchError: [String: Float]
}

/// Chooses one kept set per layer, for all positions, by how well it reproduces the full
/// model's expert branch rather than by routing popularity alone.
///
/// Bottom up, layer by layer, two hidden-state streams run over the calibration items: the
/// full model, and the pruned model with the sets chosen so far. At each layer the target is
/// the all-expert branch on the *pruned* stream's input. Each expert's importance is the error
/// increase when it alone is removed (the router re-picks from the rest). The pool is the
/// top experts by routing mass plus the most important; a greedy add to `keep`, then
/// best-improvement swaps from both the greedy set and the top-`keep`-by-mass set, picks the
/// set. Error is the weighted mean over positions of
/// `||norm(y_S) - norm(y_target)||^2 / ||norm(y_target)||^2`.
public final class ExpertSelector {
    public let model: (any HealableMoEModel)?
    public let configuration: SelectionConfiguration

    /// A selector that is driven layer by layer through ``step(layer:index:items:hf:hp:existing:)``.
    public init(configuration: SelectionConfiguration) {
        self.model = nil
        self.configuration = configuration
    }

    public init(model: any HealableMoEModel, configuration: SelectionConfiguration) {
        self.model = model
        self.configuration = configuration
    }

    // MARK: Candidate scoring

    /// Error of candidate sets on cached expert outputs.
    /// - scores: `[T, P]` router logits of the pool experts; `O`: `[T, P, D]` their outputs;
    ///   `target`: `[T, D]` the all-expert expert-branch sum; `masks`: `[C, P]` bool.
    func score(
        layer: any HealableMoELayer, scores: MLXArray, outputs O: MLXArray, target: MLXArray,
        weights: MLXArray, masks: MLXArray, pool: MLXArray
    ) -> [Float] {
        let (T, P, D) = (O.dim(0), O.dim(1), O.dim(2))
        let scaleP = layer.expertScale(pool)
        let tgt = layer.normalizeExpertBranch(target).asType(.float32)
        let tn = (tgt * tgt).sum(axis: -1)
        let flatO = O.reshaped(T * P, D)
        var errs: [MLXArray] = []
        let topK = layer.expertRouter.topK
        var b = 0
        while b < masks.dim(0) {
            let mk = masks[b ..< min(b + configuration.candidateBatch, masks.dim(0))]
            let m = min(topK, Int(mk.asType(.int32).sum(axis: -1).min().item(Int32.self)))
            let s = MLX.where(
                expandedDimensions(mk, axis: 1), expandedDimensions(scores.asType(.float32), axis: 0),
                MLXArray(routingMaskedLogit))
            let idx = argPartition(-s, kth: m - 1, axis: -1)[.ellipsis, ..<m]
            let w = softmax(takeAlong(s, idx, axis: -1), axis: -1) * scaleP[idx]
            let rows = (MLXArray.arange(T, dtype: .int32)[.newAxis, 0..., .newAxis] * Int32(P)) + idx.asType(.int32)
            let G = flatO[rows.flattened()].reshaped(mk.dim(0), T, m, D)
            let y = (expandedDimensions(w.asType(G.dtype), axis: -1) * G).sum(axis: 2)
            let h2 = layer.normalizeExpertBranch(y).asType(.float32)
            let e = ((square(h2 - expandedDimensions(tgt, axis: 0)).sum(axis: -1) / expandedDimensions(tn, axis: 0))
                * expandedDimensions(weights, axis: 0)).sum(axis: -1) / weights.sum()
            eval(e)
            errs.append(e)
            b += configuration.candidateBatch
        }
        return concatenated(errs).asArray(Float.self)
    }

    private func expertOutputs(
        _ layer: any HealableMoELayer, x: MLXArray, experts: MLXArray, chunk: Int
    ) -> MLXArray {
        let bank = layer.expertBank
        var parts: [MLXArray] = []
        var c = 0
        let T = x.dim(0)
        while c < T {
            let hi = min(T, c + chunk)
            let idx = broadcast(expandedDimensions(experts, axis: 0), to: [hi - c, experts.dim(0)])
            let o = bank(x[c ..< hi], idx, sort: idx.size >= bank.sortThreshold)
            eval(o)
            parts.append(o)
            Memory.clearCache()
            c = hi
        }
        let out = concatenated(parts)
        eval(out)
        return out
    }

    // MARK: Search at one layer

    func search(
        layer: any HealableMoELayer, layerIndex: Int, x2: MLXArray, logits: MLXArray, weights: MLXArray,
        prior: [Float]?
    ) -> LayerSelection {
        let c = configuration
        let E = layer.numExperts
        let K = c.keep
        let T = x2.dim(0)
        let (ai, aw) = routeNative(layer, logits)
        let tgtY = expertBranch(layer, x: x2, indices: ai, weights: aw, set: nil, lora: nil)
        eval(tgtY)

        var mass: [Float]
        if let prior { mass = prior } else {
            let oneHot = (expandedDimensions(ai, axis: -1) .== MLXArray.arange(E, dtype: ai.dtype)).asType(.float32)
            let m = (oneHot * expandedDimensions(aw.asType(.float32), axis: -1)
                * weights[0..., .newAxis, .newAxis]).sum(axes: [0, 1])
            mass = m.asArray(Float.self)
        }

        var rng = SplitMix64(seed: c.seed &+ UInt64(layerIndex))
        let nSub = min(c.importanceTokens, T)
        let sub = Array(0 ..< T).shuffled(using: &rng).prefix(nSub).sorted()
        let subA = MLXArray(sub.map(Int32.init))
        let allE = MLXArray.arange(E, dtype: .int32)
        let Oall = expertOutputs(layer, x: x2[subA], experts: allE, chunk: 100)
        let ones = MLXArray.ones([1, E], dtype: .bool)
        let notEye = .!(MLX.eye(E, dtype: .bool))
        let sS = logits[subA], wS = weights[subA], tS = tgtY[subA]
        let base = score(layer: layer, scores: sS, outputs: Oall, target: tS, weights: wS, masks: ones, pool: allE)[0]
        let massRank = (0 ..< E).sorted { mass[$0] != mass[$1] ? mass[$0] > mass[$1] : $0 < $1 }
        var imp = [Float](repeating: -Float.infinity, count: E)
        if let m = c.importanceCandidates, m < E {
            let cand = Array(massRank.prefix(max(m, K)))
            let errs = score(
                layer: layer, scores: sS, outputs: Oall, target: tS, weights: wS,
                masks: notEye[MLXArray(cand.map(Int32.init))], pool: allE)
            for (j, e) in cand.enumerated() { imp[e] = errs[j] - base }
        } else {
            imp = score(layer: layer, scores: sS, outputs: Oall, target: tS, weights: wS, masks: notEye, pool: allE)
                .map { $0 - base }
        }
        let impRank = (0 ..< E).sorted { imp[$0] != imp[$1] ? imp[$0] > imp[$1] : $0 < $1 }
        var pool: [Int] = []
        for e in massRank.prefix(c.poolByMass) + impRank.prefix(c.poolByImportance) where !pool.contains(e) {
            pool.append(e)
        }
        let poolSize = min(max(c.poolSize, K), E)
        for e in massRank where pool.count < poolSize && !pool.contains(e) { pool.append(e) }
        let P = pool.count
        let poolA = MLXArray(pool.map(Int32.init))
        let O = expertOutputs(layer, x: x2, experts: poolA, chunk: c.expertChunk)
        let poolScores = logits[0..., poolA]

        func ev(_ sets: [Set<Int>]) -> [Float] {
            let mk = MLXArray(sets.flatMap { s in pool.map { s.contains($0) } }, [sets.count, P])
            return score(layer: layer, scores: poolScores, outputs: O, target: tgtY, weights: weights, masks: mk, pool: poolA)
        }

        let t0 = Date()
        var chosen: [Int] = []
        while chosen.count < K {
            let cands = pool.filter { !chosen.contains($0) }
            let es = ev(cands.map { Set(chosen + [$0]) })
            chosen.append(cands[es.indices.min { es[$0] < es[$1] }!])
        }
        let greedyErr = ev([Set(chosen)])[0]

        func swaps(_ start: [Int]) -> (Set<Int>, Float, Int) {
            var cur = Set(start)
            var curErr = ev([cur])[0]
            var rounds = 0
            while rounds < c.swapRounds, c.timeLimitPerLayer.map({ Date().timeIntervalSince(t0) < $0 }) ?? true {
                var neigh: [(Int, Int)] = []
                for i in cur.sorted() { for j in pool where !cur.contains(j) { neigh.append((i, j)) } }
                if neigh.isEmpty { break }
                let es = ev(neigh.map { (cur.subtracting([$0.0])).union([$0.1]) })
                let b = es.indices.min { es[$0] < es[$1] }!
                rounds += 1
                if es[b] < curErr - 1e-6 {
                    cur = cur.subtracting([neigh[b].0]).union([neigh[b].1])
                    curErr = es[b]
                } else { break }
            }
            return (cur, curErr, rounds)
        }

        let popular = Array(massRank.prefix(K))
        let popErr = ev([Set(popular)])[0]
        let g = swaps(chosen)
        let p = swaps(popular)
        let (best, bestErr, from) = g.1 <= p.1 ? (g.0, g.1, "greedy") : (p.0, p.1, "popularity")
        let massPos = Dictionary(uniqueKeysWithValues: massRank.enumerated().map { ($1, $0) })
        return LayerSelection(
            layer: layerIndex, chosen: best.sorted(), chosenError: bestErr, start: from,
            greedyError: greedyErr, greedySwapError: g.1, popularitySwapError: p.1,
            popularity: popular.sorted(), popularityError: popErr, allExpertsError: base,
            poolSize: P, outsideMassTopK: best.filter { massPos[$0]! >= K }.count, searchTokens: T,
            layerOutputRelError: [:], validationBranchError: [:])
    }

    // MARK: Driver

    /// One layer of the bottom-up pass over `items`: the search (unless `existing` already has the layer's
    /// set), the reported errors, and both streams advanced through the layer. `hf` is the full model's
    /// state entering the layer per item and `hp` the pruned model's (earlier layers cut to their chosen sets,
    /// no adapters); on return they hold the states leaving it. ``select(items:existing:maxLayers:onLayer:)``
    /// is this in a loop; a block-sequential run calls it layer by layer while the same pass records the
    /// teacher's outputs. Each state is `[1, T, D]`.
    public func step(
        layer: any HealableMoELayer, index L: Int, items: [SelectionItem],
        hf: inout [MLXArray], hp: inout [MLXArray], existing: LayerSelection? = nil
    ) -> LayerSelection {
        var fOuts: [MLXArray] = [], pParts: [MoELayerParts] = []
        for i in items.indices {
            let pf = layer.attentionPart(hf[i], adapters: nil)
            let (fi, fw) = routeNative(layer, pf.routerLogits)
            let fo = layer.combine(
                pf, expertBranch: expertBranch(layer, x: pf.expertInput, indices: fi, weights: fw, set: nil, lora: nil))
            let pp = layer.attentionPart(hp[i], adapters: nil)
            eval(fo, pp.residual, pp.dense, pp.routerLogits, pp.expertInput)
            fOuts.append(fo)
            pParts.append(pp)
        }
        var rec: LayerSelection
        if let d = existing { rec = d } else {
            var xs: [MLXArray] = [], ss: [MLXArray] = [], ws: [MLXArray] = []
            for (i, it) in items.enumerated() where !it.isValidation {
                let p = MLXArray(it.searchPositions.map(Int32.init))
                xs.append(pParts[i].expertInput[0][p])
                ss.append(pParts[i].routerLogits[0][p])
                ws.append(MLXArray(it.searchWeights))
            }
            rec = search(
                layer: layer, layerIndex: L, x2: concatenated(xs), logits: concatenated(ss),
                weights: concatenated(ws), prior: configuration.prior?[L])
            Memory.clearCache()
        }
        let chosen = ExpertSet(keep: rec.chosen, numExperts: layer.numExperts)
        let pop = ExpertSet(keep: rec.popularity, numExperts: layer.numExperts)
        var rel: [String: (Float, Int)] = ["selection": (0, 0), "validation": (0, 0)]
        var branch: [String: [Float]] = ["chosen": [], "popularity": []]
        for i in items.indices {
            let parts = pParts[i]
            let (idx, w) = route(layer, logits: parts.routerLogits, set: chosen, tweak: nil, routerInput: nil)
            let pOut = layer.combine(
                parts, expertBranch: expertBranch(layer, x: parts.expertInput, indices: idx, weights: w, set: nil, lora: nil))
            let upd = square((fOuts[i] - hf[i]).asType(.float32)).sum(axis: -1)
            let dev = square((pOut - fOuts[i]).asType(.float32)).sum(axis: -1)
            let r = (dev / (upd + 1e-6))[0]
            let role = items[i].isValidation ? "validation" : "selection"
            rel[role]!.0 += r.sum().item(Float.self)
            rel[role]!.1 += r.size
            if items[i].isValidation {
                let (ai, aw) = routeNative(layer, parts.routerLogits)
                let tgt = layer.normalizeExpertBranch(
                    expertBranch(layer, x: parts.expertInput, indices: ai, weights: aw, set: nil, lora: nil)).asType(.float32)
                let tn = (tgt * tgt).sum(axis: -1)
                for (name, set) in [("chosen", chosen), ("popularity", pop)] {
                    let (ii, ww) = route(layer, logits: parts.routerLogits, set: set, tweak: nil, routerInput: nil)
                    let h2 = layer.normalizeExpertBranch(
                        expertBranch(layer, x: parts.expertInput, indices: ii, weights: ww, set: nil, lora: nil)).asType(.float32)
                    branch[name]!.append((square(h2 - tgt).sum(axis: -1) / tn).mean().item(Float.self))
                }
            }
            hp[i] = pOut
            hf[i] = fOuts[i]
            eval(hp[i], hf[i])
        }
        rec.layerOutputRelError = rel.mapValues { $0.0 / Float(max($0.1, 1)) }
        rec.validationBranchError = branch.mapValues { $0.isEmpty ? 0 : $0.reduce(0, +) / Float($0.count) }
        return rec
    }

    /// Runs the search layer by layer.
    ///
    /// - Parameters:
    ///   - existing: layers already searched (resumed); their sets are used as they are.
    ///   - maxLayers: stop after this many layers.
    ///   - onLayer: called after each layer, for logging or persisting.
    public func select(
        items: [SelectionItem], existing: [LayerSelection] = [], maxLayers: Int? = nil,
        onLayer: ((LayerSelection) -> Void)? = nil
    ) -> [LayerSelection] {
        guard let model else { preconditionFailure("this selector was made without a model; use step(...)") }
        let layers = model.healableLayers
        let n = min(layers.count, maxLayers ?? layers.count)
        precondition(items.contains { !$0.isValidation }, "need at least one selection item")
        var hf = items.map { model.embed(MLXArray($0.tokens)[.newAxis]) }
        var hp = hf
        eval(hf)
        var out: [LayerSelection] = []
        let done = Dictionary(uniqueKeysWithValues: existing.map { ($0.layer, $0) })
        for L in 0 ..< n {
            let rec = step(layer: layers[L], index: L, items: items, hf: &hf, hp: &hp, existing: done[L])
            out.append(rec)
            onLayer?(rec)
        }
        return out
    }
}

/// Full-model routing statistics, for building a selection prior.
public enum RoutingProbe {
    /// Routing weight mass per layer and expert from the unpruned model over `sequences`
    /// (`[layers][experts]`). With `positions` only those positions count; with `nil`, all.
    public static func mass(
        model: any HealableMoEModel, sequences: [(tokens: [Int32], positions: [Int]?)]
    ) -> [[Float]] {
        let layers = model.healableLayers
        var out = layers.map { [Float](repeating: 0, count: $0.numExperts) }
        for seq in sequences {
            var h = model.embed(MLXArray(seq.tokens)[.newAxis])
            for (L, layer) in layers.enumerated() {
                let parts = layer.attentionPart(h, adapters: nil)
                let (idx, w) = routeNative(layer, parts.routerLogits)
                var i = idx[0], ww = w[0]
                if let pos = seq.positions {
                    let p = MLXArray(pos.map(Int32.init))
                    i = i[p]
                    ww = ww[p]
                }
                let E = layer.numExperts
                let oneHot = (expandedDimensions(i, axis: -1) .== MLXArray.arange(E, dtype: i.dtype)).asType(.float32)
                let m = (oneHot * expandedDimensions(ww.asType(.float32), axis: -1)).sum(axes: [0, 1])
                let mv = m.asArray(Float.self)
                for e in 0 ..< E { out[L][e] += mv[e] }
                h = layer.combine(
                    parts, expertBranch: expertBranch(layer, x: parts.expertInput, indices: idx, weights: w, set: nil, lora: nil))
                eval(h)
            }
        }
        return out
    }

    /// Each mass table normalised to sum to 1 per layer, then summed over tables: the recipe's
    /// combination of prompt-position and generated-position mass.
    public static func normalizedSum(_ tables: [[[Float]]]) -> [[Float]] {
        guard let first = tables.first else { return [] }
        var out = first.map { [Float](repeating: 0, count: $0.count) }
        for t in tables {
            for (L, row) in t.enumerated() {
                let total = max(row.reduce(0, +), 1e-9)
                for e in row.indices { out[L][e] += row[e] / total }
            }
        }
        return out
    }
}

/// Kept sets as plain data: read a precomputed selection to skip the search, or write one.
public enum KeepSets {
    /// Reads kept expert ids per layer from JSON in any of these shapes: `[[ids], ...]`;
    /// `{"keep": [[ids], ...]}`; or a selection record `{"layers": {"0": {"chosen": [ids]}, ...}}`
    /// (which also covers files with extra fields per layer).
    public static func load(from url: URL, layerCount: Int? = nil) throws -> [[Int]] {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        var sets: [[Int]]
        if let a = json as? [[Int]] {
            sets = a
        } else if let d = json as? [String: Any], let a = d["keep"] as? [[Int]] {
            sets = a
        } else if let d = json as? [String: Any], let layers = d["layers"] as? [String: Any] {
            sets = []
            for i in 0 ..< layers.count {
                guard let rec = layers[String(i)] as? [String: Any], let ids = rec["chosen"] as? [Int] else {
                    throw HealingError.invalid("selection file has no 'chosen' set for layer \(i)")
                }
                sets.append(ids)
            }
        } else {
            throw HealingError.invalid("unrecognised keep-set file \(url.lastPathComponent)")
        }
        if let n = layerCount, sets.count != n {
            throw HealingError.invalid("keep sets cover \(sets.count) layers, the model has \(n)")
        }
        return sets.map { Array(Set($0)).sorted() }
    }

    public static func keep(from selections: [LayerSelection]) -> [[Int]] {
        selections.sorted { $0.layer < $1.layer }.map(\.chosen)
    }

    public static func save(_ keep: [[Int]], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: ["keep": keep], options: [.sortedKeys]).write(to: url)
    }
}

import Foundation
import MLX

/// Rows come from a JSON Lines file the caller prepares; one object per line:
///
/// ```json
/// {"key": "r0", "tokens": [1, 5, 9], "pinWeights": [0.15, 0.5, 1.0], "answerStart": 2,
///  "klAnchors": [{"position": 1, "weight": 2}], "klSampleRange": [1, 3],
///  "searchPositions": [1, 2], "searchWeights": [1, 1], "validation": false}
/// ```
///
/// `key` and `tokens` are required; keys must be unique. `pinWeights` is per position (default
/// the ramp that reaches 1 at `answerStart`, which defaults to the end of the row).
/// `searchPositions` and `searchWeights` feed `select` (default: every position from `answerStart` on,
/// weight 1); `validation` rows only score the reported errors.
public struct RowRecord: Decodable {
    public var key: String
    public var tokens: [Int32]
    public var pinWeights: [Float]?
    public var answerStart: Int?
    public var klAnchors: [Anchor]?
    public var klSampleRange: [Int]?
    public var searchPositions: [Int]?
    public var searchWeights: [Float]?
    public var validation: Bool?

    public struct Anchor: Decodable {
        public var position: Int
        public var weight: Float?
    }

    public var healingRow: HealingRow {
        HealingRow(
            key: key, tokens: tokens, pinWeights: pinWeights.map { MLXArray($0) }, answerStart: answerStart,
            klAnchors: (klAnchors ?? []).map { KLAnchor(position: $0.position, weight: $0.weight ?? 1) },
            klSampleRange: klSampleRange.flatMap { $0.count == 2 ? $0[0] ..< $0[1] : nil })
    }

    public var selectionItem: SelectionItem {
        let positions = searchPositions ?? Array((answerStart ?? 0) ..< tokens.count)
        return SelectionItem(
            key: key, tokens: tokens, searchPositions: positions, searchWeights: searchWeights,
            isValidation: validation ?? false)
    }
}

public enum RowFile {
    public static func load(from url: URL) throws -> [RowRecord] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        var out: [RowRecord] = []
        var seen = Set<String>()
        for (n, line) in text.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
            let r: RowRecord
            do { r = try decoder.decode(RowRecord.self, from: Data(line.utf8)) } catch {
                throw RunnerError.config("\(url.lastPathComponent) line \(n + 1): \(error)")
            }
            guard !r.tokens.isEmpty else { throw RunnerError.config("\(url.lastPathComponent) line \(n + 1): no tokens") }
            if let w = r.pinWeights, w.count != r.tokens.count {
                throw RunnerError.config("\(url.lastPathComponent) line \(n + 1): \(w.count) pinWeights for \(r.tokens.count) tokens")
            }
            if let a = r.klSampleRange, a.count != 2 {
                throw RunnerError.config("\(url.lastPathComponent) line \(n + 1): klSampleRange is [start, end]")
            }
            guard seen.insert(r.key).inserted else {
                throw RunnerError.config("\(url.lastPathComponent) line \(n + 1): duplicate key '\(r.key)'")
            }
            out.append(r)
        }
        guard !out.isEmpty else { throw RunnerError.config("\(url.lastPathComponent) has no rows") }
        return out
    }
}

/// Which row each step trains on: a pure function of the step, so a resumed run sees what an
/// uninterrupted one would.
struct OffsetRows: HealingRowSource {
    var base: HealingRowSource
    var offset: Int
    func row(forStep step: Int) -> HealingRow { base.row(forStep: step + offset) }
}

struct SequentialRows: HealingRowSource {
    var rows: [HealingRow]
    func row(forStep step: Int) -> HealingRow { rows[step % rows.count] }
}

struct ExplicitOrderRows: HealingRowSource {
    var rows: [HealingRow]
    var order: [Int]
    func row(forStep step: Int) -> HealingRow { rows[order[step % order.count]] }
}

extension RunnerConfig {
    func rowSource(rows: [HealingRow]) throws -> HealingRowSource {
        let o = order ?? Order(kind: "shuffled", seed: 0, path: nil)
        switch o.kind {
        case "shuffled": return ShuffledRows(rows: rows, seed: o.seed ?? 0)
        case "sequential": return SequentialRows(rows: rows)
        case "file":
            guard let p = o.path else { throw RunnerError.config("order.kind 'file' needs order.path") }
            let order = try JSONDecoder().decode([Int].self, from: Data(contentsOf: resolve(p)))
            guard !order.isEmpty, order.allSatisfy({ rows.indices.contains($0) }) else {
                throw RunnerError.config("the order file names rows outside 0..<\(rows.count)")
            }
            return ExplicitOrderRows(rows: rows, order: order)
        default: throw RunnerError.config("unknown order.kind '\(o.kind)' (shuffled, sequential, file)")
        }
    }
}

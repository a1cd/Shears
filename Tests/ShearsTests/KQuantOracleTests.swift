import Foundation
import XCTest

@testable import Shears

/// Compares the Swift K-quants with bytes produced by ggml itself (llama.cpp commit a3a1c47).
///
/// The reference vectors come from `ggml_quantize_chunk` in the built `libggml-base`, written to a
/// directory by a small script; set `SHEARS_KQUANT_ORACLE` to that directory (the test is skipped
/// without it). Each case is 48 rows of 512 values of one data kind, quantized with and without
/// an importance matrix.
final class KQuantOracleTests: XCTestCase {
    var dir: URL!

    override func setUpWithError() throws {
        guard let d = ProcessInfo.processInfo.environment["SHEARS_KQUANT_ORACLE"] else {
            throw XCTSkip("set TEST_RUNNER_SHEARS_KQUANT_ORACLE to the reference-vector directory")
        }
        dir = URL(fileURLWithPath: d)
    }

    func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static let kinds = ["normal", "blockscale", "heavy", "zeros", "bf16", "positive", "lattice"]

    func check(_ type: String, perRow: Int, quantize: (UnsafeBufferPointer<Float>, UnsafeBufferPointer<Float>?) -> Data,
               dequantize: (Data) -> [Float], blockBytes: Int) throws {
        var report: [String] = []
        for kind in Self.kinds {
            let x = try floats("\(kind).f32")
            let im = try floats("\(kind).imatrix.f32")
            for withIm in [false, true] {
                let tag = withIm ? "im" : "plain"
                let want = try Data(contentsOf: dir.appendingPathComponent("\(kind).\(type).\(tag).bin"))
                let got = x.withUnsafeBufferPointer { xp in im.withUnsafeBufferPointer { ip in quantize(xp, withIm ? ip : nil) } }
                XCTAssertEqual(got.count, want.count)
                var badBlocks = 0
                for b in 0 ..< want.count / blockBytes
                where got.subdata(in: b * blockBytes ..< (b + 1) * blockBytes) != want.subdata(in: b * blockBytes ..< (b + 1) * blockBytes) {
                    badBlocks += 1
                }
                if badBlocks > 0 { report.append("\(kind)/\(tag): \(badBlocks) of \(want.count / blockBytes) blocks differ") }
                // The dequantizer, on the reference bytes.
                let deq = try floats("\(kind).\(type).\(tag).deq.f32")
                let mine = dequantize(want)
                var badVals = 0
                for (a, b) in zip(deq, mine) where a.bitPattern != b.bitPattern { badVals += 1 }
                if badVals > 0 { report.append("\(kind)/\(tag): dequantize differs in \(badVals) values") }
            }
        }
        XCTAssertEqual(report, [], "\(type): \(report.joined(separator: "; "))")
    }

    func testQ4KMatchesGGML() throws {
        try check("q4_k", perRow: 512, quantize: { GGUFQ4K.quantize($0, perRow: 512, importance: $1) }, dequantize: GGUFQ4K.dequantize, blockBytes: 144)
    }

    func testQ5KMatchesGGML() throws {
        try check("q5_k", perRow: 512, quantize: { GGUFQ5K.quantize($0, perRow: 512, importance: $1) }, dequantize: GGUFQ5K.dequantize, blockBytes: 176)
    }

    func testQ6KMatchesGGML() throws {
        try check("q6_k", perRow: 512, quantize: { GGUFQ6K.quantize($0, perRow: 512, importance: $1) }, dequantize: GGUFQ6K.dequantize, blockBytes: 210)
    }

    func testQ5_0MatchesGGML() throws {
        try check("q5_0", perRow: 512, quantize: { GGUFQ5_0.quantize($0, perRow: 512, importance: $1) }, dequantize: GGUFQ5_0.dequantize, blockBytes: 22)
    }
}

extension KQuantOracleTests {
    /// Finds which multiply-adds the reference fuses: tries every on/off combination of `sites`.
    func search(_ name: String, type: String, im: Bool, sites: [Int], mismatches: () throws -> Int) throws {
        try preservingFusedSites { try searchBody(name, type: type, im: im, sites: sites, mismatches: mismatches) }
    }

    func searchBody(_ name: String, type: String, im: Bool, sites: [Int], mismatches: () throws -> Int) throws {
        var best = (Int.max, 0)
        for mask in 0 ..< (1 << sites.count) {
            for (k, s) in sites.enumerated() { fmaSites[s] = (mask >> k) & 1 == 1 }
            let m = try mismatches()
            if m < best.0 { best = (m, mask) }
            if m == 0 { break }
        }
        for (k, s) in sites.enumerated() { fmaSites[s] = (best.1 >> k) & 1 == 1 }
        print("KQSEARCH \(name): best \(best.0) mismatching blocks; fused sites \(sites.enumerated().filter { (best.1 >> $0.offset) & 1 == 1 }.map(\.element))")
    }

    func blocksDiffering(_ type: String, withIm: Bool, blockBytes: Int,
                         quantize: (UnsafeBufferPointer<Float>, UnsafeBufferPointer<Float>?) -> Data) throws -> Int {
        var bad = 0
        for kind in Self.kinds {
            let x = try floats("\(kind).f32"), im = try floats("\(kind).imatrix.f32")
            let want = try Data(contentsOf: dir.appendingPathComponent("\(kind).\(type).\(withIm ? "im" : "plain").bin"))
            let got = x.withUnsafeBufferPointer { xp in im.withUnsafeBufferPointer { ip in quantize(xp, withIm ? ip : nil) } }
            for b in 0 ..< want.count / blockBytes
            where got.subdata(in: b * blockBytes ..< (b + 1) * blockBytes) != want.subdata(in: b * blockBytes ..< (b + 1) * blockBytes) { bad += 1 }
        }
        return bad
    }

    func greedy(_ name: String, sites: [Int], mismatches: () throws -> Int) throws {
        try preservingFusedSites { try greedyBody(name, sites: sites, mismatches: mismatches) }
    }

    func greedyBody(_ name: String, sites: [Int], mismatches: () throws -> Int) throws {
        var best = try mismatches()
        var improved = true
        while improved && best > 0 {
            improved = false
            for s in sites {
                fmaSites[s].toggle()
                let m = try mismatches()
                if m < best { best = m; improved = true } else { fmaSites[s].toggle() }
            }
        }
        print("KQGREEDY \(name): \(best) mismatching blocks; fused sites \(sites.filter { fmaSites[$0] })")
    }

    func testGreedyQ5K() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_KQUANT_SEARCH"] == "greedy5k")
        let sites = Array(105 ... 114) + [126]
        try greedy("q5_k plain", sites: sites) {
            try blocksDiffering("q5_k", withIm: false, blockBytes: 176) { GGUFQ5K.quantize($0, perRow: 512, importance: $1) }
        }
        try greedy("q5_k im", sites: Array(105 ... 114) + Array(115 ... 125) + [128]) {
            try blocksDiffering("q5_k", withIm: true, blockBytes: 176) { GGUFQ5K.quantize($0, perRow: 512, importance: $1) }
        }
    }

    func testGreedyQ5() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_KQUANT_SEARCH"] == "greedy5")
        try greedy("q5_0 im", sites: [40, 41, 42, 43, 44, 31, 32]) {
            try blocksDiffering("q5_0", withIm: true, blockBytes: 22) { GGUFQ5_0.quantize($0, perRow: 512, importance: $1) }
        }
    }

    func testGreedyQ4KIm() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_KQUANT_SEARCH"] == "greedy4im")
        try greedy("q4_k im", sites: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28]) {
            try blocksDiffering("q4_k", withIm: true, blockBytes: 144) { GGUFQ4K.quantize($0, perRow: 512, importance: $1) }
        }
    }

    func testGreedyQ4K() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_KQUANT_SEARCH"] == "greedy")
        try greedy("q4_k plain", sites: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 26]) {
            try blocksDiffering("q4_k", withIm: false, blockBytes: 144) { GGUFQ4K.quantize($0, perRow: 512, importance: $1) }
        }
        try greedy("q4_k im", sites: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28]) {
            try blocksDiffering("q4_k", withIm: true, blockBytes: 144) { GGUFQ4K.quantize($0, perRow: 512, importance: $1) }
        }
    }

    func testShowMismatches() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHEARS_KQUANT_SHOW"] == "1")
        for (type, bb) in [("q6_k", 210), ("q5_0", 22), ("q4_k", 144)] {
            for withIm in [false, true] {
                for kind in Self.kinds {
                    let x = try floats("\(kind).f32"), im = try floats("\(kind).imatrix.f32")
                    let want = try Data(contentsOf: dir.appendingPathComponent("\(kind).\(type).\(withIm ? "im" : "plain").bin"))
                    let got: Data = x.withUnsafeBufferPointer { xp in im.withUnsafeBufferPointer { ip in
                        let i = withIm ? ip : nil
                        switch type {
                        case "q6_k": return GGUFQ6K.quantize(xp, perRow: 512, importance: i)
                        case "q5_0": return GGUFQ5_0.quantize(xp, perRow: 512, importance: i)
                        default: return GGUFQ4K.quantize(xp, perRow: 512, importance: i)
                        }
                    } }
                    for b in 0 ..< want.count / bb {
                        let g = got.subdata(in: b * bb ..< (b + 1) * bb), w = want.subdata(in: b * bb ..< (b + 1) * bb)
                        if g != w {
                            let diffs = (0 ..< bb).filter { g[$0] != w[$0] }
                            print("KQSHOW \(type) \(withIm ? "im" : "plain") \(kind) block \(b): \(diffs.count) bytes differ at \(diffs.prefix(12))")
                        }
                    }
                }
            }
        }
    }

    func testSearchFusedSites() throws {
        let which = ProcessInfo.processInfo.environment["SHEARS_KQUANT_SEARCH"] ?? ""
        try XCTSkipUnless(which == "qx" || which == "q4k")
        if which == "qx" {
        try search("q6_k plain", type: "q6_k", im: false, sites: [0, 1, 2, 3, 4]) {
            try blocksDiffering("q6_k", withIm: false, blockBytes: 210) { GGUFQ6K.quantize($0, perRow: 512, importance: $1) }
        }
        try search("q6_k im", type: "q6_k", im: true, sites: [0, 1, 2, 3, 4]) {
            try blocksDiffering("q6_k", withIm: true, blockBytes: 210) { GGUFQ6K.quantize($0, perRow: 512, importance: $1) }
        }
        try search("q5_0 im", type: "q5_0", im: true, sites: [0, 1, 2, 3, 4, 31, 32]) {
            try blocksDiffering("q5_0", withIm: true, blockBytes: 22) { GGUFQ5_0.quantize($0, perRow: 512, importance: $1) }
        }
        }
        if which == "q4k" {
        try search("q4_k plain", type: "q4_k", im: false, sites: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 26]) {
            try blocksDiffering("q4_k", withIm: false, blockBytes: 144) { GGUFQ4K.quantize($0, perRow: 512, importance: $1) }
        }
        }
    }
}

/// `fmaSites` is process-wide state the K-quant code reads; the searches above change it to find
/// the reference's pattern. Anything that changes it must put it back, or every later test in the
/// process quantizes differently.
func preservingFusedSites<T>(_ body: () throws -> T) rethrows -> T {
    let saved = fmaSites
    defer { fmaSites = saved }
    return try body()
}

final class FusedSitesTests: XCTestCase {
    func testTheSearchHelpersLeaveTheDefaultPatternAlone() throws {
        let before = fmaSites
        // The pattern ported from the reference (see GGUFKQuants.swift).
        var expected = [Bool](repeating: true, count: 256)
        for i in [0, 1, 2, 9, 10, 11, 20, 21, 27, 31, 40, 41, 42, 43, 44] { expected[i] = false }
        for i in 0 ..< 40 { expected[100 + i] = expected[i] }
        XCTAssertEqual(before, expected)
        let oracle = KQuantOracleTests()
        try oracle.search("probe", type: "q4_k", im: false, sites: [5, 6, 7], mismatches: { fmaSites[5] ? 1 : 0 })
        try oracle.greedy("probe", sites: [5, 6, 7], mismatches: { fmaSites[6] ? 1 : 0 })
        XCTAssertEqual(fmaSites, before, "the searches changed the global fused-site pattern")
    }
}

import Foundation
import XCTest

@testable import Shears

final class GGUFGenerationConfigTests: XCTestCase {
    func dir(_ generation: String?) throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("gen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        if let generation { try generation.write(to: u.appendingPathComponent("generation_config.json"), atomically: true, encoding: .utf8) }
        return u
    }

    func entries(_ json: String?) throws -> [String: GGUFValue] {
        Dictionary(uniqueKeysWithValues: GGUFGenerationConfig.samplingEntries(directory: try dir(json)).map { ($0.0, $0.1) })
    }

    func testCopiesWhatTheConverterCopies() throws {
        let e = try entries("""
            {"top_k": 64, "top_p": 0.95, "temperature": 1.0, "min_p": 0.05, "repetition_penalty": 1.1, "penalty_last_n": 128,
             "mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "xtc_probability": 0.5, "xtc_threshold": 0.1,
             "sequence": "top_k;top_p;temperature", "do_sample": true, "eos_token_id": [1, 106], "cache_implementation": "hybrid"}
            """)
        XCTAssertEqual(e["general.sampling.top_k"], .int32(64))
        XCTAssertEqual(e["general.sampling.top_p"], .float32(0.95))
        XCTAssertEqual(e["general.sampling.temp"], .float32(1.0))
        XCTAssertEqual(e["general.sampling.min_p"], .float32(0.05))
        XCTAssertEqual(e["general.sampling.penalty_repeat"], .float32(1.1))
        XCTAssertEqual(e["general.sampling.penalty_last_n"], .int32(128))
        XCTAssertEqual(e["general.sampling.mirostat"], .int32(2))
        XCTAssertEqual(e["general.sampling.mirostat_tau"], .float32(5.0))
        XCTAssertEqual(e["general.sampling.mirostat_eta"], .float32(0.1))
        XCTAssertEqual(e["general.sampling.xtc_probability"], .float32(0.5))
        XCTAssertEqual(e["general.sampling.xtc_threshold"], .float32(0.1))
        XCTAssertEqual(e["general.sampling.sequence"], .string("top_k;top_p;temperature"))
        XCTAssertEqual(e.count, 12, "do_sample, eos_token_id and the rest are not copied")
    }

    func testOrderAndPrecedenceFollowTheConverter() throws {
        let list = GGUFGenerationConfig.samplingEntries(directory: try dir(
            #"{"temperature": 0.7, "top_k": 40, "penalty_repeat": 1.3, "repetition_penalty": 9, "top_p": 0.9}"#))
        XCTAssertEqual(list.map(\.0), [
            "general.sampling.top_k", "general.sampling.top_p", "general.sampling.temp", "general.sampling.penalty_repeat",
        ])
        XCTAssertEqual(list.last?.1, .float32(1.3), "penalty_repeat wins over repetition_penalty")
    }

    func testMissingBadAndMistypedValuesWriteNothing() throws {
        XCTAssertEqual(try entries(nil).count, 0)
        XCTAssertEqual(try entries("not json").count, 0)
        XCTAssertEqual(try entries("[1, 2]").count, 0)
        let e = try entries(#"{"top_k": 64.5, "top_p": "0.9", "temperature": null, "min_p": true, "mirostat": 1.0}"#)
        XCTAssertEqual(Set(e.keys), ["general.sampling.mirostat"], "a whole-number float is an int; null, bool and string are not numbers")
    }

    func testExportWritesTheDefaultsAheadOfTheName() throws {
        let src = try dir(#"{"top_k": 64, "top_p": 0.95, "temperature": 1.0}"#)
        try GGUFTinyGemma4().write(to: src)
        try #"{"top_k": 64, "top_p": 0.95, "temperature": 1.0, "do_sample": true}"#.write(
            to: src.appendingPathComponent("generation_config.json"), atomically: true, encoding: .utf8)
        let out = src.appendingPathComponent("m.gguf")
        try GGUFExport.gemma4(hfDirectory: src, output: out, type: .f32)
        let r = try GGUFReader(url: out)
        XCTAssertEqual(r["general.sampling.top_k"], .int32(64))
        XCTAssertEqual(r["general.sampling.top_p"], .float32(0.95))
        XCTAssertEqual(r["general.sampling.temp"], .float32(1.0))
        let keys = r.metadata.map(\.key)
        XCTAssertLessThan(keys.firstIndex(of: "general.sampling.temp")!, keys.firstIndex(of: "general.name")!)
        // Opting out, and a directory without the file, write none.
        let off = src.appendingPathComponent("off.gguf")
        try GGUFExport.gemma4(hfDirectory: src, output: off, type: .f32, copyGenerationConfig: false)
        XCTAssertNil(try GGUFReader(url: off)["general.sampling.top_k"])
        let bare = try dir(nil)
        try GGUFTinyGemma4().write(to: bare)
        let plain = bare.appendingPathComponent("p.gguf")
        try GGUFExport.gemma4(hfDirectory: bare, output: plain, type: .f32)
        XCTAssertTrue(try GGUFReader(url: plain).metadata.allSatisfy { !$0.key.hasPrefix("general.sampling.") })
    }
}

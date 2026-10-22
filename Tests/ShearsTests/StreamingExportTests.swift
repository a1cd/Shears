import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// The export streams one layer at a time: the same tensors as before, and a peak that does not grow with
/// the model.
final class StreamingExportTests: ShearsTestCase {
    func tempDir(_ name: String = "stream-export") throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func tensors(in dir: URL) throws -> [String: MLXArray] {
        let idx = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
        var out: [String: MLXArray] = [:]
        for f in Set((idx["weight_map"] as! [String: String]).values) {
            out.merge(try loadArrays(url: dir.appendingPathComponent(f))) { $1 }
        }
        return out
    }

    /// A 4-bit checkpoint exported straight from its files equals the export of the dequantized twin model
    /// (dequantize everything, then slice): slicing the packed rows first changes no number.
    func testExportOfAQuantizedCheckpointEqualsDequantizeThenSlice() throws {
        let ckpt = try tempDir("q-ckpt")
        let twin = try QuantizedFixture.write(try HealingFixture.model(), to: ckpt)
        let keep = HealingFixture.keepSets(count: 4)
        let h = try HealingFixture.healer(model: twin, keep: keep)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        o.layersPerShard = 3  // a shard of three and one of one: the grouping must not matter
        let streamed = try tempDir("streamed")
        try MergeAndSliceExport.write(
            checkpoint: try Gemma4Checkpoint(directory: ckpt), loadDType: .float32, keep: keep, adapters: h.adapters,
            to: streamed, options: o)
        let reference = try tempDir("reference")
        o.layersPerShard = 5
        try MergeAndSliceExport.write(model: twin, keep: keep, adapters: h.adapters, to: reference, options: o)
        let a = try tensors(in: streamed), b = try tensors(in: reference)
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        XCTAssertFalse(a.keys.contains { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") })
        for (k, v) in b { assertClose(a[k]!, v, atol: 1e-6, rtol: 1e-6, k) }
        let shards = try FileManager.default.contentsOfDirectory(atPath: streamed.path).filter { $0.hasPrefix("model-layers-") }
        XCTAssertEqual(shards.sorted(), ["model-layers-00.safetensors", "model-layers-01.safetensors"])
    }

    /// The per-layer adapter provider (what `shears-cli export` uses) gives the tensors of handing over the whole
    /// `ModelAdapters`, and the record of rounding formats.
    func testLayerByLayerAdaptersEqualWholeModelAdapters() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        var ac = HealingFixture.smallAdapters
        ac.qat = QATConfiguration.q4KM(for: m, keptExperts: 4, exportDType: .float32)
        let h = try HealingFixture.healer(model: m, keep: keep, adapters: ac)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.1)
        var o = MergeAndSliceExport.Options()
        o.dtype = .float32
        let whole = try tempDir("whole")
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: h.adapters, to: whole, options: o)

        let file = try tempDir("adapters").appendingPathComponent("a.safetensors")
        try save(arrays: h.adapters.tensors(), url: file)
        let lazy = try loadArrays(url: file)
        let n = m.configuration.numExperts ?? 0
        var layersAsked: [Int] = []
        let piece = try tempDir("piece")
        try MergeAndSliceExport.write(
            source: ModelLayerSource(model: m),
            globals: {
                ["model.embed_tokens.weight": m.model.embedTokens.weight, "model.norm.weight": m.model.norm.weight]
            },
            configuration: m.configuration, keep: keep, adapterConfiguration: ac,
            layerAdapters: { i, layer, _ in
                layersAsked.append(i)
                let ma = ModelAdapters(
                    layers: [layer], firstLayer: i, sets: [ExpertSet(keep: keep[i], numExperts: n)], configuration: ac)
                try ma.load(subsetOf: lazy)
                return ma.layers[0]
            }, to: piece, options: o)
        XCTAssertEqual(layersAsked, [0, 1, 2, 3])
        let a = try tensors(in: piece), b = try tensors(in: whole)
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        for (k, v) in b { assertClose(a[k]!, v, atol: 0, rtol: 0, k) }
        let ra = try Data(contentsOf: piece.appendingPathComponent(ExportRecord.fileName))
        let rb = try Data(contentsOf: whole.appendingPathComponent(ExportRecord.fileName))
        XCTAssertEqual(try JSONDecoder().decode(ExportRecord.self, from: ra), try JSONDecoder().decode(ExportRecord.self, from: rb))
        XCTAssertFalse(try JSONDecoder().decode(ExportRecord.self, from: ra).formats.isEmpty, "the QAT formats are recorded")
    }
}

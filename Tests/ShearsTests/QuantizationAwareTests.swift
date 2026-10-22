import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

final class QuantizationAwareTests: ShearsTestCase {
    func tempDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("qat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func floats(_ a: MLXArray) -> [Float] { a.asType(.float32).reshaped(-1).asArray(Float.self) }

    /// Values with very different magnitudes per block, an all-zero block and a block of ties.
    func awkward(_ rows: Int, _ cols: Int, seed: UInt64 = 1) -> MLXArray {
        MLXRandom.seed(seed)
        let blocks = cols / 32
        let body = (MLXRandom.normal([rows, blocks, 32]) * exp(2 * MLXRandom.normal([rows, blocks, 1])))
        var parts = body.reshaped(rows, cols)
        // Block 0 is all zeros; block 1 has amax = 127 d with codes that land on x.5.
        var special = [Float](repeating: 0, count: cols)
        for (i, v) in [Float](arrayLiteral: 127, 63.5, -63.5, 0.5, -0.5, 1.5, -2.5, 126.5).enumerated() { special[32 + i] = v }
        let mask = MLXArray((0 ..< cols).map { $0 < 64 ? Float(0) : 1 })[.newAxis]
        parts = parts * mask + MLXArray(special)[.newAxis]
        return parts
    }

    func testQ8RoundTripIsTheGGUFWritersExactly() throws {
        for seed in 1 ... 4 {
            let w = awkward(16, 128, seed: UInt64(seed))
            let want = floats(w).withUnsafeBufferPointer { GGUFQ8_0.dequantize(GGUFQ8_0.quantize($0)) }
            let got = floats(QuantFormat.q8_0.roundTrip(w))
            XCTAssertEqual(got.count, want.count)
            var bad = 0
            for (g, e) in zip(got, want) where g != e { bad += 1 }
            XCTAssertEqual(bad, 0, "seed \(seed): \(bad) of \(want.count) values differ from the GGUF writer")
        }
    }

    func testStraightThroughValueAndGradient() {
        for f in [QuantFormat.q8_0, .affine(bits: 4, groupSize: 32)] {
            let w = awkward(8, 64)
            XCTAssertEqual(floats(f.fakeQuantized(w)), floats(f.roundTrip(w)), "value is exactly the round trip")
            let c = MLXRandom.normal([8, 64])
            let g = grad { (w: MLXArray) in (f.fakeQuantized(w) * c).sum() }(w)
            assertClose(g, c, atol: 0, rtol: 0, "identity gradient \(f)")
        }
    }

    func testAffineRoundTripIsMLXsAndStable() {
        let f = QuantFormat.affine(bits: 4, groupSize: 32)
        let w = awkward(8, 64)
        let (q, s, b) = MLX.quantized(w, groupSize: 32, bits: 4)
        XCTAssertEqual(floats(f.roundTrip(w)), floats(dequantized(q, scales: s, biases: b, groupSize: 32, bits: 4)))
        // A tensor that was already rounded stays put (an exported tensor re-quantizes to itself).
        let once = f.roundTrip(w)
        XCTAssertEqual(floats(f.roundTrip(once)), floats(once))
        // A last axis that is not a multiple of the group is left alone, as the writer's F16 fallback implies.
        let odd = MLXRandom.normal([4, 40])
        XCTAssertEqual(floats(QuantFormat.q8_0.roundTrip(odd)), floats(odd))
    }

    // MARK: through the model and the export

    static let q8 = QATConfiguration(.q8_0)
    func adapters(_ qat: QATConfiguration?) -> AdapterConfiguration {
        AdapterConfiguration(expertRank: 8, denseRank: 8, routerRank: 4, qat: qat)
    }

    /// A healed toy (default export dtype, no router correction: the export rounds the merged router
    /// to bfloat16, which healing does not model), its student logits, and the exported directory.
    struct Healed {
        let model: Gemma4TextModel
        let healer: PruneHealer
        let tokens: MLXArray
        let logits: MLXArray
        let hf: URL
        let source: URL
    }

    func healAndExport(_ qat: QATConfiguration?, options: ((inout MergeAndSliceExport.Options) -> Void)? = nil) throws -> Healed {
        let m = try HealingFixture.model()
        HealingFixture.makeBF16Exact(m)
        let keep = HealingFixture.keepSets(count: 4)
        var c = HealingFixture.tinyConfiguration()
        c.learningRate = 2e-3
        let ad = AdapterConfiguration(expertRank: 8, denseRank: 8, routerRank: 0, qat: qat)
        let h = try PruneHealer(model: m, keep: keep, adapterConfiguration: ad, configuration: c)
        HealingFixture.randomizeAdapters(h.adapters, scale: 0.03)
        let rows = [HealingFixture.row(length: 24, seed: 3, key: "a"), HealingFixture.row(length: 20, seed: 4, key: "b")]
        try h.run(rows: ShuffledRows(rows: rows, seed: 2), plan: PhasePlan(entries: [(.block, 0), (.freerun, 6)], total: 12))
        let tokens = MLXArray((0 ..< 24).map { Int32($0 * 7 % HealingFixture.vocab) })[.newAxis]
        let logits = h.logits(tokens: tokens, student: true)  // QAT forward, experts masked to the kept set

        // A config.json with the architecture entry the GGUF export needs; weights are written fresh.
        let src = try tempDir()
        let root: [String: Any] = [
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "eos_token_id": [1, 106], "text_config": HealingFixture.configJSON(),
        ]
        try JSONSerialization.data(withJSONObject: root).write(to: src.appendingPathComponent("config.json"))
        let dir = try tempDir()
        var o = MergeAndSliceExport.Options()  // the default dtype: bfloat16
        o.sourceDirectory = src
        options?(&o)
        try MergeAndSliceExport.write(model: m, keep: keep, adapters: h.adapters, to: dir, options: o)
        return Healed(model: m, healer: h, tokens: tokens, logits: logits, hf: dir, source: src)
    }

    func testExportedQ8ThroughGGUFReproducesTheHealedForward() throws {
        let h = try healAndExport(Self.q8)
        let gguf = h.hf.appendingPathComponent("q8.gguf")
        try GGUFExport.gemma4(hfDirectory: h.hf, output: gguf, type: .q8_0, tokenizer: .none)
        let reader = try GGUFReader(url: gguf)
        var replaced = 0
        let shipped = try HealingFixture.shippedModel(exported: h.hf, scratch: try tempDir()) { k, v in
            guard HealingFixture.isCovered(k), case .mapped(let g) = GGUFGemma4Name.map(k), reader.tensorInfo(g) != nil else { return nil }
            replaced += 1
            return MLXArray(try reader.floats(g), v.shape)
        }
        XCTAssertEqual(replaced, 4 * 6)
        assertClose(shipped(h.tokens), h.logits, atol: 2e-4, rtol: 2e-4, "Q8_0 GGUF of the default-dtype export vs the QAT-healed forward")

        // The check can fail: healing without the rounding gives a visibly different model.
        let plain = try healAndExport(nil)
        XCTAssertGreaterThan(abs(plain.logits - h.logits).max().item(Float.self), 1e-3)
    }

    /// The shipped bf16 tensor is quantized by the converter (here MLX's own quantizer, as
    /// `mlx_lm.convert` does to a bf16 checkpoint), and must equal what healing rounded to.
    func testExportedAffineQuantizedByMLXReproducesTheHealedForward() throws {
        let f = QuantFormat.affine(bits: 4, groupSize: 32)
        let h = try healAndExport(QATConfiguration(f))
        var replaced = 0
        let shipped = try HealingFixture.shippedModel(exported: h.hf, scratch: try tempDir()) { k, v in
            guard HealingFixture.isCovered(k) else { return nil }
            XCTAssertEqual(v.dtype, .bfloat16, k)
            let flat = v.reshaped(-1, v.dim(-1))
            let (q, s, b) = MLX.quantized(flat, groupSize: 32, bits: 4)
            XCTAssertEqual(s.dtype, .bfloat16, "MLX quantizes a bf16 tensor to bf16 scales")
            replaced += 1
            return dequantized(q, scales: s, biases: b, groupSize: 32, bits: 4).asType(.float32).reshaped(v.shape)
        }
        XCTAssertEqual(replaced, 4 * 6)
        assertClose(shipped(h.tokens), h.logits, atol: 2e-4, rtol: 2e-4, "MLX 4-bit of the default-dtype export vs the QAT-healed forward")
    }

    func testRoundTripAtADtypeQuantizesTheExportedBits() {
        let w = awkward(16, 128, seed: 3)
        let bf = w.asType(.bfloat16)
        // Q8_0 sees the bfloat16 values, not the float32 ones they came from.
        let want = floats(bf).withUnsafeBufferPointer { GGUFQ8_0.dequantize(GGUFQ8_0.quantize($0)) }
        XCTAssertEqual(floats(QuantFormat.q8_0.roundTrip(w, dtype: .bfloat16)), want)
        XCTAssertNotEqual(floats(QuantFormat.q8_0.roundTrip(w, dtype: .bfloat16)), floats(QuantFormat.q8_0.roundTrip(w)),
                          "the dtype matters: otherwise the test cannot tell the paths apart")
        // Affine: scales and biases in the export dtype, as MLX gives a bf16 tensor.
        let f = QuantFormat.affine(bits: 4, groupSize: 32)
        let (q, s, b) = MLX.quantized(bf, groupSize: 32, bits: 4)
        XCTAssertEqual(s.dtype, .bfloat16)
        XCTAssertEqual(floats(f.roundTrip(w, dtype: .bfloat16)),
                       floats(dequantized(q, scales: s, biases: b, groupSize: 32, bits: 4)))
    }

    // MARK: the export refuses a shipment that would not be the healed model

    func testExportRefusesADtypeOtherThanTheOneHealedFor() throws {
        XCTAssertThrowsError(try healAndExport(Self.q8) { $0.dtype = .float32 }) { e in
            XCTAssertTrue("\(e)".contains("healed against"), "\(e)")
        }
        // A float32 QAT run exports in float32.
        _ = try healAndExport(QATConfiguration(.q8_0, exportDType: .float32)) { $0.dtype = .float32 }
    }

    func testGGUFRefusesAnImportanceMatrixOnAQATModel() throws {
        let h = try healAndExport(Self.q8)
        let imatrix = GGUFImatrix(entries: ["blk.0.ffn_gate.weight": [Float](repeating: 1, count: 64)])
        XCTAssertThrowsError(
            try GGUFExport.gemma4(hfDirectory: h.hf, output: h.hf.appendingPathComponent("x.gguf"), type: .q8_0, tokenizer: .none, importance: imatrix)
        ) { XCTAssertTrue("\($0)".contains("importance"), "\($0)") }
        // Without QAT the same matrix is accepted (it is not an error to use one).
        let plain = try healAndExport(nil)
        XCTAssertNoThrow(
            try GGUFExport.gemma4(hfDirectory: plain.hf, output: plain.hf.appendingPathComponent("x.gguf"), type: .q8_0, tokenizer: .none, importance: imatrix))
    }

    func testGGUFRefusesATypeOtherThanTheOneHealedFor() throws {
        let h = try healAndExport(Self.q8)
        for wrong in [GGUFOutputType.bf16, .f16, .f32] {
            XCTAssertThrowsError(
                try GGUFExport.gemma4(hfDirectory: h.hf, output: h.hf.appendingPathComponent("x.gguf"), type: wrong, tokenizer: .none)
            ) { XCTAssertTrue("\($0)".contains("healed against"), "\($0)") }
        }
        // An MLX-affine model cannot become a GGUF at all.
        let affine = try healAndExport(QATConfiguration(.affine(bits: 4, groupSize: 32)))
        XCTAssertThrowsError(
            try GGUFExport.gemma4(hfDirectory: affine.hf, output: affine.hf.appendingPathComponent("x.gguf"), type: .q8_0, tokenizer: .none))
        // A checkpoint whose weights were re-saved in another dtype is refused too.
        let record = h.hf.appendingPathComponent(ExportRecord.fileName)
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: record)) as! [String: Any]
        json["dtype"] = "F32"
        try JSONSerialization.data(withJSONObject: json).write(to: record)
        XCTAssertThrowsError(
            try GGUFExport.gemma4(hfDirectory: h.hf, output: h.hf.appendingPathComponent("x.gguf"), type: .q8_0, tokenizer: .none))
    }

    func testSlicedStudentTrainsAgainstTheSameRoundedWeights() throws {
        try checkSlicedQAT(sortThreshold: nil)
    }

    /// The sorted gather path (real runs always take it) through the rounded expert weights.
    func testSlicedStudentTrainsAgainstTheSameRoundedWeightsWithSortedGather() throws {
        try checkSlicedQAT(sortThreshold: 1)
    }

    func checkSlicedQAT(sortThreshold: Int?) throws {
        let m = try HealingFixture.model()
        if let t = sortThreshold { HealingFixture.forceSortedGather(m, threshold: t) }
        let keep = HealingFixture.keepSets(count: 4)
        let qat = QATConfiguration(.affine(bits: 4, groupSize: 32))
        let row = HealingFixture.row()
        let store = InMemoryBoundaryStore()
        try TeacherPrepass(source: ModelLayerSource(model: m), blockEnds: m.defaultBlockEnds, options: .init(storageDType: .float32))
            .run(rows: [row], store: store)
        let masked = try PruneHealer(model: m, keep: keep, adapterConfiguration: adapters(qat), configuration: HealingFixture.tinyConfiguration())
        HealingFixture.randomizeAdapters(masked.adapters, scale: 0.03)
        let initial = HealingFixture.copyTensors(masked.adapters.tensors())
        let copy = Gemma4TextModel(m.configuration)
        copy.update(parameters: ModuleParameters.unflattened(m.parameters().flattened().map { ($0.0, $0.1 * 1) }))
        copy.train(false); eval(copy)
        if let t = sortThreshold { HealingFixture.forceSortedGather(copy, threshold: t) }
        let sliced = try PruneHealer(
            model: copy, keep: keep, adapterConfiguration: adapters(qat), configuration: HealingFixture.tinyConfiguration(),
            initialAdapters: initial, student: .sliced, teacher: store)
        for phase in [HealingPhase.block, .freerun, .end] {
            let a = masked.stepFor(phase, row, step: 1), b = sliced.stepFor(phase, row, step: 1)
            XCTAssertEqual(a.loss, b.loss, accuracy: 2e-4 * max(1, a.loss), "\(phase)")
            for (k, g) in a.grads { assertClose(b.grads[k]!, g, atol: 5e-5, rtol: 5e-3, "\(phase) \(k)") }
        }
    }

    /// Healing is judged on the shipped (rounded) model. At 3 bits the toy's rounding error is large
    /// next to what low-rank adapters can fix, and training against it recovers a clear part; at a
    /// finer format or a high learning rate the gain shrinks or reverses (the straight-through
    /// gradient is noisy), so the README says so.
    func testTrainingAgainstTheFormatBeatsRoundingAfterwards() throws {
        let m = try HealingFixture.model()
        let keep = HealingFixture.keepSets(count: 4)
        let qat = QATConfiguration(.affine(bits: 3, groupSize: 32))
        var c = HealingFixture.tinyConfiguration()
        c.learningRate = 1e-3
        c.learningRateFloor = 1e-4
        let rows = [HealingFixture.row(length: 24, seed: 3, key: "a"), HealingFixture.row(length: 22, seed: 5, key: "b"), HealingFixture.row(length: 20, seed: 6, key: "c")]
        let held = HealingFixture.row(length: 23, seed: 9, key: "held")
        let plan = PhasePlan(entries: [(.freerun, 0)], total: 80)

        func heal(_ qatTraining: QATConfiguration?) throws -> [String: MLXArray] {
            MLXRandom.seed(21)
            let h = try PruneHealer(model: m, keep: keep, adapterConfiguration: adapters(qatTraining), configuration: c)
            try h.run(rows: ShuffledRows(rows: rows, seed: 4), plan: plan)
            return HealingFixture.copyTensors(h.adapters.tensors())
        }
        // What ships is the rounded model, so both are judged with the rounding on.
        func shippedLoss(_ t: [String: MLXArray]) throws -> Float {
            let e = try PruneHealer(model: m, keep: keep, adapterConfiguration: adapters(qat), configuration: c, initialAdapters: t)
            return [held, rows[0]].map { e.probeLoss(.freerun, row: $0) }.reduce(0, +) / 2
        }
        let post = try shippedLoss(try heal(nil))
        let aware = try shippedLoss(try heal(qat))
        XCTAssertLessThan(aware, post * 0.9, "aware \(aware) vs post-hoc \(post)")
    }
}

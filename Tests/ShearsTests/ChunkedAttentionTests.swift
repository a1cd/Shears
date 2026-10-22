import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import Shears

/// Query-chunked attention must be the same function as the plain masked attention, in value and
/// in gradient, for sliding and global layers.
final class ChunkedAttentionTests: ShearsTestCase {
    func check(dtype: DType, tolerance: Float, length: Int, chunk: Int) throws {
        let m = try HealingFixture.model(dtype: dtype)
        let saved = Gemma4Attention.queryChunk
        defer { Gemma4Attention.queryChunk = saved }
        for layer in m.layers.prefix(2) {  // layer 0 sliding (window 8), layer 1 global
            MLXRandom.seed(3)
            let x = MLXRandom.normal([1, length, 64]).asType(dtype)
            let normed = layer.inputLayernorm(x)
            let window: Int? = layer.isSliding ? layer.slidingWindow : nil
            // Reference: the original path, one full boolean mask.
            let want = layer.selfAttn.attended(normed, mask: layer.mask(n: length), cache: nil)
            Gemma4Attention.queryChunk = chunk
            let got = layer.selfAttn.attendedWhole(normed, window: window)
            Gemma4Attention.queryChunk = 0
            let whole = layer.selfAttn.attendedWhole(normed, window: window)
            assertClose(got.asType(.float32), want.asType(.float32), atol: tolerance, rtol: tolerance, "chunked sliding=\(layer.isSliding)")
            assertClose(whole.asType(.float32), want.asType(.float32), atol: tolerance, rtol: tolerance, "unchunked sliding=\(layer.isSliding)")

            // Gradient with respect to the input, through a fixed cotangent.
            let w = MLXRandom.normal(want.shape)
            func gradAt(chunk: Int) -> MLXArray {
                Gemma4Attention.queryChunk = chunk
                let g = grad { (x: MLXArray) in
                    (layer.selfAttn.attendedWhole(layer.inputLayernorm(x), window: window).asType(.float32) * w).sum()
                }
                return g(x)
            }
            let gc = gradAt(chunk: chunk), g0 = gradAt(chunk: 0)
            XCTAssertGreaterThan(abs(g0.asType(.float32)).max().item(Float.self), 0)
            assertClose(gc.asType(.float32), g0.asType(.float32), atol: tolerance * 10, rtol: tolerance * 10, "grad sliding=\(layer.isSliding)")
        }
    }

    func testChunkedAttentionEqualsPlainFloat32() throws {
        try check(dtype: .float32, tolerance: 2e-5, length: 41, chunk: 8)  // ragged last chunk
        try check(dtype: .float32, tolerance: 2e-5, length: 40, chunk: 7)
    }

    func testChunkedAttentionEqualsPlainBFloat16() throws {
        try check(dtype: .bfloat16, tolerance: 6e-2, length: 41, chunk: 8)
    }
}

/// The fused attention kernel's real shapes: head dimension 256 (sliding layers) and 512 with keys
/// equal to values (global layers), a boolean mask, and key length different from query length
/// (a cached continuation; the chunked, no-gradient pre-pass). Each is compared with a plain
/// float32 softmax(q k^T) v written out by hand, not with another call of the same kernel.
final class Gemma4AttentionKernelTests: ShearsTestCase {
    func configuration() throws -> Gemma4Configuration {
        var json = HealingFixture.configJSON()
        json["head_dim"] = 256
        json["global_head_dim"] = 512
        let c = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        try c.validate()
        return c
    }

    /// `softmax(mask(q k^T)) v` per query head, scale 1, grouped keys and values repeated.
    func reference(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray) -> MLXArray {
        let rep = q.dim(1) / k.dim(1)
        let kk = repeated(k.asType(.float32), count: rep, axis: 1)
        let vv = repeated(v.asType(.float32), count: rep, axis: 1)
        var s = matmul(q.asType(.float32), kk.transposed(0, 1, 3, 2))
        s = MLX.where(mask, s, MLXArray(Float(-1e30)))
        return matmul(softmax(s, axis: -1), vv)
    }

    func layers() throws -> [(String, Gemma4Attention, Int?)] {
        let c = try configuration()
        let sliding = Gemma4Attention(c, layer: 0), global = Gemma4Attention(c, layer: 1)
        XCTAssertEqual(sliding.headDim, 256)
        XCTAssertEqual(global.headDim, 512)
        XCTAssertNil(global.vProj, "global layers have keys equal to values")
        for a in [sliding, global] { eval(a) }
        return [("head_dim 256 sliding", sliding, c.slidingWindow), ("head_dim 512 k=v global", global, nil)]
    }

    func testFullSequenceAttentionMatchesTheHandWrittenSoftmax() throws {
        for (name, attn, window) in try layers() {
            for length in [5, 8, 9, 21] {  // up to and past the fused kernel's 8-query limit
                MLXRandom.seed(UInt64(length))
                let x = MLXRandom.normal([1, length, 64])
                let (q, k, v) = attn.projected(x, offset: 0)
                let mask = gemma4AttentionMask(n: length, window: window)
                let want = reference(q: q, k: k, v: v, mask: mask).transposed(0, 2, 1, 3).reshaped(1, length, -1)
                assertClose(attn.attended(x, mask: mask, cache: nil), want, atol: 1e-3, rtol: 1e-3, "\(name) L=\(length)")
            }
        }
    }

    func testCachedContinuationWithFewerQueriesThanKeysMatches() throws {
        for (name, attn, window) in try layers() {
            MLXRandom.seed(5)
            let total = 17, prefix = 14
            let x = MLXRandom.normal([1, total, 64])
            let (q, k, v) = attn.projected(x, offset: 0)
            let full = reference(q: q, k: k, v: v, mask: gemma4AttentionMask(n: total, window: window))
            let cache = Gemma4KVCache()
            _ = attn.attended(x[0..., ..<prefix], mask: gemma4AttentionMask(n: prefix, window: window), cache: cache)
            // 3 queries against 17 keys, with a boolean [3, 17] mask.
            let n = total - prefix
            let got = attn.attended(x[0..., prefix...], mask: gemma4AttentionMask(n: n, offset: prefix, window: window), cache: cache)
            XCTAssertEqual(cache.offset, total)
            let want = full[0..., 0..., prefix..., 0...].transposed(0, 2, 1, 3).reshaped(1, n, -1)
            assertClose(got, want, atol: 1e-3, rtol: 1e-3, "\(name) cached")
        }
    }

    func testChunkedNoGradientAttentionMatchesAtTheRealHeadDimensions() throws {
        let saved = Gemma4Attention.queryChunk
        defer { Gemma4Attention.queryChunk = saved }
        for (name, attn, window) in try layers() {
            let length = 29  // ragged chunks; each chunk sees fewer keys than the whole sequence
            MLXRandom.seed(9)
            let x = MLXRandom.normal([1, length, 64])
            let (q, k, v) = attn.projected(x, offset: 0)
            let want = reference(q: q, k: k, v: v, mask: gemma4AttentionMask(n: length, window: window))
                .transposed(0, 2, 1, 3).reshaped(1, length, -1)
            for chunk in [8, 11, 0] {
                Gemma4Attention.queryChunk = chunk
                assertClose(attn.attendedWhole(x, window: window), want, atol: 1e-3, rtol: 1e-3, "\(name) chunk \(chunk)")
            }
        }
    }

    /// The rope `factor` read from the config reaches the attention layer's rotation.
    func testConfiguredRopeFactorChangesTheGlobalLayersOutput() throws {
        func layer(factor: Float) throws -> Gemma4Attention {
            var json = HealingFixture.configJSON()
            json["rope_parameters"] = [
                "sliding_attention": ["rope_type": "default", "rope_theta": 10000.0],
                "full_attention": ["rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1.0e6, "factor": factor],
            ]
            let c = try JSONDecoder().decode(Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: json))
            MLXRandom.seed(4)  // the same weights for both
            let a = Gemma4Attention(c, layer: 1)
            eval(a)
            return a
        }
        MLXRandom.seed(8)
        let x = MLXRandom.normal([1, 12, 64])
        let mask = gemma4AttentionMask(n: 12)
        let a = try layer(factor: 1).attended(x, mask: mask, cache: nil)
        let b = try layer(factor: 1).attended(x, mask: mask, cache: nil)
        let scaled = try layer(factor: 16).attended(x, mask: mask, cache: nil)
        XCTAssertEqual(abs(a - b).max().item(Float.self), 0, "same weights, same output")
        XCTAssertGreaterThan(abs(a - scaled).max().item(Float.self), 1e-4)
    }
}

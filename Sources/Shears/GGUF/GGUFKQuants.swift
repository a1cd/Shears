import Foundation

// ggml's K-quants and Q5_0, reproduced from `ggml-quants.c` (llama.cpp commit a3a1c47) so that a
// tensor quantized here is byte-identical to what `llama-quantize` writes.
//
// Floating-point behaviour matters: the reference is compiled with clang's default contraction,
// which fuses `a * b + c` inside one expression into a fused multiply-add, and the scale search
// compares errors that differ in the last bit. Every place a multiply feeds an add or subtract in
// the C source is written here with `addingProduct`, and the sums run in the same order.

@inline(__always) private func nearestInt(_ f: Float) -> Int32 {
    // The reference's bit trick: round half to even for |f| < 2^22.
    let v = f + 12582912
    return Int32(bitPattern: v.bitPattern & 0x007F_FFFF) - 0x0040_0000
}

@inline(__always) private func clampInt(_ v: Int32, _ lo: Int32, _ hi: Int32) -> Int32 { max(lo, min(hi, v)) }

private let groupMaxEps: Float = 1e-15

/// Which multiply-adds are fused (one flag per site in this file). Found by comparing against the
/// reference's own output until nothing differed: the reductions inside the search loops
/// (`make_qx_quants` first pass, `make_qkx*_quants` iteration sums, the whole-row `sum_x2` of the
/// weighted Q4_K path, Q5_0's `sum_x2`) are not fused, everything else is.
nonisolated(unsafe) var fmaSites: [Bool] = {
    var f = [Bool](repeating: true, count: 256)
    for i in [0, 1, 2, 9, 10, 11, 20, 21, 27, 31, 40, 41, 42, 43, 44] { f[i] = false }
    for i in 0 ..< 40 { f[100 + i] = f[i] }
    return f
}()

@inline(__always) private func fm(_ site: Int, _ c: Float, _ a: Float, _ b: Float) -> Float {
    fmaSites[site] ? c.addingProduct(a, b) : c + a * b
}

/// `make_qx_quants` with `rmse_type == 1` (weights `x^2`, or `qw` when given).
private func makeQxQuants(
    n: Int, nmax: Int32, x: UnsafePointer<Float>, L: UnsafeMutablePointer<Int8>, qw: UnsafePointer<Float>?, siteBase: Int = 0
) -> Float {
    var max: Float = 0
    var amax: Float = 0
    for i in 0 ..< n {
        let ax = abs(x[i])
        if ax > amax { amax = ax; max = x[i] }
    }
    if amax < groupMaxEps {
        for i in 0 ..< n { L[i] = 0 }
        return 0
    }
    var iscale = -Float(nmax) / max
    var sumlx: Float = 0
    var suml2: Float = 0
    for i in 0 ..< n {
        var l = nearestInt(iscale * x[i])
        l = clampInt(l, -nmax, nmax - 1)
        L[i] = Int8(l + nmax)
        let w = qw.map { $0[i] } ?? x[i] * x[i]
        let lf = Float(l)
        sumlx = fm(siteBase + 0, sumlx, w * x[i], lf)
        suml2 = fm(siteBase + 1, suml2, w * lf, lf)
    }
    var scale: Float = suml2 != 0 ? sumlx / suml2 : 0
    var best = scale * sumlx
    for iss in -9 ... 9 {
        if iss == 0 { continue }
        iscale = -fm(siteBase + 2, (Float(nmax)), 0.1, Float(iss)) / max
        sumlx = 0
        suml2 = 0
        for i in 0 ..< n {
            var l = nearestInt(iscale * x[i])
            l = clampInt(l, -nmax, nmax - 1)
            let w = qw.map { $0[i] } ?? x[i] * x[i]
            let lf = Float(l)
            sumlx = fm(siteBase + 3, sumlx, w * x[i], lf)
            suml2 = fm(siteBase + 4, suml2, w * lf, lf)
        }
        if suml2 > 0 && sumlx * sumlx > best * suml2 {
            for i in 0 ..< n {
                var l = nearestInt(iscale * x[i])
                l = nmax + clampInt(l, -nmax, nmax - 1)
                L[i] = Int8(l)
            }
            scale = sumlx / suml2
            best = scale * sumlx
        }
    }
    return scale
}

/// `make_qkx2_quants` (`weights` required) or `make_qkx3_quants` (`weights` optional, the
/// importance-matrix path), selected by `v3`.
private func makeQkxQuants(
    v3: Bool, n: Int, nmax: Int32, x: UnsafePointer<Float>, weights: UnsafePointer<Float>?,
    L: UnsafeMutablePointer<UInt8>, theMin: inout Float, Laux: UnsafeMutablePointer<UInt8>,
    rmin: Float, rdelta: Float, nstep: Int, useMad: Bool, base: Int
) -> Float {
    @inline(__always) func weight(_ i: Int) -> Float { weights.map { $0[i] } ?? x[i] * x[i] }
    var mn = x[0]
    var mx = x[0]
    var sumW = weight(0)
    var sumX = sumW * x[0]
    for i in 1 ..< n {
        if x[i] < mn { mn = x[i] }
        if x[i] > mx { mx = x[i] }
        let w = weight(i)
        sumW += w
        sumX = fm(base + 5, sumX, w, x[i])
    }
    if mn > 0 { mn = 0 }
    if v3 ? (mx <= mn) : (mx == mn) {
        for i in 0 ..< n { L[i] = 0 }
        theMin = -mn
        return 0
    }
    var iscale = Float(nmax) / (mx - mn)
    var scale = 1 / iscale
    var bestError: Float = 0
    for i in 0 ..< n {
        let l = nearestInt(iscale * (x[i] - mn))
        L[i] = UInt8(clampInt(l, 0, nmax))
        var diff = fm(base + 6, mn, scale, Float(L[i])) - x[i]
        diff = useMad ? abs(diff) : diff * diff
        bestError = fm(base + 7, bestError, weight(i), diff)
    }
    if nstep < 1 {
        theMin = -mn
        return scale
    }
    for iss in 0 ... nstep {
        iscale = (fm(base + 8, rmin, rdelta, Float(iss)) + Float(nmax)) / (mx - mn)
        var sumL: Float = 0, sumL2: Float = 0, sumXL: Float = 0
        for i in 0 ..< n {
            let l: Int32
            if v3 {
                let v = iscale * (x[i] - mn)
                l = v > 0 ? nearestInt(min(v, Float(nmax))) : 0
            } else {
                l = clampInt(nearestInt(iscale * (x[i] - mn)), 0, nmax)
            }
            Laux[i] = UInt8(l)
            let w = weight(i)
            let lf = Float(l)
            sumL = fm(base + 9, sumL, w, lf)
            sumL2 = fm(base + 10, sumL2, w * lf, lf)
            sumXL = fm(base + 11, sumXL, w * lf, x[i])
        }
        let d = fmaForSub(sumW, sumL2, sumL, sumL, base)
        if d > 0 {
            var thisScale = fmaForSub(sumW, sumXL, sumX, sumL, base) / d
            var thisMin = fmaForSub(sumL2, sumX, sumL, sumXL, base) / d
            if thisMin > 0 {
                thisMin = 0
                thisScale = sumXL / sumL2
            }
            var err: Float = 0
            for i in 0 ..< n {
                var diff = fm(base + 12, thisMin, thisScale, Float(Laux[i])) - x[i]
                diff = useMad ? abs(diff) : diff * diff
                err = fm(base + 13, err, weight(i), diff)
            }
            if err < bestError {
                for i in 0 ..< n { L[i] = Laux[i] }
                bestError = err
                scale = thisScale
                mn = thisMin
            }
        }
    }
    theMin = -mn
    return scale
}

/// `a * b - c * d` as clang contracts it: `fma(a, b, -(c * d))`.
@inline(__always) private func fmaForSub(_ a: Float, _ b: Float, _ c: Float, _ d: Float, _ base: Int) -> Float {
    fm(base + 14, (-(c * d)), a, b)
}

/// `make_qp_quants`.
private func makeQpQuants(
    n: Int, nmax: Int32, x: UnsafePointer<Float>, L: UnsafeMutablePointer<UInt8>, qw: UnsafePointer<Float>, base: Int
) -> Float {
    var mx: Float = 0
    for i in 0 ..< n { mx = max(mx, x[i]) }
    if mx < groupMaxEps {
        for i in 0 ..< n { L[i] = 0 }
        return 0
    }
    var iscale = Float(nmax) / mx
    for i in 0 ..< n { L[i] = UInt8(truncatingIfNeeded: nearestInt(iscale * x[i])) }
    var scale = 1 / iscale
    var bestMse: Float = 0
    for i in 0 ..< n {
        let diff = fm(base + 15, x[i], -scale, Float(L[i]))
        bestMse = fm(base + 16, bestMse, qw[i] * diff, diff)
    }
    for iss in -4 ... 4 {
        if iss == 0 { continue }
        let iscaleIs = fm(base + 17, Float(nmax), 0.1, Float(iss)) / mx
        let scaleIs = 1 / iscaleIs
        var mse: Float = 0
        for i in 0 ..< n {
            var l = nearestInt(iscaleIs * x[i])
            l = min(nmax, l)
            let diff = fm(base + 18, x[i], -scaleIs, Float(l))
            mse = fm(base + 19, mse, qw[i] * diff, diff)
        }
        if mse < bestMse {
            bestMse = mse
            iscale = iscaleIs
        }
    }
    var sumlx: Float = 0
    var suml2: Float = 0
    for i in 0 ..< n {
        var l = nearestInt(iscale * x[i])
        l = min(nmax, l)
        L[i] = UInt8(truncatingIfNeeded: l)
        let w = qw[i]
        sumlx = fm(base + 20, sumlx, w * x[i], Float(l))
        suml2 = fm(base + 21, suml2, w * Float(l), Float(l))
    }
    for _ in 0 ..< 5 {
        var nChanged = 0
        for i in 0 ..< n {
            let w = qw[i]
            var slx = fm(base + 22, sumlx, -(w * x[i]), Float(L[i]))
            var sl2 = fm(base + 23, suml2, -(w * Float(L[i])), Float(L[i]))
            if slx > 0 && sl2 > 0 {
                var newL = nearestInt(x[i] * sl2 / slx)
                newL = min(nmax, newL)
                if newL != Int32(L[i]) {
                    slx = fm(base + 24, slx, w * x[i], Float(newL))
                    sl2 = fm(base + 25, sl2, w * Float(newL), Float(newL))
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        L[i] = UInt8(truncatingIfNeeded: newL)
                        sumlx = slx
                        suml2 = sl2
                        nChanged += 1
                    }
                }
            }
        }
        if nChanged == 0 { break }
    }
    return suml2 > 0 ? sumlx / suml2 : 0
}

private func putHalf(_ v: Float, _ p: UnsafeMutablePointer<UInt8>) {
    let h = GGUFFloat.float16Bits(v)
    p[0] = UInt8(h & 0xFF)
    p[1] = UInt8(h >> 8)
}

private func getHalf(_ p: UnsafePointer<UInt8>) -> Float {
    GGUFFloat.float(fromFloat16Bits: UInt16(p[0]) | (UInt16(p[1]) << 8))
}

// MARK: Q4_K and Q5_K

/// The shared implementation of `Q4_K` and `Q5_K`: super-blocks of 256 as 8 sub-blocks of 32, each
/// with a 6-bit scale and minimum (`d`, `dmin` in half precision), 4- or 5-bit codes.
///
/// Which multiply-adds the compiler fused differs between the two (each is its own inlined copy),
/// so each has its own set of flags (`base` 0 and 100).
private enum QKQuant {
    @inline(__always) static func scaleMin(_ j: Int, _ q: UnsafePointer<UInt8>) -> (UInt8, UInt8) {
        if j < 4 { return (q[j] & 63, q[j + 4] & 63) }
        return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }

    static func blockBytes(five: Bool) -> Int { five ? 176 : 144 }

    static func quantize(
        _ values: UnsafeBufferPointer<Float>, perRow: Int, importance: UnsafeBufferPointer<Float>?, five: Bool
    ) -> Data {
        precondition(perRow % 256 == 0 && values.count % perRow == 0)
        let bytes = blockBytes(five: five)
        let nb = values.count / 256
        var out = Data(count: nb * bytes)
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: UInt8.self).baseAddress!
            let blocksPerRow = perRow / 256
            for b in 0 ..< nb {
                let x = values.baseAddress! + b * 256
                let y = dst + b * bytes
                if let importance {
                    weighted(x, y, importance.baseAddress! + (b % blocksPerRow) * 256, five: five)
                } else {
                    reference(x, y, five: five)
                }
            }
        }
        return out
    }

    /// Codes from the final scales, then the packing.
    static func finish(
        _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<UInt8>, _ L: UnsafeMutablePointer<UInt8>, five: Bool
    ) {
        let scales = y + 4
        let d16 = getHalf(y), dmin16 = getHalf(y + 2)
        let top: Int32 = five ? 31 : 15
        for j in 0 ..< 8 {
            let (sc, m) = scaleMin(j, UnsafePointer(scales))
            let d = d16 * Float(sc)
            if d == 0 { continue }
            let dm = dmin16 * Float(m)
            for ii in 0 ..< 32 {
                let l = nearestInt((x[32 * j + ii] + dm) / d)
                L[32 * j + ii] = UInt8(clampInt(l, 0, top))
            }
        }
        if five {
            let qh = y + 16, ql = y + 48
            for k in 0 ..< 32 { qh[k] = 0 }
            var m1: UInt8 = 1, m2: UInt8 = 2
            var n = 0, o = 0
            while n < 256 {
                for j in 0 ..< 32 {
                    var l1 = L[n + j]
                    if l1 > 15 { l1 -= 16; qh[j] |= m1 }
                    var l2 = L[n + j + 32]
                    if l2 > 15 { l2 -= 16; qh[j] |= m2 }
                    ql[o + j] = l1 | (l2 << 4)
                }
                m1 <<= 2
                m2 <<= 2
                o += 32
                n += 64
            }
        } else {
            let q = y + 16
            var j = 0, o = 0
            while j < 256 {
                for l in 0 ..< 32 { q[o + l] = L[j + l] | (L[j + l + 32] << 4) }
                o += 32
                j += 64
            }
        }
    }

    static func packScales(_ y: UnsafeMutablePointer<UInt8>, _ ls: [UInt8], _ lm: [UInt8]) {
        let s = y + 4
        for k in 0 ..< 12 { s[k] = 0 }
        for j in 0 ..< 8 {
            if j < 4 {
                s[j] = ls[j]
                s[j + 4] = lm[j]
            } else {
                s[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4)
                s[j - 4] |= (ls[j] >> 4) << 6
                s[j] |= (lm[j] >> 4) << 6
            }
        }
    }

    static func reference(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<UInt8>, five: Bool) {
        let base = five ? 100 : 0
        var L = [UInt8](repeating: 0, count: 256)
        var Laux = [UInt8](repeating: 0, count: 32)
        var weights = [Float](repeating: 0, count: 32)
        var mins = [Float](repeating: 0, count: 8)
        var scales = [Float](repeating: 0, count: 8)
        var maxScale: Float = 0, maxMin: Float = 0
        for j in 0 ..< 8 {
            var sumX2: Float = 0
            for l in 0 ..< 32 { sumX2 = fm(base + 26, sumX2, x[32 * j + l], x[32 * j + l]) }
            let avX = (sumX2 / 32).squareRoot()
            for l in 0 ..< 32 { weights[l] = avX + abs(x[32 * j + l]) }
            var mn: Float = 0
            scales[j] = L.withUnsafeMutableBufferPointer { lp in
                weights.withUnsafeBufferPointer { wp in
                    Laux.withUnsafeMutableBufferPointer { ap in
                        makeQkxQuants(
                            v3: false, n: 32, nmax: five ? 31 : 15, x: x + 32 * j, weights: wp.baseAddress,
                            L: lp.baseAddress! + 32 * j, theMin: &mn, Laux: ap.baseAddress!,
                            rmin: five ? -0.5 : -1, rdelta: 0.1, nstep: five ? 15 : 20, useMad: false, base: base)
                    }
                }
            }
            mins[j] = mn
            if scales[j] > maxScale { maxScale = scales[j] }
            if mins[j] > maxMin { maxMin = mins[j] }
        }
        let invScale: Float = maxScale > 0 ? 63 / maxScale : 0
        let invMin: Float = maxMin > 0 ? 63 / maxMin : 0
        var ls = [UInt8](repeating: 0, count: 8), lm = ls
        for j in 0 ..< 8 {
            ls[j] = UInt8(min(63, UInt8(truncatingIfNeeded: nearestInt(invScale * scales[j]))))
            lm[j] = UInt8(min(63, UInt8(truncatingIfNeeded: nearestInt(invMin * mins[j]))))
        }
        packScales(y, ls, lm)
        putHalf(maxScale / 63, y)
        putHalf(maxMin / 63, y + 2)
        L.withUnsafeMutableBufferPointer { finish(x, y, $0.baseAddress!, five: five) }
    }

    static func weighted(
        _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<UInt8>, _ qw: UnsafePointer<Float>, five: Bool
    ) {
        let base = five ? 100 : 0
        var L = [UInt8](repeating: 0, count: 256)
        var Laux = [UInt8](repeating: 0, count: 32)
        var Ls = [UInt8](repeating: 0, count: 8), Lm = Ls
        var weights = [Float](repeating: 0, count: 32)
        var sw = [Float](repeating: 0, count: 8)
        var mins = sw, scales = sw
        var sumX2: Float = 0
        for l in 0 ..< 256 { sumX2 = fm(base + 27, sumX2, x[l], x[l]) }
        let sigma2 = 2 * sumX2 / 256
        for j in 0 ..< 8 {
            for l in 0 ..< 32 {
                weights[l] = qw[32 * j + l] * fm(base + 28, sigma2, x[32 * j + l], x[32 * j + l]).squareRoot()
            }
            var sumw: Float = 0
            for l in 0 ..< 32 { sumw += weights[l] }
            sw[j] = sumw
            var mn: Float = 0
            scales[j] = L.withUnsafeMutableBufferPointer { lp in
                weights.withUnsafeBufferPointer { wp in
                    Laux.withUnsafeMutableBufferPointer { ap in
                        makeQkxQuants(
                            v3: true, n: 32, nmax: five ? 31 : 15, x: x + 32 * j, weights: wp.baseAddress,
                            L: lp.baseAddress! + 32 * j, theMin: &mn, Laux: ap.baseAddress!,
                            rmin: -0.9, rdelta: 0.05, nstep: 36, useMad: false, base: base)
                    }
                }
            }
            mins[j] = mn
        }
        let dBlock = scales.withUnsafeBufferPointer { s in
            sw.withUnsafeBufferPointer { w in Ls.withUnsafeMutableBufferPointer { l in
                makeQpQuants(n: 8, nmax: 63, x: s.baseAddress!, L: l.baseAddress!, qw: w.baseAddress!, base: base)
            } }
        }
        let mBlock = mins.withUnsafeBufferPointer { s in
            sw.withUnsafeBufferPointer { w in Lm.withUnsafeMutableBufferPointer { l in
                makeQpQuants(n: 8, nmax: 63, x: s.baseAddress!, L: l.baseAddress!, qw: w.baseAddress!, base: base)
            } }
        }
        packScales(y, Ls.map { min(63, $0) }, Lm.map { min(63, $0) })
        putHalf(dBlock, y)
        putHalf(mBlock, y + 2)
        L.withUnsafeMutableBufferPointer { finish(x, y, $0.baseAddress!, five: five) }
    }

    static func dequantize(_ data: Data, five: Bool) -> [Float] {
        let bytes = blockBytes(five: five)
        precondition(data.count % bytes == 0)
        let nb = data.count / bytes
        var out = [Float](repeating: 0, count: nb * 256)
        data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt8.self).baseAddress!
            for i in 0 ..< nb {
                let x = src + i * bytes
                let d = getHalf(x), mn = getHalf(x + 2)
                let scales = UnsafePointer(x + 4)
                var q = x + (five ? 48 : 16)
                let qh = x + 16
                var isc = 0
                var o = i * 256
                var u1: UInt8 = 1, u2: UInt8 = 2
                var j = 0
                while j < 256 {
                    let (s1, m1) = scaleMin(isc, scales)
                    let (s2, m2) = scaleMin(isc + 1, scales)
                    let d1 = d * Float(s1), mm1 = mn * Float(m1)
                    let d2 = d * Float(s2), mm2 = mn * Float(m2)
                    for l in 0 ..< 32 {
                        let c1 = Int32(q[l] & 0xF) + (five && (qh[l] & u1) != 0 ? 16 : 0)
                        let c2 = Int32(q[l] >> 4) + (five && (qh[l] & u2) != 0 ? 16 : 0)
                        out[o + l] = (-mm1).addingProduct(d1, Float(c1))
                        out[o + 32 + l] = (-mm2).addingProduct(d2, Float(c2))
                    }
                    q += 32
                    isc += 2
                    o += 64
                    u1 <<= 2
                    u2 <<= 2
                    j += 64
                }
            }
        }
        return out
    }
}

/// `Q4_K`, 144 bytes per 256 values. With `importance` (one weight per column, the importance
/// matrix slice for this tensor) the search weights each column, as `llama-quantize --imatrix`
/// does; without it the reference row quantizer runs.
public enum GGUFQ4K {
    public static let blockSize = 256
    public static let blockBytes = 144
    public static func quantize(
        _ values: UnsafeBufferPointer<Float>, perRow: Int, importance: UnsafeBufferPointer<Float>? = nil
    ) -> Data { QKQuant.quantize(values, perRow: perRow, importance: importance, five: false) }
    public static func dequantize(_ data: Data) -> [Float] { QKQuant.dequantize(data, five: false) }
}

/// `Q5_K`, 176 bytes per 256 values.
public enum GGUFQ5K {
    public static let blockSize = 256
    public static let blockBytes = 176
    public static func quantize(
        _ values: UnsafeBufferPointer<Float>, perRow: Int, importance: UnsafeBufferPointer<Float>? = nil
    ) -> Data { QKQuant.quantize(values, perRow: perRow, importance: importance, five: true) }
    public static func dequantize(_ data: Data) -> [Float] { QKQuant.dequantize(data, five: true) }
}

// MARK: Q6_K

/// `Q6_K`: 16 sub-blocks of 16 with int8 scales, 6-bit codes, one half-precision `d`. 210 bytes.
public enum GGUFQ6K {
    public static let blockSize = 256
    public static let blockBytes = 210

    public static func quantize(
        _ values: UnsafeBufferPointer<Float>, perRow: Int, importance: UnsafeBufferPointer<Float>? = nil
    ) -> Data {
        precondition(perRow % blockSize == 0 && values.count % perRow == 0)
        let nb = values.count / blockSize
        var out = Data(count: nb * blockBytes)
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: UInt8.self).baseAddress!
            let blocksPerRow = perRow / blockSize
            for b in 0 ..< nb {
                quantizeBlock(
                    values.baseAddress! + b * blockSize, dst + b * blockBytes,
                    importance.map { $0.baseAddress! + (b % blocksPerRow) * blockSize })
            }
        }
        return out
    }

    private static func quantizeBlock(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<UInt8>, _ qw: UnsafePointer<Float>?) {
        var L = [Int8](repeating: 0, count: 256)
        var scales = [Float](repeating: 0, count: 16)
        var maxScale: Float = 0, maxAbs: Float = 0
        for ib in 0 ..< 16 {
            let s = L.withUnsafeMutableBufferPointer {
                makeQxQuants(n: 16, nmax: 32, x: x + 16 * ib, L: $0.baseAddress! + 16 * ib, qw: qw.map { $0 + 16 * ib })
            }
            scales[ib] = s
            if abs(s) > maxAbs { maxAbs = abs(s); maxScale = s }
        }
        // ql [0,128), qh [128,192), scales int8 [192,208), d [208,210)
        for k in 0 ..< 210 { y[k] = 0 }
        if maxAbs < groupMaxEps {
            putHalf(0, y + 208)
            return
        }
        let iscale = -128 / maxScale
        putHalf(1 / iscale, y + 208)
        var sc = [Int8](repeating: 0, count: 16)
        for ib in 0 ..< 16 {
            sc[ib] = Int8(min(127, nearestInt(iscale * scales[ib])))
            y[192 + ib] = UInt8(bitPattern: sc[ib])
        }
        let dh = getHalf(y + 208)
        for j in 0 ..< 16 {
            let d = dh * Float(sc[j])
            if d == 0 { continue }
            for ii in 0 ..< 16 {
                let l = clampInt(nearestInt(x[16 * j + ii] / d), -32, 31)
                L[16 * j + ii] = Int8(l + 32)
            }
        }
        var ql = y, qh = y + 128
        var j = 0
        while j < 256 {
            for l in 0 ..< 32 {
                let q1 = UInt8(L[j + l] & 0xF), q2 = UInt8(L[j + l + 32] & 0xF)
                let q3 = UInt8(L[j + l + 64] & 0xF), q4 = UInt8(L[j + l + 96] & 0xF)
                ql[l] = q1 | (q3 << 4)
                ql[l + 32] = q2 | (q4 << 4)
                qh[l] = UInt8(L[j + l] >> 4) | UInt8(L[j + l + 32] >> 4) << 2 | UInt8(L[j + l + 64] >> 4) << 4 | UInt8(L[j + l + 96] >> 4) << 6
            }
            ql += 64
            qh += 32
            j += 128
        }
    }

    public static func dequantize(_ data: Data) -> [Float] {
        precondition(data.count % blockBytes == 0)
        let nb = data.count / blockBytes
        var out = [Float](repeating: 0, count: nb * blockSize)
        data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt8.self).baseAddress!
            for i in 0 ..< nb {
                let b = src + i * blockBytes
                let d = getHalf(b + 208)
                var ql = b, qh = b + 128
                var sc = b + 192
                var o = i * blockSize
                var n = 0
                while n < 256 {
                    for l in 0 ..< 32 {
                        let isx = l / 16
                        let q1 = Int8(bitPattern: UInt8((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32
                        let q2 = Int8(bitPattern: UInt8((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32
                        let q3 = Int8(bitPattern: UInt8((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32
                        let q4 = Int8(bitPattern: UInt8((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32
                        func s(_ k: Int) -> Float { Float(Int8(bitPattern: sc[isx + k])) }
                        out[o + l] = d * s(0) * Float(q1)
                        out[o + l + 32] = d * s(2) * Float(q2)
                        out[o + l + 64] = d * s(4) * Float(q3)
                        out[o + l + 96] = d * s(6) * Float(q4)
                    }
                    o += 128
                    ql += 64
                    qh += 32
                    sc += 8
                    n += 128
                }
            }
        }
        return out
    }
}

// MARK: Q5_0

/// `Q5_0`: blocks of 32, a half-precision scale, 5-bit codes. 22 bytes.
public enum GGUFQ5_0 {
    public static let blockSize = 32
    public static let blockBytes = 22

    public static func quantize(
        _ values: UnsafeBufferPointer<Float>, perRow: Int, importance: UnsafeBufferPointer<Float>? = nil
    ) -> Data {
        precondition(perRow % blockSize == 0 && values.count % perRow == 0)
        let nb = values.count / blockSize
        var out = Data(count: nb * blockBytes)
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: UInt8.self).baseAddress!
            let perRowBlocks = perRow / blockSize
            var sigma2: Float = 0
            for b in 0 ..< nb {
                let x = values.baseAddress! + b * blockSize
                let y = dst + b * blockBytes
                var codes = [UInt8](repeating: 0, count: 32)
                if let importance {
                    if b % perRowBlocks == 0 {
                        var s: Float = 0
                        for j in 0 ..< perRow { s = fm(31, s, x[j], x[j]) }
                        sigma2 = s / Float(perRow)
                    }
                    var weight = [Float](repeating: 0, count: 32)
                    let qw = importance.baseAddress! + (b % perRowBlocks) * blockSize
                    for j in 0 ..< 32 { weight[j] = qw[j] * (fm(32, sigma2, x[j], x[j])).squareRoot() }
                    var L = [Int8](repeating: 0, count: 32)
                    let d = weight.withUnsafeBufferPointer { w in
                        L.withUnsafeMutableBufferPointer { makeQxQuants(n: 32, nmax: 16, x: x, L: $0.baseAddress!, qw: w.baseAddress!, siteBase: 40) }
                    }
                    putHalf(d, y)
                    for j in 0 ..< 32 { codes[j] = UInt8(bitPattern: L[j]) }
                } else {
                    var amax: Float = 0, mx: Float = 0
                    for j in 0 ..< 32 where amax < abs(x[j]) { amax = abs(x[j]); mx = x[j] }
                    let d = mx / -16
                    let id: Float = d != 0 ? 1 / d : 0
                    putHalf(d, y)
                    for j in 0 ..< 16 {
                        codes[j] = UInt8(min(31, Int(x[j] * id + 16.5)))
                        codes[j + 16] = UInt8(min(31, Int(x[j + 16] * id + 16.5)))
                    }
                }
                var qh: UInt32 = 0
                for j in 0 ..< 16 {
                    let xi0 = codes[j], xi1 = codes[j + 16]
                    y[6 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4)
                    qh |= UInt32((xi0 & 0x10) >> 4) << UInt32(j)
                    qh |= UInt32((xi1 & 0x10) >> 4) << UInt32(j + 16)
                }
                y[2] = UInt8(qh & 0xFF); y[3] = UInt8((qh >> 8) & 0xFF); y[4] = UInt8((qh >> 16) & 0xFF); y[5] = UInt8(qh >> 24)
            }
        }
        return out
    }

    public static func dequantize(_ data: Data) -> [Float] {
        precondition(data.count % blockBytes == 0)
        let nb = data.count / blockBytes
        var out = [Float](repeating: 0, count: nb * blockSize)
        data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt8.self).baseAddress!
            for i in 0 ..< nb {
                let b = src + i * blockBytes
                let d = getHalf(b)
                let qh = UInt32(b[2]) | UInt32(b[3]) << 8 | UInt32(b[4]) << 16 | UInt32(b[5]) << 24
                for j in 0 ..< 16 {
                    let xh0 = UInt8(truncatingIfNeeded: ((qh >> UInt32(j)) << 4) & 0x10)
                    let xh1 = UInt8(truncatingIfNeeded: (qh >> UInt32(j + 12)) & 0x10)
                    let x0 = Int32((b[6 + j] & 0x0F) | xh0) - 16
                    let x1 = Int32((b[6 + j] >> 4) | xh1) - 16
                    out[i * 32 + j] = Float(x0) * d
                    out[i * 32 + j + 16] = Float(x1) * d
                }
            }
        }
        return out
    }
}

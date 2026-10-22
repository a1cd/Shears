import Foundation

/// Scalar float conversions and the Q8_0 block format, in plain Swift.
///
/// These are independent of MLX so the reader (and the tests) can decode a file without
/// trusting the same array library that produced it.
public enum GGUFFloat {
    /// IEEE half bits from a float, rounding to nearest even.
    public static func float16Bits(_ f: Float) -> UInt16 {
        let x = f.bitPattern
        let sign = UInt16((x >> 16) & 0x8000)
        let exp = Int((x >> 23) & 0xFF)
        let mant = x & 0x7F_FFFF
        if exp == 0xFF {
            return sign | 0x7C00 | (mant != 0 ? 0x0200 : 0)
        }
        let e = exp - 127 + 15
        if e >= 0x1F { return sign | 0x7C00 }
        if e <= 0 {
            if e < -10 { return sign }
            let m = mant | 0x80_0000
            let shift = UInt32(14 - e)
            var half = UInt16(m >> shift)
            let rem = m & ((1 << shift) - 1)
            let halfway: UInt32 = 1 << (shift - 1)
            if rem > halfway || (rem == halfway && (half & 1) == 1) { half += 1 }
            return sign | half
        }
        var half = UInt16(e << 10) | UInt16(mant >> 13)
        let rem = mant & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) { half += 1 }
        return sign | half
    }

    public static func float(fromFloat16Bits h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32((h >> 10) & 0x1F)
        let mant = UInt32(h & 0x3FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }
            let v = Float(mant) * 0x1p-24
            return sign != 0 ? -v : v
        }
        if exp == 0x1F { return Float(bitPattern: sign | 0x7F80_0000 | (mant << 13)) }
        return Float(bitPattern: sign | ((exp + 112) << 23) | (mant << 13))
    }

    public static func float(fromBFloat16Bits b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    /// bfloat16 bits from a float, rounding to nearest even.
    public static func bfloat16Bits(_ f: Float) -> UInt16 {
        if f.isNaN { return 0x7FC0 }
        let x = f.bitPattern
        return UInt16((x &+ 0x7FFF &+ ((x >> 16) & 1)) >> 16)
    }
}

/// The `Q8_0` format: blocks of 32 values stored as one half-precision scale followed by 32
/// signed bytes. Matches `quantize_row_q8_0_ref` in ggml: `d = max|x| / 127`, `q = round(x / d)`
/// with halves rounded away from zero, and the scale rounded to half precision only when stored.
public enum GGUFQ8_0 {
    public static let blockSize = 32
    public static let blockBytes = 34

    /// Quantizes `values` (count must be a multiple of 32) to a Q8_0 byte stream.
    public static func quantize(_ values: UnsafeBufferPointer<Float>) -> Data {
        precondition(values.count % blockSize == 0, "Q8_0 needs a multiple of 32 values")
        let blocks = values.count / blockSize
        var out = Data(count: blocks * blockBytes)
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: UInt8.self)
            for b in 0 ..< blocks {
                let src = values.baseAddress! + b * blockSize
                var amax: Float = 0
                for j in 0 ..< blockSize {
                    let a = abs(src[j])
                    if a > amax { amax = a }
                }
                let d = amax / 127
                let id: Float = d == 0 ? 0 : 1 / d
                let h = GGUFFloat.float16Bits(d)
                let o = b * blockBytes
                dst[o] = UInt8(h & 0xFF)
                dst[o + 1] = UInt8(h >> 8)
                for j in 0 ..< blockSize {
                    let r = (src[j] * id).rounded(.toNearestOrAwayFromZero)
                    let q: Int8 = r.isFinite ? Int8(max(-127, min(127, r))) : 0
                    dst[o + 2 + j] = UInt8(bitPattern: q)
                }
            }
        }
        return out
    }

    /// Decodes a Q8_0 byte stream to floats.
    public static func dequantize(_ data: Data) -> [Float] {
        precondition(data.count % blockBytes == 0)
        let blocks = data.count / blockBytes
        var out = [Float](repeating: 0, count: blocks * blockSize)
        data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt8.self)
            for b in 0 ..< blocks {
                let o = b * blockBytes
                let d = GGUFFloat.float(fromFloat16Bits: UInt16(src[o]) | (UInt16(src[o + 1]) << 8))
                for j in 0 ..< blockSize {
                    out[b * blockSize + j] = d * Float(Int8(bitPattern: src[o + 2 + j]))
                }
            }
        }
        return out
    }
}

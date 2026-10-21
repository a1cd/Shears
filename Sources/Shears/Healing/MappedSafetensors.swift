#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Foundation
import MLX

/// Read-only, zero-copy access to the tensors of a safetensors file.
///
/// Each tensor is mapped on its own (`mmap` with `PROT_READ`, `MAP_SHARED`) and handed to MLX as a
/// buffer that wraps the mapped pages: no bytes are copied, nothing is ever written back, and
/// the operating system may drop the pages whenever it needs the memory, since they are clean and
/// file-backed. The mapping is released when the array is. MLX counts the wrapped bytes in
/// `Memory.activeMemory` (it cannot tell they are file-backed), while ``ProcessMemory`` reports them
/// as file-backed and outside the process's `phys_footprint`.
///
/// `MAP_SHARED` is deliberate: a `MAP_PRIVATE` mapping of the same file is charged to the
/// footprint as if it were anonymous memory.
public enum MappedSafetensors {
    public struct Entry: Sendable {
        public var dtype: DType
        public var shape: [Int]
        /// Absolute byte range in the file.
        public var begin: Int
        public var end: Int
    }

    /// The tensors a file holds, from its header alone.
    public static func index(of url: URL) throws -> [String: Entry] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let lengthBytes = try handle.read(upToCount: 8) ?? Data()
        guard lengthBytes.count == 8 else { throw HealingError.invalid("\(url.lastPathComponent) is not a safetensors file") }
        let n = lengthBytes.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
        guard n < 1 << 28 else { throw HealingError.invalid("\(url.lastPathComponent) has an implausible header") }
        let json = try handle.read(upToCount: n) ?? Data()
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw HealingError.invalid("\(url.lastPathComponent) has an unreadable header")
        }
        var out: [String: Entry] = [:]
        for (name, v) in root where name != "__metadata__" {
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let dtype = dtype(named: dt),
                let shape = d["shape"] as? [Int], let offs = d["data_offsets"] as? [Int], offs.count == 2
            else { throw HealingError.invalid("\(url.lastPathComponent): tensor \(name) has an unreadable entry") }
            out[name] = Entry(dtype: dtype, shape: shape, begin: 8 + n + offs[0], end: 8 + n + offs[1])
        }
        return out
    }

    static func dtype(named s: String) -> DType? {
        switch s {
        case "F32": .float32
        case "F16": .float16
        case "BF16": .bfloat16
        case "F64": .float64
        case "I8": .int8
        case "I16": .int16
        case "I32": .int32
        case "I64": .int64
        case "U8": .uint8
        case "U16": .uint16
        case "U32": .uint32
        case "U64": .uint64
        case "BOOL": .bool
        default: nil
        }
    }

    /// `MAP_FAILED`, which Glibc does not import into Swift.
    private static let mapFailed = UnsafeMutableRawPointer(bitPattern: -1)

    /// Whether this system hands mapped pages to MLX without copying them. Checked once, on one
    /// page of anonymous memory: the wrapped array sees a later write to the page only if it
    /// shares it.
    public static let zeroCopySupported: Bool = {
        let page = Int(getpagesize())
        guard let p = mmap(nil, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), p != mapFailed else { return false }
        p.storeBytes(of: Float(1), as: Float.self)
        final class Released { var value = false }
        let released = Released()
        let a = MLXArray(rawPointer: p, [page / 4], dtype: .float32) {
            released.value = true
            munmap(p, page)
        }
        eval(a)
        // An allocator that cannot wrap the pointer (every one but Metal's) copies the bytes and lets
        // go of the page at once, so it is already unmapped here.
        if released.value { return false }
        p.storeBytes(of: Float(2), as: Float.self)
        return a[0].item(Float.self) == 2
    }()

    /// The tensor as an array over the file's own pages, or a copy read into memory when the
    /// system does not share the pages or the tensor is not aligned to its element size.
    public static func tensor(_ e: Entry, fd: Int32, in url: URL) throws -> MLXArray {
        let nbytes = e.end - e.begin
        if nbytes == 0 { return MLXArray.zeros(e.shape, dtype: e.dtype) }
        let page = Int(getpagesize())
        if zeroCopySupported, e.begin % e.dtype.size == 0 {
            let start = e.begin & ~(page - 1)
            let length = e.end - start
            if let p = mmap(nil, length, PROT_READ, MAP_SHARED, fd, off_t(start)), p != mapFailed {
                return MLXArray(rawPointer: p.advanced(by: e.begin - start), e.shape, dtype: e.dtype) {
                    munmap(p, length)
                }
            }
        }
        var data = Data(count: nbytes)
        let got = data.withUnsafeMutableBytes { pread(fd, $0.baseAddress, nbytes, off_t(e.begin)) }
        guard got == nbytes else { throw HealingError.invalid("short read of a tensor in \(url.lastPathComponent)") }
        return MLXArray(data, e.shape, dtype: e.dtype)
    }

    /// The tensor read into ordinary memory (no mapping).
    public static func read(_ e: Entry, fd: Int32, in url: URL) throws -> MLXArray {
        let nbytes = e.end - e.begin
        if nbytes == 0 { return MLXArray.zeros(e.shape, dtype: e.dtype) }
        var data = Data(count: nbytes)
        let got = data.withUnsafeMutableBytes { pread(fd, $0.baseAddress, nbytes, off_t(e.begin)) }
        guard got == nbytes else { throw HealingError.invalid("short read of a tensor in \(url.lastPathComponent)") }
        return MLXArray(data, e.shape, dtype: e.dtype)
    }

    /// Rows `rows` (ascending or not) of the tensor's first axis, read straight from the file into
    /// one `[rows.count, ...]` array: a tensor `[E, a, b]` stores expert `e`'s rows contiguously
    /// from `begin + e * rowBytes`, so only the requested rows are ever read or held. Runs of
    /// consecutive rows are one read.
    public static func gatherRows(_ e: Entry, rows: [Int], fd: Int32, in url: URL) throws -> MLXArray {
        guard let n = e.shape.first, n > 0, (e.end - e.begin) % n == 0 else {
            throw HealingError.invalid("\(url.lastPathComponent): a tensor of shape \(e.shape) has no whole rows")
        }
        let rowBytes = (e.end - e.begin) / n
        for r in rows where r < 0 || r >= n { throw HealingError.invalid("row \(r) out of range 0..<\(n)") }
        var data = Data(count: rows.count * rowBytes)
        var failed = false
        data.withUnsafeMutableBytes { buf in
            var i = 0
            while i < rows.count {
                var j = i + 1
                while j < rows.count, rows[j] == rows[j - 1] + 1 { j += 1 }
                let want = (j - i) * rowBytes
                let got = pread(fd, buf.baseAddress! + i * rowBytes, want, off_t(e.begin + rows[i] * rowBytes))
                if got != want { failed = true }
                i = j
            }
        }
        guard !failed else { throw HealingError.invalid("short read of rows of a tensor in \(url.lastPathComponent)") }
        return MLXArray(data, [rows.count] + Array(e.shape.dropFirst()), dtype: e.dtype)
    }

    /// Asks the kernel to start reading the byte ranges into the page cache. Returns at once.
    public static func prefetch(_ entries: [Entry], fd: Int32) {
        let chunk = 1 << 30
        for e in entries where e.end > e.begin {
            var at = e.begin
            while at < e.end {
                ReadAhead.advise(fd: fd, offset: at, count: min(e.end - at, chunk))
                at += chunk
            }
        }
    }
}

#if canImport(Darwin)
import Darwin
#endif
import Foundation

/// The operating system's view of this process's memory, as opposed to MLX's allocator counters.
///
/// `footprint` is `phys_footprint` from `task_vm_info`: the number Activity Monitor's "Memory"
/// column and `footprint(1)` report. It counts anonymous (dirty) memory, compressed memory and
/// wired pages, and leaves out clean file-backed pages the system can drop for free.
/// `Memory.peakMemory` only counts bytes MLX's allocator handed out, so the two differ: the
/// allocator number includes memory-mapped weights (as buffers) and excludes everything else the
/// process holds (the Swift heap, safetensors readers, the mapped files' page cache).
///
/// On Linux the same three numbers come from `/proc/self/status`: `footprint` is `RssAnon + VmSwap`
/// (anonymous memory, resident or swapped), `fileBacked` is `RssFile + RssShmem`, `anonymous` is
/// `RssAnon`.
public enum ProcessMemory {
    public struct Snapshot: Sendable, Equatable {
        /// `phys_footprint` in bytes.
        public var footprint: Int
        /// Resident file-backed (clean, droppable) pages in bytes (`task_vm_info.external`).
        public var fileBacked: Int
        /// Resident anonymous pages in bytes (`task_vm_info.internal`); the part that can only be
        /// compressed or swapped, never just dropped.
        public var anonymous: Int
    }

    /// The process's resident set (`resident_size`): every page of the process in physical memory, anonymous or
    /// file-backed, including clean mapped pages that `footprint` leaves out. Bytes; 0 if unavailable.
    public static func residentBytes() -> Int {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.resident_size) : 0
        #else
        guard let status = try? String(contentsOfFile: "/proc/self/status", encoding: .utf8),
            let line = status.split(separator: "\n").first(where: { $0.hasPrefix("VmRSS:") }),
            let kB = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).dropFirst().first.flatMap({ Int($0) })
        else { return 0 }
        return kB * 1024
        #endif
    }

    #if canImport(Darwin)
    public static func snapshot() -> Snapshot {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return Snapshot(footprint: 0, fileBacked: 0, anonymous: 0) }
        // `external` and `internal` are byte counts in task_vm_info.
        return Snapshot(
            footprint: Int(info.phys_footprint), fileBacked: Int(info.external), anonymous: Int(info.internal))
    }
    #else
    public static func snapshot() -> Snapshot {
        guard let status = try? String(contentsOfFile: "/proc/self/status", encoding: .utf8) else {
            return Snapshot(footprint: 0, fileBacked: 0, anonymous: 0)
        }
        // Lines look like "RssAnon:\t   12345 kB".
        var kB: [Substring: Int] = [:]
        for line in status.split(separator: "\n") {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let fields = line[line.index(after: colon)...].split(whereSeparator: { $0 == " " || $0 == "\t" })
            if let v = fields.first.flatMap({ Int($0) }) { kB[line[..<colon]] = v }
        }
        let anon = (kB["RssAnon"] ?? 0) * 1024
        return Snapshot(
            footprint: anon + (kB["VmSwap"] ?? 0) * 1024,
            fileBacked: ((kB["RssFile"] ?? 0) + (kB["RssShmem"] ?? 0)) * 1024,
            anonymous: anon)
    }
    #endif
}

import Darwin
import Foundation

/// The operating system's view of this process's memory, as opposed to MLX's allocator counters.
///
/// `footprint` is `phys_footprint` from `task_vm_info`: the number Activity Monitor's "Memory"
/// column and `footprint(1)` report. It counts anonymous (dirty) memory, compressed memory and
/// wired pages, and leaves out clean file-backed pages the system can drop for free.
/// `Memory.peakMemory` only counts bytes MLX's allocator handed out, so the two differ: the
/// allocator number includes memory-mapped weights (as buffers) and excludes everything else the
/// process holds (the Swift heap, safetensors readers, the mapped files' page cache).
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
}

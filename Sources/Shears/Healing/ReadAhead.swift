#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Asks the kernel to start reading a byte range of an open file into the page cache. Returns at
/// once; a failure is ignored, since this is only a hint. `F_RDADVISE` on Darwin,
/// `posix_fadvise(POSIX_FADV_WILLNEED)` elsewhere.
enum ReadAhead {
    static func advise(fd: Int32, offset: Int, count: Int) {
        #if canImport(Darwin)
        var r = radvisory(ra_offset: off_t(offset), ra_count: Int32(clamping: count))
        _ = fcntl(fd, F_RDADVISE, &r)
        #else
        _ = posix_fadvise(fd, off_t(offset), off_t(count), POSIX_FADV_WILLNEED)
        #endif
    }
}

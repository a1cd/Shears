import Foundation
import Shears

let status = ShearsRunner.main(arguments: CommandLine.arguments)

#if os(Linux)
// Leave without running C++ static destructors and atexit handlers. With the CUDA backend those run after the CUDA runtime
// may already be gone, and a destructor that throws then ends a run that had finished its work in std::terminate (an abort,
// exit status 134). Everything a command writes is written and closed by now. SHEARS_NORMAL_EXIT=1 exits the ordinary way,
// to check whether the exit itself crashes (the notebook's CUDA diagnostics do).
fflush(stdout)
fflush(stderr)
if ProcessInfo.processInfo.environment["SHEARS_NORMAL_EXIT"] == nil {
    _exit(status)
}
#endif
exit(status)

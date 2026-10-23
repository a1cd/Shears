#include "CShearsDiag.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <typeinfo>

#if defined(__has_include)
#if __has_include(<execinfo.h>)
#include <execinfo.h>
#define SHEARS_HAVE_BACKTRACE 1
#endif
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define SHEARS_HAVE_CXXABI 1
#endif
#endif

namespace {

volatile int entered = 0;

[[noreturn]] void shears_terminate() {
    if (entered++ > 0) {
        // The handler itself threw or crashed: say so in the simplest way possible and stop.
        static const char msg[] = "shears: terminate called again while reporting the first one\n";
        fwrite(msg, 1, sizeof(msg) - 1, stderr);
        std::abort();
    }
    fputs("shears: std::terminate called\n", stderr);
    if (std::exception_ptr p = std::current_exception()) {
        try {
            std::rethrow_exception(p);
        } catch (const std::exception &e) {
            const char *name = typeid(e).name();
#ifdef SHEARS_HAVE_CXXABI
            int status = 0;
            char *demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
            if (status == 0 && demangled) name = demangled;
#endif
            fprintf(stderr, "shears: uncaught C++ exception of type %s\nshears: what(): %s\n", name, e.what());
        } catch (...) {
            fputs("shears: uncaught C++ exception of a type that is not std::exception\n", stderr);
        }
    } else {
        fputs("shears: no C++ exception is active (terminate was called directly, for example by a noexcept function or a thread)\n", stderr);
    }
#ifdef SHEARS_HAVE_BACKTRACE
    void *frames[64];
    int n = backtrace(frames, 64);
    fputs("shears: backtrace (innermost first):\n", stderr);
    fflush(stderr);
    backtrace_symbols_fd(frames, n, 2);
#endif
    fflush(stderr);
    std::abort();
}

// Priority 101: before the other static initializers of this program (the default runs after them), because an exception
// thrown by one of those is exactly what this is for.
__attribute__((constructor(101))) void install_before_main() { std::set_terminate(shears_terminate); }

}  // namespace

extern "C" void shears_install_terminate_handler(void) {
    std::set_terminate(shears_terminate);
    // A test hook, so that CI can prove the handler reports an exception thrown where nothing can catch it (a thread).
    if (std::getenv("SHEARS_TEST_TERMINATE")) {
        std::thread t([] { throw std::runtime_error("test exception from SHEARS_TEST_TERMINATE"); });
        t.join();
    }
}

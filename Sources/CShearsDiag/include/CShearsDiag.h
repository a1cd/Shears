#ifndef CSHEARSDIAG_H
#define CSHEARSDIAG_H

#ifdef __cplusplus
extern "C" {
#endif

/// Installs a std::terminate handler that prints the active C++ exception's type and what() text and a backtrace to
/// standard error before aborting. Without it a C++ exception that nothing catches (thrown on a worker thread, in a
/// destructor at exit, or before main) ends in "terminate called after throwing an instance of ..." with no message,
/// or with "terminate called recursively". Safe to call more than once. It is also installed before main.
void shears_install_terminate_handler(void);

#ifdef __cplusplus
}
#endif

#endif

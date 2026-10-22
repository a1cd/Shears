#!/usr/bin/env python3
"""Read the log of `swift test`, write the counts to the job summary, and decide whether the job passes.

usage: summarize_swift_test.py <swift-test.log> <exit-status>

Passes when swift test exited 0, or when every failure is in KNOWN_FLAKES (numeric-tolerance
misses documented in docs/linux.md) and nothing else went wrong (no crash, no failure outside the list).
Any other failure, a crash, or a log with no test results fails the job.
"""
import os
import re
import sys

# Numeric-tolerance misses on CPU, documented in docs/linux.md ("Ran: test results" and "CI"). Float32 sums
# in a different order than on the Mac's GPU; which of these misses depends on the machine. A listed test is
# only allowed when every failure message it printed is a closeness check ("max |diff|"); any other message
# (a thrown error, a crash, a wrong shape) still fails the job.
KNOWN_FLAKES = {
    # adapters of the merged and routed layouts drift 7.4e-4 apart against atol 5e-4 / rtol 5e-3; losses agree
    "MergedExpertTests.testPipelineTrainingIsTheSameInMergedAndRoutedLayouts",
    # layer-0 adapter gradient differs by 7e-3 between shared-prefix and whole-row passes against atol 2e-3 / rtol 2e-3
    "SharedPrefixTests.testNetworkAdapterGradientsEqualWholeRows",
}
TOLERANCE_MESSAGE = "max |diff|"

log = open(sys.argv[1], errors="replace").read()
status = int(sys.argv[2])

# XCTest on Linux: "Test Case 'MergedExpertTests.testFoo' passed (0.5 seconds)." (or '-[Mod.Class method]' on Darwin)
cases = {}
for m in re.finditer(r"Test Case '(?:-\[)?([^']+?)\]?' (passed|failed|skipped) \(", log):
    name = m.group(1)
    name = re.sub(r"^\w+\.(\w+) (\w+)$", r"\1.\2", name)  # "-[Mod.Class method]" -> "Class.method"
    cases[name] = m.group(2)
passed = sum(1 for v in cases.values() if v == "passed")
skipped = sum(1 for v in cases.values() if v == "skipped")
failed = sorted(k for k, v in cases.items() if v == "failed")

# swift-testing, if any suite uses it.
st_fail = sorted(set(re.findall(r'^[^\n]*✘ Test "?([^"\n]+?)"? (?:with \d+ arguments? )?failed', log, re.M)))
executed = re.findall(r"Executed (\d+) tests?, with (?:(\d+) tests? skipped and )?(\d+) failures?", log)
total_line = executed[-1] if executed else None
crashed = bool(re.search(r"Fatal error|Illegal instruction|Segmentation fault|signal \d+|Bus error", log))

def only_tolerance_messages(name):
    msgs = [l for l in log.splitlines() if re.search(r"error: " + re.escape(name) + r" : ", l)]
    return bool(msgs) and all(TOLERANCE_MESSAGE in l for l in msgs)

unknown = [f for f in failed + st_fail if f not in KNOWN_FLAKES or not only_tolerance_messages(f)]
known = [f for f in failed + st_fail if f in KNOWN_FLAKES and only_tolerance_messages(f)]

if status == 0:
    verdict, ok = "passed", True
elif not cases and not st_fail:
    verdict, ok = "FAILED: no test results in the log (build error?)", False
elif crashed:
    verdict, ok = "FAILED: the test process crashed", False
elif unknown or not known:
    verdict, ok = f"FAILED: {', '.join(unknown) or 'swift test exited non-zero with no failing test named'}", False
else:
    verdict, ok = f"passed with {len(known)} known tolerance failure(s) allowed", True

lines = [
    "### swift test (Linux x86_64, CPU, debug)",
    "",
    f"**{verdict}**",
    "",
    "| | |",
    "|---|---|",
    f"| tests run | {len(cases)} |",
    f"| passed | {passed} |",
    f"| skipped | {skipped} |",
    f"| failed | {len(failed) + len(st_fail)} |",
]
if total_line:
    lines.append(f"| XCTest's own last line | Executed {total_line[0]} tests, {total_line[1] or 0} skipped, {total_line[2]} failures |")
if known:
    lines += ["", "Known failures, allowed (numeric tolerance, see docs/linux.md):"] + [f"- `{k}`" for k in known]
if unknown:
    lines += ["", "Unexpected failures:"] + [f"- `{k}`" for k in unknown]
text = "\n".join(lines) + "\n"
print(text)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(text)
sys.exit(0 if ok else 1)

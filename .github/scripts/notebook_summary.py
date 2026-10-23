#!/usr/bin/env python3
"""Print the notebook's phase table (phases.json, written by its last cell) to the job summary.
Fails unless the notebook set up through the prebuilt path, ran the tiny model, and generated text from its export."""
import json, os, sys

d = json.load(open(sys.argv[1]))
lines = [
    "### Notebook on a clean runner (no Swift)",
    "",
    f"model: **{d.get('model')}**, device: **{d.get('device')}**, setup path: **{d['mode']}**; setup took **{d['setup_seconds']:.1f} s**; all phases {d['total_seconds']:.1f} s",
    "",
    "| phase | seconds |",
    "|---|---|",
] + [f"| {n} | {s:.1f} |" for n, s in d["phases"]]
text = "\n".join(lines) + "\n"
print(text)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text)
if d["mode"] != "prebuilt":
    sys.exit("the notebook did not use the prebuilt path")
if d.get("model") != "tiny":
    sys.exit(f"CI must run the tiny model, not {d.get('model')!r}")
if not d.get("generated"):
    sys.exit("the tiny run did not generate anything with transformers")
print("generated:", repr(d["generated"]))

# iOS and visionOS

[Shears docs](README.md)

`xcodebuild build -scheme Shears -destination 'generic/platform=iOS'` (and `generic/platform=visionOS`)
succeed; nothing macOS-only was found (CryptoKit, Foundation and MLX are all available). Nothing has
been run on a device or simulator. Needs runtime testing: the **increased memory limit** and
**extended virtual addressing** entitlements (the per-process limit kills a training run long before
the Mac's), `MemoryConfiguration` limits chosen for the device, and **background kills** (an
interrupted run resumes through the checkpoint and pre-pass resume paths, but nothing has exercised
that on iOS). Keep stores and checkpoints under Application Support, marked excluded from backup,
not tmp or Caches (the system purges those); the boundary store is 91 MB per row at 26B scale, so
check free storage first. GGUF export's CPU K-quant search spreads over the cores and will heat a
phone. Metal's shader compile happens at first use.

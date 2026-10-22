# Notes and status

[Shears docs](README.md)

## Things worth knowing

- **Training must use the sorted path.** `GatherMM`'s weight gradient on unsorted indices builds
  an `[out, in]` matrix for every token and then scatter-adds them. The sorted path uses a
  segmented matmul instead. `SwitchGLU` therefore always sorts while `training` is true (the
  MLXNN default). Call `train(false)` for inference.
- **`Module.update(parameters:)` rewrites the existing `MLXArray` objects in place.** Holding a
  parameter array does not keep its old value. Snapshot by value (`w * 1`) before training if you
  want to compare with the start (healing's resume and equivalence tests do), or use
  `copyParameters(into:)` to clone a module. For the same reason the loader
  hands out a fresh `MLXArray` object for every tensor; otherwise a teacher and a student loaded
  from one checkpoint would share objects, and pruning one would prune the other.
- **`CustomFunction` VJPs must return exactly one gradient per primal.** MLX indexes the returned
  list by input position and then appends the cotangents, so a short list silently misattributes
  gradients. `vjp(_:primals:cotangents:)` does return one per primal.
- **`valueAndGrad(model:)` is generic over a concrete `Model: Module`.** A closure taking
  `any Block` doesn't type-check, so generic training code is generic over the concrete module type.
- Checked against 0.32.3: `dim(-1)`, `flattened(start:end:)`, `floorDivide`, `.==` and
  `[.ellipsis, ..<k]` all exist as written. `MLXArray(0..<n)` does not: use
  `MLXArray.arange(n, dtype:)`.

## Not done yet

- Healing is ported for Gemma 4 only (the layer decomposition and the Hugging Face tensor names);
  a Qwen-style MoE needs a `HealableMoELayer` conformance and its own export name map.
- Not ported from the recipe, because they are specific to one dataset or machine: its metric
  suites (answer-distribution and classification agreement, coverage, per-class monitors), data
  assembly and held-out splits, class-balanced sampling (`GroupBalancedRows` is the generic
  form), and the system watchdogs (replaced by `MemoryConfiguration` and the pause/stop hooks).

- No real Qwen3-MoE checkpoint has been loaded. The loader is tested against synthetic checkpoints in all
  three layouts. Its tensor names match the real indexes of `Qwen/Qwen3-30B-A3B` (HF per-expert)
  and `mlx-community/Qwen3-30B-A3B-4bit` (stacked, 4-bit, quantized router, bf16 scales). The
  bf16-scale case has its own test.
- Gemma 4 26B-A4B has been run for real on one machine (an M3 Max with 48 GB; see
  [Benchmarks](benchmarks.md#real-gemma-4-26b-a4b-m3-max-48-gb-2026-10)), but only part of it is measured: peak memory
  is measured for block 0 only, and the whole-model two-task run is preliminary (small samples, an unexplained loss on other
  languages). The memory figures for a whole 26B run in [Memory](memory.md#estimate-for-the-real-26b-a4b-derived-not-measured)
  are still estimates.
- Nothing has been run on an iOS or visionOS device or simulator (see [iOS and visionOS](ios-visionos.md)), and Linux and CUDA have
  not been tried.
- K-quant bit-exactness against `ggml` is tested only behind `TEST_RUNNER_SHEARS_KQUANT_ORACLE` / `TEST_RUNNER_SHEARS_LLAMA_BIN`; reference vectors for the default suite have not been checked in (they must come from the reference build, never from this package's own quantizer).
- The loader ignores `quantization.mode`, so mxfp4/nvfp4 conversions fail loudly rather than load.
- For Qwen3-MoE, attention isn't mapped, so healing covers the MoE MLP half of each decoder layer only.
- The fixed overhead in checkpointing (see [Checkpointing](benchmarks.md#checkpointing)).

# Shears on Linux, Colab and CUDA

Shears targets MLX on Apple Silicon. Linux is a secondary, CPU-only port that exists so the pipeline can be tried without a Mac, for example from the [Colab notebook](../Notebooks/Shears.ipynb). Speed, memory behaviour (no zero-copy mapping, no unified memory) and numerics differ from the Mac, so do not compare numbers across the two. The CUDA backend has never been run.

The notebook runs the commands below for you; this page is the reference for what was verified and what was not.

Tested 2026-10-21 on Ubuntu 24.04.5 LTS, x86_64, 4 cores, 15 GB RAM, **no GPU** (`nvidia-smi` and `nvcc` not installed). Everything under "Ran" was run; everything under "Read only" comes from reading source and was not executed.

## The Colab notebook

[`Notebooks/Shears.ipynb`](../Notebooks/Shears.ipynb) runs the commands on this page for you and then the tiny pipeline (`shears-cli sample`, `select`, `prepass`, `heal`, `export`), and loads the export with transformers. It also runs on an Apple Silicon Mac (where it uses `xcodebuild`). Facts about it:

- **Run headless on a Mac** (`jupyter nbconvert --execute`, M3 Max): about 90 s for a cold `xcodebuild` of `shears-cli`, about 15 s for the whole tiny pipeline, 2 minutes in all including the transformers cell.
- **Not run on Colab, and not run on x86_64 Linux in the notebook.** The Linux cells are the commands below, assembled; the pieces were run on an x86_64 Ubuntu 24.04 machine, the assembled notebook was not. Colab's image (Ubuntu 22.04 or 24.04, its Python and preinstalled packages) has not been tried; on 22.04 the notebook adds a PPA for `libstdc++-13`, which is untested.
- **Setup time on Colab is unmeasured.** From the 4-core x86_64 numbers below (library about 2 min, release CLI about 5.5 min cold) expect 15 to 30 minutes on Colab's 2 vCPUs, mostly the MLX C++ build.
- **The checkout directory must be called `Shears`.** SwiftPM names a path dependency after its directory, so `CLI/Package.swift`'s `.package(path: "..")` only resolves when the directory is `Shears`. The notebook copies or clones into `<work>/Shears` for that reason.
- **transformers.** The export loads with `AutoModelForCausalLM` (a text-only `Gemma4ForCausalLM`; Gemma 4 appears in transformers 5.5.0, the notebook pins 5.20.2, the version tested). In float32 its top-5 next-token log-probabilities match `shears-serve` on the same export to four decimals. See [Quantization and export](quantization-and-export.md#loading-the-export-with-transformers).

## Ran: toolchain (verified commands)

```bash
# Swift 6.3.3 via swiftly (about 45 s). Run from a scratch directory: `swiftly install --use`
# writes a .swift-version file into the current directory.
curl -sSLO https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz
tar zxf swiftly-$(uname -m).tar.gz
./swiftly init --quiet-shell-followup --assume-yes --skip-install
. ~/.local/share/swiftly/env.sh
swiftly install --use --assume-yes 6.3.3

apt-get install -y libcurl4-openssl-dev libxml2-dev libz3-dev \
  libstdc++-13-dev libopenblas-dev liblapacke-dev gfortran ninja-build
```

`gfortran` is required: mlx-swift's Linux manifest links `-lgfortran` and the link fails without it. `g++` must also be present at run time (MLX's CPU backend JIT-compiles fused kernels with it into `/tmp/mlx`).

## Ran: mlx-swift 0.32.3 needs two local fixes on Linux CPU

Both are bugs in mlx-swift's SwiftPM Linux path (still present on its `main` as of today), not in Shears.

1. **MLXNN does not compile.** With `SPM_CUDA=0` the manifest excludes `MLXFast.swift` and `mlx-c/.../fast.cpp`, but MLXNN uses `MLXFast` (RMSNorm, LayerNorm, RoPE, attention). Fix: [`Notebooks/mlx-swift-linux-cpu.patch`](../Notebooks/mlx-swift-linux-cpu.patch) (two deleted lines). With it, `MLXFast` links and runs on CPU.
2. **`compile` crashes on x86_64.** The shipped `mlx-generated/compiled_preamble.cpp` was generated on arm64 (`using ::float16_t;`), so every compiled function fails in g++ with "'float16_t' has not been declared". Fix: regenerate it on the host with MLX's own script.

```bash
git clone --depth 1 --branch 0.32.3 --recurse-submodules --shallow-submodules \
  https://github.com/ml-explore/mlx-swift /opt/mlx-swift
cd /opt/mlx-swift && git apply /path/to/Shears/Notebooks/mlx-swift-linux-cpu.patch
cd Source/Cmlx && bash mlx/mlx/backend/cpu/make_compiled_preamble.sh \
  mlx-generated/compiled_preamble.cpp g++ mlx FALSE x86_64

export SPM_CUDA=0          # required: without it the build looks for cuda.h and fails
cd /path/to/Shears && swift package resolve && swift package edit mlx-swift --path /opt/mlx-swift
swift build && swift build --build-tests && swift test
cd CLI && swift package resolve && swift package edit mlx-swift --path /opt/mlx-swift && swift build -c release
cd ../Server && swift package resolve && swift package edit mlx-swift --path /opt/mlx-swift && swift build -c release
```

The patch applies to a fresh recursive clone and the preamble it regenerates is byte-identical to the one tested; the builds ran against an identical copy from `.build/checkouts`.

Times on this 4-core machine: library debug build about 2 min cold, CLI release about 5.5 min cold.

## Ran: test results (CPU, debug build, serial `swift test`)

363 tests: **327 passed, 35 skipped, 1 failed**, 561 s in XCTest (9 min 23 s wall).

- 32 skips are the existing opt-in ones (benchmarks, oracle vectors, llama-quantize, tokenizer in the HF cache, the new sample-workspace writer).
- 3 skips are new and Linux-only, each with its reason in the skip message: two `ResidencyTests` mapping tests (only MLX's Metal allocator wraps external memory; CPU and CUDA copy it) and `TopKFormTests.testTiesBreakAsInMLXLM` (expected orders are Metal's tie-breaking).
- 1 failure, not fixed: `MergedExpertTests.testPipelineTrainingIsTheSameInMergedAndRoutedLayouts`, max |diff| 7.4e-4 on `layers.0.mlp_up.a` (keep 2) against atol 5e-4 / rtol 5e-3. Losses agree; the merged and routed layouts' adapters drift slightly more on CPU than the test allows. Cause not investigated; tolerance left as is.

## Ran: end to end through the binaries

`RunnerTests/testWriteSampleWorkspace` (new, opt-in: `SHEARS_SAMPLE_WORKSPACE=<dir> swift test --filter RunnerTests/testWriteSampleWorkspace`) writes the tiny model, `rows.jsonl` and `Examples/tiny-config.json`. On it, the release `shears-cli` ran `select`, `prepass`, `heal` (stages A, B, C: 12 + 12 + 5 steps), `export` and `gguf`, all exit 0, in under 2 s total; output includes `keep.json`, three stages' adapters, the export and a GGUF file. `shears-serve serve --port` answered `/health` and a chat completion (bf16), and `shears-serve bench` ran.

The progress line's "mlx peak" reads 0.00 GB on CPU; use the process footprint there.

## Failures found and fixed

| Problem | Fix |
|---|---|
| `import CryptoKit` (TeacherPrepass) | swift-crypto 3.x/4.x on non-Darwin only (`Package.swift`), `#if canImport(CryptoKit)` |
| `import Darwin`, `mmap`/`MAP_FAILED`, `F_RDADVISE` | Glibc; `MAP_ANONYMOUS`; `MAP_FAILED` spelled as a bit pattern; new `ReadAhead` uses `posix_fadvise(WILLNEED)` |
| `task_info` footprint | `/proc/self/status`: footprint = RssAnon + VmSwap, file-backed = RssFile + RssShmem |
| `CFGetTypeID`/`CFBooleanGetTypeID`/`CFNumberIsFloatType` | `NSNumber.isBoolean` / `isFloatingPoint` in `Portability.swift` (CF on Darwin, identity and `objCType` elsewhere; checked on Linux) |
| `autoreleasepool` | no-op shim without the Objective-C runtime |
| HTTP server: `SOCK_STREAM` type, `SHUT_RDWR`, `SO_NOSIGPIPE`, `Darwin.send` | Glibc types; `MSG_NOSIGNAL` on each send |
| Zero-copy probe wrote to an unmapped page (segfault): non-Metal MLX copies and frees at once | probe returns false; loads fall back to `pread` |
| `[GatherMM::eval] Currently only supports float32` aborts every bf16 MoE path on CPU | `GatherMM.swift` shadows `gatherMM` in the module: float32 on a CPU device, MLX's own otherwise |
| Tests: `Darwin.*`, `URLSession` (FoundationNetworking), `XCTestCase()` has no `init()` in corelibs XCTest, `getrusage` type, `proc_pid_rusage` | per-platform aliases and imports; helper init; footprint benchmark skips off macOS |

Files changed: `.gitignore`, `Package.swift`, `Sources/Shears/{GatherMM,Portability}.swift` (new), `Sources/Shears/Healing/{ReadAhead}.swift` (new), `Sources/Shears/Healing/{MappedSafetensors,ProcessMemory,StreamedBlockHealing,TeacherPrepass}.swift`, `Sources/Shears/GGUF/{GGUFGemma4,GGUFGenerationConfig}.swift`, `Sources/Shears/Inference/ChatTemplate.swift`, `Sources/Shears/Inference/Server/HTTPServer.swift`, `Tests/ShearsTests/{InferenceServerTests,KQuantOracleTests,PrepareMemoryBenchmark,ResidencyBenchmarkTests,ResidencyTests,RunnerTests,TestSupport,TopKFormTests}.swift`, plus this page and the patch. `Package.resolved` files were not committed (Linux adds swift-crypto and swift-argument-parser pins).

## Read only: CUDA

- **SwiftPM CUDA path likely fails as shipped.** The `.cu` sources include CUTLASS/CuTe (`cutlass_utils.cuh`, `device/cute_dequant.cuh`, `gemms/grouped_gemm_unaligned.cu`, `quantized/qmm/fp_qmv.cu`) and `cudnn_utils.cpp` uses cudnn-frontend; `CudaBuild.json` adds no include path for either, and only the CMake path FetchContents them. Include paths are hard-coded to `/usr/local/cuda/include` and `/usr/local/cuda/lib64`. `CUDA_ARCH` is passed to nvcc as `-arch`.
- **CMake path:** rejects CUDA 13.1 exactly (`>= 13.1 AND < 13.2`); default arch is the build machine's native one (`MLX_CUDA_ARCHITECTURES` overrides). mlx-swift CI: Ubuntu 24.04, CUDA 12.9, a T4, Swift **6.2.3**, CMake; it builds static libraries and examples, not something a SwiftPM package like Shears links directly.
- **`gatherQuantizedMM` on CUDA** (correcting the prior note): the N%128 rule applies only to the sm80 fast path (compute capability >= 8, N%128, K%max(64, group), fp16/bf16, transposed, 4 or 8 bits). Below that there is a naive kernel (transposed needs K%max(64, group)) and a `qmv` kernel used when M*B < 8 (K%8, transposed). "No implementation for problem shape" comes only when all three refuse, so a T4 should take the naive path.
- Attention on compute capability < 8 takes the unfused fallback (`scaled_dot_product_attention.cpp`).
- CUDA's allocator does not wrap external memory either, so memory-mapped loading copies there too.

## Still unverified

Anything on a GPU: whether the SwiftPM CUDA build works with CUTLASS/cudnn-frontend added; CMake-built mlx-swift linked into Shears; CUDA 13.0 on Colab; T4/L4/A100/H100 numerics and the bf16 paths; `gatherQuantizedMM`/`vjp` CUDA-vs-CPU agreement. On CPU: the one failing test above; memory behaviour of real-size models; Colab's own image.

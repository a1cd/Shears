# Shears on Linux, Colab and CUDA

Shears targets MLX on Apple Silicon. Linux is a secondary, CPU-only port that exists so the pipeline can be tried without a Mac, for example from the [Colab notebook](../Notebooks/Shears.ipynb). Speed, memory behaviour (no zero-copy mapping, no unified memory) and numerics differ from the Mac, so do not compare numbers across the two. **A CUDA build of `shears-cli` is compiled in CI (see [the CUDA binary](#the-cuda-binary-built-in-ci-never-run-on-a-gpu)), but it has never been run on a GPU.**

The notebook runs the commands below only in its optional build-from-source cell; by default it downloads a prebuilt binary (see the next section). This page is the reference for what was verified and what was not.

Tested 2026-10-21 on Ubuntu 24.04.5 LTS, x86_64, 4 cores, 15 GB RAM, **no GPU** (`nvidia-smi` and `nvcc` not installed). Everything under "Ran" was run; everything under "Read only" comes from reading source and was not executed.

## The Colab notebook

[`Notebooks/Shears.ipynb`](../Notebooks/Shears.ipynb) downloads a prebuilt `shears-cli` (or, in its optional last cell, runs the commands on this page to build it) and then the tiny pipeline (`shears-cli sample`, `select`, `prepass`, `heal`, `export`), and loads the export with transformers. It also runs on an Apple Silicon Mac (where it uses `xcodebuild`). Facts about it:

- **Run headless on a Mac** (`jupyter nbconvert --execute`, M3 Max): about 100 s for a cold `xcodebuild` of `shears-cli`, 2 to 15 s for the whole tiny pipeline (the first run compiles Metal kernels), under 2 minutes in all including the transformers cell.
- **Two modes, `MODEL = "gemma"` (default) or `"tiny"`** (the first code cell). `tiny` is everything described below: a random 4-layer model, the CPU or the CUDA binary, no download. `gemma` is the real `google/gemma-4-26B-A4B-it`: the notebook checks the free disk and prints the download size (about 52 GB) and an estimate of the time, stops with a message if it will not fit, downloads, then runs a short real pipeline and generates from the export with a real prompt (see [the real model on Colab](#the-real-model-on-colab-gemma-mode)). CI runs `tiny` (the notebook picks it when `GITHUB_ACTIONS` is set, or when `SHEARS_MODEL=tiny`), and `linux.yml` is unchanged.
- **Not run on Colab, and not run on a GPU.** The assembled notebook has run on a Mac (tiny mode, Metal) and in CI on an x86_64 Ubuntu 24.04 runner with no GPU and no Swift (tiny mode, CPU binary). It has never run on Colab. **The CUDA path and `gemma` mode have never run anywhere**; the CUDA binary has only been compiled. Colab's current GPU image, from [googlecolab/backend-info](https://github.com/googlecolab/backend-info) (read 2026-10-22), is Ubuntu 24.04.4, CUDA toolkit 13.0 (nvcc 13.0.88, cuBLAS 13.1.1.3), cuDNN 9.14 for CUDA 13, PyTorch 2.11 for CUDA 13, g++ 13.3 and libstdc++ 14, so the prebuilt's glibc 2.39 requirement is met. Colab's pinned past runtimes (2026.01, 2026.04, 2026.07) are Ubuntu 22.04 and the prebuilt will not run on them.
- **Prebuilt binary.** On x86_64 Linux the notebook downloads a prebuilt `shears-cli` and does not build anything (`PREBUILT_URL` and `PREBUILT_SHA256` are in the settings cell; the SHA-256 is otherwise read from the `.sha256` file next to the tarball). If the download, the checksum or a first run of the binary fails, it stops and says so. Building from source is one optional cell at the very end (`BUILD_FROM_SOURCE = True`, off by default), for changing Shears itself or for a system the binary does not run on. [`.github/workflows/linux.yml`](../.github/workflows/linux.yml) builds the binary on every push: release build with `--static-swift-stdlib`, the few shared libraries it still needs (OpenBLAS, LAPACK, libgfortran, libstdc++, libgcc) copied into `lib/` next to it with `$ORIGIN` run paths, a SHA-256 file, and an upload as a workflow artifact. The same workflow runs `swift test` and then the notebook headless, on a runner with Swift removed, against that artifact. It is built on Ubuntu 24.04 and needs glibc 2.39 or newer and `g++` at run time (MLX compiles fused CPU kernels with it); on an older image (an Ubuntu 22.04 Colab) it will not run, and the notebook stops and points at the source-build cell. CPU only.
- **`swift test` in CI.** One failure is allowed by name, `MergedExpertTests.testPipelineTrainingIsTheSameInMergedAndRoutedLayouts` (the numeric-tolerance miss described under "test results" below); any other failure, a crash or a build error fails the job. The counts are in the job summary.
- **Setup time on Colab is unmeasured.** From the 4-core x86_64 numbers below (library about 2 min, release CLI about 5.5 min cold) expect 15 to 30 minutes on Colab's 2 vCPUs, mostly the MLX C++ build.
- **The checkout directory must be called `Shears`.** SwiftPM names a path dependency after its directory, so `CLI/Package.swift`'s `.package(path: "..")` only resolves when the directory is `Shears`. The notebook copies or clones into `<work>/Shears` for that reason.
- **transformers.** The export loads with `AutoModelForCausalLM` (a text-only `Gemma4ForCausalLM`; Gemma 4 appears in transformers 5.5.0, the notebook pins 5.20.2, the version tested). In float32 its top-5 next-token log-probabilities match `shears-serve` on the same export to four decimals. See [Quantization and export](quantization-and-export.md#loading-the-export-with-transformers).

## The CUDA binary, built in CI, never run on a GPU

[`.github/workflows/linux-cuda.yml`](../.github/workflows/linux-cuda.yml) (separate from `linux.yml`) compiles `shears-cli` with MLX's CUDA backend on an `ubuntu-24.04` runner. **The runner has no GPU, so this only proves the thing compiles and links.** Nothing in CI starts a CUDA kernel; the first real test is a Colab GPU runtime. It publishes the result as the **pre-release** `linux-x86_64-cuda-experimental` (`shears-cli-linux-x86_64-cuda.tar.gz` and its `.sha256`) on pushes to `cuda-binary` and `main`. The CPU release `linux-x86_64-latest` is untouched.

What the workflow does, and why:

- **CUDA 13.0 from NVIDIA's apt repository** (nvcc, cudart, NVRTC, driver stubs, cuBLAS, cuFFT, nvtx, cuDNN 9.14), because that is what Colab's GPU image has, and a binary linked against CUDA 12 would need `libcudart.so.12` and `libcublas.so.12`, which Colab does not put on the library path. MLX's CMake path rejects exactly 13.1; 13.0 passes.
- **Headers the SwiftPM build does not fetch**, at the versions MLX's CMake pins: CUTLASS v4.4.2 (`cutlass`, `cute`), cudnn-frontend v1.16.0 (its whole `include/` directory), CCCL v3.1.3 (merged over the toolkit's own `include/cccl`). mlx-swift's `CudaBuild.json` and manifest add no include path for them, hard-code `/usr/local/cuda`, and do not give the generated `cuda_jit_sources.h` a search path, so the workflow puts all of that under `/usr/local/cuda/include`.
- **`SPM_CUDA=1`** (it is the default on Linux; the CPU workflow sets `0`). mlx-swift's `CudaBuild` plugin turns each `.cu` file into C++ with `nvcc -cuda -rdc=true`, using the Swift toolchain's `clang++` as the host compiler, and links the device code.
- **Architectures: sm_75 (T4), sm_80 (A100; also runs on 8.6 and 8.9, so L4 and A10), sm_90 (H100), plus PTX for 9.0.** mlx-swift's plugin passes one `-arch` from `CUDA_ARCH`. A small `nvcc` wrapper placed first on `PATH` replaces it with one `-gencode` per architecture in `SHEARS_CUDA_ARCHS` (the compile and the device-link step get the same list) and adds `-allow-unsupported-compiler`. This matters: MLX picks its sm_80 and sm_90 kernels at run time from the GPU's compute capability, and their bodies are compiled in only for device code built for 8.0 or newer, so a T4-only build would be wrong, not just slow, on an A100.
- **`--static-swift-stdlib`**, as for the CPU binary, and `--jobs 3` because nvcc on CUTLASS sources is memory hungry on a 16 GB runner.
- **No NVIDIA library is bundled.** [`package-linux-cuda.sh`](../.github/scripts/package-linux-cuda.sh) copies only what the CPU tarball copies (OpenBLAS, LAPACK, gfortran, libstdc++, libgcc) and writes the NVIDIA ones it finds in the binary's `NEEDED` entries into `NVIDIA_LIBRARIES.txt` and `BUILD_INFO.txt` inside the tarball.

**Required at run time, not in the tarball** (see the CI job summary for the exact list of the latest build): `libcuda.so.1` (the NVIDIA driver; any machine with a working GPU), and from the CUDA 13 toolkit, `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcufft.so.12`, `libnvrtc.so.13` and `libcudnn.so.9`, plus glibc 2.39 or newer and `g++` (as for the CPU build). On Colab's GPU image these are installed system-wide (`/usr/local/cuda-13.0`) and also come as `nvidia-*` pip wheels; the notebook looks in both and adds what it finds to `LD_LIBRARY_PATH`. MLX also compiles some kernels with NVRTC at run time against CCCL headers, which it looks for in `<bundle>/include/cccl` first; the notebook downloads CCCL 3.1.3 there (the build was made against the same headers).

**`shears-cli device`** runs one 4x4 matrix product on MLX's default device and prints `mlx device: gpu` or `mlx device: cpu`; every other command prints the same line first. **That line is how to tell a GPU run from a CPU run.** On a Mac `gpu` is Metal. The notebook uses the command as its test of the CUDA binary: if it fails (no driver, a library of the wrong version, no kernel for this GPU) or answers `cpu`, the notebook prints the reason and falls back to the CPU build.

**Not yet known:** whether it starts on a T4 or any other GPU; whether kernels built for 7.5/8.0/9.0 plus PTX 9.0 cover Colab's other GPUs (L4 uses the 8.0 code; a Blackwell part would depend on the PTX); whether MLX's run-time (NVRTC) kernels work against CCCL 3.1.3 beside Colab's CUDA 13.0 headers; whether `bf16` and the quantized paths give correct results; and the speed and memory of any of it. The release is labelled experimental for that reason.

## The real model on Colab (`gemma` mode)

`MODEL = "gemma"` in the notebook is a short smoke-level run of the real pipeline, not a recipe for a good model. All of it is **untested**: nothing here has run on Colab, a GPU, or Linux.

1. **Disk check.** It sums the file sizes from the Hugging Face API (51.6 GB for `*.safetensors`, `*.json`, the tokenizer and the chat template; the model is Apache 2.0 and not gated), subtracts what is already on disk, adds 14 GB for the export (about 4B parameters in bf16, roughly 8 GB; an estimate, not measured), the frontier files and slack, prints the time at 50 to 150 MB/s, and stops with a message if `WORK` has less free. A free Colab GPU runtime may not have enough disk.
2. **Device gate.** If MLX ended up on the CPU (no GPU, or the CUDA binary failed) the notebook stops before the download. `ALLOW_GEMMA_ON_CPU = True` overrides.
3. **Rows.** 32 short chat exchanges written for the notebook, tokenized with the model's own chat template (`answerStart` at the reply).
4. **`shears-cli run`**, not `select` plus `heal`: only `run` streams the weights a layer at a time, so the 52 GB never has to fit in memory (`select` and the `heal` stages load the whole model). It selects 8 of 128 experts per layer (`importanceCandidates` 32, one swap round, 60 s per layer) and heals each of the 5 blocks of 6 layers for 10 steps. **This deviates from "heal the first block only"**: `export` needs adapters and a keep set for all 30 layers, and `run` with `maxBlocks: 1` leaves the other layers without either, so every block gets the same small number of steps. The memory budget (`memory.budgetGB`) is 85% of the GPU's memory on a GPU; the run is refused up front, with the amount, if it needs more. The compute dtype is float32 on a T4 (no bf16 hardware) and on the CPU, bfloat16 on compute capability 8.0 or newer.
5. **`shears-cli export`**, then the notebook loads the export with transformers (on the GPU if torch sees one) and generates 120 tokens from a real prompt.

The learning rate, the ranks, the number of steps and the selection settings are guesses in the notebook's config cell, not tuned values.

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

Anything on a GPU (the CUDA binary compiles in CI, see above; it has never run); CMake-built mlx-swift linked into Shears; CUDA 13.0 on Colab; T4/L4/A100/H100 numerics and the bf16 paths; `gatherQuantizedMM`/`vjp` CUDA-vs-CPU agreement. On CPU: the one failing test above; memory behaviour of real-size models; Colab's own image.

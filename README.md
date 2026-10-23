# Shears

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a1cd/Shears/blob/main/Notebooks/Shears.ipynb)

Shears prunes a Mixture-of-Experts model (one built from many specialist sub-networks, called experts, of which each token uses only a few) by removing experts, then heals it by retraining what is left to recover the quality the pruning cost. It is written in Swift on [mlx-swift](https://github.com/ml-explore/mlx-swift), with no Python and nothing that can't ship in an app.

Needs macOS 14+, Swift 6.0 tools and mlx-swift 0.31 or later (it also builds on x86_64 Linux on the CPU backend, as a way to try the pipeline on a tiny model: [Linux and Colab](docs/linux.md)); iOS 17+ and visionOS 1+ build but are untested on a device ([iOS and visionOS](docs/ios-visionos.md)). This is a young project: see [Status](#status) for what has and hasn't been verified.

## What it does

- **Prune and heal** a MoE to a fixed number of experts per layer. Healing trains small add-on weight matrices (LoRA adapters: per-expert, dense, and a low-rank correction to the router) against the original, unpruned model, called the teacher, so the pruned model's outputs move back toward the teacher's ([healing](docs/healing.md)).
- **Block-sequential pipeline** (`shears-cli run`): heals a few layers at a time instead of the whole model at once, so only one block is in memory. Resumable, with a `--memory-budget` ceiling ([command line and pipeline](docs/cli.md)).
- **Low memory:** a streamed block phase, fewer layers resident, memory-mapped weights, and the teacher's outputs computed once and stored on disk instead of recomputed every step ([memory](docs/memory.md)).
- **Quantization-aware healing** (QAT: training against the rounding that the shipped 4-bit format will apply, so healing accounts for it) and a **Q4_K_M GGUF export** (GGUF is the file format llama.cpp loads) that writes the file `llama-quantize` would ([quantization and export](docs/quantization-and-export.md)).
- **Gemma 4** text, dense and MoE (e.g. 26B-A4B), ported from mlx-swift-lm ([Gemma 4](docs/gemma4.md)).
- **Inference without llama.cpp:** tokenizer, chat template, KV-cache decode, sampling, and `shears-serve`, an OpenAI-style server ([inference and serve](docs/inference-and-serve.md)).
- **MoE building blocks:** `SwitchGLU`, `TopKRouter`, `MoEBlock`, `ExpertPruner`, gradient checkpointing and more ([what's here](docs/library.md)).

## Measured

Two kinds of evidence, kept apart. The setup, the noise and the limits are in the linked pages.

**On the real Gemma 4 26B-A4B** (October 2026, M3 Max with 48 GB; every layer pruned to 8 experts, healed in 6-layer blocks; [details](docs/benchmarks.md#real-gemma-4-26b-a4b-m3-max-48-gb-2026-10)):

- **Memory.** Block 0 (layers 0 to 5), 50 steps on the same rows: peak process footprint 8.0 GB with `fast` keeping (all layers keep their intermediates), 6.0 GB (mid) and 4.8 GB (lowMemory); the three settings trade memory for recompute time, see `keepIntermediates` in the [command line page](docs/cli.md); MLX peak 6.75, 4.71 and 3.87 GB. The reference Python/MLX recipe this was ported from used about 24 GB on the same task.
- **Quality and speed.** The same held-out loss in every arm (0.366, against 0.368 for the reference recipe). With `fast` keeping, 0.72 s/step against 0.86 s/step for the reference recipe (about 1.2x faster per step). The reference recipe also runs a live teacher pass (+0.35 s/step); Shears computes the teacher once up front, about 0.29 s/step spread over these 50 steps. Counted together that is about 1.0 against 1.2 s/step, so about 1.2x faster end to end, not 2x.
- **Energy.** Net energy per step (powermetrics, idle subtracted): 9% less than the reference recipe with `fast` keeping, 4% more with mid, 20% more with lowMemory.
- **Parity with the reference recipe.** The pruned student with the reference recipe's adapters reproduces its outputs exactly. The teacher matches except for float-noise router ties. The Q4_K_M export is byte-identical to `llama-quantize` output, tensor by tensor.
- **A limit.** Training is not lower memory than running the pruned Q4_K_M model in llama.cpp (about 3.7 to 4 GB).
- **Preliminary, one run, small samples:** healing the whole model for two tasks improved each adapter on its own task, but the translation adapter made other languages worse and the cause is not yet understood ([details](docs/benchmarks.md#whole-model-two-task-run-preliminary)).

**On toy models** (a small Gemma 4 MoE, Apple silicon):

- **Memory.** On a toy Gemma 4 MoE (M3 Max, fp32, 2026-10-17) the streamed peak barely grows with depth (188 to 201 MB for twice the layers), while the main path's resident set grows with it ([memory](docs/memory.md)).
- **Speed.** A streamed step on the toy block went from 0.117 to 0.076 s/step (small, whole block resident) and 0.162 to 0.111 (big) with compiled steps; the machine was not quiet, about 10 to 15% noise ([benchmarks](docs/benchmarks.md)).
- **Q4_K_M export.** Verified byte for byte, for every tensor, and whole files against `llama-quantize` on tiny scratch models (hidden 256, 12 layers; 8 and 4 experts) ([quantization and export](docs/quantization-and-export.md)).

## Status

Healing is ported for Gemma 4 only. The real Gemma 4 26B-A4B has been loaded, pruned, healed and exported on one machine (an M3 Max with 48 GB); the numbers above come from that, and its whole-model quality results are preliminary. Memory was measured on one block; the figures for a whole 26B run in the docs are still estimates ([memory](docs/memory.md#estimate-for-the-real-26b-a4b-derived-not-measured)). Nothing has been run on an iOS or visionOS device or simulator. Linux builds and passes its tests on the CPU backend (one test fails), but no CUDA GPU has been tried: CI compiles an experimental CUDA build of `shears-cli`, whose first test on a GPU (a Colab T4) crashed at start-up ([Linux and Colab](docs/linux.md)). The full list is in [notes and status](docs/notes.md).

## Install

Add it as a Swift package (there are no tags yet, so track `main`):

```swift
.package(url: "https://github.com/a1cd/Shears.git", branch: "main")
// target dependency: .product(name: "Shears", package: "Shears")
```

## Quickstart

Prune and heal from one JSON config (`Examples/tiny-config.json` is the sample), then run the result:

```bash
cd CLI && xcodebuild build -scheme shears-cli -destination 'platform=macOS'
shears-cli run --config path/to/config.json                       # the block-sequential pipeline
shears-cli run --config path/to/config.json --memory-budget 8GB   # a ceiling: refused up front, with the amount, if the run needs more

cd Server && xcodebuild build -scheme shears-serve -destination 'platform=macOS'
shears-serve serve --model DIR --socket /path/to/sock        # or --port 8080 [--host 127.0.0.1]; --alias, --dtype, --max-tokens
```

```swift
let runtime = try InferenceRuntime.load(directory: dir)               // bf16, a Shears export, or a 4-bit QAT checkpoint
let prompt = try runtime.promptTokens(messages: [["role": "user", "content": "hello"]])
let result = try runtime.generator.generate(runtime.request(promptTokens: prompt, maxTokens: 256, logprobs: 5)) { event in
    print(event.text, terminator: ""); return true                    // stream; return false to stop
}
```

## Try it without a Mac

[`Notebooks/Shears.ipynb`](Notebooks/Shears.ipynb) builds Shears, prunes and heals a tiny random Gemma 4 MoE, exports it and loads the export with Hugging Face transformers. It runs top to bottom on an Apple Silicon Mac or on Colab (CPU, Linux). It shows the pipeline working; the real target is MLX on a Mac with Gemma 4 26B-A4B. Details and what is unverified: [Linux and Colab](docs/linux.md).

## Build and test

For contributors. `swift test` can't compile MLX's Metal shaders. Use Xcode (needs the Metal Toolchain:
`xcodebuild -downloadComponent MetalToolchain`):

```bash
xcodebuild test -scheme Shears -destination 'platform=macOS'
```

Memory benchmarks are skipped unless enabled:

```bash
TEST_RUNNER_SHEARS_BENCH=1 xcodebuild test -scheme Shears -destination 'platform=macOS' -only-testing:ShearsTests/MemoryBenchmarkTests -only-testing:ShearsTests/CheckpointScalingBenchmark
```

## Docs

[docs/README.md](docs/README.md) is the index: [healing](docs/healing.md), [memory](docs/memory.md), [command line and pipeline](docs/cli.md), [quantization and export](docs/quantization-and-export.md), [Gemma 4](docs/gemma4.md), [inference and serve](docs/inference-and-serve.md), [benchmarks](docs/benchmarks.md), [what's here](docs/library.md), [iOS and visionOS](docs/ios-visionos.md), [Linux and Colab](docs/linux.md), [notes and status](docs/notes.md). Third-party code is credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT, see [LICENSE](LICENSE). Code adapted from other projects is listed with its licenses in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Where it came from

Shears came out of work on optimizing on-device models for [Tolby](https://tolby.app), a macOS app.

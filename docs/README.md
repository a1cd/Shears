# Shears docs

The front page is the [README](../README.md). The detail lives here, split by topic.

| Page | What's in it |
|---|---|
| [Pruning and healing](healing.md) | Pruning is a mask, the block phase, low-memory mode, the pre-computed teacher and sliced student, notes on faithfulness. |
| [Memory](memory.md) | Fewer layers resident (`residentLayers`, `memoryBudget`), memory-mapped weights, the two memory numbers, measured peak memory on toys, what the on-disk teacher and kept-experts-only student save, and the estimate for a whole 26B-A4B run (derived, not measured). |
| [Command line, long runs and the pipeline](cli.md) | Long runs: schedule, mid-block checkpoints, stop, command line. The block-sequential pipeline (`shears-cli run`), auto residency, merged expert layout and shared expert LoRA, the learning-rate sweep (`shears-cli sweep`). |
| [Quantization-aware healing and GGUF export](quantization-and-export.md) | Quantization-aware healing, the Q4_K_M export (bit-exact), sampling defaults. |
| [Gemma 4](gemma4.md) | The Gemma 4 port: layers, config defaults, router, healing, checkpoints, quantized checkpoints. |
| [Inference and `shears-serve`](inference-and-serve.md) | Tokenizer, chat template, decode, unmerged adapters, and the OpenAI-style server. |
| [Benchmarks](benchmarks.md) | The real Gemma 4 26B-A4B runs (memory, speed, energy, parity, a two-task whole-model run), speed on toys, and checkpointing. |
| [What's here](library.md) | The table of types: what each one does. |
| [iOS and visionOS](ios-visionos.md) | What builds, and what needs runtime testing. |
| [Linux and Colab](linux.md) | The Linux CPU port, the Colab notebook, the verified install commands, and what is only read, not run (CUDA). |
| [Notes and status](notes.md) | Things worth knowing, and not done yet. |

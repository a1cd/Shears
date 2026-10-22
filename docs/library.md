# What's here

[Shears docs](README.md)

| Type | What it does |
|---|---|
| `checkpoint(_:)`, `checkpoint(_:_:)` | Gradient checkpointing from `CustomFunction` (mlx-swift doesn't wrap `mlx_checkpoint`, and `Cmlx` isn't a public product). The module form threads trainable parameters through as explicit inputs. |
| `SwitchLinear`, `SwitchGLU` | Ports of mlx-lm's `switch_layers`. `weight` is `[E, out, in]`, computed with `gatherMM`. Rows are sorted by expert once `indices.size >= 64`, and always in training mode (see [Things worth knowing](notes.md#things-worth-knowing)). |
| `ExpertRouter`, `MixtureOfExperts` | The protocols everything below works through: a router (`numExperts`, `topK`, `route`, `keepExperts`) and a router-plus-expert-bank unit. |
| `TopKRouter` | softmax → `argPartition` top-k → `takeAlong`, with optional renormalisation. |
| `MoEBlock` | Router plus experts, weighted sum over k. Keys `gate` / `switch_mlp` match mlx-lm Qwen3-MoE. |
| `MoEResidualLayer` | `x + moe(rmsNorm(x))`: the MLP half of a decoder layer. Used for toy models and healing. |
| `ExpertStats` | Calibration hook: per-expert pick count, routing-weight mass, and REAP-style saliency (`score × ‖expert output‖`). |
| `ExpertPruner` | Slices router rows and expert weights to a keep list in place, and clamps `topK`. |
| `QuantizedSwitchLinear`, `ExpertQuantizer` | Frozen quantized experts via `gatherQuantizedMM`, with optional per-expert LoRA. |
| `MoETraining` | Freezing helpers: router+experts, router only, router+LoRA. |
| `PruneHealer` and friends | Pruning by routing mask and healing with adapters: see [Pruning and healing](healing.md). |
| `Qwen3MoECheckpoint` | Loads a Qwen3-MoE checkpoint's sparse MLP layers one at a time (HF per-expert, mlx-lm stacked, or mlx-lm quantized layouts). |
| `Gemma4TextModel`, `Gemma4Checkpoint` | Gemma 4 text (dense and MoE, e.g. 26B-A4B), ported from mlx-swift-lm. See [Gemma 4](gemma4.md). |
| `InferenceRuntime`, `Generator`, `BPETokenizer`, `ChatTemplate`, `InferenceServer` | Run a Gemma 4 model without llama.cpp: tokenizer, chat template, KV-cache decode, sampling, an OpenAI-style server. See [Inference and `shears-serve`](inference-and-serve.md). |

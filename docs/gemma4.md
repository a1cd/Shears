# Gemma 4

[Shears docs](README.md)

`Sources/Shears/Models/Gemma4/` is a port of mlx-swift-lm's `Gemma4Text.swift` (MIT; see
`THIRD_PARTY_NOTICES.md`).

- **Layers.** Sliding-window and full attention. Full attention uses `global_head_dim`,
  proportional partial RoPE and `attention_k_eq_v`, so it has no `v_proj`. A dense GELU MLP
  and, with `enable_moe_block`, a parallel sparse branch. Final logit softcapping. A
  simple append-only KV cache.
- **Config defaults.** Absent keys follow Hugging Face: no `final_logit_softcapping` is no cap; no `layer_types` is a 5:1 sliding:full pattern (Hugging Face fixes the period at 6 and ignores `sliding_window_pattern`; here an explicit key is honoured as the period, as mlx-lm does); the last layer is always full attention; no `rope_parameters` is a 0.25 rotary factor on full attention. `rope_type` (proportional or default) and `factor` are honoured. Real configs set all of these.
- **Router.** `Gemma4Router` takes the top-k over raw `proj` logits, softmaxes them, then
  multiplies by `per_expert_scale`. It reads the post-attention residual, while the
  experts read `pre_feedforward_layernorm_2` of it. `Gemma4MoEDecoderLayer` is itself the
  `MixtureOfExperts`, so its `router.*` and `experts.switch_glu.*` keys sit at layer level
  as in Hugging Face. Pruning slices `proj` rows and `per_expert_scale`, and leaves
  `scale` (`[hidden]`) alone.
- **Healing.** A decoder layer is a `UnaryLayer`. Called with one argument, it treats its
  input as a whole sequence from position 0 and builds its own causal or sliding mask,
  so a whole decoder layer is usable as a block at any sequence length.
- **Checkpoints.** `Gemma4Checkpoint` reads the Hugging Face multimodal layout
  (`model.language_model.…`, fused `experts.gate_up_proj`), the text-only layout and
  mlx-lm's sanitised layout. It loads the whole model or one layer at a time and skips
  vision and audio tensors. `huggingFaceWeights(of:)` maps a model or a layer back to
  Hugging Face names.
- **Quantized checkpoints.** mlx-community-layout checkpoints (`weight`/`scales`/`biases`, a
  config `quantization` entry with per-module overrides such as the 26B-A4B QAT 4-bit
  conversion's 8-bit dense MLPs and routers) load as `QuantizedLinear`, `QuantizedEmbedding` and
  `QuantizedSwitchLinear`, whole or one layer at a time. Group size and bits are inferred from the tensor shapes and the module's logical width (never assumed) and must match the config entry if there is one; the global tensors of `loadModel(keeping:)` are verified like `loadModel()`'s. Top-k routing uses mlx-lm's `argpartition(x, kth=-k)[..., -k:]` form (`topKIndices`). A `dtype` cast touches floating-point tensors only, never packed integers. Adapters are
  sized from logical widths. `MergeAndSliceExport` dequantizes every quantized tensor before
  folding adapters and writes plain bf16 (or fp32) with no `scales`/`biases` left over. Tensor
  names were checked against the published index of `mlx-community/gemma-4-26B-A4B-it-qat-4bit`.
  Only the `affine` mode is supported.
- **Not supported.** Per-layer embeddings and KV sharing (the small dense models) are
  rejected at config load. Bidirectional attention over image tokens isn't ported (text only).

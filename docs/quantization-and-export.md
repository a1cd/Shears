# Quantization-aware healing and GGUF export

[Shears docs](README.md)

**Quantization-aware healing.** Set `AdapterConfiguration.qat` and the forward pass of the
student uses, for each covered module, the merged weight rounded to the shipping format:
`fakeQuantized(base + scale * (a @ b)^T)`, with the gradient passed straight through to the
full-precision adapters (`rt + (w - stopGradient(w))`, whose value is exactly the round trip).
`MergeAndSliceExport` writes the merged weight **unrounded**, cast to the export dtype, and the
converter that ships it (GGUF, MLX) does the rounding. Healing rounds `w.asType(exportDType)` with
the same quantizer, so the converter sees the same bits and picks the same codes. (Pre-rounding in
the export was dropped: rounding and then casting to bfloat16 is not a fixed point of the quantizer.)
`QATConfiguration.exportDType` (default `.bfloat16`, the export default) is that dtype. Guards, each
tested: `MergeAndSliceExport.write` throws if its dtype differs from the QAT one; the export carries
`shears-export.json` (dtype and per-tensor formats), and `GGUFExport.gemma4` throws when given an
importance matrix for such a model (healing never used one), when a covered tensor would be written
in a type other than the one it was rounded for, when the stored dtype differs, or for an affine
(MLX) model. Tests run at the default dtype: Q8_0 and Q4_K_M through GGUF, affine through
`MLX.quantized` on the bf16 export (bf16 scales and biases), logits equal to the student's.

- *Formats:* `.affine(bits:groupSize:)` (MLX's own quantizer), `.q8_0` (matches
  `quantize_row_q8_0_ref`, tested value for value against this package's GGUF writer, including
  all-zero blocks and exact ties), and `.gguf(.q4_k / .q5_k / .q6_k / .q5_0)`, the K-quants as
  `llama-quantize` writes them (see "Q4_K_M export" below). `QATConfiguration.q4KM(for:keptExperts:)`
  gives each tensor the type the Q4_K_M mix will give it, so healing sees exactly the weights the
  exported file holds: the test loads the Q4_K_M GGUF of a healed toy model, dequantizes the covered
  tensors, and its logits match the QAT student's (bit-exact, 0.0, with a bf16 export). The K-quant search runs on the CPU, so
  these formats are rounded once per step before the traced forward pass
  (`ModelAdapters.refreshRounding`, called by `PruneHealer` and `StreamedBlockHealer`); the forward
  then adds the zero-valued `w - stopGradient(w)` to that result. That costs CPU time per step
  (a few seconds per block on the 26B, not measured) and one float32 copy of the covered merged
  weights. No importance matrix is used while training or when exporting such a model.
- *Covered:* dense MLP gate/up/down, attention output, and the kept experts' gate/up/down, in the
  masked and the sliced student alike (they agree, tested). *Not covered:* the router (GGUF ships it
  as float32), norms, q/k/v and embeddings (frozen, so their rounding is a constant healing cannot
  reduce). 
- *Does it help?* On the toy (freerun, judged with the rounding on): 3-bit affine, rank 8, lr 1e-3,
  80 steps: shipped loss 0.56 against 0.66 for healing in float and rounding afterwards (2-bit:
  0.72 against 0.85). It is not free: at 4-bit with lr 3e-3 the straight-through noise made it
  worse (0.60 against 0.55), and the toy's untouched rounding is small next to the pruning damage.
  Use a smaller learning rate than for float healing, and measure on the shipped model.
- *Cost:* the merged kept-expert and dense weights are materialised in float32 each forward
  (recomputed under checkpointing). Toy block training: 131 MB plain, 136 MB with rounding. On the
  26B that is about 190 MB per layer for the eight kept experts' three projections, doubled for
  their gradients, so roughly +0.5 to 1 GB at the peak of the streamed block phase (estimate).

**Q4_K_M export, bit-exact.** `GGUFExport.gemma4(..., type: .q4_k_m, importance:)` writes the file
`llama-quantize ... Q4_K_M` would, and `GGUFTensorType` now has `Q4_K`, `Q5_K`, `Q6_K` and `Q5_0`
(writer, reader, dequantizer). The quantizers are ports of `ggml-quants.c` and the per-tensor mix of
`llama-quant.cpp` (llama.cpp commit a3a1c47; an importance matrix in `llama-imatrix`'s legacy `.dat`
layout loads with `GGUFImatrix.load`). Verified byte for byte, for every tensor: the quantizers against
`ggml_quantize_chunk` from the built library (7 data kinds, with and without an importance matrix,
3 data sets of up to 400 rows; also the dequantizers bit for bit), and whole files against
`llama-quantize` on tiny scratch models (hidden 256, 12 layers; 8 and 4 experts; with and without an
importance matrix that leaves some tensors out); on the real 26B-A4B the export was also found byte-identical to `llama-quantize` tensor by tensor (see [Benchmarks](benchmarks.md#real-gemma-4-26b-a4b-m3-max-48-gb-2026-10)). The mix covers the Gemma / MoE cases that
matter: eight experts put `attn_v` and `attn_k` at Q8_0 and `attn_output` at Q5_K; Q6_K goes to
tied embeddings and to `use_more_bits` layers; rows that are not a multiple of 256 (the real model's
expert and dense `ffn_down`) fall back to Q5_0 or Q8_0; the layer counters run in file order.
Caveat: the reference is compiled with fused multiply-adds in some expressions and not others, and
the port reproduces the pattern found by comparing against the reference output (`fmaSites` in
`GGUFKQuants.swift`); a differently built `llama-quantize` could differ in rare last-bit cases.
`Q5_K` rows that fall back to `Q5_1` are not supported. The oracle tests need the reference
(`TEST_RUNNER_SHEARS_KQUANT_ORACLE`, `TEST_RUNNER_SHEARS_LLAMA_BIN`) and skip without it.

**Sampling defaults.** The reference converter copies `generation_config.json` into
`general.sampling.*` keys (and `suppress_tokens` for the unified Gemma 4 architecture); so does the
export, with the converter's mapping, key order and precedence (`GGUFGenerationConfig`).

## Loading the export with transformers

`shears-cli export` writes a Hugging Face checkpoint, and `AutoModelForCausalLM.from_pretrained(dir)` loads it (transformers 5.20.2 tested; Gemma 4 first appears in 5.5.0). Four details of the export exist for that reason, each found by trying to load one:

- **Floats stay floats.** `config.json` writes `final_logit_softcapping` as `30.0`, not `30`: transformers validates it as a float and refuses an integer. Foundation's `JSONSerialization` writes `30.0` as `30`, so the exporter writes this file itself (`ConfigJSON`).
- **Tensor names follow the config.** A source `config.json` with a `text_config` (the multimodal `google/gemma-4-26B-A4B-it`, and every mlx-community conversion of it) is exported text-only by default (`export.textOnly`, default `true`): the config is flat (`model_type: gemma4_text`, `architectures: ["Gemma4ForCausalLM"]`, no `vision_config`) and the tensors are named `model.*`, which is what `Gemma4ForCausalLM` expects. `"textOnly": false` keeps the multimodal layout: the config is copied with its `text_config`, the tensors are `model.language_model.*`, and transformers builds `Gemma4ForConditionalGeneration` with a vision tower that has no weights in the export. That tower is initialized at random: about 0.6 billion parameters of noise that no text prompt uses, and a load report with a MISSING line per vision key (found on a Colab run of the real 26B; the text output was not affected). A config that is already flat, such as the one `shears-cli sample` writes, always gives `model.*`. Before, a text-only config got the multimodal names and loaded with every weight missing.
- **No default rope `factor`.** It is written only when it is not 1.0, because transformers warns about a `factor` on the `proportional` rope type.
- **Vision and audio weights are not exported.** Neither layout carries them; the text model is what Shears healed. Text-only export (the default) leaves the vision and audio configs out as well, so nothing is built that has no weights. Checked on the real 26B (QAT 4-bit source, 8 experts kept): transformers 5.20.2 loads the export as `Gemma4ForCausalLM` with no missing and no unexpected key.

Checked on the tiny sample model: in float32, transformers' top-5 next-token log-probabilities for a prompt equal `shears-serve`'s on the same export to four decimals (the top token and its probability, -1.6193). In bf16 they differ in the second decimal, which is bf16 rounding on a random model.


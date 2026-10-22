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

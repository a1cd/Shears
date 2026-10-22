# Memory

[Shears docs](README.md)

## Fewer layers resident: `residentLayers`, `memoryBudget` (library; the pipeline chooses automatically)

By default a streamed block is loaded whole. `HealingConfiguration.residentLayers` (1 up to the
block's length) holds only that many of its layers at once, with the same result:

- The forward walk goes up the block one window of layers at a time, keeping each window's input
  activation (`T x D`) and dropping its weights. The top window runs the loss in the same pass as
  its gradient, so it stays loaded from forward into backward.
- The backward walk goes down: each window is loaded again and each of its layers, from the top, gets one
  `valueAndGrad` that returns the layer's adapter gradients and the gradient of its input. The input's
  gradient goes down and the layer's adapters get their AdamW update at once, so below the top window one
  layer's gradients exist at a time (the top window is one fused `valueAndGrad`, so its `residentLayers`
  layers' gradients exist together). The cotangent is computed before the update, so the math is the
  whole-block update's. The optimizer is the block's one `HealingAdamW`, its step count advanced once per
  step, and checkpoints have the whole-block format (a windowed run resumes in either path). Adapter
  initialisation draws the same random numbers as the whole-block path.
- Compute is the whole-block path's, and a test counts it: the top window runs forward and loss together
  under the per-layer checkpointing (one forward, one recompute, one backward per layer), and a lower
  layer's forward walk keeps only its input, then one `valueAndGrad` per layer whose forward is the
  recompute, with no checkpoint wrapper (a wrapper would run the layer a third time). Layer forwards
  built per step are equal to the whole-block path's for every window. The extra cost is I/O and
  synchronisation: each layer outside the top window is loaded twice per step instead of once per block,
  `LayerSource.prefetchLayers` asks the kernel to read the next window while the current one computes
  (`F_RDADVISE`; readahead into the page cache only, the gather of kept experts and the `eval` still run
  synchronously, and the effect is unmeasured because the toy runs always had a warm cache), and
  every layer ends in an `eval`. With `checkpointBlocks` off the top window keeps its activations as
  the whole-block path does, but lower layers always recompute: that is the price of dropping them.
- Tested: losses and adapters equal the whole-block path for windows of 1, 2, 4 (does not divide 6),
  5 and 6 layers, with and without checkpointing, on sliced and masked students; skipped steps apply
  no update in any window; checkpoints cross paths; layer forwards built per step equal the whole-block
  path's for every window (it failed, 17 against 12, while lower layers still ran a third forward); an
  overshoot on a block's last step is reported and stops the run; mapped tensors share the file's pages and
  are not charged to the footprint (both fail if mapping is switched off).

`HealingConfiguration.memoryBudget` (bytes; `shears-cli heal --memory-budget 3GB`) chooses the window
for you. `ResidencyPlanner` estimates peak MLX memory from the loaded layer sizes, the adapters (fp32,
plus two Adam moments), the sequence length and a per-token workspace (x1.25 margin plus 8 MB), and takes
the largest window that fits. If even one resident layer does not fit, the run is refused before any
training (exit status 2) with the estimate. **MLX's own memory limit is only a guideline** (read in the
allocator: it makes MLX reclaim its buffer cache and never refuses an allocation), so the budget is kept by
the estimate and by measurement: the run sets the limit to the budget, checks the measured peak before every
step and after each block, and when it is over, saves a checkpoint and stops with the reason (a finished last
block cannot stop; `Report.peakOverLimit` and a CLI warning say it happened). A step can therefore overshoot
before the run notices. The estimate is a model, calibrated on toys (it covers the measured toy peaks,
tested) and has not been checked against the real 26B except for the one block measured in [Benchmarks](benchmarks.md#real-gemma-4-26b-a4b-m3-max-48-gb-2026-10). Memory-mapped weights are counted in the budget (MLX counts
them), so the budget bounds MLX's number, not Activity Monitor's.

**Weights are memory-mapped where that works.** `Gemma4LayerSource(mapWeights: true)` (the default;
`"memory": {"mapWeights": false}` in the CLI config turns it off) hands each tensor to MLX as a
read-only view of the checkpoint file (`mmap` with `PROT_READ`, `MAP_SHARED`, wrapped through
`MLXArray(rawPointer:)`; Metal wraps the pages without copying them, verified on macOS 27 by
overwriting the file and watching the array change). The pages are clean and file-backed, so the system can
drop them with no write, and nothing about weights is ever written. Limits, measured: it needs a tensor
to start on a multiple of its element size (Hugging Face files do; MLX's own `save` does not pad its header,
so such a file is read into memory instead, still correct); it uses `MAP_SHARED` because a `MAP_PRIVATE`
mapping is charged to the process footprint as if anonymous; and slicing a layer to its kept experts
*gathers* them into ordinary memory, so with a sliced student only the dense and attention tensors and the
read of the experts are file-backed, the kept experts are not. Memory that is not file-backed can be
compressed or swapped by the system under pressure; the only protection is how short a layer is held.

**Two memory numbers.** `Memory.peakMemory` ("MLX peak") counts bytes MLX's allocator holds, memory-mapped
weights included. The "process peak" is the process's `phys_footprint`, what Activity Monitor's *Memory*
column shows: it counts anonymous memory (and the process's own baseline, ~0.25 GB in a test host) and
leaves out clean file-backed pages. Progress lines, `steps.jsonl` (`mlxPeakBytes`, `footprintPeakBytes`,
`fileBackedBytes`) and `StepRecord` carry both; `fileBackedBytes` is the resident file-backed total of the
whole process (libraries included), not only weights.

Measured on a toy Gemma 4 MoE (one block of 6 layers, 16 experts cut to 8, top-2, fp32, adapters 32/16/8;
one process per arm on Apple silicon, 8 steps, median of steps 2 to 8; the machine was not quiet, so
repeat runs vary by about 10 to 15%):

| toy | resident layers | MLX peak | process peak | s / step | CPU s / step |
|---|---|---|---|---|---|
| small (hidden 256, expert 256, dense 512, T 512) | 6 (whole block) | 320 MB | 581 MB | 0.115 | 0.127 |
| | 3 | 281 MB | 532 MB | 0.190 | 0.173 |
| | 2 | 255 MB | 496 MB | 0.205 | 0.188 |
| | 1 | 154 MB | 390 MB | 0.217 | 0.198 |
| | 1, weights copied (`mapWeights: false`) | 154 MB | 375 MB | 0.223 | 0.212 |
| large (hidden 512, expert 512, dense 1024, T 1024) | 6 (whole block) | 1060 to 1089 MB | 1369 to 1409 MB | 0.155 to 0.177 | 0.171 to 0.209 |
| | 3 | 926 MB | 1172 MB | 0.270 | 0.237 |
| | 2 | 847 MB | 910 MB | 0.291 | 0.246 |
| | 1 | 509 MB | 781 to 837 MB | 0.409 to 0.447 | 0.365 to 0.407 |
| | 1, weights copied | 509 MB | 722 MB | 0.365 | 0.351 |

**Time is not equal on these toys: one resident layer costs about 1.9 times (small) to 2.5 times (large)
the whole-block step, and the compute is the same (the forward count is tested).** What is left is
overhead: about 11 layer loads per step for one resident layer (a few ms each, each one a gather of the kept
experts), a synchronisation per layer, and AdamW updates dispatched per layer instead of once. These toy
layers are small enough that a step is mostly kernel launches and waiting, so overhead that would be a small
share of a compute-bound 26B step is a large share here; that is an argument, not a measurement, and a real
run may or may not close the gap. Memory-mapping did not lower the process peak on the toys (it was 15 to
60 MB *higher* than copying, within noise): the sliced student gathers its experts into ordinary memory, which is
most of a layer. Energy could not be measured (`powermetrics` needs root); CPU seconds per step are the
proxy (about equal to wall time, so the GPU wait is not visible in them) and MLX exposes no GPU-busy time.
"File-backed" in the progress line is the whole process's resident file-backed pages (libraries included,
about 61 MB in every run above); it does not show how much of that is weights, because pages read only by the
GPU do not appear in it. So how much of a run's memory is droppable file-backed weight is not answered by a
column; the footprint difference between mapped and copied runs is the honest signal, and here it is none.

Estimate for a 26B-A4B-class model (hidden 2816, 30 layers in blocks of 6, 8 kept experts of width 704,
dense MLP 2112, T = 2700, 4-bit base with group 64, adapters at the recipe's ranks 128 / 64 / 16, assumed
16-head 256-wide attention; `ResidencyBenchmarkTests.testEstimateAtLargeModelShape`; a model, not a measurement of
the real weights; block 0 of the real 26B was measured later, see the update under [the estimate](#estimate-for-the-real-26b-a4b-derived-not-measured)):

| resident layers | sliced weights | adapters + Adam | gradients | activations | workspace | load transient | total (x1.25 + 64 MB) |
|---|---|---|---|---|---|---|---|
| 6 (whole block) | 0.32 GB | 0.82 GB | 0.82 GB | 0.28 GB | 0.85 GB | 0.43 GB | 4.5 GB |
| 3 | 0.16 | 0.82 | 0.27 | 0.28 | 0.85 | 0.43 | 3.6 GB |
| 2 | 0.11 | 0.82 | 0.18 | 0.28 | 0.85 | 0.43 | 3.4 GB |
| 1 | 0.05 | 0.82 | 0.09 | 0.28 | 0.85 | 0.43 | 3.2 GB |

The weight bytes are analytic (parameter counts at 4.5 bits). On the toy a loaded layer's allocator bytes equal its parameters' sizes with each tensor over 16 KB rounded up to whole pages (checked), so rounding is negligible at 26B tensor sizes; but `Memory.activeMemory` read right after a load also still counts the uncut tensors, which MLX frees a moment later when the GPU finishes (about 0.2 MB of 0.5 MB on the toy), and the planner measures its per-layer weights that way, so its numbers lean high.

With a 4-bit base the sliced weights of a block are only about 0.3 GB, so residency saves at most about
1.3 GB: what is left is the block's fp32 adapters and Adam moments (0.82 GB), one layer's working memory at
T = 2700 (0.85 GB) and the uncut layer that is counted while it is cut (file-backed). Going lower needs
smaller adapters, a shorter sequence, or optimizer state in a smaller dtype, which would change results and is
not offered.

## Measured peak memory (M3 Max, fp32 toy, 2026-10-17)

Toy Gemma 4 MoE: hidden 256, 16 experts pruned to 8 (top-2, width 256), dense MLP 512, sequence
512, adapter ranks 32 / 16 / 8, blocks of two layers. `Memory.peakMemory` while one training step
(loss and gradients) runs, resident = model + adapters before the step.

| Arm | Resident before | Peak | Peak - resident |
|---|---|---|---|
| main path, block phase, 8 layers | 130 MB | 282 MB | 152 MB |
| main path, block phase, checkpointed | 130 MB | 338 MB | 208 MB |
| main path, freerun (checkpointed) | 130 MB | 400 MB | 270 MB |
| main path, end-to-end KL (checkpointed) | 130 MB | 404 MB | 274 MB |
| main path, block phase, 16 layers | 259 MB | 436 MB | 177 MB |
| streamed block phase, 8 layers (layers read from a checkpoint dir) | 0 | 188 MB | 188 MB |
| streamed block phase, 16 layers | 0 | 201 MB | 201 MB |

The streamed peak does not grow with depth (188 to 201 MB for twice the layers), while the main
path's resident set grows with it. On a toy this small, checkpointing a two-layer block costs more
than it saves; it pays off once a block holds many layers or sequences are long.

## Measured: pre-computed teacher and sliced student (Apple silicon, fp32 toy, 2026-10-17)

Toy with the real model's proportions (hidden 176 : expert width 44 : dense 132 = 2816 : 704 : 2112),
32 experts pruned to 2 (16x; the real model is 128 to 8), 12 layers in blocks of six, sequence 512,
adapter ranks 32 / 16 / 8. `StoredTeacherBenchmark`.

| | Masked, all experts (live teacher) | Sliced + stored teacher |
|---|---|---|
| one block's weights | 20.8 MB | 4.7 MB (4.5x less; the non-expert part does not shrink) |
| streamed block training, peak | 208 MB | 192 MB (the 16 MB saved is exactly the weights) |
| whole student, freerun, resident / peak | 45 / 205 MB | 14 / 174 MB |
| whole student, end (KL), resident / peak | 45 / 205 MB | 14 / 164 MB |

Activations and adapter state are identical between the forms (the peak minus resident of the
whole-student phases agrees to 1 MB); slicing removes the unkept experts and nothing else. Pre-pass,
one block's weights plus the batch (`rows per batch`: 1 / 2 / 4 / 8, 512 tokens each): 33 / 45 / 70 / 118
MB, linear in rows (the first row is a warm-up outlier). Against sequence length, one row, peak
above the weights (`chunk off` is the plain masked attention, `512` the default):

| tokens | pre-pass, off | pre-pass, 512 | sliced block training, off | training, 512 |
|---|---|---|---|---|
| 512 | 12.6 MB | 12.6 MB | 127 MB | 123 MB |
| 1024 | 34 MB | 27 MB | 317 MB | 249 MB |
| 2048 | 104 MB | 56 MB | 796 MB | 462 MB |
| 4096 | 353 MB | 113 MB | 1489 MB | 913 MB |

Without chunking the cost has a T^2 term (18 bytes per T^2 for these 4 heads: the attention scores
are materialised); with it the cost is flat per token, 28 KB in the pre-pass and 228 KB in training
(fp32 toy). Why chunking was needed although the port already calls
`MLXFast.scaledDotProductAttention`: MLX's Metal build uses the fused kernel only for head dimensions
64 to 256 and, even then, chooses the unfused path whenever a gradient is being traced and for
queries longer than 8 unless the head dimension is 192 or 256. So training always scores
`heads x L x L`, and so do the toy's head dim 32 and the real model's 512-wide global layers.
`Gemma4Attention.attendedWhole` splits the queries into blocks of `Gemma4Attention.queryChunk`
(512, 0 turns it off); a block sees only the keys it can attend to (a sliding layer costs
`L x window`) and is recomputed in the backward pass. Gemma 4 has no attention softcap and a scale
of 1, so the same SDPA call expresses it exactly; chunked equals plain to 2e-5 in fp32 (values and
gradients, tested), and `checkpointBlocks` is now on by default.

## Estimate for the real 26B-A4B (derived, not measured)

Update: one block of the real 26B has since been measured (block 0, peak process footprint 8.0 / 6.0 / 4.8 GB for the three
keeping settings; see [Benchmarks](benchmarks.md#real-gemma-4-26b-a4b-m3-max-48-gb-2026-10)). The estimate below was not revised
against it, and the rest of this section is still derived.

Config: hidden 2816, 30 layers, 128 experts (keep 8), expert width 704, dense MLP 2112, vocab
262144, 16 heads (head dim 256, 512 on global layers), sequence 2700. Parameter counts:
experts 22.84B (1.43B kept), attention 1.11B, embeddings 0.74B, dense MLP 0.54B; 25.2B total,
3.81B after pruning to 8.

Adapters at the recipe's ranks (per-expert 128 on 8 kept experts, dense 64, router 16):
324.4M + 43.0M + 1.4M = 368.8M parameters. In float32 with gradients and both AdamW moments:
1.5 GB parameters + 1.5 GB gradients + 2.9 GB moments = **5.9 GB**.

| Base weights | Main path (all phases) | Streamed block phase (one 6-layer block) |
|---|---|---|
| 4-bit, about 14 GB (13.5 GB measured) | about 14 + 5.9 + activations = **26 to 30 GB peak** | about 3 + 1.2 + activations = **10 to 13 GB** (6 to 8 GB with checkpointed blocks) |
| bf16, about 50 GB | about 50 + 5.9 + activations = **62 to 66 GB** | about 10 + 1.2 + activations = **17 to 20 GB** (13 to 15 GB checkpointed) |

Activations: one hidden state is 2700 x 2816 x 2 B = 15 MB. A block step without checkpointing
holds roughly 1 to 1.5 GB per layer for backward (the expert path gathers 21.6k token-expert rows,
and the float32 LoRA deltas add several hundred MB), so about 6 to 9 GB for six layers; the
end-to-end phases checkpoint per layer and hold about 30 hidden states plus one layer's
recomputation. The 4-bit main-path figure agrees with the recipe's own run (22 to 23 GB active,
about 30 GB peak, 13.5 GB base, sequences up to 2700); the other rows follow from the same
arithmetic. The boundary store is 6 x 15 MB = 91 MB per row on disk (about 270 GB
for 3,000 rows), not in RAM.

**With the pre-computed teacher and a sliced student** (scaling each part separately; activations
and adapter state do not change, measured above). Per layer, 4-bit with bf16 scales and biases is
about 0.56 B per weight: all 128 experts 428 MB, the 8 kept 27 MB, dense and attention about 50 MB.

| Stage | Weights resident | Peak (estimate) |
|---|---|---|
| Pre-pass, one block, 1 / 4 / 8 rows of 2700 tokens | 2.9 GB (six layers, all experts) | about 3.5 / 5.3 / 7.7 GB: 28 KB per token on the toy is 40 hidden states, about 225 KB per token in bf16 at 2816 wide, 0.6 GB per row; no T^2 term |
| Streamed block phase, sliced, checkpointed (default) | 0.46 GB (was 2.9 GB) | about 5 to 7 GB: 0.46 + 1.2 adapters and Adam + 3 to 5 activations (the toy's 228 KB per token is 330 hidden states; 5 GB at 2700 tokens if bf16 scaled directly, less because the toy computes everything in fp32). Was 10 to 13 GB unchunked and unsliced |
| Freerun / dirty / end / answerKL, sliced student | about 2.4 GB (was 13.5 GB): kept experts 0.8, attention 0.6, dense 0.6, embedding and head 0.4 | about 14 to 18 GB (was 26 to 30): 2.4 + 5.9 adapters and Adam state + 6 to 10 activations |

The whole-student phases also stop paying for a live 30-layer teacher forward at every step (read
from the store instead); that time saving is not measured here.

The 4-bit rows are estimates (the quantized loader is tested on a tiny model, not on the 26B
itself); the bf16 rows are the same arithmetic. A bf16 main path needs a Mac with well over
64 GB; the streamed block phase is the route for smaller machines.

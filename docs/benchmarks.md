# Benchmarks

[Shears docs](README.md)

Three parts: runs on the real Gemma 4 26B-A4B, then speed and checkpointing on toy models.

## Real Gemma 4 26B-A4B (M3 Max, 48 GB, 2026-10)

Setup: the real Gemma 4 26B-A4B with every layer pruned to 8 experts, healed in 6-layer blocks, on an M3 Max with 48 GB (October 2026).
One machine. The "reference recipe" is the Python/MLX recipe that Shears' healing was ported from. The toy-model sections
below remain toy-model numbers.

**Block 0 (layers 0 to 5), 50 steps, the same rows in every arm.** The three Shears arms are the three keeping settings of
`keepIntermediates` ([command line](cli.md)): `fast`, mid (a layer count), and `lowMemory`.

| | `fast` | mid | lowMemory |
|---|---|---|---|
| Peak process footprint | 8.0 GB | 6.0 GB | 4.8 GB |
| MLX peak | 6.75 GB | 4.71 GB | 3.87 GB |
| Net energy per step, against the reference recipe (powermetrics, idle subtracted) | 9% less | 4% more | 20% more |

- **Memory.** The reference recipe used about 24 GB on the same task.
- **Quality.** The same held-out loss in every arm (0.366, against 0.368 for the reference recipe).
- **Speed.** With `fast` keeping, 0.72 s/step against 0.86 s/step for the reference recipe (about 1.2x faster per step). The reference recipe
  also runs a live teacher pass (+0.35 s/step) at every step; Shears computes the teacher once beforehand, about 0.29 s/step spread over
  these 50 steps. Counted together that is about 1.0 against 1.2 s/step: about 1.2x faster end to end, not 2x. The reference recipe's
  own timed window was longer than its step times add up to (1.64 s/step); that gap is unexplained and is not counted here.

**An honest limit.** Training is not lower memory than running the pruned Q4_K_M model in llama.cpp, which takes about 3.7 to 4 GB.
The 4.8 GB footprint of the `lowMemory` arm is above that.

**Parity with the reference recipe, on the real 26B.**

- The pruned student with the reference recipe's adapters reproduces the reference outputs exactly.
- The teacher matches, except for float-noise ties in the router (experts whose router scores tie to within float noise).
- The Q4_K_M export is byte-identical to `llama-quantize` output, tensor by tensor.

### Whole-model two-task run (preliminary)

Preliminary: one run, small samples, and a result whose cause is not yet understood.

Setup: all 30 layers pruned to 8 experts and healed in 5 blocks of 100 steps each, one pass over 100 training rows per task,
for two separate tasks (a title task and a translation task). Each adapter set was scored on 20 held-out rows per set by
teacher-forced agreement with the unpruned teacher (top-1 agreement, and KL divergence where noted).

| | before healing | after healing |
|---|---|---|
| Title task, top-1 agreement with the teacher | 0.55 | 0.70 |
| Translation task, top-1 agreement with the teacher | 0.71 | 0.77 |

- Healing improves each adapter on its own task.
- No sign of overfitting: train and held-out block loss are within 1 to 4% of each other.
- **Other languages get worse with the translation adapter.** Top-1 agreement falls from 0.27 to 0.06 on Korean and from 0.17 to 0.06 on
  Spanish, and KL against the teacher rises by 9 to 14 nats on Korean, Thai, Hindi and Spanish and by 3.8 on Swahili.
- Off its task, the pruned model is far from the teacher.
- The cause of the other-language degradation is not yet understood. With 20 held-out rows per set, small differences should not
  be trusted.

## Speed

Measured on the toy block of ["Fewer layers resident"](memory.md#fewer-layers-resident-residentlayers-memorybudget-library-the-pipeline-chooses-automatically) (one process per arm, median of steps 2 to 8, machine not quiet,
about 10 to 15% noise), `StreamedBlockHealer` before this work (HEAD `c7dc766`) and after:

| toy | resident layers | s / step before | after | what changed |
|---|---|---|---|---|
| small (hidden 256, T 512) | 6 (whole block) | 0.117 | 0.076 | compiled step |
| | 3 | 0.185 | 0.106 | |
| | 1 | 0.219 | 0.113 | |
| big (hidden 512, T 1024) | 6 | 0.162 | 0.111 | |
| | 3 | 0.266 | 0.183 | |
| | 1 | 0.302 | 0.233 | |

- *Per-layer work.* The optimizer update touches only the layer it belongs to (it used to walk and rewrite the whole adapter tree
  after every layer) and evaluates only that layer's tensors; the AdamW arithmetic is one compiled kernel per tensor (equal to the
  uncompiled update to float rounding, tested); the lowest window stays loaded from one step's backward walk into the next step's
  forward; one cache clear per step.
- *Kept experts only.* `Gemma4Checkpoint.loadDecoderLayer(_:keeping:)` reads only the kept experts' rows (`pread` of each expert's
  contiguous slab of the `[E, out, in]` tensors, fused or split, quantized scales and biases included; equal to load-then-slice,
  tested for float and 4-bit) so a layer is never materialised with every expert; the teacher pass still uses the zero-copy full map.
- *Compiled layer steps* (`CompiledLayerStep.swift`, on by default, `StreamedBlockHealer.compileLayerSteps`): each layer's forward,
  the top window's forward + loss + gradient with the per-layer recompute, and each lower layer's backward are `mlx.compile`d with
  the weights and adapters as explicit inputs, sequence lengths padded to buckets (at most 25% padding) with the pin loss masked to
  the row's true length, so graphs are traced once per bucket. Equal to the uncompiled path (losses, gradients, adapters). The first
  step pays 0.1 to 0.3 s of tracing. With three layers resident the MLX peak is 10 to 20% higher than uncompiled; the pipeline's
  stacked-row (several rows) step is not compiled.
- *Shared prefix* (4 rows of 1024 tokens sharing 896, whole block resident, hidden 512, 6 layers): 0.319 s/step without, 0.227 with,
  and the teacher pass shrinks likewise.
- *Not done:* reading the loss back every step is still needed for the skipped-update rule; a background loader for the next layer.
  One resident layer remains slower than the whole block (loading a layer is about 7 ms on the big toy).

## Checkpointing

Checkpoint gradients match plain gradients to within 1e-4 (tested). How much memory
it saves depends on depth. Whole-model value+grad, peak minus resident:

| Layers | 2 | 4 | 8 | 12 |
|---|---|---|---|---|
| Plain | 298 MB | 593 MB | 1075 MB | 1373 MB |
| Checkpointed | 359 MB | 715 MB | 929 MB | 1042 MB |

Checkpointing cuts the per-layer slope from about 75 MB to about 28 MB per layer, but it adds a
large fixed overhead, so at this size it only wins past roughly 6–8 layers. The VJP makes the
recomputation `depends` on the cotangents, standing in for MLX core's dependence on the forward
outputs (the Swift `VJP` closure never sees the outputs). Removing that made no measurable
difference. Where the fixed overhead comes from is still open.

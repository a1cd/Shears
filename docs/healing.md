# Pruning and healing

[Shears docs](README.md)

`Sources/Shears/Healing/` ports a Python/MLX recipe for pruning a MoE to a fixed number of
experts per layer and healing the damage with adapters. The pruned model is exported as a
standalone dense-structure model (the same experts for every position), so a GGUF or Core ML
export is exact.

**Pruning is a mask.** The model is loaded once. The teacher is that model with all experts
routable and every adapter off; the student restricts each layer's router to its kept set and
applies adapters. Weights are sliced only at export.

| Piece | Type |
|---|---|
| Layer decomposition (attention half, routing, combine) | `HealableMoELayer`, `MoELayerParts`; `Gemma4MoEDecoderLayer` conforms. A Qwen-style MoE implements the same protocol. |
| Routing restricted to a kept set, padded to the native top-k | `ExpertSet`, `route(...)` (internal) |
| Per-expert LoRA (rank 128 on gate/up/down), dense LoRA (rank 64 on MLP and attention output), low-rank router correction (rank 16) | `ExpertLoRA`, `LoRAPair`, `RouterTweak`, `LayerAdapters`, `ModelAdapters` |
| Teacher/student forward over any layer range | `HealingNetwork` |
| Position-weighted, per-dimension-normalised hidden-state pin; KL; weight ramp | `HealingLoss` |
| Phases `block`, `freerun`, `dirty`, `end`, `answerKL`, plan strings like `block@0,freerun@600` | `PruneHealer`, `PhasePlan`, `HealingPhase` |
| Warmup + cosine schedule, AdamW with saveable state | `LearningRateSchedule`, `HealingAdamW` |
| Atomic checkpoints, resume | `Checkpointing` (used by `PruneHealer.run`) |
| Expert selection (importance by single removal, pool, greedy + swaps) | `ExpertSelector`, `SelectionItem`, `LayerSelection` |
| Merge adapters and slice to an HF-layout model | `MergeAndSliceExport` (then `GGUFExport.gemma4`) |
| Healing against the shipping format (straight-through rounding of the merged weights) | `QATConfiguration`, `QuantFormat`, `AdapterConfiguration.qat` |
| Low-memory block phase | `StreamedBlockHealer`, `LayerSource`, `BoundaryStore` |
| Teacher computed once, block by block, to disk; student holding only its kept experts | `TeacherPrepass`, `StudentForm`, `PruneHealer(student:teacher:)`, `Gemma4Checkpoint.loadModel(keeping:)` |

Pin weights default to the recipe's ramp (0.15 floor, rising to 1 over the 64 positions before
`HealingRow.answerStart`), never to ones. A precomputed selection (`KeepSets.load`, which reads
the recipe's `select-*.json` records, plain `[[ids]]` or `{"keep": ...}`) can be passed straight
to `PruneHealer(keep:)` to skip selection. `RoutingProbe` builds the recipe's selection prior
(full-model routing mass, prompt and generated positions normalised and summed).

Callers supply what is specific to their data: rows (`HealingRow` with tokens, per-position pin
weights and optional KL positions), a `HealingRowSource` (a pure function of the step, so resume
is exact), selection items with search positions and weights, and any evaluation through
`HealingCallbacks`. Memory is configured with `MemoryConfiguration` (MLX memory and cache limits,
an optional peak-memory stop) plus optional `shouldPause` / `shouldStop` hooks.

**Block phase.** Blocks are spans of decoder layers ending at the global-attention layers
(`defaultBlockEnds`); for Gemma 4 26B-A4B that is layers 5, 11, 17, 23, 29. Each step takes one
sequence, computes the teacher's state at every block end, trains every block on its
teacher-forced input, and applies one optimizer step to the gradients averaged over blocks.

**Low-memory mode.** `StreamedBlockHealer` runs the block phase with one block resident: each
block is loaded once and trained through all steps against targets read from a `BoundaryStore`
(memory or a directory). It covers **the block phase only**. With the same rows, schedule and
`1/blockCount` gradient scale, and no clipping, it matches the in-memory block phase (tested).

**The teacher is never computed during healing.** `TeacherPrepass` runs it once, block-major:
each block's teacher weights (all experts, as stored, 4-bit stays 4-bit) are loaded once, every
row is pushed through in batches, the block-end states are written to the store, and the block is
released; block `b` reads its inputs from block `b - 1`'s stored ends (boundary 0 is the
embedding). Then:

```swift
let store = try DiskBoundaryStore(directory: dir)
try TeacherPrepass(source: Gemma4LayerSource(checkpoint: cp), blockEnds: ends,
                   options: .init(storageDType: .bfloat16, maxBatchTokens: 8192))
    .run(rows: rows, store: store)          // resumable; safe to rerun

// block phase, one sliced block at a time:
var streamed = StreamedBlockHealer(source: src, keep: keep, blockEnds: ends)  // slicedStudent = true
try streamed.train(rows: source, steps: n, store: store)

// freerun / dirty / end / answerKL on a student that holds only its kept experts:
let student = try cp.loadModel(keeping: keep)          // layers cut as they load
let healer = try PruneHealer(model: student, keep: keep, student: .preSliced(numExperts: 128),
                             teacher: store,
                             teacherFingerprint: try src.teacherFingerprint())   // optional: refuses another model's store
try healer.run(rows: source, plan: plan)
```

- *Batching is exact.* Rows are sorted by length, packed under `maxBatchTokens` / `maxBatchRows`
  and right-padded; attention is causal and everything else is per token, so each stored row equals
  a row-by-row teacher (tested against it, with mixed lengths, one shorter than the sliding window).
  Padding is cut off before storing (the pin loss's scale averages over positions).
- *What is stored.* The embedding and each block's end, per row. The KL phases' teacher
  log-probabilities (full vocabulary, at the anchors and at positions the `end` phase samples
  anew every step) are a function of the last stored state, so they are recomputed from it with the
  output head each step rather than stored: exact, and a vocabulary-sized vector per position
  would dwarf the states.
- *Disk.* `sum over rows of T x D x bytes(dtype) x (blocks + 1)`
  (`TeacherPrepass.storageBytes`). 26B, bf16, T = 2700, 5 blocks: 2700 x 2816 x 2 x 6 = **91 MB per
  row**, about 270 GB for 3,000 rows. The storage dtype defaults to the model's compute dtype (bf16
  for the real model, lossless); `.float32` is the explicit option, and `.bfloat16` on an fp32 model
  rounds the targets and each next block's input.
- *Resume.* Files appear atomically; a stage is marked done in the store's manifest after its last
  row; a rerun skips finished stages and rows, and rejects a store written for different rows,
  blocks, dtype or *teacher model* instead of reusing it. The manifest holds `LayerSource.teacherFingerprint()`:
  for a checkpoint, SHA-256 over `config.json` (quantization included), the index, every shard's name, size
  and safetensors header, and the load dtype; for an in-memory model, over every parameter's name, shape
  and dtype (it cannot tell two same-shape weight sets apart). Manifests without the field are refused.
  A pre-pass also refuses a source whose layers no longer have all their experts (cut in place by a
  sliced run). `DiskBoundaryStore` names files by a hash of the row key (case-insensitive volumes), with
  a layout marker; stores of the old layout are refused.
- *Sliced student.* The student's layers are cut to the kept experts (router rows,
  `per_expert_scale`, expert bank, quantized scales and biases included); adapters keep exactly
  the format they have behind a mask (router `b` is still `[r, E]`, its kept columns are gathered),
  so checkpoints, resume and `MergeAndSliceExport` are unchanged. Losses and every adapter gradient
  equal the masked formulation's in all five phases (tested, fp32 and 4-bit, including a layer
  with fewer kept experts than top-k). A sliced student has no teacher and refuses to run one.
- *The one thing that needs a live teacher* is the dirty phase with `dirtyLambda > 0`: the
  teacher's block applied to the student's own input. It needs a `.masked` student (the teacher is
  the same weights); a sliced student refuses that plan. With `dirtyLambda = 0` (the default) the
  stored targets are all it uses.
- `.sliced` cuts the model you pass in place, so it can no longer serve as a teacher; run the
  pre-pass first, or load the student with `loadModel(keeping:)`.

**Notes on faithfulness.**
- `answerKL` freezes layers below `trainFromLayer` for the whole run, as the recipe's equivalent
  phase does, even if the plan later returns to other phases.
- The first optimizer update runs at learning rate 0 when there is a warmup: the schedule is read
  at the update count before it increments, as in `mlx.optimizers`.
- Exact reproducibility: MLX's gather-matmul gradients are not bit-reproducible, so two identical
  runs differ at about 1e-5. Tests compare at tolerances well under how far a run moves the adapters.

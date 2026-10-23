# Command line, long runs and the pipeline

[Shears docs](README.md)

## Long runs: schedule, mid-block checkpoints, stop, command line

- **Schedule.** `HealingConfiguration.scheduleLength` (default: the run's steps), `scheduleOffset` and
  `scheduleLengthChanges` (`[.init(from: position, total: length)]`) place a run on a warmup-and-cosine
  schedule of its own: continue an earlier run's schedule (offset = updates already made), restart from
  adapters with a fresh optimizer and a new schedule (offset 0), or re-length it part-way. Values are
  tested against `mlx.optimizers` (`join_schedules` of `linear_schedule` and `cosine_decay`).
- **Mid-block checkpoints.** `StreamedBlockHealer.train(..., checkpointDirectory:)` saves the block in
  progress (adapters, optimizer moments, update count, step; the row for a step is a pure function of
  the step) every `checkpointEvery` steps, atomically, and resumes there. Each block's adapters start
  from MLX's generator seeded with `configuration.seed` plus the block's first layer, so a resumed run
  initialises later blocks as an uninterrupted one does. A save interrupted between its two renames
  resumes from `ckpt.old`.
- **Stop and pause.** `HealingCallbacks.shouldStop`, `shouldPause`, `stopFile` and `pauseFile` are
  honoured between steps by `PruneHealer.run` and `StreamedBlockHealer.train` (a stop saves a checkpoint
  and deletes the stop file); `MemoryConfiguration.peakStopBytes` stops both; `TeacherPrepass.run`
  takes `shouldStop` between batches. Streamed runs return `Report.outcome`.
- **`shears-cli`.** `ShearsRunner` (in the library) drives `select`, `prepass`, `heal`, `run`, `sweep`, `export` and `gguf`
  from one JSON config (strict: unknown keys are errors; relative paths resolve against the config's
  directory). Rows come from a JSON Lines file: `key`, `tokens`, and optionally `pinWeights`,
  `answerStart`, `klAnchors`, `klSampleRange`, `searchPositions`, `searchWeights`, `validation`.
  `heal` runs `stages`, each with its own schedule, row offset, `initialAdapters` (a path or `previous`),
  fresh optimizer, checkpoints and `done.json`; a stage is `block` (streamed) or `plan` (phases on a
  student holding only its kept experts). Every command is resumable: rerun it. Progress goes to
  stderr (step, loss, lr, s/step, MLX peak and process peak memory) and to `steps.jsonl`; SIGINT, SIGTERM or the stop file
  stop at the next step with exit status 3. `Examples/tiny-config.json` is the sample, exercised by
  `RunnerTests`. Build the executable (macOS) from its own package so the library keeps one Xcode scheme:

```bash
cd CLI && xcodebuild build -scheme shears-cli -destination 'platform=macOS'
shears-cli heal --config path/to/config.json            # also: select, prepass, export, gguf; --stage NAME
shears-cli run --config path/to/config.json                       # the block-sequential pipeline (below)
shears-cli sweep --config path/to/config.json                     # learning-rate / variant grid on block 0
shears-cli sample DIR                                             # a tiny random Gemma 4 MoE (DIR/model) plus DIR/rows.jsonl; copy Examples/tiny-config.json to DIR/config.json and run the commands on it
shears-cli run --config path/to/config.json --memory-budget 8GB   # a ceiling: refused up front, with the amount, if the run needs more
```
There is no resident-layers option: training always holds as many layers as fit under the run's own memory floor (see [Auto residency](#auto-residency)).

## The block-sequential pipeline (`shears-cli run`)

`BlockPipeline` heals one block at a time, end to end. There is no full-model teacher pre-pass and no store of every
block's states: only the **frontier** (the states entering the current block, teacher-clean and healed-dirty, for every row)
is on disk, and the previous frontier is deleted once the next is written.

```
embed rows  ->  frontier 0 (clean)
for block b:
  PREPARE     (teacherPass "block", the default) teacher walks the block's layers ONE layer at a time over EVERY row of the
              block, activations in RAM, in as few walks as the state budget allows (teacherStateBudgetGB); in the first walk
              each layer's experts are SELECTED from the first batch's rows (the pruned stream's input against the all-expert
              target), and the pruned stream advances with the new keep set; teacher block ends -> frontier b+1
  for each rolling data batch k of batchSteps steps:                      <- resumable per batch
    TRAIN     steps on the batch: input = clean (teacher) and, from block 1, dirty (healed blocks below) stacked as a
              batch of two, target = the teacher's clean block end; loss = mean of the pins; one update per step
  FRONTIER    healed block over every row -> dirty states for block b+1 (+ any teacher ends not written); delete frontier b
  save block b's adapters; block b+1 starts from the frontier
```

- **Teacher pass (`teacherPass`).** `block` (default) computes the teacher's outputs for all of a block's rows before its
  training starts, so the block's full-expert layers (all 128 experts each: the largest reads of the run) are read once per
  walk, not once per rolling batch. The cost is that the hidden states of the rows in a walk are in RAM together
  (`teacherStateBudgetGB`, default 2: rows beyond it take another walk) and that training starts only after the whole
  block's teacher is done. `batch` is the old behaviour (one walk per batch, just before its steps): least RAM, but the
  block is read again at every batch boundary. Either way the numbers are the same (tested). Not done: overlapping the
  teacher walk of the next block with training. Each walk logs a line such as
  `block 2/5, all 3 batches: teacher outputs 41.3 s = read 9.8 s (1.71 GB) + first-call/compile 30.1 s (4 new kernel shapes) + compute 1.4 s; 32 rows in 1 walk, 6 layer loads`:
  *read* is the time of the layer loads and the bytes of weights loaded (on a mapped checkpoint the pages are read on first
  use, inside *compute*; the line then adds the growth of file-backed pages), *first-call/compile* is the time of the first
  evaluation of each kernel shape (rows x tokens x hidden, dtype, attention shape) this process has not run before, which
  on a backend that compiles at run time includes the compile (it also includes that call's compute), and *compute* is
  every later evaluation. The classification is a heuristic by shape; it answers "disk, compile or compute".
- **Fixed shapes (`sequenceLength`, `padWalks`).** Teacher walks and the frontier pad their batches to the rows' bucket
  length and a power-of-two row count (`padWalks`, default on); `sequenceLength: N | "auto"` runs everything (walks,
  frontier, compiled steps) at exactly one length with a fixed row count per walk, `"buckets"` keeps the fine bucket set.
  Padding is masked out (attention is causal, targets are cut to the row's length, the loss is masked), so no value changes
  beyond float rounding (tested). `auto` is the longest row rounded up to a multiple of 64 and is the default on Linux with
  a GPU, where MLX's CUDA backend compiles kernels at run time. The run ends with its count of compiled-function traces and
  distinct teacher-walk shapes. Not done: packing several rows into one sequence with a block-diagonal mask (it needs
  segment-aware attention and per-row RoPE positions in the Gemma 4 attention); padding wastes up to the gap between a row
  and the fixed length.
- **Clean + dirty (`mix`).** `both` (default) runs the row's clean and dirty input through one forward/backward
  (`[2, T, D]`), so the last updates always cover both; `clean` is the old block phase, `dirty` the healed stream only.
  Block 0's dirty input equals its clean one, so it runs a single input (same update, half the cost).
- **Rows per step.** `rowsPerStep` rows are stacked (right-padded; attention is causal, so each row's own positions are
  exact and each is scored on its own length) and the loss is the mean. `steps` counts updates.
- **Shared prefix.** Rows that begin with the same tokens (`sharedPrefix`: `auto`, `off` or a count) share that work: the
  teacher computes the prefix activations once per layer for a whole batch, and a step of several rows with the block
  resident runs the prefix once, forward and backward, the suffixes attending to the prefix's K/V
  (`healingLayerForwardShared`, `HealingNetwork.forwardShared`). Exactly equal to running each row whole (tested, gradients
  included). A windowed step runs its rows whole.
- **Selection** per block from the first batch's rows that have a selection item (`ExpertSelector.step` is the per-layer
  body of `select`; the in-pass sets equal the separate run's, tested). `select.importanceCandidates` limits the
  remove-one-expert tests to the experts with the highest routing mass (default: all, the exhaustive recipe); on a toy with
  skewed routing the pruned search picks the same sets.
- **Plateau annealing** (`pipeline.plateau`): every `evalEvery` updates the `validation` rows are scored; after `patience`
  scores that fail to beat the best by `threshold`, the cosine schedule is re-lengthened to end `annealSteps` later (the
  existing `scheduleLengthChanges`), once per block, and the trigger is logged.
- **Compute dtype.** `dtype: bfloat16` in the config loads the model in bf16 while adapters, gradients and the optimizer
  stay float32 (the LoRA products already run in float32). Off unless set; the sweep compares it (`variants[].dtype`).
- **Resumable** per selected layer, per batch, per block (`pipeline.json` in the pipeline directory), with mid-block
  checkpoints; `pipeline.directory` is where the frontier lives (about `2 x rows x tokens x hidden x bytes`; the run reports
  its size).

### Auto residency

By default **the whole block stays resident**: a pruned layer is small, and a smaller window reloads its layers every step
(per-layer overhead on top of the read). Fewer layers are held only when `--memory-budget` / `memory.budgetGB` forces it, in
this order: smaller token chunks of the expert math first (exact, no reloads; `memory.tokenChunk` sets one by hand), then
fewer resident layers (the largest window whose estimate, `ResidencyPlanner`'s model with its 1.25 margin, fits). A run whose
floor (`max(measured MLX peak of the other stages, training's one-layer estimate at the smallest chunk)`) is above the budget
is refused before training, naming the amount (`ResidencyPlanner.chooseAuto`). The choice and its reason are logged per block
and kept in the manifest (a resume picks the same one). The MLX buffer-cache limit defaults to an eighth of the estimate
(64 MB to 1 GB; `memory.cacheLimitGB` overrides) so freed buffers do not inflate the footprint, and the next step's stored
states are prefetched from disk while the current step runs. The stages before training (embedding, the residency
measurement, the teacher walk, selection) bound the cache too: sixteen batches' hidden states, 64 MB to 1 GB, never above a limit
already set. Left at MLX's default (the whole memory limit) the cache grew with every distinct buffer size the walk allocated and
held several times the bytes in use; it counts toward the process footprint, which is what showed. Two bit-identical step options
are off by default because on a toy that is dominated by activations (whole block resident, one 2,048-token row) each raised
MLX's peak by about a fifth: applying and freeing each layer's gradients one layer at a time
(`StreamedBlockHealer.freeGradientsEarly`, `BlockPipelineOptions.freeGradientsEarly`) and reading the loss after the backward is
queued (`overlapLossCheck`).

**Recompute or keep (`keepIntermediates`).** With per-layer checkpointing (the default), each layer's forward is rebuilt in the
backward pass. `keepIntermediates` (config `pipeline.keepIntermediates`, `--keep-intermediates`, `HealingConfiguration.keepIntermediates`)
chooses how many of a block's layers skip that and keep their forward intermediates: `lowMemory` (none; least memory, slowest),
`fast` (all; no recompute, most memory), or a layer count (the top layers, whose backward runs first). The arithmetic is the same
for every setting, so the compiled step's result is bit-identical across them; only memory and time move. The default `auto` is
`lowMemory` unless the run has a memory budget: then the window and token chunk are chosen first (they cost no extra memory), and
the most layers whose estimated peak still fits the budget are kept. A windowed step keeps only within its top window.
Turning `checkpointBlocks` off is `fast`.

### Merged expert layout and shared expert LoRA

When the kept count is at most `ExpertCompute.mergedMaxKeptRatio` (default 1.5) times the router's top-k, the expert branch
runs as one wide MLP: the kept experts' gate and up stacked into `hidden -> K*width` projections, each expert's hidden chunk
scaled by its routing weight (exactly 0 for an expert the token did not pick), and the down projections stacked into one
`K*width -> hidden` projection: three dense matmuls, no gather, sort or unsort. It equals the routed path in forward and
gradients (fp32 tight, bf16 loose; quantized banks use `quantizedMM`), per-expert LoRAs run batched in it with the existing
adapter format, and inference (`Gemma4MoEDecoderLayer.sparse`, prompts and batches; a single decoded token stays routed unless
kept == top-k) uses it for pruned layers. `adapters.sharedExperts: true` (a sweep variant too) trains one LoRA of rank
`expertRank` on the merged expert MLP instead of one per expert; export folds it into the per-expert weights exactly.

### Learning-rate sweep (`shears-cli sweep`)

`LearningRateSweep` runs short warmup-and-cosine healing on block 0 for a grid of peak learning rates x step counts x training
variants (`rowsPerStep`, plateau annealing, compute dtype), each from the same seeded adapters with a fresh optimizer and a
schedule exactly as long as its steps, and scores it by the pin loss on held-out rows (the `validation` rows, or
`sweep.heldOutRows`). The block's data is prepared once by the pipeline (resumable) and reused by every point and later sweep.
It prints a table (best marked, annealing trigger shown, the unhealed block as the reference) and writes the JSON
(`sweep.output`). Config: `{"sweep": {"learningRates": [...], "steps": [...], "variants": [{"name": "rows2", "rowsPerStep": 2}, ...]}}`
(see `Examples/tiny-config.json`).

The router correction is its own knob: `adapters.routerRank` (0 turns it off, whatever the other adapters are) and
`adapters.routerLRMultiplier` (scales its learning rate and, with it, its decoupled weight decay; 1 is exact). A sweep variant
can override both (`variants[].routerRank`, `variants[].routerLRMultiplier`), and each variant with a router correction also
reports `routing` on the held-out rows: `KL(original routing || corrected routing)` over the kept experts and the top-k
overlap, where the original routing is the uncorrected router applied to the student's own router input.

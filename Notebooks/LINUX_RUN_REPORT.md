# Linux run report

2026-10-22. Ubuntu 24.04.5 LTS, x86_64, 4 cores, 15 GB RAM, no GPU. Python 3.13.16.

**Result: `Shears.ipynb` runs top to bottom headless (`jupyter nbconvert --execute`) with no errors.** No library or notebook fixes were needed.

## Clean run (no Swift installed, empty work directory)

| Phase | Time |
|---|---|
| apt packages | 3.8 s (packages already present from an earlier attempt) |
| Swift 6.3.3 via swiftly | 87 s |
| mlx-swift 0.32.3 clone + patch + preamble | 7 s |
| `swift build -c release` shears-cli | 285 s |
| pipeline: sample, select, prepass, heal, export | 33 s (heal 31 s: first run, MLX compiles its CPU kernels) |
| pip install transformers + torch | 255 s (torch was not installed; Colab has it) |
| total | 671 s, about 11 min |

The export (4 of 8 experts per layer, bf16) loaded as `Gemma4ForCausalLM` in transformers 5.20.2 and generated tokens.

## Change made

The tiny run was cut to a smoke test: the notebook now writes 3 heal steps per stage (stage C plan `freerun@0,end@2`) and generates 5 tokens. `Examples/tiny-config.json` is unchanged (the tests use it). The trimmed cells were then rerun against the same build: pipeline 1.2 s, no errors.

## Not done

- The `swift test` suite was not run here, to keep this a fast check. The earlier run on the same kind of machine (docs/linux.md) had 327 passed, 35 skipped and 1 failed (`MergedExpertTests.testPipelineTrainingIsTheSameInMergedAndRoutedLayouts`, a tolerance miss).
- The EXPERIMENTAL CUDA cell was skipped (off by default). CUDA is still untested.
- Colab itself: its image, its 2 vCPUs (build time), and the Ubuntu 22.04 PPA path are untested.

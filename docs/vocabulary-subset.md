# Vocabulary subset: not keeping the 262K-row embedding table loaded

[Shears docs](README.md)

Gemma 4's embedding table has 262,144 rows of 2,816 values (0.74 billion parameters: 1.48 GB in bf16, 0.42 GB in 4-bit). It is
*tied*: the same table turns token ids into vectors going in, and scores every token as the next one coming out. In the pruned
26B-A4B it is a fifth of the weights in bf16, and every decoded token reads all of it.

`VocabularyWindow` does not load it. The table stays in the safetensors files, and the model runs on a small **window** of its rows.

- **Going in**, the rows of the tokens in the request are read from the file for that request (`pread`, only those rows).
- **Coming out**, logits are computed over the window only. Every other token's logit is `-inf`, so a token outside the window
  cannot be chosen. The result is spread back to a full-width `[.., 262144]` vector, so the sampler, the JSON constraint, stop
  tokens and log probabilities need no change.

What is in the window:

| Part | What | Rows |
|---|---|---|
| Byte-fallback tokens | `<0x00>` ... `<0xFF>`: any text can be spelled, whatever else is missing | 256 |
| Special tokens | `<eos>`, `<turn|>`, `<|channel>`, ... | 24 |
| JSON, verdict, ASCII and punctuation tokens | the pieces of `{"state": "ON_TASK", "confidence": 0.9, "reason": "..."}`, the four verdicts, digits, every printable ASCII character alone and after a space, and every token with no letter or digit in it (operators, brackets, runs of spaces and newlines) | about 4,600 (4,845 with the two rows above) |
| A broad English-and-code base | the most frequent tokens of a text corpus, half of the room given to whole English words (a token that is a dictionary word or an inflection of one) | the rest of the chosen size (a setting) |
| **Every token in the request's context** | added for each request: the prompt's tokens, plus each one's word-start variant (`Claude` and `▁Claude` are two tokens) | a few hundred to a few thousand |

The last row is the answer to "what if it says a name or a window title we never considered": names and titles nearly always
come from the screen the model is reading, and those rows are loaded for the request anyway. The static base does not need to
know them.

## Using it

```bash
# 1. Build the static list from a text corpus (files or directories) and a dictionary; --emit writes other sizes from the same pass.
shears-cli vocab build --tokenizer MODEL_DIR --corpus corpus/ --dictionary /usr/share/dict/web2 \
    --size 65536 --out static-64k.json --emit 16384:static-16k.json --emit 32768:static-32k.json
shears-cli vocab info static-64k.json --tokenizer MODEL_DIR          # sections, build summary, tokenizer match
# How much of some real answer text a list would not hold (tokenization only, no model runs):
shears-cli vocab coverage --tokenizer MODEL_DIR --prompts prompts.jsonl --vocab static-16k.json --vocab static-64k.json --show 20

# 2. Run on a window.
shears-serve serve --model MODEL_DIR --vocab static-64k.json --socket /path/to/sock --cache-limit-mb 256
```

```swift
let vocabulary = try StaticVocabulary.load(URL(fileURLWithPath: "static-64k.json"))
let runtime = try InferenceRuntime.load(directory: dir, vocabulary: vocabulary)   // no embedding table in memory
// generate(...) prepares the window for each request: it reads the rows of the prompt's tokens.
```

`Examples/vocab/gemma4-static-64k.json` is a 64K-row list for the Gemma 4 tokenizer, built as described under
[the static list](#the-static-list); `Examples/vocab/gemma4-verdict-only-515.json` is the 515 always-included tokens without the
punctuation set (for scoring a few verdict tokens, not for generating text). A list is made for one tokenizer and refused for
another (a SHA-256 over the vocabulary is stored in it). The model must have tied embeddings, which Gemma 4's do.

`--vocab` works with `serve`, `generate`, `bench`, `decode-bench` and with a 4-bit checkpoint (the rows are then read quantized and
scored with `quantizedMM`). The other commands used below: `shears-serve decode-bench` (a real prompt, memory and speed),
`ab-bench` (the same model with and without the window, alternating, in one process), `verdict-bench`, `vocab-eval`, and
`shears-serve quantize --model DIR --out DIR` (writes a plain affine 4-bit, group size 64 checkpoint, one layer in memory at a
time; this is how a 4-bit model was made to measure; it is **not** Q4_K_M and not quantization-aware).

**How the rows are read.** The static rows are read once into one page-aligned anonymous buffer that MLX wraps without copying
(`MappedSafetensors.gatherRowsInPlace`); the context rows are read the same way for every request, replacing the last request's.
A first version read through an ordinary buffer and left hundreds of MB of it in the footprint for a large list; reading in place
fixed that. Rows are not memory-mapped from the model file: a gathered subset is not contiguous there. Writing the static rows to a
file of their own and mapping it would make them clean, droppable pages; that is not done.

## What was measured

Everything in this section is **measured** unless it says **estimated** or **cited**.

**Machine and model.** M3 Max, 48 GB, macOS 27.2, Release builds. The Mac was not quiet: an unrelated llama-server doing
background work was running (load average up to 23), so speeds are taken by alternating the two arms in one process (below) and memory
by repeating each arm in a fresh process (three runs, medians shown; the runs differed by at most 0.17 GB, in the footprint columns). The model is the pruned
Gemma 4 26B-A4B, 8 experts per layer, 3.81B parameters, bf16 export (7.6 GB). **It is not the classifier's own export**, which is not on this
machine: it is the English-to-Japanese translator-healed export (`export-healed`; the unhealed twin writes only `【A】`). Memory and
speed depend on the shapes only, so they carry over; what the model writes does not (it writes malformed JSON for classifier prompts),
which is why most of the coverage evidence below does not use it. The 4-bit model was made from the same export with
`shears-serve quantize` (plain affine, group size 64, embedding included). The prompt is a 1,853-token classifier-style prompt
(system prompt, declared task, activity JSON, a screen description), 80 tokens decoded, `--cache-limit-mb 256` in every arm.

**Why the cache limit is in every arm.** MLX keeps a cache of freed buffers that, uncapped, grew to a 20.9 GB footprint against a
8.7 GB MLX peak on a first run (a Debug build). That would swamp a 1 GB saving. All arms cap it at 256 MB.

### Memory

Three numbers, because they answer different questions. *Footprint* is `phys_footprint` (Activity Monitor's Memory column) once the model
is loaded and MLX's cache is cleared; *resident set* is every page in RAM; *lifetime peak* is `/usr/bin/time -l`'s peak footprint over the
whole run, load and a 1,853-token prefill included. Decimal GB. In brackets, the change against the whole table.

**bf16** pruned model (7.6 GB export; table 1.48 GB)

| window rows | footprint, loaded | resident set, loaded | lifetime peak footprint |
|---|---|---|---|
| whole table (262,144) | 7.77 | 7.84 | 9.12 |
| 515 (verdict only) | 6.47 (-1.30) | 6.37 (-1.47) | 7.66 (-1.47) |
| 16,384 | 6.56 (-1.21) | 6.46 (-1.38) | 7.75 (-1.37) |
| 32,768 | 6.66 (-1.12) | 6.55 (-1.28) | 7.84 (-1.29) |
| 65,536 | 6.84 (-0.93) | 6.73 (-1.10) | 8.02 (-1.10) |
| 131,072 | 7.21 (-0.56) | 7.11 (-0.73) | 8.46 (-0.66) |

**4-bit** pruned model (affine g64; table 0.42 GB)

| window rows | footprint, loaded | resident set, loaded | lifetime peak footprint |
|---|---|---|---|
| whole table (262,144) | 2.30 | 2.36 | 3.88 |
| 515 (verdict only) | 2.06 (-0.24) | 1.95 (-0.41) | 3.55 (-0.34) |
| 16,384 | 2.08 (-0.22) | 1.98 (-0.38) | 3.50 (-0.39) |
| 32,768 | 2.11 (-0.19) | 2.00 (-0.36) | 3.52 (-0.36) |
| 65,536 | 2.16 (-0.14) | 2.06 (-0.31) | 3.56 (-0.32) |
| 131,072 | 2.27 (-0.03) | 2.17 (-0.19) | 3.68 (-0.21) |


- The saving is what the arithmetic says: a bf16 row is 5,632 bytes, a 4-bit row 1,584 (weight plus scales and biases), so a list of
  N rows saves (262,144 - N) x that, less the few thousand context rows. The resident-set and MLX numbers follow it closely.
- The footprint saves about 0.15 GB less than the resident set in the windowed arms. It shows up as "Owned physical footprint
  (unmapped) (graphics)" in `footprint(1)`: graphics memory MLX has freed that the system has not yet taken back. Where it comes
  from is not understood; it comes and goes between runs (0 to 160 MB).
- Peak is higher than the loaded model by prefill (activations and the KV cache). `--prefill-chunk 128` (instead of 512) lowered the 4-bit 64K
  arm's lifetime peak from 3.56 to 3.26 GB (and the whole-table arm's from 3.88 to 3.59), at about the same speed; that is unrelated to the window.
- **llama.cpp** (cited, not measured): this README reports about 3.7 to 4 GB for the pruned Q4_K_M model in llama.cpp. llama.cpp is not installed
  here and Tolby's bundled one was not touched. The 4-bit MLX model above is not Q4_K_M (4.5 bits a weight against about 4.85), so the
  comparison is only rough: the whole-table 4-bit model peaks at 3.9 GB in this setup, and a 64K window at 3.6 GB (3.3 GB with `--prefill-chunk 128`), while sitting at 2.2 GB once loaded.

### Speed

Alternating whole table / window rounds in one process (median of 9, 4-bit; 7, bf16; min to max in the raw output). Prefill is
unchanged; decode gains a little:

| | decode, whole table | decode, window | prefill |
|---|---|---|---|
| 4-bit, 16K rows | 56.8 tok/s | 59.2 (+4%) | 1,592 / 1,593 tok/s |
| 4-bit, 64K rows | 56.9 | 58.6 (+3%) | 1,591 / 1,596 |
| 4-bit, 123K rows | 56.7 | 57.8 (+2%) | 1,564 / 1,567 |
| bf16, 64K rows | 21.0 | 22.3 (+6%) | 1,359 / 1,373 |

The gain is smaller than the bytes saved would suggest (the head is a fifth of a bf16 decode's weight reads, and a 64K window cuts it by three
quarters, which would be 15%): decode here is not only bandwidth-bound. Time to first token is unchanged (1.17 s) and the first request
pays nothing visible for reading its context rows (they are a few MB).
Load time is about 1 s with or without a window.

**Verdict-only prefill** (`verdict-bench`: prefill, then the scores of just the four verdict first-tokens `ON`, `NOT`, `ID`, `NO`
instead of logits for a whole vocabulary): 1.268 s against 1.272 s (whole table, 4-bit) and 1.264 s against 1.267 s (64K window). The head is computed for
one position only, so it is a millisecond against 1.27 s of prefill: scoring four tokens saves no time. What it saves is memory: with the
515-row list the output side needs nothing, and the window is then the context's rows only.

### How often would the model choose a token outside the set?

**Evidence 1, on real replies, no model run (the main one).** `shears-cli vocab coverage` tokenizes 5,460 logged replies of Tolby's real
classifier (the 12B and the unpruned 26B-A4B, sampled at temperature 1; stratified by verdict and date; 115,855 tokens) and counts the
tokens a list would not hold, with the context of the request added: the request's own tokens (the system prompt, the declared task, the
activity JSON: 983 distinct tokens on average) and then their word-start variants. The static lists were built from public text, not from
these logs.

| list | outside the list alone | + the request's tokens | + their word-start variants | replies with any |
|---|---|---|---|---|
| 8K | 17.9% | 6.3% | 4.6% | 2,795 of 5,460 |
| 16K | 7.0% | 3.1% | 2.1% | 1,697 |
| 32K | 3.3% | 1.7% | 0.91% | 915 |
| **64K** | 0.65% | 0.20% | **0.16%** | 175 |
| 96K | 0.033% | 0.016% | 0.010% | 12 |
| 131K | 0.017% | 0.009% | 0.003% | 4 |

Two things make these numbers pessimistic: the prompts were rebuilt from the logged fields (declared task, app, window titles) and the screen
description where one was logged, and **have no OCR text** (the log keeps only its length), which is where many of the names a reply uses live. What is
left outside at 64K is mostly names that appear nowhere in the context here (`Wren`, app names, `Obsidian`, `Xcode`). And two kinds of things were added after looking at what a first list left outside, so these numbers were
seen while developing: the word-start variants (a name written `Claude` in a title and ` Claude` in a sentence are two tokens), and
credit for inflected forms of dictionary words (`aligns`, `transitioned`). No token from the logs went into any list, and the sizes, the
50% share for English words and the corpus were not tuned on them (one other share, 100%, was tried once and was no better).

**Evidence 2, a model run (weak, because the model is not the classifier's).** `shears-serve vocab-eval` runs the model with its whole
table, under the JSON-object constraint the classifier uses, greedy, and at every step checks whether the token it chose is in the list plus the
context so far; it also adds up the probability the full softmax puts outside, and the share of the top-64, top-95% sampling distribution
outside. Then it runs the window and compares the token sequences. On the 4-bit model, 40 prompts with a screen description and 40
without (64K list), and 30 without (16K list):

| | 64K list | 16K list |
|---|---|---|
| prompts, generated tokens | 80, 5,264 | 30, 1,915 |
| greedy tokens outside list + context | **0** | **0** |
| ... outside the list alone | 5 | 9 |
| probability outside the set, mean per step | 0.06% | 0.18% |
| sampled-reply chance of >= 1 token outside (top-k 64, top-p 0.95), mean | 1.1% to 2.3% | 4.0% |
| window decode equals whole-table decode (token for token) | **80 of 80** | 30 of 30 |

The model's replies are poor ("echoes the schema, then a sentence about the task"), so these show the mechanism working more than they
estimate the classifier's rate. A 515-row list, used as a generator (it was built to score verdicts), diverges at once in all 30 prompts,
as it should, mostly on runs of newline tokens: the reason all punctuation and whitespace-only tokens are now always included.

**Verdict scoring.** The model never wrote a parseable `"state"` value, so scoring four candidates **could not be compared with the model's own free
verdict** (0 of 80 comparable). What was compared: after the forced start `{"state": "`, the four candidate tokens hold 96.6% of the full softmax,
and the best of the four equals the full-vocabulary argmax in every prompt (40, 40 and 30): that only says the argmax was a candidate, which
is close to circular. Verdict accuracy against labelled data, on the classifier's own export, is **not measured**.

### The static list

Built by `shears-cli vocab build`. The shipped example (`Examples/vocab/gemma4-static-64k.json`) and the lists measured above came from, at
these sizes, public data downloaded for this purpose (about 1.2 GB compressed, a slice of each):

| Source | Used for | Slice | License |
|---|---|---|---|
| [allenai/c4](https://huggingface.co/datasets/allenai/c4), `en/c4-train.00000-of-01024` | web English | 356K documents, 774 MB of text | ODC-By 1.0 (the text is from Common Crawl) |
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-103-raw-v1` train shard 0 | encyclopedic English | 134K articles, 257 MB (lightly detokenized) | CC BY-SA 3.0 and GFDL |
| [codeparrot/codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid), shard 54 | code | 61K Python files, 597 MB | the dataset card names none; the files carry their authors' licenses. Only token counts are kept, no text |
| `/usr/share/dict/web2`, `propernames` (macOS) | word forms for the English share | 237K words | Webster's Second (public domain) and a list of first names |

Counting 1.63 GB of text took about a minute (Release build, all cores). A tenth of the files (every tenth) is held out. The share of the
held-out text's tokens a list holds: 8K 81.1%, 16K 90.5%, 32K 95.6%, 64K 98.7%, 96K 99.7%, 131K 99.9%. The always-included part is 4,845 rows (7.4% of the 64K
list); sizes below that are refused.

Choosing a size is the main knob, and the tables above are what it costs: 16K saves the most (1.4 GB of 1.48 in bf16) and leaves 2.1% of
the real replies' tokens outside; 64K saves about 1.1 GB and leaves 0.16%; 96K saves about 0.9 GB and leaves 0.01%.
A token outside the window is not an error: the model's next-best allowed token is chosen, so the cost of a miss is one different
word, not a failure. The 64K list is the default recommendation here; it is a judgment call, not a measured optimum.

## Things this changes, and what it does not do

- **Log probabilities and `top_logprobs` are normalized over the window**, not the 262,144 tokens. Anything that treats them as the
  model's full-vocabulary probabilities (Tolby's #1167 soft labels do) sees a slightly different distribution; with the mass outside at
  0.1% or less that is a small difference, but it is not the same thing.
- A constraint that allows no token the window holds ends the request with an error (`no token the constraint allows is in the vocabulary
  window`), rather than choosing token 0. The JSON-object constraint cannot do that with these lists, since every ASCII character is in them.
- Untied models (an `lm_head`) are refused. One request at a time, like the generator.
- The window is built from the model's table rows, so a model whose table was changed by healing is read as it is on disk; nothing else
  about adapters is affected.

## Not verified

- The classifier's own pruned export was not available: generation-level numbers come from another pruned export of the same shape.
  Coverage on the classifier's real replies (Evidence 1) needs no model but lacks the OCR text of the screen.
- No llama.cpp number was measured (cited above). The 4-bit model is plain affine, not Q4_K_M.
- Whether the verdicts or the coach's replies are as good with a window as without (a labelled comparison) was not measured.
- Nothing was run on Linux, CUDA, iOS or visionOS; the new code uses only calls the Linux build already uses, and CI will say.
- The unexplained 0 to 160 MB of graphics memory in the windowed footprint; mapping the static rows from a file of their own (estimated
  to save the rest of the footprint gap and make the rows droppable) was not tried.
- The server (`shears-serve serve --vocab`) was queried by hand only: a JSON-object request with `logprobs`, and a streamed one, both
  answered correctly. There is no automated server test under a window.

# Data for the Shears notebook's translator demo

`Notebooks/Shears.ipynb` (mode `gemma`) heals the pruned model to translate English into Japanese and checks it against the real Gemma 4 26B-A4B. These files are what the check uses. The notebook reads them from its copy of the sources.

| File | What it is |
|---|---|
| `translate-system-prompt-ja.txt` | The system prompt (1,358 bytes, no trailing newline): the "always translate" variant A of the red-teamed prompt (v5), with the language slot filled as "Japanese". Training rows, the saved Gemma translations and every check use it verbatim as the system message. |
| `gemma-en-ja-200.jsonl` | 200 English segments with Gemma 4 26B-A4B's Japanese translation of each, plus the human reference. About 120 KB. |
| `make_gemma_en_ja.py` | The script that picked the 200 segments and made the translations. It also holds the filter the notebook uses to pick its training sentences. |

## `gemma-en-ja-200.jsonl`

One JSON object per line: `id`, `domain`, `document_id`, `segment_id` (these identify the WMT24++ segment), `english` (the source), `gemma_ja` (Gemma's translation), `wmt_ref_ja` (the WMT24++ human reference, the post-edited `target`), `model`, `provider`, `date`.

- **English and `wmt_ref_ja`**: [google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp), config `en-ja_JP`, revision `fd7405c06494bc66a57b25f55d217a72f96e60dc`, Apache 2.0. Deutsch et al., "WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects", 2025, [arXiv:2502.12404](https://arxiv.org/abs/2502.12404). The canary row and the `is_bad_source` rows are dropped. Segments are 4 to 40 words, without URLs, and without a list of offensive, political or violent words (`AVOID` in the script); 121 social, 58 literary, 16 news, 5 speech (the speech domain has few segments that pass).
- **`gemma_ja`**: `google/gemma-4-26b-a4b-it` on [OpenRouter](https://openrouter.ai), served by the provider NextBit (the reply names it; the script fixes it after the first call), on 2026-10-22. The system message is `translate-system-prompt-ja.txt`, the user message is the English, temperature 0, reasoning off. One call per segment, no retries; the run stopped on any unusable reply, and none occurred. Cost $0.0047. Gemma 4 is released under the Apache 2.0 license.
- **Full precision versus 4-bit.** OpenRouter serves the full-precision model. The model the notebook prunes, and heals against, is the 4-bit QAT checkpoint (`mlx-community/gemma-4-26B-A4B-it-qat-4bit`), so its translations can differ slightly from these.
- **Not for training.** The notebook leaves these 200 segments out of its training rows (it picks its training sentences from the other segments of the same WMT24++ file, with the same filter, and uses the human reference as the reply text).

Rebuild (needs an OpenRouter key; the script reads it from `OR_KEY`): `OR_KEY=... python3 -I make_gemma_en_ja.py`. `--dry-run` only lists what it would translate.

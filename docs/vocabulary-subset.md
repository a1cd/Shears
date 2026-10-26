# Vocabulary subset: not keeping the 262K-row embedding table loaded

[Shears docs](README.md)

Work in progress (draft PR). The plan, to be replaced by the write-up when the measurements are in:

1. **Input side:** keep the full embedding table on disk and read only the rows of the tokens in the current context.
2. **Output side:** compute logits only over an allowed set: a static English/code base, JSON, verdict and special tokens,
   the 256 byte-fallback tokens, and every token in the current context.
3. **Tools:** `shears-cli vocab`, to build the static base from a corpus, at a configurable size.
4. **Measured:** resident and peak memory and tokens/s with and without, on the pruned Gemma 4 26B-A4B export.

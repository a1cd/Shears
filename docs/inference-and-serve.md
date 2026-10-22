# Inference and `shears-serve`

[Shears docs](README.md)

`Sources/Shears/Inference` runs a Gemma 4 model (a Hugging Face directory, or a healed model still in memory) with no
llama.cpp:

```swift
let runtime = try InferenceRuntime.load(directory: dir)               // bf16, a Shears export, or a 4-bit QAT checkpoint
let prompt = try runtime.promptTokens(messages: [["role": "user", "content": "hello"]])
let result = try runtime.generator.generate(runtime.request(promptTokens: prompt, maxTokens: 256, logprobs: 5)) { event in
    print(event.text, terminator: ""); return true                    // stream; return false to stop
}
// A healed model with its adapters left unmerged (masked or sliced):
let model = try InferenceModel(model: healedModel, overlay: AdapterOverlay(healer: healer))
```

- **Tokenizer** (`BPETokenizer`): reads `tokenizer.json` itself rather than through swift-transformers, which would put
  its hub client, networking and crypto dependencies in an app, and keys the vocabulary by UTF-8 bytes (a Swift
  `[String: Int]` merges tokens that differ only in Unicode normalization). It covers byte-fallback BPE with `Replace`
  normalizers, `Split` pre-tokenizers, added tokens and the `ByteFallback` decoder; anything else throws. Encoding matches
  Hugging Face `tokenizers` on the real Gemma 4 `tokenizer.json` (checked when a copy is in the local Hugging Face cache).
- **Chat template** (`ChatTemplate`): the model's `chat_template.jinja` rendered with
  [swift-jinja](https://github.com/huggingface/swift-jinja) (the package's one new dependency). swift-jinja 2.5 does not
  apply `{{-` / `-}}` whitespace control next to braces, so the source is normalized first; the output then equals Python
  jinja2's byte for byte on system prompts, thinking, content parts, tool declarations, calls and replies (tested).
- **Decode**: chunked prefill, then one token at a time over a KV cache (sliding layers mask to the window, so they are
  right past it); the previous request's cache is reused for a shared prompt prefix. Sampling is greedy, or temperature,
  top-k and top-p with a seed, defaulting to the model's `generation_config.json`. Stops on EOS ids and stop strings;
  log probabilities (chosen token and top alternatives) per generated token. Sliding layers keep every key, which is
  correct and costs memory on very long contexts.
- **Unmerged adapters**: the layer forward is the healer's own routing and expert code over the cache, so the output equals
  the healer's student and, to rounding, the merge-and-slice export (both tested).

`shears-serve` (`Server/`, macOS, built like `shears-cli`):

```bash
cd Server && xcodebuild build -scheme shears-serve -destination 'platform=macOS'
shears-serve serve --model DIR --socket /path/to/sock        # or --port 8080 [--host 127.0.0.1]; --alias, --dtype, --max-tokens
shears-serve bench --model DIR --prompt-tokens 512 --decode-tokens 128 --repeats 3 [--json]
```

It answers `GET /health`, `GET /props` (with `model_path`), `GET /v1/models` and `POST /v1/chat/completions` (also
`POST /apply-template`), shaped after `llama-server`: `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`stream` (SSE, with `stream_options.include_usage`), `logprobs` / `top_logprobs`, `tools` (rendered into the prompt; a
call the model writes comes back as content), `chat_template_kwargs`, and `response_format: {"type": "json_object"}`,
which is enforced token by token. A reasoning section (`<|channel>thought ... <channel|>`) comes back as
`reasoning_content`. Unknown request fields are ignored; `n` other than 1, `response_format` `json_schema` and `grammar`
get a 400. One generation runs at a time; others queue, and `/health` answers meanwhile. `bench` reports prefill and decode
tokens per second, time to first token, peak MLX memory and peak physical footprint.

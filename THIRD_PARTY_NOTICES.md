# Third-party notices

## mlx-swift-lm

The Gemma 4 model code in `Sources/Shears/Models/Gemma4/` is ported from
[ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
(`Libraries/MLXLLM/Models/Gemma4Text.swift`, `Libraries/MLXLLM/Models/Gemma4.swift`,
`Libraries/MLXLMCommon/RoPEUtils.swift`, `Libraries/MLXLMCommon/KVCache.swift`), which is
distributed under the MIT License:

```
MIT License

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mlx-swift-lm (inference)

The generation loop, prefill chunking, KV-cache handling and sampling chain in
`Sources/Shears/Inference/` (`Generator.swift`, `InferenceModel.swift`, `Sampling.swift`) follow
[ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
(`Libraries/MLXLMCommon/Evaluate.swift`, `KVCache.swift`), distributed under the MIT License
(Copyright (c) 2024 ml-explore; the license text is above).

## swift-jinja

Chat templates are rendered with [huggingface/swift-jinja](https://github.com/huggingface/swift-jinja)
(MIT License), added as a package dependency; no source is copied.

## Gemma 4 chat template (test fixture)

`Tests/ShearsTests/InferenceTemplateFixtures.swift` embeds the Gemma 4 instruct `chat_template.jinja`
(Apache License 2.0, Google) so the template tests need no download.

## llama.cpp / ggml

The K-quant and Q5_0 quantizers in `Sources/Shears/GGUF/GGUFKQuants.swift` are reproduced from
`ggml/src/ggml-quants.c`, and the Q4_K_M per-tensor type choice in `Sources/Shears/GGUF/GGUFQuantMix.swift`
from `src/llama-quant.cpp`, both in [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
(commit a3a1c47), which is distributed under the MIT License:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

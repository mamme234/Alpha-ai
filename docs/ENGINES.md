# AlphaAI Engines

An **engine** is a real local runtime that AlphaAI can drive to generate text.
AlphaAI owns every engine listed here. The **model** an engine runs may belong to
someone else — DeepSeek, Alibaba, Meta, Mistral AI, Google or Moonshot AI — and
the engine reports that owner honestly (see [`../ATTRIBUTION.md`](../ATTRIBUTION.md)).

## Registered engines

| Engine key | Class | Runtime | Notes |
| ---------- | ----- | ------- | ----- |
| `deepseek` | `DeepSeekEngine` | vendored DeepSeek-V3 reference implementation (torch) | FP8 (triton) or BF16; CUDA required for real generation |
| `transformers` | `TransformersEngine` | Hugging Face `transformers` | generic safetensors path |
| `qwen` | `QwenEngine` | `transformers` | Qwen family defaults |
| `kimi` | `KimiEngine` | `transformers` | Kimi family defaults |
| `llama` | `LlamaEngine` | `transformers` | Llama family defaults |
| `mistral` | `MistralEngine` | `transformers` | Mistral family defaults |
| `gemma` | `GemmaEngine` | `transformers` | Gemma family defaults |
| `alphaai` | `AlphaAIEngine` | `transformers` | for future AlphaAI-trained checkpoints (none trained yet) |
| `llama_cpp` | `LlamaCppEngine` | `llama-cpp-python` | GGUF quantised models |
| `qwen_gguf` | `QwenGgufEngine` | `llama-cpp-python` | GGUF, Qwen defaults |
| `deepseek_gguf` | `DeepSeekGgufEngine` | `llama-cpp-python` | GGUF, DeepSeek defaults |

The mapping from key to class lives in `alphaai/engines/__init__.py`
(`ENGINE_CLASSES`). Classes are imported lazily, so a missing runtime never
breaks `alphaai version`.

## Models

Model metadata lives in `configs/models/*.json`. Each entry records the engine,
the **model owner**, the license, capabilities, context length and (optionally) a
weights URL. AlphaAI registers the models it knows about but only reports a model
as **usable** when its runtime, weights and hardware are actually present.

```shell
alphaai models list          # every registered model + honest availability
alphaai models show deepseek-v3
alphaai models check deepseek-v3
```

## Running DeepSeek-V3

DeepSeek-V3 has 671B total parameters (37B active) — roughly **685 GB** of
weights. It needs a multi-GPU machine with the FP8 kernels (triton) or a BF16
conversion. On a laptop or a CPU-only box it cannot be loaded, and AlphaAI says
so instead of pretending:

```
engine_unavailable — DeepSeek-V3 weights are not present / hardware insufficient
```

Real steps on adequate hardware:

1. **Get the weights.** Download DeepSeek-V3 from DeepSeek and accept the
   DeepSeek Model License (`LICENSE-MODEL`) first.
2. **Convert** to the reference format:

   ```shell
   python inference/convert.py \
     --hf-ckpt-path /path/to/DeepSeek-V3 \
     --save-path models/deepseek-v3 \
     --n-experts 256 --model-parallel 16
   ```

   (BF16 instead? `python inference/fp8_cast_bf16.py --input-fp8-hf-path ... --output-bf16-hf-path ...`)
3. **Check** what AlphaAI sees:

   ```shell
   alphaai models check deepseek-v3
   ```
4. **Generate** through AlphaAI:

   ```shell
   alphaai infer deepseek-v3 "Explain multi-head latent attention." --max-tokens 200
   ```

The engine refuses to emulate DeepSeek-V3 on CPU. If you want a model that runs
on modest hardware, use a GGUF build of a smaller model through the `llama_cpp`
engine.

## Running a small model locally (GGUF)

This is how AlphaAI talks on a normal machine. One command does the whole job:

```shell
alphaai doctor                                       # hardware + recommended size
alphaai models install qwen2.5-0.5b-instruct-gguf     # install, verify, load test
alphaai chat "Hello AlphaAI"                          # streaming real inference
alphaai chat --no-stream "What is 1234 × 5678?"       # calculator tool + model phrasing
```

What `models install` does, in order:

1. reads `configs/models/<id>.json` (source repo, filename, format, quantisation,
   declared size and SHA-256);
2. prints **Model / Size / Parameters / Quantization / Runtime / Required RAM /
   Available RAM / Required storage / Available storage** and stops with a
   smaller-model recommendation if the model does not fit (`--dry-run` prints
   only this report);
3. downloads the file from the authoritative repository (`--force` overrides a
   failing hardware check; AlphaAI never downloads DeepSeek-V3 weights here);
4. verifies size, SHA-256 and the `GGUF` magic bytes;
5. writes provenance (source, revision, format, quantisation, size, checksum,
   runtime, install time) back into `configs/models/<id>.json`;
6. detects the runtime for that format;
7. **loads the model and generates text** — a model is only reported `AVAILABLE`
   after it has really produced a token.

Measured on a 1-core, 1.9 GB CPU-only machine with
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 MB): load ~1 s, ~9 tokens/s decode,
first streamed token ~0.3–2 s. `llama-cpp-python` from the official CPU wheel
(`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`) is the
practical installation path there.

Generation goes through `llama_cpp.Llama.create_chat_completion`, so the chat
template embedded in the GGUF is applied. AlphaAI's tool schemas are appended to
the system message (the same mechanism every AlphaAI engine uses) and only the
tool categories relevant to the request are offered — prompt size dominates
latency on CPU-only machines.

Weights are always attributed to their owner. For this model AlphaAI reports
`created by Alibaba`; AlphaAI owns the engine integration and nothing else.

## Availability and honesty

`ModelEngine.health()` reports one of: `ready`, `available`, `disabled`,
`unavailable`. When an engine is unavailable it also reports **why**
(missing runtime, missing weights, insufficient hardware) and a remediation.
`alphaai doctor` prints the whole picture, including which optional runtimes
(`torch`, `transformers`, `llama_cpp`, `triton`, `safetensors`) are installed.

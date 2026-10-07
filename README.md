<!-- AlphaAI — rebranded system layer over the DeepSeek-V3 reference implementation. -->

<div align="center">

# ALPHA AI

### Intelligence, built from the ground up.

AlphaAI is an open **AI system layer**: a model router over local inference
engines, a real tool and skill runtime, persistent memory, an agent
orchestrator, and a training foundation for future AlphaAI-owned weights.

`AlphaAI powered by DeepSeek-V3`

[Architecture](docs/ARCHITECTURE.md) · [Engines](docs/ENGINES.md) ·
[Tools](docs/TOOLS.md) · [Skills](docs/SKILLS.md) · [Training](docs/TRAINING.md) ·
[API](docs/API.md)

</div>

---

## What AlphaAI is

AlphaAI is the **engine and system layer** around local language models. It
ships:

* **AlphaAI Core** — a model registry and router that picks the best available
  engine for each task, a conversation engine, a context manager, persistent
  memory and an agent orchestrator.
* **An engine layer** — real runtimes for DeepSeek-V3, Hugging Face
  `transformers` models (Qwen, Kimi, Llama, Mistral, Gemma) and GGUF models
  through `llama-cpp-python`.
* **A tool engine** — permission-checked, sandboxed, timeout-bounded tools with
  an audit log.
* **A skill runtime** — twelve real skills (calculator, reasoning, data
  analysis, text, summarization, translation, files, documents, code, tool
  calling, web research, agent workflow).
* **A training foundation** — datasets, tokenizer, checkpoints and evaluation so
  AlphaAI can eventually own its own weights.

AlphaAI does **not** call any external inference API (no OpenAI, Anthropic,
Gemini, or hosted endpoint), requires no API keys, and never fabricates a
result. When something is missing — no engine, no weights, no permission, not
enough hardware — it returns a structured error with a remediation instead of a
plausible-looking answer.

## Attribution — please read

AlphaAI is an independent project and is **not** affiliated with DeepSeek.

* AlphaAI owns the system layer: router, engines, conversation engine, context
  manager, tool engine, skills, memory, orchestrator, API, CLI and training
  foundation.
* The current flagship engine, **`AlphaAI DeepSeek Engine`**, runs
  **DeepSeek-V3** — a model **created by DeepSeek**, not by AlphaAI, and **not an
  AlphaAI-trained model**.
* The DeepSeek-V3 reference implementation is vendored unchanged in
  [`inference/`](inference) and remains **Copyright (c) 2023 DeepSeek**, under
  the MIT License (see [`LICENSE-CODE`](LICENSE-CODE)).
* DeepSeek-V3 model weights are released by DeepSeek under the **DeepSeek Model
  License** (see [`LICENSE-MODEL`](LICENSE-MODEL)). Accept that license before
  downloading or using the weights.

See [`NOTICE`](NOTICE) and [`ATTRIBUTION.md`](ATTRIBUTION.md) for the full
statement. AlphaAI never claims to have created DeepSeek-V3.

## Quick start

```shell
# 1. install AlphaAI Core + the API
pip install -e .

# 2. see what AlphaAI and this machine can actually do
alphaai version
alphaai doctor                # measured hardware + recommended model size
alphaai models list          # every registered model + honest availability
alphaai skills list
alphaai tools list

# 3. install the lightweight open-weight model and talk to it (no API keys)
alphaai models install qwen2.5-0.5b-instruct-gguf
alphaai chat "Hello AlphaAI. Tell me what you are."

# 4. run the HTTP API + dashboard
alphaai serve --host 0.0.0.0 --port 8000
```

`alphaai models list` reports every engine as `available` / `unavailable` with a
reason. On a machine without weights, they are honestly `unavailable`:

```
no_suitable_model — No available AlphaAI engine can handle a 'chat' request.
                    Put weights under models/<id>/ and run `alphaai models check`.
```

That is the intended behaviour, not a bug — and `alphaai models install` is how
you fix it for real.

## The lightweight local model (real inference)

AlphaAI talks through a **real, open-weight instruct model running locally** —
no external inference API, no API key, no Ollama, no canned replies. The default
model is:

| | |
|---|---|
| Model | **Qwen2.5-0.5B-Instruct** (created by Alibaba, Apache-2.0) |
| Build | `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491 MB, SHA-256 verified |
| Source | `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF` (official Qwen repo) |
| Runtime | `llama.cpp` via `llama-cpp-python` (CPU) |
| AlphaAI role | router, engine integration, context, memory, tools, API, CLI |

```shell
# what can this machine hold?
alphaai doctor

# hardware + source + size check, download, checksum, register, real load test
alphaai models install qwen2.5-0.5b-instruct-gguf

# real generation through AlphaAI Core
alphaai chat --no-stream "Hello AlphaAI. Tell me what you are."
alphaai chat "Explain what Python is in simple terms."          # streaming
alphaai chat "What is 1234 × 5678?"                              # calculator tool
```

Installation is never silent: `alphaai models install` prints model, size,
parameters, quantisation, runtime, required vs available RAM and required vs
available storage **before** downloading anything, stops with a smaller-model
recommendation when the model will not fit, verifies the file (size, SHA-256,
GGUF magic bytes), records provenance into `configs/models/<id>.json`, and only
reports the model as `AVAILABLE` after it has **loaded and generated text**.
A model is never marked available because files exist.

Every answer names the model that produced it:

```
$ alphaai chat --no-stream "Hello AlphaAI. Tell me what you are."
Model: Qwen2.5-0.5B-Instruct-GGUF
Runtime: llama_cpp/cpu
Mode: Local
AlphaAI powered by Qwen2.5-0.5B-Instruct-GGUF
Engine: qwen2.5-0.5b-instruct-gguf · AlphaAI llama.cpp Engine (…) — underlying model: Qwen2.5-0.5B-Instruct-GGUF (created by Alibaba)
```

### AlphaAI's own model is *not* trained yet

AlphaAI is the **system and engine layer**; the weights above were trained by
Alibaba. There are **no AlphaAI-trained weights** in this repository — the
`alphaai-x` model stays registered as `not_trained`. AlphaAI never claims to have
created Qwen, DeepSeek, Llama, Gemma, Mistral or Kimi.

## Running a model manually

**Any GGUF model (works on CPU):** place a `.gguf` file under
`models/<model-id>/` and go:

```shell
alphaai models check qwen2.5-0.5b-instruct-gguf
alphaai chat "hello" --engine qwen2.5-0.5b-instruct-gguf
```

**DeepSeek-V3 (needs a multi-GPU machine):** 671B total / 37B active parameters,
roughly 685 GB of weights. On adequate hardware:

```shell
python inference/convert.py \
  --hf-ckpt-path /path/to/DeepSeek-V3 \
  --save-path models/deepseek-v3 \
  --n-experts 256 --model-parallel 16

alphaai models check deepseek-v3
alphaai infer deepseek-v3 "Explain multi-head latent attention." --max-tokens 200
```

AlphaAI never emulates DeepSeek-V3 on CPU. Full details, including the BF16
conversion path, are in [`docs/ENGINES.md`](docs/ENGINES.md).

## Training foundation

AlphaAI can train its own small models end to end on CPU to prove the pipeline is
real:

```shell
alphaai train validate  --dataset alphaai-sample
alphaai train prepare   --dataset alphaai-sample
alphaai train tokenize  --dataset alphaai-sample --vocab-size 4096
alphaai train finetune  --dataset alphaai-sample --max-steps 30
alphaai train evaluate  --dataset alphaai-sample
```

Loss and perplexity are measured, never invented. No AlphaAI weights exist yet;
the `alphaai-x` model is registered as `not_trained`. See
[`docs/TRAINING.md`](docs/TRAINING.md).

## Repository layout

```
alphaai/            AlphaAI system layer (core, engines, tools, skills, api, cli)
configs/            configuration + model metadata
docs/               documentation
datasets/ tokenizer/ checkpoints/ training/ evaluation/
scripts/            the training foundation
tests/              pytest suite
inference/          DeepSeek-V3 reference implementation (vendored — DeepSeek, MIT)
```

## Using the API

```shell
curl -s localhost:8000/api/health
curl -s localhost:8000/api/models
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message": "hello"}'
```

Full endpoint list in [`docs/API.md`](docs/API.md).

## Testing

```shell
python -m pytest
```

The suite uses a deterministic test engine (`tests/conftest.py`) so the router,
conversation engine, tools, skills, orchestrator, API, CLI and training
foundation are exercised without weights. Nothing in `alphaai` itself falls back
to a fake engine.

## License

* **AlphaAI system layer** (`alphaai/`, `docs/`, `configs/`, training
  foundation): MIT.
* **DeepSeek-V3 code** (`inference/`): MIT — [`LICENSE-CODE`](LICENSE-CODE),
  Copyright (c) 2023 DeepSeek.
* **DeepSeek-V3 model weights**: the DeepSeek Model License —
  [`LICENSE-MODEL`](LICENSE-MODEL). DeepSeek-V3 series (Base and Chat) supports
  commercial use.
* Other model families remain under their own licenses, recorded per model in
  `configs/models/*.json`.

## Citation

DeepSeek-V3, the model the current AlphaAI DeepSeek engine runs, is described in:

```bibtex
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

## Related reading

* [`README_WEIGHTS.md`](README_WEIGHTS.md) — the DeepSeek-V3 main-model and
  Multi-Token-Prediction weight layout (DeepSeek).
* [`docs/ENGINES.md`](docs/ENGINES.md) — every AlphaAI engine and how to run it.
* [`ATTRIBUTION.md`](ATTRIBUTION.md) — the AlphaAI / DeepSeek ownership split.

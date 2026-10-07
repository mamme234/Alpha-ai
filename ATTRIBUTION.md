# Attribution

AlphaAI is a **system layer**, not a model. This document records exactly what
AlphaAI owns, what it does not own, and how the underlying models must be
attributed wherever AlphaAI exposes them.

## The one rule

> AlphaAI owns the **engine and system layer**. The **model owner** trained the
> weights. AlphaAI never claims to have created a model it did not train.

## What AlphaAI owns

AlphaAI is the independent system layer around local inference:

* the model registry and **model router**,
* the **engine layer** (device/format selection, resource assessment, prompt
  rendering, streaming, usage accounting, error taxonomy),
* the **conversation engine** and context manager,
* the **tool engine** and the **skill runtime**,
* persistent **memory** and the **agent orchestrator**,
* the **FastAPI server** and the **CLI**,
* the **training foundation** (datasets, tokenizer, checkpoints, evaluation) for
  future AlphaAI-owned weights.

Copyright: the AlphaAI project. License: MIT (`LICENSE-CODE`).

## What AlphaAI does not own

### DeepSeek-V3

The `AlphaAI DeepSeek Engine` runs **DeepSeek-V3**, which is created and released
by **DeepSeek**.

* DeepSeek-V3 is **not** an AlphaAI-trained model. It was not created by AlphaAI
  and is not owned by AlphaAI.
* The DeepSeek-V3 reference implementation is vendored unchanged in `inference/`:
  **Copyright (c) 2023 DeepSeek**, MIT License (`LICENSE-CODE`).
* DeepSeek-V3 model weights are released by DeepSeek under the **DeepSeek Model
  License** (`LICENSE-MODEL`). Accept that license before downloading or using
  the weights.
* Technical report: *DeepSeek-V3 Technical Report*, DeepSeek-AI, 2024,
  **arXiv:2412.19437** — <https://arxiv.org/abs/2412.19437>.

Every user-visible surface that names DeepSeek-V3 must present it as the
underlying model of the AlphaAI DeepSeek engine, for example:

```
AlphaAI DeepSeek Engine — underlying model: DeepSeek-V3 (created by DeepSeek; not an AlphaAI-trained model)
```

### Other model families

| Model family | Owner        | Governed by                    |
| ------------ | ------------ | ------------------------------ |
| DeepSeek-V3  | DeepSeek     | DeepSeek Model License         |
| Qwen2.5      | Alibaba      | Qwen License                   |
| Kimi-K2      | Moonshot AI  | Modified MIT                   |
| Llama 3.1    | Meta         | Llama 3.1 Community License    |
| Mistral      | Mistral AI   | Apache-2.0                     |
| Gemma 2      | Google       | Gemma Terms of Use             |

Each entry is recorded per model in `configs/models/*.json` (`model_owner`,
`license`, `license_url`).

## Future AlphaAI weights

No AlphaAI-trained weights exist yet. When they do, the `alphaai` engine and the
AlphaAI-X model card will carry `model_owner: "AlphaAI"`, and AlphaAI will then
be able to say "owned by AlphaAI" for that model only.

## Machine-readable attribution

The strings that enforce this policy live in `alphaai/branding.py`. They are
surfaced through the CLI (`alphaai attribution`), the API
(`GET /api/attribution`) and the dashboard footer, and they are checked by
`tests/test_branding.py`.

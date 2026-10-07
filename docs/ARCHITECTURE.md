# AlphaAI Architecture

> **ALPHA AI** — *Intelligence, built from the ground up.*
>
> AlphaAI is an independent system layer. Its current flagship engine runs
> **DeepSeek-V3**, a model created and released by DeepSeek (see
> [`../ATTRIBUTION.md`](../ATTRIBUTION.md)). AlphaAI does not own DeepSeek-V3.

## Layers

```
                 ┌───────────────────────────────────────────────┐
   surfaces      │  CLI (alphaai)      FastAPI (/api)   dashboard │
                 └───────────────────────────────────────────────┘
                                        │
                 ┌───────────────────────────────────────────────┐
   system        │  AlphaRuntime — creates and wires everything   │
   layer         ├───────────────────────────────────────────────┤
                 │  Orchestrator │ Conversation │ Memory │ Skills │
                 │  Context      │ Tool engine  │ Router │ Registry│
                 └───────────────────────────────────────────────┘
                                        │
                 ┌───────────────────────────────────────────────┐
   engine        │  ModelEngine implementations (real runtimes)   │
   layer         │  deepseek · transformers · llama_cpp · family  │
                 └───────────────────────────────────────────────┘
                                        │
                 ┌───────────────────────────────────────────────┐
   weights       │  models/<id>/  (safetensors or gguf)           │
                 └───────────────────────────────────────────────┘
```

Everything above the weights is AlphaAI code. Nothing in AlphaAI fabricates a
result: when a capability is missing — no engine, no weights, no permission, not
enough hardware — the system returns a **structured error** with a remediation
hint instead of a plausible-looking answer.

## Core components

| Component | Module | Responsibility |
| --------- | ------ | -------------- |
| `AlphaRuntime` | `alphaai/core/runtime.py` | One object that owns the whole system: creates the registry, router, engines, tools, skills, memory, conversation engine and orchestrator, and exposes the API surface the CLI and server call. |
| `ModelRegistry` | `alphaai/core/registry.py` | Loads `configs/models/*.json`, instantiates the right engine per model, and answers `usable()` / `health()`. |
| `ModelRouter` | `alphaai/core/router.py` | Classifies a request (coding, mathematics, reasoning, summarization, translation, tool-calling, …) and picks the best **available** engine, or raises `NoSuitableModelError` with the candidates it considered. |
| `ConversationEngine` | `alphaai/core/conversation.py` | Real chat loop: routing → prompt assembly → engine call → tool loop → memory write. Both one-shot `chat()` and streaming `stream()`. |
| `ContextManager` | `alphaai/core/context.py` | Trims history to the model's context length with a deterministic budget. |
| `MemoryStore` | `alphaai/core/memory.py` | Persistent (SQLite) key/value memory with namespaces and search. |
| `ToolExecutor` | `alphaai/core/tools/executor.py` | Permission-checked, sandboxed, timeout-bounded tool calls with an audit log. |
| `SkillManager` | `alphaai/core/skills/manager.py` | Runs skills with schema validation, availability checks and timeouts. |
| `Orchestrator` | `alphaai/core/orchestrator.py` | Declared plans (`run_plan`) and goal-driven loops (`run_goal`) over skills, tools and the model. |
| Error taxonomy | `alphaai/core/errors.py` | Every failure has a stable `code`, a message, an optional `remediation` and `details`. |

## Engines

An **engine** is a real local runtime that can load weights and generate text.
`alphaai/engines/` ships these (see [`ENGINES.md`](ENGINES.md)):

* `deepseek` — the vendored DeepSeek-V3 reference implementation (FP8 / BF16),
* `transformers` — Hugging Face `transformers`,
* `qwen`, `kimi`, `llama`, `mistral`, `gemma`, `alphaai` — the same
  transformers path with family defaults,
* `llama_cpp` — GGUF through `llama-cpp-python`,
* `qwen_gguf`, `deepseek_gguf` — GGUF with family defaults.

Heavy runtimes (`torch`, `transformers`, `llama_cpp`) are imported lazily, so
`alphaai version` and `alphaai doctor` stay fast even when nothing is installed.

## Request lifecycle (`POST /api/chat`)

0. **Model** — `alphaai models install` put a real open-weight GGUF model on disk
   (source, size and SHA-256 recorded in `configs/models/<id>.json`) and proved it
   loads and generates. The engine layer is `llama.cpp`; the weights belong to
   their publisher, and AlphaAI names them.
1. **Route** — the router classifies the task and selects an engine (or raises
   `no_suitable_model`). `configs/alphaai.toml → routing.task_preferences` makes
   the lightweight local model the default for the tasks it can serve.
2. **Assemble** — the context manager trims history; the engine renders the chat
   prompt (the template embedded in the GGUF, plus the tool schemas the tool
   policy considers relevant to this request).
3. **Generate** — the engine produces real tokens; usage is counted.
4. **Tool loop** — if the model emits a tool call and tools are enabled, the tool
   engine checks permission, executes, and feeds the result back; the loop is
   bounded.
5. **Persist** — the turn is written to conversation history and memory.
6. **Return** — a `ChatOutcome` (data only). The API adds the `ok` envelope.

## Configuration

`configs/alphaai.toml` plus `configs/runtime/*` and `configs/models/*.json`.
Precedence: built-in defaults → config files → `ALPHAI_*` environment variables →
explicit arguments. `alphaai config show`, `config paths` and `config validate`
inspect it; the loader validates types and known keys.

## Verification

`tests/` (pytest) exercises core, routing, conversation, tools, skills,
orchestration, training, the API and the CLI with a deterministic **test engine**
(`tests/conftest.py`). Nothing in `alphaai` itself falls back to a fake engine —
the fake engine exists only inside the test suite.

# AlphaAI HTTP API

A single FastAPI application exposes the AlphaAI runtime. Start it with:

```shell
alphaai serve --host 0.0.0.0 --port 8000
```

or programmatically via `alphaai.api.app:create_app`. A small dashboard is
served at `/`.

Every JSON error uses the same envelope:

```json
{ "code": "no_suitable_model", "message": "...", "remediation": "...", "details": { } }
```

There is **no upstream AI provider and no fallback answer**: if no engine can
serve a request, you get a structured error.

## Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/api/health` | runtime health: hardware, engines, tools, memory |
| `GET` | `/api/models` | every registered engine + live status |
| `GET` | `/api/models/{id}` | one engine |
| `GET` | `/api/skills` | every skill, availability and schema |
| `GET` | `/api/tools` | every tool, permission decision and schema |
| `GET` | `/api/tools/calls` | recent tool-call audit log |
| `GET` | `/api/runtime` | runtime summary (versions, counts) |
| `GET` | `/api/config` | redacted effective configuration |
| `GET` | `/api/attribution` | branding + DeepSeek-V3 attribution strings |
| `POST` | `/api/chat` | one conversation turn (real inference) |
| `POST` | `/api/chat/stream` | streaming conversation (NDJSON) |
| `POST` | `/api/tools/execute` | execute a tool under policy |
| `POST` | `/api/skills/execute` | execute a skill under policy |
| `POST` | `/api/orchestrate` | declared plan or goal-driven run |

`GET` list endpoints return `{"ok": true, "count": <n>, ...}`.

## Chat

```shell
curl -s localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message": "Explain the AlphaAI router.", "use_tools": true}'
```

`ChatRequest` fields: `message` (required), `session_id`, `engine_id`, `task`,
`system_prompt`, `use_tools`, `use_memory`, `max_tokens`, `temperature`, `top_p`,
`seed`. The response is the `ChatOutcome` data (`text`, `engine_id`, `usage`,
`routing`, `context`, `attribution`, …) plus an `ok` field.

### Streaming

`POST /api/chat/stream` returns newline-delimited JSON events:
`route`, `delta`, `tool_call`, `tool_result`, `tool_loop_limit`, `done`,
`error`.

## Tools and skills

```shell
curl -s localhost:8000/api/tools/execute \
  -H 'content-type: application/json' \
  -d '{"tool_id": "calculator.evaluate", "arguments": {"expression": "6*7"}}'

curl -s localhost:8000/api/skills/execute \
  -H 'content-type: application/json' \
  -d '{"skill_id": "skill.calculator", "inputs": {"expression": "6*7"}}'
```

Both return `{"ok": <bool>, "result": {...}}`.

## Orchestration

```shell
curl -s localhost:8000/api/orchestrate \
  -H 'content-type: application/json' \
  -d '{"mode": "goal", "goal": "compute 21 * 2", "max_steps": 3}'
```

`mode` is `goal` (model-driven loop) or `declared` (run the supplied `steps`).

## Status codes

| Code | HTTP |
| ---- | ---- |
| `unknown_model`, `tool_not_found`, `skill_not_found` | 404 |
| `tool_permission_denied`, `skill_permission_denied` | 403 |
| `tool_invalid_input`, `skill_invalid_input` | 400 |
| `context_overflow` | 413 |
| `tool_timeout`, `skill_timeout` | 504 |
| `engine_unavailable`, `engine_load_failed`, `no_suitable_model` | 503 |
| `generation_failed` | 502 |
| `model_incompatible` | 409 |
| `conversation_error` | 404 |
| `memory_error`, `skill_execution_failed`, `tool_execution_failed` | 500 |

The mapping lives in `alphaai/api/app.py` (`STATUS_BY_CODE`).

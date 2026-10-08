# Deploying AlphaAI

AlphaAI serves real inference from a real local model: the GGUF weights are
loaded by llama.cpp on a host AlphaAI controls. There is no external AI
provider, no AI API key and no mock response anywhere in the path.

Two tiers are supported, and they are the same application:

| Tier | What runs there | Where it lives in this repo |
| --- | --- | --- |
| **Frontend** | the AlphaAI dashboard (static HTML/CSS/JS, same-origin `fetch`) | `alphaai/api/static/index.html` |
| **Backend API** | the FastAPI application (all `/api/*` endpoints) | `alphaai/api/app.py`, served by `alphaai/api/__init__.py:serve` |
| **Inference** | llama.cpp + the Qwen2.5-0.5B-Instruct GGUF, driven by AlphaAI Core | `alphaai/engines/llama_cpp.py` via `alphaai/core/runtime.py` |

```
Browser
  │  GET /                      → frontend service  (static dashboard)
  │  fetch /api/*               → backend service   (FastAPI)
  ▼
FastAPI backend (Vercel Services, stateless gateway)
  │  ALPHAI_INFERENCE_URL
  ▼
AlphaAI inference server (container host, `alphaai serve`, persistent volume)
  │
  ▼
llama.cpp + Qwen2.5-0.5B-Instruct GGUF  →  real generated text
```

Two deployment shapes use the same code:

1. **Single host (simplest).** One container runs everything: `alphaai serve`
   serves the dashboard at `/`, the API at `/api/*` and performs inference in
   the same process. See sections 1–4.
2. **Split (frontend + API on managed hosting).** Vercel Services run the static
   frontend and the FastAPI backend; the backend is a **stateless gateway** that
   forwards `/api/*` to the inference server. See sections 5–7.

---

## 1. Resource requirements for the inference host

| Resource | Required | Notes |
| --- | --- | --- |
| RAM | ~1 GB free (model needs ~0.9 GB resident) | a 1 GB instance is the practical minimum |
| Disk | ~1 GB **persistent** (model is 491 MB + runtime state) | must survive restarts, do **not** bake weights into an image |
| CPU | 1+ cores | ~9 tokens/s per core on a 0.5B Q4_K_M model |
| Network | outbound HTTPS on first boot | only to download the model from Hugging Face |

## 2. Startup command

The repository already ships the server entrypoint:

```bash
alphaai serve            # binds api.host:api.port from configs/alphaai.toml
```

`serve` resolves its bind in this order: explicit `--host`/`--port`, then the
platform-provided `$PORT`, then the configured `api.port` (8090). The log level
comes from `--log-level`, then `$ALPHAI_LOG_LEVEL`, then `info`.

Without an installed console script, `python -m alphaai.cli serve` is equivalent.

## 3. Install the project and the model

```bash
pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu ".[api,llama]"
alphaai models install qwen2.5-0.5b-instruct-gguf   # ~0.5 GB, one time
alphaai models list                                 # must print AVAILABLE
```

* `api` — FastAPI + uvicorn (already used by the existing API).
* `llama` — `llama-cpp-python` + `huggingface_hub`, the runtime for the GGUF model.
* The extra index provides **prebuilt CPU wheels**, so no compiler is needed.

The installer verifies hardware, disk space and the authoritative source before
downloading, checks the SHA-256 and GGUF magic, then records provenance and runs
a real load test. A model is never marked available on files alone. Weights are
gitignored (`models/*/`) and are never part of an image, the repo or a deployment
bundle.

`deploy/Dockerfile` does all of this and is the supported inference-host image:

```bash
docker build -f deploy/Dockerfile -t alphaai-inference .
docker run -d --name alphaai -p 8090:8090 \
  -v alphaai-data:/data \
  -e ALPHAI_CORS_ORIGINS=https://your-project.vercel.app \
  -e ALPHAI_INFERENCE_TOKEN=<shared-secret> \
  alphaai-inference
```

`deploy/entrypoint.sh` installs the model on first boot, skips it on every later
start (the weights persist on the `/data` volume), then serves.

## 4. Environment variables (inference host)

All of these are optional; defaults come from `configs/alphaai.toml`.

| Variable | Purpose | Example |
| --- | --- | --- |
| `PORT` | bind port injected by most hosts | `8090` |
| `ALPHAI_HOST` | bind interface | `0.0.0.0` |
| `ALPHAI_CORS_ORIGINS` | the deployed frontend origin(s), comma-separated | `https://your-project.vercel.app` |
| `ALPHAI_INFERENCE_TOKEN` | shared secret the gateway must present | `<shared-secret>` |
| `ALPHAI_MODELS_DIR` | persistent directory for weights | `/data/models` |
| `ALPHAI_STATE_DIR` | persistable runtime state | `/data/state` |
| `ALPHAI_LOG_DIR` | log output directory | `/data/logs` |
| `ALPHAI_LOG_LEVEL` | uvicorn log level | `info` |
| `ALPHAI_MAX_REQUEST_BYTES` | maximum accepted request body | `1048576` |
| `ALPHAI_CPU_THREADS` | llama.cpp thread count | `2` |
| `ALPHAI_GPU_LAYERS` | layers to offload (0 on CPU-only hosts) | `0` |
| `ALPHAI_MAX_TOKENS` | default max output tokens | `512` |
| `ALPHAI_TEMPERATURE` / `ALPHAI_TOP_P` | sampling defaults | `0.3` / `0.9` |
| `ALPHAI_ALLOW_MODEL_DOWNLOAD` | allow the installer to fetch weights | `true` |

There are **no AI API keys** and no AI provider secrets anywhere in AlphaAI.
`ALPHAI_INFERENCE_TOKEN`, when set, is a shared secret between two AlphaAI
services — it authorises nothing external.

Set `ALPHAI_CORS_ORIGINS` to the real frontend origin when the browser and the
API are on different origins. With the same-origin routing in section 5 the
browser never makes a cross-origin request and CORS is not exercised at all.

> Do not set `ALPHAI_CORS_ORIGINS` to an empty string. An empty value resolves to
> an empty allow-list, which blocks every browser request. Leave it unset to keep
> the configured default, or set the exact frontend origin.

## 5. Vercel Services: frontend + FastAPI backend

`vercel.json` (repository root) declares exactly two services:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "services": {
    "frontend": { "root": "alphaai/api/static/" },
    "backend": { "root": ".", "framework": "fastapi", "entrypoint": "asgi:app" }
  },
  "rewrites": [
    { "source": "/api/(.*)", "destination": { "service": "backend" } },
    { "source": "/docs", "destination": { "service": "backend" } },
    { "source": "/redoc", "destination": { "service": "backend" } },
    { "source": "/openapi.json", "destination": { "service": "backend" } },
    { "source": "/(.*)", "destination": { "service": "frontend" } }
  ]
}
```

* `frontend` — the dashboard directory, served as static files. It calls the API
  with same-origin relative paths (`/api/health`, `/api/chat/stream?format=sse`),
  so no backend hostname is hard-coded in the browser code.
* `backend` — `asgi.py` at the repository root (`app = create_app()`), the FastAPI
  application. `requirements.txt` at the repository root declares its
  dependencies (`fastapi`, `uvicorn`); it deliberately does **not** include
  llama.cpp, torch or any weights.
* `rewrites` — the only public routing table. `/api/*` and the API documentation
  routes go to the FastAPI service; everything else goes to the frontend.
  Services are internal by default, so both rules are required.

### Setup on Vercel

1. Import the repository and **set the project framework to “Services”** in
   *Build and Deployment* settings — the `services` key is only honoured when
   this setting is selected.
2. Add the project environment variables (they apply to the `backend` service):

   | Variable | Value |
   | --- | --- |
   | `ALPHAI_INFERENCE_URL` | `https://<your-inference-host>` — the base URL of the container from section 3 (alias: `ALPHA_INFERENCE_URL`) |
   | `ALPHAI_INFERENCE_TOKEN` | the same shared secret you set on the inference host (optional but recommended) |
   | `ALPHAI_CORS_ORIGINS` | only needed if a browser calls the API from another origin |

3. Deploy. Then verify (section 8) that
   `GET https://<project>.vercel.app/api/health` reports
   `inference.ready: true` **and** `gateway.reachable: true`.

`ALPHAI_INFERENCE_URL` is the only wiring between the two tiers. Nothing in the
source hard-codes a host: no `localhost`, no fixed port, no service hostname.
It is never set to an AI provider — it must point at another AlphaAI server.

### Why there is no `inference` service in `vercel.json`

Vercel's function limits make a resident GGUF model unsound (figures from the
Vercel documentation, retrieved for this change):

| Limit | Value | Consequence for this model |
| --- | --- | --- |
| Function bundle, uncompressed | 250 MB (500 MB for Python); “Large functions” beta raises this to 5 GB | the Qwen GGUF is 491 MB on its own |
| Memory | Hobby 2 GB / 1 vCPU; Pro 4 GB / 2 vCPU | ~0.9 GB resident just for the weights, no GPU, no headroom for concurrency |
| Max duration | 300 s (Hobby); 300 s default, 800 s max (Pro) | a long stream on a 1-vCPU instance can approach the cap |
| Request/response body | 4.5 MB | fine for chat, but it is a hard ceiling |
| Instances | scale down after 5 minutes idle (container images) | every cold start re-reads the weights instead of keeping them resident |
| Filesystem | ephemeral per instance | `models/` cannot persist, and AlphaAI's sqlite memory store would silently reset |

A serverless platform therefore cannot satisfy AlphaAI's storage and residency
requirements, and pretending otherwise would mean shipping a fake path. So the
repository runs the frontend and the FastAPI backend on Vercel, and keeps the
**real** inference process on a container host with a persistent volume, selected
through `ALPHAI_INFERENCE_URL`. If a future deployment does try to host inference
on Vercel, it must show `inference.ready: true` from a real generation before
being called ready — the health payload exists to make that impossible to fake.

## 6. Health semantics

`GET /api/health` separates “the server is up” from “inference is ready”:

```json
{
  "ok": true,
  "inference": {
    "ready": true,
    "model_loaded": false,
    "model_unavailable": false,
    "models_registered": 9,
    "usable_models": 1,
    "detail": "1 model(s) can serve local inference."
  }
}
```

* `ok` — the API server answered.
* `inference.ready` — at least one model can actually serve requests. **Never**
  report the service as inference-ready on `ok` alone.
* `inference.model_unavailable` — a registered model exists but cannot run here.
* `model_loaded` — weights are resident right now (they load lazily on the first
  request, so this can be `false` while `ready` is `true`).

On a gateway deployment the same payload is the inference server's, plus the
gateway's own facts:

```json
{
  "gateway": {
    "enabled": true,
    "stateless": true,
    "inference_host": "inference.example.com",
    "inference_url": "https://inference.example.com",
    "token_required": true,
    "reachable": true
  }
}
```

If the inference server cannot be reached the gateway answers with HTTP 503 and
a structured `inference_unreachable` error (host + remediation). It never
substitutes a local model or a canned answer.

## 7. Verify a deployment

```bash
API=https://your-project.vercel.app     # or the container's own URL

curl -s $API/api/health | python3 -m json.tool          # inference.ready must be true
curl -s $API/api/models | python3 -m json.tool          # qwen2.5-0.5b-instruct-gguf AVAILABLE
curl -s $API/api/attribution | python3 -m json.tool     # who the weights belong to

curl -s -X POST $API/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello AlphaAI. Introduce yourself."}'

curl -sN -X POST "$API/api/chat/stream?format=ndjson" \
  -H 'Content-Type: application/json' \
  -d '{"message": "Count to three."}'
```

Required endpoints (all already implemented — nothing was recreated):

```
GET  /api/health      GET  /api/models       GET  /api/models/{id}
GET  /api/skills      GET  /api/tools        GET  /api/tools/calls
GET  /api/runtime     GET  /api/config       GET  /api/attribution
POST /api/chat        POST /api/chat/stream
POST /api/tools/execute                      POST /api/skills/execute
POST /api/orchestrate
```

## 8. Security notes

* The inference service is only reachable where it must be: the gateway declares
  the target through `ALPHAI_INFERENCE_URL`, and the container host can restrict
  ingress to Vercel (or require `ALPHAI_INFERENCE_TOKEN` on both sides).
* Model files are never exposed: no route serves `models/`, `.vercelignore`
  keeps the weights (and local state) out of any upload, and the repository
  ignores `*.gguf`, `*.safetensors`, `*.pt`, `*.bin`.
* No AI provider keys exist in this project, and no AlphaAI code depends on a
  hosting platform: the same `alphaai serve` runs on a laptop, a container or a
  VM.

## 9. Platform notes

This service needs a **long-running container with a persistent volume**. It is
not a fit for serverless/static-only hosting: a 491 MB GGUF model must be
downloaded once, kept resident between requests, and reloaded from local disk
after a restart. Serverless functions re-read the weights on every cold start and
have no persistent disk.

Some managed platforms build only Node.js projects, or serve static output plus
`api/*.py` serverless functions with no `pip`/`python` build step. Those can host
the dashboard alone. The API and the inference server need a container host
(Render, Railway, Fly.io, Hugging Face Spaces, or any Docker host) using
`deploy/Dockerfile` and the variables above. Nothing in `alphaai` depends on any
hosting platform: the same `alphaai serve` command runs everywhere.

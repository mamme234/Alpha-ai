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
| RAM | ~1 GB free (model needs ~0.9 GB resident) | a 1 GB instance is the practical minimum; 2 GB leaves headroom |
| Disk | ~1 GB **persistent** (model is 491 MB + runtime state) | must survive restarts; weights are never committed to git |
| CPU | 1+ cores | ~9 tokens/s per core on a 0.5B Q4_K_M model |
| Network | outbound HTTPS **during the image build** | only to fetch the model from Hugging Face (`deploy/prefetch_model.py`) |

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
gitignored (`models/*/`) and are **never** committed, pushed or uploaded as part
of a repository or deployment bundle.

`deploy/Dockerfile` does all of this and is the supported inference-host image.
While the image is built, `deploy/prefetch_model.py` fetches the weights from the
URL recorded in `configs/models/<id>.json` and verifies size **and** SHA-256 — so
a broken or tampered download fails the *build*, visibly, instead of becoming a
quiet runtime surprise:

```bash
docker build -f deploy/Dockerfile -t alphaai-inference .
docker run -d --name alphaai -p 8090:8090 \
  -v alphaai-data:/data \
  -e ALPHAI_CORS_ORIGINS=https://your-project.vercel.app \
  -e ALPHAI_INFERENCE_TOKEN=<shared-secret> \
  alphaai-inference
```

`deploy/entrypoint.sh` copies those weights onto the `/data` volume on the first
boot, then runs `alphaai models install` to **verify and load-test** them (the
copy is never assumed good), and skips that on every later start — the weights
persist on the volume. With `ALPHAI_INFERENCE_TOKEN` set the container requires
that secret on `/api/*` (section 4) — the gateway presents it, and only direct
inspection needs the header.

## 4. Environment variables (inference host)

All of these are optional; defaults come from `configs/alphaai.toml`.

| Variable | Purpose | Example |
| --- | --- | --- |
| `PORT` | bind port injected by most hosts | `8090` |
| `ALPHAI_HOST` | bind interface | `0.0.0.0` |
| `ALPHAI_CORS_ORIGINS` | the deployed frontend origin(s), comma-separated | `https://your-project.vercel.app` |
| `ALPHAI_INFERENCE_TOKEN` | shared secret this server requires from callers (the gateway) | `<shared-secret>` |
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

It works on both sides of the hop:

* on the **gateway**, it is sent with every forwarded request as
  `Authorization: Bearer …`;
* on the **inference host**, setting it makes the server *require* that header on
  `/api/*` (and the API documentation routes) and answer `401 unauthorized` —
  in AlphaAI's usual JSON error shape — to anything else. An unset token means
  the API is served to any caller.

Setting the token therefore locks the whole API of the inference host to callers
that hold the secret, which is the point of a private inference host: the gateway
holds it. A single-host deployment that also serves the browser dashboard must
leave it unset, because a browser cannot present a shared secret — its `/api/*`
calls would answer `401` (the dashboard page itself still loads, and shows that
message).

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
   | `DATABASE_URL` | **optional** — conversation history. Use Supabase's *transaction pooler* string here (port 6543), or leave it unset and let the inference host own the database. See section 5b and [`DATABASE.md`](./DATABASE.md) |

   No Supabase **API** key is needed anywhere (`SUPABASE_URL`,
   `SUPABASE_ANON_KEY` and `SUPABASE_SERVICE_ROLE_KEY` are unused by AlphaAI),
   and none may be exposed to the browser.

3. **Redeploy** so the variables reach the `backend` service: *Deployments* →
   the latest deployment → *Redeploy* (a new commit on the connected branch does
   the same). Environment variables are read when a function is built, so a
   deployment that started before the variable was added keeps running without
   it.

4. Verify (section 7) that `GET https://<project>.vercel.app/api/health`
   reports `inference.ready: true` **and** a `gateway` block with
   `"reachable": true`. Until the variable is set the same endpoint answers 200
   with **no** `gateway` block (the deployment is running its own engine) and
   `inference.ready: false` — the two states are distinguishable on purpose, so
   a half-configured deployment cannot look healthy.

`ALPHAI_INFERENCE_URL` is the only wiring between the two tiers. Nothing in the
source hard-codes a host: no `localhost`, no fixed port, no service hostname.
It is never set to an AI provider — it must point at another AlphaAI server.

### 5b. Conversation history (Supabase PostgreSQL) — optional

History is a database concern, not an inference one, so it can live on either
tier — whichever one has `DATABASE_URL` set. Full detail (schema, security, RLS,
verification) is in [`DATABASE.md`](./DATABASE.md).

The short version:

1. Create a Supabase project and copy *Project Settings → Database → Connection
   string → URI*, with your database password and `?sslmode=require`.
2. Set `DATABASE_URL` on **one** deployment:
   * on the **Vercel gateway**, use the **transaction pooler** (`…pooler.supabase.com:6543`) —
     serverless functions are short-lived, so session state is not safe there; the
     gateway then serves `/api/conversations*` itself;
   * on the **inference host**, use the direct connection (`db.<ref>.supabase.co:5432`)
     or Supavisor's session mode (port 5432).
   Setting it on both makes each side own a different set of threads — pick one.
3. Apply the schema:
   * `alphaai db migrate` on the host (the image includes `supabase/`), or
   * `supabase db push`, or
   * `POST /api/database/migrate` against the running server.
4. **Redeploy** the Vercel project if you set the variable there (environment
   variables are read at build time), then confirm
   `GET /api/database` reports `reachable: true` and `migrations.pending: []`.

Until `DATABASE_URL` is set, nothing is broken and nothing is faked: `/api/health`
adds `database: {configured: false}`, persistence requests answer 503
`database_not_configured` with the fix, chat still answers, and every chat
response says whether the turn was stored.

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

### What the deployed backend guarantees

Managed hosting fails loudly and unhelpfully: a function that crashes while
starting is replaced by the platform's own HTML error page, which no JSON client
can read. The backend is therefore built so that a failure is always the API's
own structured JSON:

* **The entrypoint never dies silently.** `asgi.py` builds the app and, if that
  fails (missing dependency, invalid `ALPHAI_INFERENCE_URL`, packaging mistake),
  serves the failure itself as `application/json` with the code
  `startup_failed`, the exception and its traceback in `error.details`.
* **An unhandled error is JSON too.** The app converts any unexpected exception
  into `{"ok": false, "error": {"code": "internal_error", ...}}` with
  `Content-Type: application/json` instead of HTML.
* **A read-only project root does not break the API.** Serverless hosts mount
  the deployment read-only. AlphaAI's *volatile* directories (runtime state,
  logs, the tool sandbox) then move to the platform's writable temp directory —
  see `alphaai/config/loader.py:relocate_volatile_paths` — while the code,
  configs and weights stay where they are. The API keeps reporting the machine's
  real state: with no weights and no llama.cpp on this host, every engine is
  honestly `unavailable`, and `inference.ready` is `false`.
* **Dependencies are declared twice on purpose.** `requirements.txt` and
  `pyproject.toml` both list FastAPI and uvicorn, because a platform may build
  from either manifest and `asgi:app` imports FastAPI at module load. Neither
  file includes llama.cpp, torch or weights.
* **The function ships AlphaAI's Python modules, not the model catalogue.** The
  Python build bundles what the entrypoint reaches, so `configs/models/*.json`
  is *not* part of the deployed function — the model metadata and the weights
  belong to the inference host. Until `ALPHAI_INFERENCE_URL` is set, the
  deployed API therefore reports an empty model list (`{"count": 0,
  "models": []}`) and `inference.ready: false`, and the dashboard renders that
  emptiness as a stated fact instead of an empty table. With the variable set,
  `/api/models` is the inference server's real list, statuses included.
* **Real inference only ever comes from `ALPHAI_INFERENCE_URL`.** Without it the
  backend serves the API surface (health, models, skills, tools, runtime,
  config, attribution) with inference reported unavailable; with it, `api/*` is
  forwarded to the real inference server. No endpoint fabricates an answer.

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

A real answer carries the model that produced it (`"text"`, `"engine"`,
`"model"`, `"attribution"`). `503` with `no_suitable_model` or
`inference_unreachable` means the chain is not connected yet — never that the
answer was generated.

Calling the **inference host** directly, when it requires the token, needs the
header (`ALPHAI_INFERENCE_URL` + `/api/health` from the gateway needs nothing):

```bash
INFER=https://<your-inference-host>
curl -s $INFER/api/health -H "Authorization: Bearer $ALPHAI_INFERENCE_TOKEN" | python3 -m json.tool
# without the header the same endpoint answers 401 in JSON — that is the shared
# secret working, not a broken deployment
curl -s -o /dev/null -w '%{http_code}\n' $INFER/api/health   # 401 when a token is set
```

Routing and content type — every one of these must answer `application/json`,
never `text/html` and never a platform error page:

```bash
for path in /api/health /api/models /api/skills /api/tools /api/runtime \
            /api/config /api/attribution /openapi.json; do
  printf '%-20s ' "$path"
  curl -s -o /dev/null -w '%{http_code} %{content_type}\n' "$API$path"
done
```

If a path answers `500 text/plain` (or `FUNCTION_INVOCATION_FAILED`), the
backend service did not start. Read the JSON error body first — a
`startup_failed` payload carries the exception and traceback — then the
function logs of that deployment. Two conditions make it start cleanly:

1. the project framework is **“Services”** (otherwise `services` is ignored), and
2. the `backend` service can import `asgi:app`, which means FastAPI and uvicorn
   are installed from `requirements.txt` / `pyproject.toml`.

`/api/chat` and `/api/chat/stream` answer `503` with
`{"error": {"code": "inference_unreachable"}}` until a real inference server is
reachable through `ALPHAI_INFERENCE_URL`. That is the honest answer, not a
failure of the deployment: the gateway never invents text.

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

The container host must provide:

* a **Dockerfile build** from this repository (so `pip` can install
  `.[api,llama]` and the CPU llama.cpp wheel),
* a **persistent volume** mounted at `/data` (the 491 MB GGUF is downloaded on
  first boot and must survive restarts, along with `/data/state`),
* **HTTPS ingress** on the port the container serves (8090, or `$PORT`), and
* at least **~1 GB RAM** with a long-running (not scale-to-zero) service.

### Shortest path: Render (Blueprint already in this repository)

`render.yaml` at the repository root describes the inference host exactly: it
builds `deploy/Dockerfile` with the repository root as the build context,
attaches a 2 GB persistent disk at `/data`, asks for `plan: standard`, and
prompts for the two variables a deployment has to decide.

**Cost first (nothing is charged until you press Apply on an account with a
payment method):**

| Item | Price | Why it is needed |
| --- | --- | --- |
| Render **Standard** instance (2 GB RAM / 1 CPU) | ~$25 / month | the model needs ~1.3 GB resident; the 512 MB plans (`free`, `starter`) cannot hold it, and the free plan has no persistent disk |
| Persistent disk, 2 GB | ~$0.50 / month | holds the 491 MB GGUF across restarts and redeploys |
| **Total** | **~$25.50 / month** | no per-request or bandwidth charge applies at this size |

**Steps**

1. Create (or sign in to) a Render account and connect it to GitHub, granting
   Render access to this repository — *Dashboard → GitHub → Connect*.
2. **New → Blueprint**, select this repository and the branch to deploy. Render
   reads `render.yaml` and shows the `alphaai-inference` service it will create.
3. Fill the prompted variables — `ALPHAI_INFERENCE_TOKEN` (recommended: a long
   random string; set the same value on the Vercel project) and
   `ALPHAI_CORS_ORIGINS` (not needed with same-origin routing) — confirm the
   **Standard** instance and the 2 GB disk, and add a payment method if the
   account has none. Then **Apply**.
4. The build installs `.[api,llama,db]` with the CPU llama.cpp wheel and fetches
   the 491 MB GGUF into the image (verified against its SHA-256). First boot
   copies it onto the disk, verifies it again and runs a real load test — about
   a minute end to end.
5. Copy the service URL (`https://<service>.onrender.com`). That is
   `ALPHAI_INFERENCE_URL`. Render terminates TLS and forces HTTPS on that host,
   and it injects `$PORT`, which `alphaai serve` already honours.

**No health check path is declared, deliberately.** Render's probe is an
unauthenticated `GET`, while a token-protected host answers `401` to every
`/api/*` request without the shared secret — the probe would declare a healthy
deployment dead. Render instead marks the deploy live when the port opens, and
`deploy/entrypoint.sh` only starts the server after the weights have been
copied, verified and load-tested, so "the port is open" really means "the API
is ready". Verify the model-level state yourself:

```bash
INFER=https://<service>.onrender.com

# 401 without the shared secret — the auth is working, not a broken deploy
curl -s -o /dev/null -w '%{http_code}\n' $INFER/api/health

curl -s $INFER/api/health -H "Authorization: Bearer $ALPHAI_INFERENCE_TOKEN" \
  | python3 -m json.tool          # inference.ready must be true
```

If the service stays unhealthy, read *Logs* in the Render dashboard: the
entrypoint prints every step (seed, install, verification) and a failed model
install is reported as `inference.ready: false` rather than hidden.

### Alternative: Fly.io

Fly runs the same image as a micro-VM with a volume. `fly.toml`:

```toml
app = "alphaai-inference"
primary_region = "ord"

[build]
  dockerfile = "deploy/Dockerfile"

[env]
  PORT = "8090"          # keep the app and internal_port in agreement

[http_service]
  internal_port = 8090
  force_https = true
  auto_stop_machines = "off"   # never scale to zero: the weights stay resident
  auto_start_machines = true

[mounts]
  source = "alphaai_data"
  destination = "/data"
```

```sh
fly volumes create alphaai_data --size 2 --region ord --app alphaai-inference
fly deploy --app alphaai-inference --dockerfile deploy/Dockerfile
fly open --app alphaai-inference           # https://alphaai-inference.fly.dev
```

## 10. What still has to happen outside this repository

Everything in this repository is ready; three things are account-level actions
that cannot be done from the source tree:

1. **Run the inference image on a container host.** `deploy/Dockerfile` is the
   supported image; build it from this repository and mount a volume at `/data`.
   The GGUF weights are fetched while the image builds and copied onto the
   volume on the first boot; they are never committed to git.
2. **Record its public HTTPS URL.** That URL is the value of
   `ALPHAI_INFERENCE_URL`. Never use a temporary/workspace preview URL or an
   AI provider — it must be the `alphaai serve` instance from step 1.
3. **Set the variables on the Vercel project and redeploy** (section 5, step 3):
   `ALPHAI_INFERENCE_URL`, plus `ALPHAI_INFERENCE_TOKEN` if the inference host
   requires it. Conversation history is a separate, optional action: set
   `DATABASE_URL` (Supabase) on one deployment and apply the migrations —
   section 5b and [`DATABASE.md`](./DATABASE.md).

Until step 3 is done the deployed gateway stays honest: `GET /api/health`
answers 200 with `inference.ready: false`, `POST /api/chat` answers 503
`no_suitable_model`, and `POST /api/chat/stream` emits a structured error event.
No endpoint fabricates text, so there is nothing to "unfake" later.

"""AlphaAI HTTP API.

A single FastAPI application that exposes the AlphaAI runtime:

============================  ==============================================
endpoint                      purpose
============================  ==============================================
``GET  /api/health``          runtime health (hardware, engines, tools, memory)
``GET  /api/models``          every registered engine + live status
``GET  /api/models/{id}``     one engine
``GET  /api/skills``          every skill, availability and schema
``GET  /api/tools``           every tool, permission decision and schema
``POST /api/chat``            one conversation turn (real inference)
``POST /api/chat/stream``     streaming SSE/NDJSON conversation
``POST /api/tools/execute``   execute a tool under policy
``POST /api/skills/execute``  execute a skill under policy
============================  ==============================================

If no engine can serve a request the API returns a structured error
(``engine_unavailable`` / ``no_suitable_model``) with the remediation AlphaAI
computed. There is no fallback answer and no upstream AI provider.

When ``api.inference_url`` is set (``$ALPHAI_INFERENCE_URL``) the app runs in
*gateway* mode: it loads no model of its own and forwards ``/api/*`` to a
separate, always-on AlphaAI inference server. See :mod:`alphaai.api.gateway`.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..branding import (
    ATTRIBUTION_ENGINE,
    ATTRIBUTION_LONG,
    ATTRIBUTION_SHORT,
    DESCRIPTION,
    DISPLAY_NAME,
    NAME,
    TAGLINE,
    attribution_lines,
    powered_by,
)
from ..config.loader import public_config_view
from ..config.schema import AlphaAIConfig
from ..core.errors import AlphaAIError
from ..core.runtime import AlphaRuntime
from ..version import CORE_INTERFACE_VERSION, __version__
from .gateway import configure_gateway, inference_host, normalize_inference_url

logger = logging.getLogger("alphaai.api")

STATIC_DIR = Path(__file__).parent / "static"

#: AlphaAI error codes -> HTTP status codes.
STATUS_BY_CODE = {
    "unknown_model": 404,
    "tool_not_found": 404,
    "skill_not_found": 404,
    "tool_permission_denied": 403,
    "skill_permission_denied": 403,
    "tool_invalid_input": 400,
    "skill_invalid_input": 400,
    "context_overflow": 413,
    "tool_timeout": 504,
    "skill_timeout": 504,
    "engine_unavailable": 503,
    "engine_load_failed": 503,
    "inference_unreachable": 503,
    "generation_failed": 502,
    "no_suitable_model": 503,
    "model_incompatible": 409,
    "conversation_error": 404,
    "memory_error": 500,
    "skill_execution_failed": 500,
    "tool_execution_failed": 500,
}


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=200_000)
    session_id: str | None = None
    engine_id: str | None = None
    task: str | None = None
    system_prompt: str | None = None
    use_tools: bool = True
    use_memory: bool = True
    max_tokens: int | None = Field(default=None, ge=1, le=131_072)
    temperature: float | None = Field(default=None, ge=0.0, le=4.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None


class ToolExecuteRequest(BaseModel):
    tool_id: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class SkillExecuteRequest(BaseModel):
    skill_id: str = Field(min_length=1, max_length=200)
    inputs: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class OrchestrateRequest(BaseModel):
    goal: str = Field(default="", max_length=20_000)
    mode: str = Field(default="goal", pattern="^(goal|declared)$")
    steps: list[dict[str, Any]] = Field(default_factory=list)
    engine_id: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=64)
    use_tools: bool = True


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so responses are always strict JSON.

    Error details are built from live values (for example a router candidate
    score of ``-inf``). Strict JSON encoders reject those, which would turn a
    structured error into a 500.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# application factory
# ---------------------------------------------------------------------------
def dashboard_response(*, enabled: bool = True) -> Any:
    """The AlphaAI dashboard page — the front end this API ships with."""

    if not enabled:
        return JSONResponse(
            {
                "ok": True,
                "name": NAME,
                "version": __version__,
                "detail": "AlphaAI dashboard is disabled (api.enable_dashboard = false).",
            }
        )
    index = STATIC_DIR / "index.html"
    if not index.exists():  # pragma: no cover - packaging problem
        return JSONResponse(
            {"ok": False, "error": {"code": "dashboard_missing", "message": str(index)}}
        )
    return HTMLResponse(index.read_text(encoding="utf-8"))


def create_app(
    config: AlphaAIConfig | str | None = None,
    *,
    runtime: AlphaRuntime | None = None,
    project_root: str | None = None,
    inference_url: str | None = None,
) -> FastAPI:
    """Build the AlphaAI FastAPI application.

    ``inference_url`` (default: ``api.inference_url``, i.e.
    ``$ALPHAI_INFERENCE_URL``) switches the app into gateway mode: it forwards
    ``/api/*`` to a separate, always-on AlphaAI inference server instead of
    loading a model on this host.
    """

    # Resolve the effective config before anything else: CORS, the request-size
    # limit and gateway mode all follow configuration (configs/alphaai.toml plus
    # the ALPHAI_* environment overrides) even when the caller only passed a
    # path. A production deploy must never silently fall back to a ``*`` origin.
    active_config = config if isinstance(config, AlphaAIConfig) else None
    if active_config is None:
        from ..config.loader import load_config

        try:
            active_config = load_config(
                config if isinstance(config, (str, Path)) else None,
                project_root=project_root,
            )
        except Exception:  # noqa: BLE001 - the API still boots and reports the failure
            logger.warning("could not load config for CORS/limits; using defaults", exc_info=True)
            active_config = None
    api_config = active_config.api if active_config else None
    gateway_base = normalize_inference_url(
        inference_url if inference_url is not None else (api_config.inference_url if api_config else "")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if gateway_base:
            # Gateway mode owns no model and touches no disk: every request is
            # served by the inference server behind ALPHAI_INFERENCE_URL.
            logger.info(
                "%s HTTP API ready in gateway mode -> %s", NAME, inference_host(gateway_base)
            )
            for line in attribution_lines():
                logger.info("%s", line)
            yield
            return
        app.state.runtime = runtime or AlphaRuntime.create(
            config, project_root=project_root, create_dirs=True
        )
        logger.info("%s HTTP API ready", NAME)
        for line in attribution_lines():
            logger.info("%s", line)
        try:
            yield
        finally:
            if runtime is None:  # only close what we created
                app.state.runtime.close()

    app = FastAPI(
        title=f"{DISPLAY_NAME} API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # In gateway mode the inference server owns the API schema, so the
        # gateway proxies its /docs and /openapi.json instead of documenting
        # the pass-through route.
        docs_url=None if gateway_base else "/docs",
        redoc_url=None if gateway_base else "/redoc",
        openapi_url=None if gateway_base else "/openapi.json",
    )

    origins = list(active_config.api.cors_origins) if active_config else ["*"]
    max_request_bytes = active_config.api.max_request_bytes if active_config else 1_048_576
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _enforce_body_limit(request: Request, call_next):
        """Reject oversized payloads before they reach the chat/tool handlers."""

        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_request_bytes:
            return JSONResponse(
                {
                    "ok": False,
                    "error": {
                        "code": "request_too_large",
                        "message": f"Request body exceeds api.max_request_bytes ({max_request_bytes}).",
                        "remediation": "Send a smaller payload or raise api.max_request_bytes.",
                    },
                },
                status_code=413,
            )
        return await call_next(request)

    if gateway_base:
        # Remote-inference deployment: the public API surface below is served by
        # the inference server, not by this process.
        if origins == ["*"]:
            logger.warning(
                "gateway mode is serving CORS origin '*' - set ALPHAI_CORS_ORIGINS to "
                "the deployed frontend origin (same-origin routing needs no wildcard)"
            )
        configure_gateway(
            app,
            base_url=gateway_base,
            token=api_config.inference_token if api_config else "",
            timeout_s=api_config.inference_timeout_s if api_config else 300.0,
            dashboard=api_config.enable_dashboard if api_config else True,
        )
        return app

    def get_runtime(request: Request) -> AlphaRuntime:
        return request.app.state.runtime

    def error_response(exc: AlphaAIError) -> JSONResponse:
        payload = {"ok": False, "error": _json_safe(exc.to_dict())}
        return JSONResponse(payload, status_code=STATUS_BY_CODE.get(exc.code, 500))

    # -- meta -------------------------------------------------------------
    @app.get("/api", tags=["meta"])
    def api_index() -> dict[str, Any]:
        return {
            "ok": True,
            "name": NAME,
            "display_name": DISPLAY_NAME,
            "tagline": TAGLINE,
            "version": __version__,
            "core_interface_version": CORE_INTERFACE_VERSION,
            "attribution": {
                "short": ATTRIBUTION_SHORT,
                "engine": ATTRIBUTION_ENGINE,
                "long": ATTRIBUTION_LONG,
            },
            "endpoints": [
                "GET /api/health",
                "GET /api/models",
                "GET /api/models/{model_id}",
                "GET /api/skills",
                "GET /api/tools",
                "POST /api/chat",
                "POST /api/chat/stream",
                "POST /api/tools/execute",
                "POST /api/skills/execute",
                "GET /api/runtime",
                "GET /api/config",
                "POST /api/orchestrate",
                "GET /api/tools/calls",
            ],
            "note": "AlphaAI performs local inference only: no external AI provider, no API keys.",
        }

    @app.get("/api/health", tags=["meta"])
    def health(request: Request) -> dict[str, Any]:
        return get_runtime(request).health()

    @app.get("/api/runtime", tags=["meta"])
    def runtime_info(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        return {
            "ok": True,
            "capabilities": runtime_state.capabilities(),
            "doctor": runtime_state.doctor(),
            "hardware": runtime_state.hardware.to_dict(),
        }

    @app.get("/api/config", tags=["meta"])
    def config_view(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        return {
            "ok": True,
            "source": runtime_state.config.source,
            "config": public_config_view(runtime_state.config),
        }

    @app.get("/api/attribution", tags=["meta"])
    def attribution(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        usable = runtime_state.registry.usable()
        payload = {
            "ok": True,
            "short": ATTRIBUTION_SHORT,
            "engine": ATTRIBUTION_ENGINE,
            "long": ATTRIBUTION_LONG,
            "lines": attribution_lines(),
            "preserved_files": ["LICENSE-CODE", "LICENSE-MODEL", "NOTICE", "ATTRIBUTION.md"],
            # Which model is actually serving requests right now, named plainly:
            # AlphaAI owns the engine integration, the model owner keeps the weights.
            "active_models": [
                {
                    "id": engine.id,
                    "model": engine.model,
                    "model_owner": engine.spec.model_owner,
                    "engine_owner": engine.spec.engine_owner,
                    "runtime": engine.runtime,
                    "attribution": engine.attribution,
                }
                for engine in usable
            ],
            "powered_by": [powered_by(engine.model) for engine in usable],
        }
        return payload

    # -- models -----------------------------------------------------------
    @app.get("/api/models", tags=["models"])
    def list_models(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        models = runtime_state.models()
        return {
            "ok": True,
            "count": len(models),
            "usable": len([item for item in models if item["status"]["usable"]]),
            "models": models,
        }

    @app.get("/api/models/{model_id}", tags=["models"])
    def get_model(model_id: str, request: Request, refresh: bool = False) -> Any:
        runtime_state = get_runtime(request)
        try:
            engine = runtime_state.registry.get(model_id)
        except AlphaAIError as exc:
            return error_response(exc)
        if refresh:
            engine.refresh_status()
        return {"ok": True, "model": engine.info(redact_paths=runtime_state.config.api.redact_paths)}

    # -- skills & tools ---------------------------------------------------
    @app.get("/api/skills", tags=["skills"])
    def list_skills(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        skills = runtime_state.skills_view()
        return {
            "ok": True,
            "count": len(skills),
            "enabled": len([item for item in skills if item.get("enabled")]),
            "skills": skills,
        }

    @app.get("/api/tools", tags=["tools"])
    def list_tools(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        tools = runtime_state.tools_view()
        return {
            "ok": True,
            "count": len(tools),
            "permitted": len([item for item in tools if item["permissions"].get("allowed")]),
            "tools": tools,
        }

    @app.get("/api/tools/calls", tags=["tools"])
    def tool_calls(request: Request) -> dict[str, Any]:
        runtime_state = get_runtime(request)
        return {"ok": True, "log": runtime_state.tools.log.to_dict()}

    # -- chat -------------------------------------------------------------
    def _chat_kwargs(payload: ChatRequest, runtime_state: AlphaRuntime) -> dict[str, Any]:
        session = None
        if payload.session_id:
            session = runtime_state.conversation.get_session(payload.session_id)
        elif payload.system_prompt:
            session = runtime_state.create_session(system_prompt=payload.system_prompt)
        elif payload.engine_id:
            session = runtime_state.create_session(engine_id=payload.engine_id)
        sampling = {}
        if payload.max_tokens is not None:
            sampling["max_tokens"] = payload.max_tokens
        if payload.temperature is not None:
            sampling["temperature"] = payload.temperature
        if payload.top_p is not None:
            sampling["top_p"] = payload.top_p
        if payload.seed is not None:
            sampling["seed"] = payload.seed
        return {
            "session": session,
            "engine_id": payload.engine_id,
            "task": payload.task,
            "use_tools": payload.use_tools,
            "use_memory": payload.use_memory,
            "max_new_tokens": payload.max_tokens,
            "sampling": sampling or None,
        }

    @app.post("/api/chat", tags=["chat"])
    def chat(payload: ChatRequest, request: Request) -> Any:
        runtime_state = get_runtime(request)
        try:
            outcome = runtime_state.chat(payload.message, **_chat_kwargs(payload, runtime_state))
        except AlphaAIError as exc:
            return error_response(exc)
        body = outcome.to_dict()
        body["ok"] = True
        return body

    @app.post("/api/chat/stream", tags=["chat"])
    def chat_stream(payload: ChatRequest, request: Request, format: str = "sse"):
        """Stream real tokens. ``format`` is ``sse`` (default) or ``ndjson``."""

        runtime_state = get_runtime(request)
        kwargs = _chat_kwargs(payload, runtime_state)
        use_sse = format != "ndjson"

        def event_stream() -> Iterator[str]:
            try:
                for event in runtime_state.stream(payload.message, **kwargs):
                    if use_sse:
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    else:
                        yield json.dumps(event, ensure_ascii=False) + "\n"
            except AlphaAIError as exc:
                event = {"type": "error", "error": exc.to_dict()}
                if use_sse:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                else:
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            if use_sse:
                yield "event: alphaai-done\ndata: [DONE]\n\n"

        media = "text/event-stream" if use_sse else "application/x-ndjson"
        return StreamingResponse(event_stream(), media_type=media)

    # -- execution --------------------------------------------------------
    @app.post("/api/tools/execute", tags=["tools"])
    def execute_tool(payload: ToolExecuteRequest, request: Request) -> Any:
        runtime_state = get_runtime(request)
        result = runtime_state.execute_tool(
            payload.tool_id, payload.arguments, run_id=payload.run_id or "api"
        )
        return {"ok": result.ok, "result": result.to_dict()}

    @app.post("/api/skills/execute", tags=["skills"])
    def execute_skill(payload: SkillExecuteRequest, request: Request) -> Any:
        runtime_state = get_runtime(request)
        result = runtime_state.execute_skill(
            payload.skill_id, payload.inputs, run_id=payload.run_id or "api"
        )
        return {"ok": result.ok, "result": result.to_dict()}

    @app.post("/api/orchestrate", tags=["orchestrate"])
    def orchestrate(payload: OrchestrateRequest, request: Request) -> Any:
        runtime_state = get_runtime(request)
        try:
            if payload.mode == "declared":
                report = runtime_state.orchestrator.run_plan(payload.steps, goal=payload.goal)
            else:
                report = runtime_state.orchestrator.run_goal(
                    payload.goal,
                    max_steps=payload.max_steps,
                    engine_id=payload.engine_id,
                    use_tools=payload.use_tools,
                )
        except AlphaAIError as exc:
            return error_response(exc)
        body = report.to_dict()
        body["ok"] = report.ok
        return body

    # -- dashboard --------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request) -> Any:
        runtime_state = get_runtime(request)
        return dashboard_response(enabled=runtime_state.config.api.enable_dashboard)

    @app.exception_handler(AlphaAIError)
    async def alphaai_error_handler(_request: Request, exc: AlphaAIError) -> JSONResponse:
        return error_response(exc)

    return app


__all__ = [
    "ChatRequest",
    "OrchestrateRequest",
    "SkillExecuteRequest",
    "STATUS_BY_CODE",
    "ToolExecuteRequest",
    "create_app",
]

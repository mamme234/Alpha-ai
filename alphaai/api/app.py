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

import hmac
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
from ..core.errors import (
    AlphaAIError,
    DatabaseNotFoundError,
    DatabaseNotConfiguredError,
    RuntimeUnavailableError,
)
from ..core.runtime import AlphaRuntime
from ..db import open_database, validate_client_id
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
    "unauthorized": 401,
    "engine_unavailable": 503,
    "engine_load_failed": 503,
    "inference_unreachable": 503,
    "runtime_unavailable": 503,
    "generation_failed": 502,
    "no_suitable_model": 503,
    "model_incompatible": 409,
    "conversation_error": 404,
    "memory_error": 500,
    "skill_execution_failed": 500,
    "tool_execution_failed": 500,
    "invalid_request": 400,
    "conversation_not_found": 404,
    "database_not_configured": 503,
    "database_unavailable": 503,
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
    #: Anonymous dashboard id. Supplying it (with persistence available) is what
    #: saves the turn to PostgreSQL; without it the turn is answered and not stored.
    client_id: str | None = Field(default=None, max_length=200)
    #: Continue an existing stored thread instead of starting a new one.
    conversation_id: str | None = Field(default=None, max_length=64)
    #: ``False`` opts out of persistence for this turn even when client_id is set.
    persist: bool = True


class ConversationCreateRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=200)
    engine_id: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = Field(default=None, max_length=64)


class MessageCreateRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    role: str = Field(default="user", max_length=32)
    content: str = Field(min_length=1, max_length=200_000)
    engine_id: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=200)
    finish_reason: str | None = Field(default=None, max_length=32)
    latency_ms: float | None = Field(default=None, ge=0.0)
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferencesRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    preferences: dict[str, Any] = Field(default_factory=dict)


class MigrateRequest(BaseModel):
    directory: str | None = Field(default=None, max_length=1024)


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
# persistence (PostgreSQL / Supabase)
#
# Where does the database live? On exactly the deployment that has DATABASE_URL
# set. That is normally the inference host (it already has a persistent disk and
# a long-lived process, and the chat turns it produces are what gets stored),
# but a serverless gateway with its own DATABASE_URL can serve history too — a
# database is not an inference concern. The routes below are registered by
# whichever process owns the store; the other one proxies them.
# ---------------------------------------------------------------------------
def _store_for(request: Request) -> Any:
    """The store this deployment uses, or the unconfigured placeholder."""

    store = getattr(request.app.state, "database", None)
    return store if store is not None else UnconfiguredStore()


def database_health_decorator(store: Any) -> Any:
    """Add a gateway's *own* database block to the proxied health payload.

    The inference host may report a database too. Renaming that one keeps
    ``database`` unambiguous: it is always the store this API answers with.
    """

    def decorate(payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload.get("database"), dict):
            payload["inference_database"] = payload.pop("database")
        payload["database"] = store.status()
        return payload

    return decorate


def conversation_not_found(conversation_id: str) -> JSONResponse:
    """404 in the same error shape as every other AlphaAI failure."""

    return JSONResponse(
        {
            "ok": False,
            "error": {
                "code": "conversation_not_found",
                "message": "No conversation with that id exists for this client.",
                "details": {"conversation_id": conversation_id},
            },
        },
        status_code=404,
    )


def _persist_turn(
    store: Any,
    payload: ChatRequest,
    *,
    assistant: dict[str, Any],
) -> dict[str, Any]:
    """Save one chat turn, or explain why it was not saved.

    Called only when the client asked for persistence (``client_id`` set and
    ``persist`` true). It never raises: the answer has already been produced by
    real inference, so a persistence failure is reported in the response as
    ``persistence`` rather than thrown away as a 500. Nothing is ever reported
    as saved unless it was.
    """

    if store is None or not store.configured:
        return {
            "requested": True,
            "persisted": False,
            "error": DatabaseNotConfiguredError(
                "This AlphaAI server has no database configured, so the turn was not saved."
            ).to_dict(),
        }
    try:
        conversation_id = payload.conversation_id
        if conversation_id:
            conversation = store.get_conversation(
                conversation_id, client_id=payload.client_id, include_messages=False
            )
            if conversation is None:
                raise DatabaseNotFoundError(
                    "That conversation does not exist for this client.",
                    remediation=(
                        "Start a new conversation by omitting conversation_id, then send the "
                        "id it returns with the next turn."
                    ),
                    details={"conversation_id": conversation_id},
                )
        else:
            conversation = store.create_conversation(
                client_id=payload.client_id,
                session_id=assistant.get("session_id"),
                engine_id=assistant.get("engine_id"),
                model=assistant.get("model"),
                metadata={"task": payload.task} if payload.task else {},
            )
        identifier = conversation["id"]
        store.append_message(
            identifier, role="user", content=payload.message, client_id=payload.client_id
        )
        saved = store.append_message(
            identifier,
            role="assistant",
            content=assistant.get("text") or "",
            client_id=payload.client_id,
            engine_id=assistant.get("engine_id"),
            model=assistant.get("model"),
            finish_reason=assistant.get("finish_reason"),
            latency_ms=assistant.get("latency_ms"),
            usage=assistant.get("usage"),
            metadata={"runtime": assistant["runtime"]} if assistant.get("runtime") else {},
        )
        store.record_usage(
            client_id=payload.client_id,
            conversation_id=identifier,
            engine_id=assistant.get("engine_id"),
            model=assistant.get("model"),
            usage=assistant.get("usage"),
            latency_ms=assistant.get("latency_ms"),
            metadata={"source": "chat"},
        )
    except AlphaAIError as exc:
        return {"requested": True, "persisted": False, "error": _json_safe(exc.to_dict())}
    return {
        "requested": True,
        "persisted": True,
        "conversation_id": identifier,
        "message_id": saved.get("id"),
    }


def _persistence_for(
    request: Request, payload: ChatRequest, assistant: dict[str, Any]
) -> dict[str, Any]:
    """The persistence block of a chat response (skipped unless requested)."""

    if not payload.persist:
        return {"requested": False, "persisted": False, "detail": "persist=false"}
    if not payload.client_id:
        return {
            "requested": False,
            "persisted": False,
            "detail": (
                "no client_id supplied: the turn was answered and not stored. Send "
                "client_id to save conversations."
            ),
        }
    return _persist_turn(_store_for(request), payload, assistant=assistant)


def register_persistence_routes(
    app: FastAPI,
    *,
    store: Any,
    config: AlphaAIConfig | None,
    error_response: Any,
) -> None:
    """Register ``/api/conversations``, ``/api/preferences``, ``/api/usage``.

    Registered whether or not a database is configured: an unconfigured store
    answers every call with a structured ``database_not_configured`` error, which
    is far more useful than a 404 that looks like "no history yet".
    """

    def config_view() -> dict[str, Any]:
        if config is None:
            return {"configured": False}
        return public_config_view(config)["database"]

    def client_of(raw: str) -> str:
        """Validate the anonymous dashboard id before it reaches any store.

        The id scopes rows to one browser. It is not authentication, so it is
        only ever used as a parameter — and a malformed one is a 400 here rather
        than a driver error later.
        """

        return validate_client_id(raw)

    @app.get("/api/conversations", tags=["persistence"])
    def list_conversations(
        request: Request, client_id: str, limit: int = 50, offset: int = 0
    ) -> Any:
        active = _store_for(request)
        try:
            items = active.list_conversations(
                client_id=client_of(client_id), limit=limit, offset=offset
            )
        except AlphaAIError as exc:
            return error_response(exc)
        return {
            "ok": True,
            "count": len(items),
            "conversations": items,
            "persistence": {
                "configured": bool(active.configured),
                "client_id": client_id,
                "limit": limit,
                "offset": offset,
            },
        }

    @app.post("/api/conversations", tags=["persistence"])
    def create_conversation(payload: ConversationCreateRequest, request: Request) -> Any:
        try:
            conversation = _store_for(request).create_conversation(
                client_id=client_of(payload.client_id),
                title=payload.title,
                session_id=payload.session_id,
                engine_id=payload.engine_id,
                model=payload.model,
                metadata=payload.metadata,
                conversation_id=payload.conversation_id,
            )
        except AlphaAIError as exc:
            return error_response(exc)
        return {"ok": True, "conversation": conversation}

    @app.get("/api/conversations/{conversation_id}", tags=["persistence"])
    def get_conversation(
        conversation_id: str, request: Request, client_id: str | None = None
    ) -> Any:
        try:
            conversation = _store_for(request).get_conversation(
                conversation_id, client_id=client_of(client_id) if client_id else None
            )
        except AlphaAIError as exc:
            return error_response(exc)
        if conversation is None:
            return conversation_not_found(conversation_id)
        return {"ok": True, "conversation": conversation}

    @app.delete("/api/conversations/{conversation_id}", tags=["persistence"])
    def delete_conversation(
        conversation_id: str, request: Request, client_id: str | None = None
    ) -> Any:
        try:
            deleted = _store_for(request).delete_conversation(
                conversation_id, client_id=client_of(client_id) if client_id else None
            )
        except AlphaAIError as exc:
            return error_response(exc)
        if not deleted:
            return conversation_not_found(conversation_id)
        return {"ok": True, "deleted": conversation_id}

    @app.post("/api/conversations/{conversation_id}/messages", tags=["persistence"])
    def append_message(
        conversation_id: str, payload: MessageCreateRequest, request: Request
    ) -> Any:
        try:
            message = _store_for(request).append_message(
                conversation_id,
                role=payload.role,
                content=payload.content,
                client_id=client_of(payload.client_id),
                engine_id=payload.engine_id,
                model=payload.model,
                finish_reason=payload.finish_reason,
                latency_ms=payload.latency_ms,
                usage=payload.usage,
                metadata=payload.metadata,
            )
        except AlphaAIError as exc:
            return error_response(exc)
        return {"ok": True, "message": message}

    @app.get("/api/preferences", tags=["persistence"])
    def get_preferences(request: Request, client_id: str) -> Any:
        try:
            stored = _store_for(request).get_preferences(client_id=client_of(client_id))
        except AlphaAIError as exc:
            return error_response(exc)
        return {"ok": True, **stored}

    @app.put("/api/preferences", tags=["persistence"])
    def set_preferences(payload: PreferencesRequest, request: Request) -> Any:
        try:
            stored = _store_for(request).set_preferences(
                client_id=client_of(payload.client_id), preferences=payload.preferences
            )
        except AlphaAIError as exc:
            return error_response(exc)
        return {"ok": True, **stored}

    @app.get("/api/usage", tags=["persistence"])
    def usage(request: Request, client_id: str | None = None, limit: int = 50) -> Any:
        try:
            summary = _store_for(request).usage_summary(
                client_id=client_of(client_id) if client_id else None, limit=limit
            )
        except AlphaAIError as exc:
            return error_response(exc)
        return {"ok": True, **summary}

    @app.get("/api/database", tags=["persistence"])
    def database_status(request: Request) -> dict[str, Any]:
        """Connection, schema and migration state. Never fails."""

        return {
            "ok": True,
            "database": _store_for(request).status(),
            "config": config_view(),
        }

    @app.post("/api/database/migrate", tags=["persistence"])
    def database_migrate(request: Request, payload: MigrateRequest | None = None) -> Any:
        """Apply pending SQL migrations.

        An explicit operator action (and, on an inference host that sets
        ``ALPHAI_INFERENCE_TOKEN``, already protected by the token check).
        """

        try:
            report = _store_for(request).migrate(
                directory=payload.directory if payload else None
            )
        except AlphaAIError as exc:
            return error_response(exc)
        return {"ok": bool(report.get("ok", True)), "migration": report}

    app.state.persistence_routes = True


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


def local_runtime(
    config: AlphaAIConfig | str | None = None,
    *,
    project_root: str | None = None,
) -> AlphaRuntime:
    """Build the full local AlphaAI runtime, tolerating a read-only code root.

    A managed host (serverless platforms mount everything except a temp
    directory read-only) cannot keep AlphaAI's volatile directories next to the
    code. Rather than failing every request, those directories move to a
    writable location — see :func:`alphaai.config.loader.relocate_volatile_paths`
    — and the API keeps answering with the machine's *real* state: no engine is
    reported as usable until weights and a runtime actually exist, and
    ``ALPHAI_INFERENCE_URL`` points inference at the server that owns them.
    """

    from ..config.loader import load_config, relocate_volatile_paths, resolve_paths

    active = config if isinstance(config, AlphaAIConfig) else load_config(
        config, project_root=project_root
    )
    try:
        resolve_paths(active, create=True)
    except OSError as exc:
        moved = relocate_volatile_paths(active)
        logger.warning(
            "%s: %s is not writable (%s); moving runtime state to %s",
            NAME,
            active.paths.project_root,
            exc,
            moved.get("paths.state_dir", "a temporary directory"),
        )
        resolve_paths(active, create=True)
    return AlphaRuntime.create(active)


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
        try:
            app.state.runtime = runtime or local_runtime(config, project_root=project_root)
        except Exception as exc:  # noqa: BLE001 - the API must still answer in JSON
            # A runtime that cannot start is a real failure, but it must never
            # turn the API into a hosting platform's HTML error page: every
            # request then reports it as a structured 503 ("runtime_unavailable").
            logger.error("%s runtime could not start: %s", NAME, exc, exc_info=True)
            app.state.runtime = None
        logger.info("%s HTTP API ready", NAME)
        for line in attribution_lines():
            logger.info("%s", line)
        store = getattr(app.state, "database", None)
        if store is not None and store.configured:
            status = store.status()
            if status.get("reachable"):
                logger.info(
                    "persistence ready: %s/%s (%s)",
                    status.get("host"),
                    status.get("database"),
                    status.get("mode"),
                )
            else:
                logger.error(
                    "persistence is configured but unreachable: %s",
                    (status.get("error") or {}).get("message"),
                )
            if active_config is not None and active_config.database.migrate_on_start:
                try:
                    report = store.migrate()
                    logger.info(
                        "database migrations: applied %s, already applied %s",
                        report.get("applied") or "none",
                        len(report.get("already_applied") or []),
                    )
                except AlphaAIError as exc:  # the API still serves inference
                    logger.error("database migrations failed: %s", exc.message)
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

    # Persistence is created up front but connects lazily: a deployment whose
    # database is unreachable still boots and answers, reporting the database as
    # unreachable. Without DATABASE_URL this is an UnconfiguredStore, which gives
    # every persistence request a structured `database_not_configured` error.
    app.state.database = (
        open_database(active_config) if active_config is not None else open_database(AlphaAIConfig())
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

    @app.middleware("http")
    async def _json_errors(request: Request, call_next):
        """Report an unexpected failure as JSON, never as an HTML error page.

        A hosting platform replaces a crashed function with a page of its own,
        which a JSON client cannot read. The API therefore answers an
        unhandled exception with the same structured error shape as every other
        failure, so a deployment problem is visible and machine-readable.
        """

        try:
            return await call_next(request)
        except AlphaAIError as exc:
            return JSONResponse(
                {"ok": False, "error": _json_safe(exc.to_dict())},
                status_code=STATUS_BY_CODE.get(exc.code, 500),
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.exception("unhandled error in %s %s", request.method, request.url.path)
            return JSONResponse(
                {
                    "ok": False,
                    "error": {
                        "code": "internal_error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "remediation": (
                            "This is an AlphaAI failure, not a missing model. The traceback "
                            "for this request is in the server log."
                        ),
                    },
                },
                status_code=500,
            )

    if not gateway_base and api_config is not None and api_config.inference_token:
        # An inference host that owns weights can require the shared secret that
        # the gateway presents ($ALPHAI_INFERENCE_TOKEN). Without this the token
        # would only be *sent* by the gateway and never checked, which protects
        # nothing. The check covers the API the gateway forwards to (and the API
        # documentation) and answers 401 in the usual JSON error shape.
        # Consequences, by design: only a caller holding the token can talk to
        # this server. That is the point on a private inference host — the
        # gateway holds the token. A self-hosted deployment that also serves the
        # browser dashboard must therefore leave the token unset (or the
        # dashboard's own /api/* requests are refused), which is why the default
        # is empty.
        required_token = api_config.inference_token
        protected_routes = ("/api", "/docs", "/redoc", "/openapi.json")

        def _is_protected(path: str) -> bool:
            return path in protected_routes or path.startswith("/api/")

        @app.middleware("http")
        async def _require_inference_token(request: Request, call_next):
            if request.method == "OPTIONS" or not _is_protected(request.url.path):
                # CORS preflights never carry credentials; the browser only
                # preflights a cross-origin call, which the gateway never makes.
                return await call_next(request)
            presented = request.headers.get("authorization") or ""
            scheme, _, value = presented.partition(" ")
            if scheme.lower() == "bearer" and hmac.compare_digest(value.strip(), required_token):
                return await call_next(request)
            return JSONResponse(
                {
                    "ok": False,
                    "error": {
                        "code": "unauthorized",
                        "message": (
                            "This AlphaAI inference server requires the shared "
                            "ALPHAI_INFERENCE_TOKEN."
                        ),
                        "remediation": (
                            "Send `Authorization: Bearer <token>` with the same value as "
                            "ALPHAI_INFERENCE_TOKEN on this host (the AlphaAI gateway does "
                            "this automatically). Unset the variable to serve the API "
                            "without a token."
                        ),
                    },
                },
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

    def get_runtime(request: Request) -> AlphaRuntime:
        active = getattr(request.app.state, "runtime", None)
        if active is None:
            raise RuntimeUnavailableError(
                "The AlphaAI runtime is not available in this process.",
                remediation=(
                    "Check the server log for the startup error (usually a filesystem or "
                    "configuration problem) and restart the service."
                ),
            )
        return active

    def error_response(exc: AlphaAIError) -> JSONResponse:
        payload = {"ok": False, "error": _json_safe(exc.to_dict())}
        return JSONResponse(payload, status_code=STATUS_BY_CODE.get(exc.code, 500))

    if gateway_base:
        # Remote-inference deployment: the public API surface below is served by
        # the inference server, not by this process.
        if origins == ["*"]:
            logger.warning(
                "gateway mode is serving CORS origin '*' - set ALPHAI_CORS_ORIGINS to "
                "the deployed frontend origin (same-origin routing needs no wildcard)"
            )
        # A gateway that has its own DATABASE_URL serves history itself: a database
        # is not an inference concern, and this is what makes history available
        # even while the inference host is down. Without one, the routes stay
        # unregistered and the catch-all below forwards them to the inference
        # host, which owns the store in that deployment.
        decorate_health = None
        if app.state.database.configured:
            logger.info(
                "DATABASE_URL is set on this gateway: /api/conversations* are served "
                "here (connections use the pooler-friendly settings; a long-lived "
                "inference host would use the direct/session connection instead)"
            )
            register_persistence_routes(
                app,
                store=app.state.database,
                config=active_config,
                error_response=error_response,
            )
            decorate_health = database_health_decorator(app.state.database)
        configure_gateway(
            app,
            base_url=gateway_base,
            token=api_config.inference_token if api_config else "",
            timeout_s=api_config.inference_timeout_s if api_config else 300.0,
            dashboard=api_config.enable_dashboard if api_config else True,
            decorate_health=decorate_health,
        )
        return app

    # Local (inference) deployment: history is served by this process, and with
    # no database configured every persistence request answers with a structured
    # `database_not_configured` error rather than an empty, fabricated history.
    register_persistence_routes(
        app, store=app.state.database, config=active_config, error_response=error_response
    )

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
                "GET /api/conversations",
                "POST /api/conversations",
                "GET /api/conversations/{conversation_id}",
                "DELETE /api/conversations/{conversation_id}",
                "POST /api/conversations/{conversation_id}/messages",
                "GET /api/preferences",
                "PUT /api/preferences",
                "GET /api/usage",
                "GET /api/database",
                "POST /api/database/migrate",
            ],
            "note": "AlphaAI performs local inference only: no external AI provider, no API keys.",
        }

    @app.get("/api/health", tags=["meta"])
    def health(request: Request) -> dict[str, Any]:
        """Runtime health **and** database health.

        The database block is additive: AlphaAI's health endpoint keeps working
        when nothing is configured (``database.configured: false``) or when the
        database is down (``database.reachable: false`` with the real error).
        """

        payload = get_runtime(request).health()
        payload["database"] = _store_for(request).status()
        return payload

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
        body["persistence"] = _persistence_for(request, payload, body)
        return body

    @app.post("/api/chat/stream", tags=["chat"])
    def chat_stream(payload: ChatRequest, request: Request, format: str = "sse"):
        """Stream real tokens. ``format`` is ``sse`` (default) or ``ndjson``."""

        runtime_state = get_runtime(request)
        kwargs = _chat_kwargs(payload, runtime_state)
        use_sse = format != "ndjson"

        def event_stream() -> Iterator[str]:
            collected: list[str] = []
            done: dict[str, Any] | None = None
            try:
                for event in runtime_state.stream(payload.message, **kwargs):
                    if event.get("type") == "delta" and event.get("text"):
                        collected.append(str(event["text"]))
                    elif event.get("type") == "done":
                        done = event
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
            if done is not None:
                # Persistence happens after the last token reached the client, and
                # the outcome is streamed as its own event so the dashboard can say
                # whether the thread was saved — without inventing a save that did
                # not happen. Streamed usage is whatever the engine actually
                # reported: no token counts are estimated here.
                assistant = {
                    **done,
                    "text": "".join(collected),
                    "usage": {
                        "source": "stream",
                        "approximate_context_tokens": done.get("approximate_context_tokens"),
                        "token_source": done.get("token_source"),
                    },
                }
                outcome = _persistence_for(request, payload, assistant)
                saved = {"type": "persisted", **outcome}
                if use_sse:
                    yield f"data: {json.dumps(saved, ensure_ascii=False)}\n\n"
                else:
                    yield json.dumps(saved, ensure_ascii=False) + "\n"
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
    "ConversationCreateRequest",
    "MessageCreateRequest",
    "MigrateRequest",
    "OrchestrateRequest",
    "PreferencesRequest",
    "SkillExecuteRequest",
    "STATUS_BY_CODE",
    "ToolExecuteRequest",
    "create_app",
]

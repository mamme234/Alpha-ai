"""Remote-inference gateway.

AlphaAI always runs inference on hardware it owns: a GGUF model loaded by
``llama-cpp-python`` on a host with a persistent disk. Some hosting targets can
serve the dashboard and the HTTP API but cannot keep a model resident — a
serverless platform has no persistent filesystem and recycles instances. For
those targets the API can run in *gateway* mode: it forwards ``/api/*`` to a
separate, always-on AlphaAI inference server instead of loading weights itself.

Enable it by setting ``ALPHAI_INFERENCE_URL`` (alias: ``ALPHA_INFERENCE_URL``)
to the base URL of that inference server. In gateway mode:

* no model is loaded locally and nothing is written to disk,
* every public endpoint — including ``POST /api/chat/stream`` — is served by the
  inference server, so the API surface, the error format and the streaming
  behaviour are identical to a self-hosted AlphaAI,
* an unreachable inference server is reported as ``inference_unreachable``;
  the gateway never fabricates an answer.

``ALPHAI_INFERENCE_TOKEN`` (alias ``ALPHA_INFERENCE_TOKEN``) can be set on both
sides to require a shared bearer token, which is how the inference server stays
private while still reachable from the gateway. It is an AlphaAI-to-AlphaAI
shared secret, not an AI provider API key: no external provider is involved.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..config.loader import ConfigurationError
from ..core.errors import InferenceUnreachableError

logger = logging.getLogger("alphaai.api.gateway")

#: Headers that describe a single hop and must never be forwarded. The gateway
#: forwards bodies byte for byte, so it also asks the inference server for an
#: unencoded response instead of promising an encoding it cannot produce.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "accept-encoding",
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

#: Routes that document or describe the inference server's real API.
PROXIED_DOCUMENTATION_ROUTES = ("/openapi.json", "/docs", "/redoc")

_REMEDIATION = (
    "Run the real AlphaAI inference server on a host with a persistent disk "
    "(`alphaai serve` after `alphaai models install <model-id>`) and point "
    "ALPHAI_INFERENCE_URL at its base URL."
)


# ---------------------------------------------------------------------------
# url helpers
# ---------------------------------------------------------------------------
def normalize_inference_url(raw: str | None) -> str:
    """Return the gateway target without a trailing slash, or ``""`` if unset."""

    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ConfigurationError(
            f"api.inference_url must be an http(s) base URL, got {value!r}"
        )
    return value


def inference_host(url: str) -> str:
    """The host (with port) of an inference URL — safe to log or return."""

    parts = urlsplit(url)
    return parts.netloc or url


def _json_error(exc: InferenceUnreachableError) -> dict[str, Any]:
    return {"ok": False, "error": exc.to_dict()}


def _unreachable(cause: BaseException, base_url: str) -> InferenceUnreachableError:
    reason = getattr(cause, "reason", cause)
    return InferenceUnreachableError(
        f"AlphaAI inference server at {inference_host(base_url)} could not be reached: {reason}",
        remediation=_REMEDIATION,
        details={"inference_host": inference_host(base_url)},
    )


# ---------------------------------------------------------------------------
# passthrough
# ---------------------------------------------------------------------------
async def _forward(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_s: float,
):
    """Open the upstream request off the event loop (urllib is blocking)."""

    def send():
        request = urlrequest.Request(url, data=body or None, headers=headers, method=method)
        return urlrequest.urlopen(request, timeout=timeout_s)  # noqa: S310 - scheme validated

    return await run_in_threadpool(send)


def configure_gateway(
    app: FastAPI,
    *,
    base_url: str,
    token: str = "",
    timeout_s: float = 300.0,
    dashboard: bool = True,
) -> str:
    """Turn ``app`` into a stateless gateway in front of ``base_url``.

    Returns the normalized base URL that was registered.
    """

    base = normalize_inference_url(base_url)
    if not base:
        raise ConfigurationError("configure_gateway requires an AlphaAI inference URL")
    host = inference_host(base)

    def build_headers(request: Request) -> dict[str, str]:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers.setdefault("accept", "*/*")
        if token:
            # AlphaAI-to-AlphaAI shared secret; never an AI provider API key.
            headers["authorization"] = f"Bearer {token}"
        return headers

    async def proxy(request: Request, path: str) -> Response:
        target = f"{base}/{path.lstrip('/')}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        body = await request.body()
        try:
            upstream = await _forward(
                method=request.method,
                url=target,
                headers=build_headers(request),
                body=body or None,
                timeout_s=timeout_s,
            )
        except urlerror.HTTPError as exc:
            # The inference server answered with a real error status/body: pass
            # both through unchanged so clients see the authoritative response.
            payload = await run_in_threadpool(exc.read)
            media = exc.headers.get("content-type", "application/json") if exc.headers else "application/json"
            return Response(content=payload, status_code=exc.code, media_type=media)
        except (urlerror.URLError, OSError, TimeoutError) as exc:
            failure = _unreachable(exc, base)
            logger.warning("%s", failure.message)
            return JSONResponse(_json_error(failure), status_code=503)

        media = upstream.headers.get("content-type", "application/json")
        if "text/event-stream" in media or "ndjson" in media:
            # Stream event by event. Line iteration on the open response is what
            # keeps this incremental: the client sees each token as the
            # inference server produces it instead of one buffered blob.
            def relay() -> Iterator[bytes]:
                try:
                    for line in upstream:
                        yield line
                finally:
                    upstream.close()

            return StreamingResponse(
                relay(),
                status_code=upstream.status,
                media_type=media,
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )

        payload = await run_in_threadpool(upstream.read)
        upstream.close()
        return Response(content=payload, status_code=upstream.status, media_type=media)

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    async def alphaai_gateway(path: str, request: Request) -> Response:
        response = await proxy(request, f"/api/{path}")

        if path.strip("/") != "health" or response.status_code != 200:
            return response
        # Health is the one response the gateway enriches: it must show both the
        # inference server's real state and the fact that this API is a gateway.
        raw = getattr(response, "body", b"")
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return response
        if not isinstance(payload, dict):
            return response
        payload["gateway"] = {
            "enabled": True,
            "stateless": True,
            "inference_host": host,
            "inference_url": base,
            "token_required": bool(token),
            "reachable": True,
        }
        return JSONResponse(payload, status_code=response.status_code)

    for route in PROXIED_DOCUMENTATION_ROUTES:

        async def _documentation(request: Request, route: str = route) -> Response:
            return await proxy(request, route)

        app.add_api_route(route, _documentation, methods=["GET"], include_in_schema=False)

    if dashboard:
        from .app import dashboard_response

        def dashboard_view() -> Any:
            return dashboard_response()

        app.add_api_route("/", dashboard_view, methods=["GET"], include_in_schema=False)

    logger.info(
        "%s gateway mode: /api/* -> %s (token: %s)",
        "AlphaAI",
        base,
        "required" if token else "not set",
    )
    return base


__all__ = [
    "HOP_BY_HOP_HEADERS",
    "PROXIED_DOCUMENTATION_ROUTES",
    "configure_gateway",
    "inference_host",
    "normalize_inference_url",
]

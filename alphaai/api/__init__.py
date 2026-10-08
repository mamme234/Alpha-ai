"""AlphaAI HTTP API package.

``from alphaai.api import create_app`` returns the FastAPI application. FastAPI
and uvicorn live in the ``api`` extra (``pip install -e '.[api]'``), so this
module imports them lazily through :func:`create_app`.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

__all__ = ["create_app", "resolve_bind", "resolve_mode", "serve"]

DEFAULT_LOG_LEVEL = "info"


def create_app(*args: Any, **kwargs: Any):
    """Build the AlphaAI FastAPI app (see :mod:`alphaai.api.app`)."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def resolve_bind(
    host: str | None = None,
    port: int | None = None,
    *,
    fallback_host: str = "0.0.0.0",
    fallback_port: int = 8090,
    log_level: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, int, str]:
    """Resolve the production bind ``(host, port, log_level)``.

    Explicit arguments win, then the platform-provided ``$PORT`` (used by most
    hosting providers), then the configured ``api.port``. Resolution is pure so
    it can be unit tested without starting uvicorn.
    """

    source = os.environ if env is None else env
    resolved_port = port
    if resolved_port is None:
        raw_port = (source.get("PORT") or "").strip()
        if raw_port:
            try:
                resolved_port = int(raw_port)
            except ValueError as exc:
                raise ValueError(f"Invalid PORT environment variable: {raw_port!r}") from exc
    if resolved_port is None:
        resolved_port = fallback_port
    resolved_level = (log_level or source.get("ALPHAI_LOG_LEVEL") or DEFAULT_LOG_LEVEL).lower()
    return host or fallback_host, int(resolved_port), resolved_level


def resolve_mode(config: Any, *, create_dirs: bool = False) -> tuple[Any, bool]:
    """Return ``(config, gateway_mode)`` for a loaded AlphaAI configuration.

    A deployment that sets ``ALPHAI_INFERENCE_URL`` serves the API as a gateway
    in front of another AlphaAI inference server: it loads no model and creates
    no local directories (an inference host owns the real state and weights).
    """

    from ..config.loader import load_config, resolve_paths
    from ..config.schema import AlphaAIConfig

    active = config if isinstance(config, AlphaAIConfig) else load_config(config)
    gateway_mode = bool(active.api.inference_url.strip())
    if create_dirs and not gateway_mode:
        resolve_paths(active, create=True)
    return active, gateway_mode


def serve(
    config: Any = None,
    *,
    host: str | None = None,
    port: int | None = None,
    log_level: str | None = None,
) -> None:
    """Run the AlphaAI API with uvicorn (blocking). Used by ``alphaai serve``.

    With ``ALPHAI_INFERENCE_URL`` set this starts the stateless gateway instead
    of loading a local model.
    """

    import uvicorn

    from ..core.runtime import AlphaRuntime

    active, gateway_mode = resolve_mode(config, create_dirs=True)
    if gateway_mode:
        app = create_app(active)
    else:
        runtime = AlphaRuntime.create(active, create_dirs=True)
        active = runtime.config
        app = create_app(active, runtime=runtime)
    resolved_host, resolved_port, resolved_level = resolve_bind(
        host,
        port,
        fallback_host=active.api.host,
        fallback_port=active.api.port,
        log_level=log_level,
    )
    uvicorn.run(app, host=resolved_host, port=resolved_port, log_level=resolved_level)

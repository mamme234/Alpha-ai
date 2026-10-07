"""AlphaAI HTTP API package.

``from alphaai.api import create_app`` returns the FastAPI application. FastAPI
and uvicorn live in the ``api`` extra (``pip install -e '.[api]'``), so this
module imports them lazily through :func:`create_app`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "serve"]


def create_app(*args: Any, **kwargs: Any):
    """Build the AlphaAI FastAPI app (see :mod:`alphaai.api.app`)."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def serve(
    config: Any = None,
    *,
    host: str | None = None,
    port: int | None = None,
    log_level: str = "info",
) -> None:
    """Run the AlphaAI API with uvicorn (blocking). Used by ``alphaai serve``."""

    import uvicorn

    from ..config.loader import load_config
    from ..core.runtime import AlphaRuntime

    runtime = AlphaRuntime.create(config, create_dirs=True)
    active = runtime.config
    uvicorn.run(
        create_app(active, runtime=runtime),
        host=host or active.api.host,
        port=int(port or active.api.port),
        log_level=log_level,
    )

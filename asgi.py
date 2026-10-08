"""ASGI entrypoint for managed hosting.

``vercel.json`` points the ``backend`` service at this module
(``"entrypoint": "asgi:app"``), so hosting imports it and gets one ASGI
application:

* with ``ALPHAI_INFERENCE_URL`` set the app is the stateless AlphaAI gateway —
  it loads no model and forwards ``/api/*`` to a separate AlphaAI inference
  server (see :mod:`alphaai.api.gateway`);
* without it the app is the full AlphaAI API, which loads the local
  llama.cpp/GGUF model on the host it runs on.

Both surfaces are the same API: the endpoints, the error format and the
streaming behaviour never change.

A hosting platform answers a crashed function with a page of its own (HTML, or
plain text), which a JSON client cannot read. So if the application cannot even
be built — a missing dependency, an invalid ``ALPHAI_INFERENCE_URL``, a
packaging mistake — this module serves the failure itself as the API's usual
JSON error shape instead of handing the client a platform error page. The
traceback is included so the cause is visible from the deployment itself.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Awaitable, MutableMapping

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class StartupFailureApp:
    """Last-resort ASGI app: report a startup failure as readable JSON."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.detail = f"{type(error).__name__}: {error}"
        self.traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-4000:]

    def payload(self) -> dict[str, Any]:
        details: dict[str, Any] = {"exception": type(self.error).__name__}
        if self.traceback:
            details["traceback"] = self.traceback
        return {
            "ok": False,
            "error": {
                "code": "startup_failed",
                "message": f"AlphaAI API could not start: {self.detail}",
                "remediation": (
                    "This deployment failed while building the API, so no request can be "
                    "served. Check the deployed Python dependencies (requirements.txt / "
                    "pyproject.toml), the service entrypoint (asgi:app) and any ALPHAI_* "
                    "environment variable for the cause named above."
                ),
                "details": details,
            },
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return
        if scope["type"] != "http":
            return
        body = json.dumps(self.payload(), ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"" if scope.get("method") == "HEAD" else body,
            }
        )


def build_app() -> Any:
    """Build the AlphaAI ASGI app, or return the JSON startup-failure app."""

    try:
        from alphaai.api import create_app

        return create_app()
    except BaseException as exc:  # noqa: BLE001 - the failure must stay readable
        return StartupFailureApp(exc)


app = build_app()

__all__ = ["StartupFailureApp", "app", "build_app"]

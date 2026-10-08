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
"""

from __future__ import annotations

from alphaai.api import create_app

app = create_app()

__all__ = ["app"]

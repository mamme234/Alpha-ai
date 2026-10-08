"""Remote-inference gateway tests.

A deployment that cannot keep a model resident (serverless hosting) must serve
the real AlphaAI API by *forwarding* to an AlphaAI inference server. These tests
drive that code path against a real HTTP server on a real socket, so they
verify proxying, streaming, error pass-through and the "never fabricate an
answer" guarantee without mocking inference itself.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib import request as urlrequest

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from alphaai.api import resolve_mode
from alphaai.api.app import create_app
from alphaai.api.gateway import inference_host, normalize_inference_url
from alphaai.config.loader import load_config

# ---------------------------------------------------------------------------
# a real HTTP server standing in for the AlphaAI inference host
# ---------------------------------------------------------------------------


class _Upstream(BaseHTTPRequestHandler):
    """Records every request and answers like an AlphaAI server would."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        pass

    # -- helpers ----------------------------------------------------------
    def _body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, status: int, payload: Any, media: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", media)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, path: str, body: bytes) -> None:
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": path,
                "authorization": self.headers.get("authorization"),
                "accept": self.headers.get("accept"),
                "body": body,
            }
        )

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._record(self.path, b"")
        if self.path.startswith("/api/health"):
            self._send(200, {"ok": True, "inference": {"ready": True, "usable_models": 1}})
        elif self.path.startswith("/api/models"):
            self._send(200, {"ok": True, "count": 1, "models": [{"id": "qwen-test"}]})
        elif self.path == "/openapi.json":
            self._send(200, {"openapi": "3.1.0", "info": {"title": "AlphaAI API"}})
        else:
            self._send(404, {"ok": False, "error": {"code": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self._body()
        self._record(self.path, body)
        if self.path.startswith("/api/chat/stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            self._chunk(b'data: {"type":"delta","text":"one"}\n\n')
            # Only a *streaming* proxy delivers event one before event two.
            observed = self.server.first_event_read.wait(timeout=5)  # type: ignore[attr-defined]
            self.server.streaming_observed.append(observed)  # type: ignore[attr-defined]
            self._chunk(b'data: {"type":"delta","text":"two"}\n\n')
            self._chunk(b"event: alphaai-done\ndata: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        elif self.path.startswith("/api/chat"):
            self._send(200, {"ok": True, "text": "real answer from the inference host"})
        elif self.path.startswith("/api/tools/execute"):
            self._send(403, {"ok": False, "error": {"code": "tool_permission_denied"}})
        else:
            self._send(404, {"ok": False, "error": {"code": "not_found"}})

    def _chunk(self, payload: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
        self.wfile.flush()


@pytest.fixture
def upstream() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    server.daemon_threads = True
    server.requests = []  # type: ignore[attr-defined]
    server.streaming_observed = []  # type: ignore[attr-defined]
    server.first_event_read = threading.Event()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def base_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def gateway_client(config, server: ThreadingHTTPServer, **kwargs: Any) -> TestClient:
    app = create_app(config, inference_url=base_url(server), **kwargs)
    return TestClient(app)


@contextlib.contextmanager
def running_server(app: FastAPI) -> Iterator[str]:
    """Serve ``app`` over real HTTP (a real socket), like a deployment does.

    ``TestClient`` buffers responses, so streaming behaviour has to be verified
    against an actual ASGI server instead of an in-process transport.
    """

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:  # pragma: no cover - startup failure is a test failure
        server.should_exit = True
        raise RuntimeError("gateway server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ---------------------------------------------------------------------------
# url handling
# ---------------------------------------------------------------------------


def test_inference_url_normalisation() -> None:
    assert normalize_inference_url("") == ""
    assert normalize_inference_url(None) == ""
    assert normalize_inference_url("  https://infer.example.com/  ") == "https://infer.example.com"
    assert inference_host("https://infer.example.com:8443/api") == "infer.example.com:8443"


# ---------------------------------------------------------------------------
# proxying
# ---------------------------------------------------------------------------


def test_gateway_proxies_requests_verbatim(config, upstream) -> None:
    with gateway_client(config, upstream) as client:
        models = client.get("/api/models")
        assert models.status_code == 200
        assert models.json()["models"][0]["id"] == "qwen-test"

        chat = client.post("/api/chat", json={"message": "hello"})
        assert chat.status_code == 200
        assert chat.json()["text"] == "real answer from the inference host"

    seen = {item["path"].split("?")[0]: item for item in upstream.requests}
    assert set(seen) == {"/api/models", "/api/chat"}
    # The body reached the inference server unchanged...
    assert json.loads(seen["/api/chat"]["body"]) == {"message": "hello"}
    # ...and the path was not rewritten.
    assert seen["/api/chat"]["path"] == "/api/chat"


def test_gateway_forwards_query_strings_and_token(config, upstream) -> None:
    config.api.inference_token = "shared-secret"
    with gateway_client(config, upstream) as client:
        client.post("/api/chat/stream?format=ndjson", json={"message": "hi"})
    recorded = upstream.requests[-1]
    assert recorded["path"] == "/api/chat/stream?format=ndjson"
    assert recorded["authorization"] == "Bearer shared-secret"


def test_gateway_health_shows_remote_state_and_gateway_facts(config, upstream) -> None:
    with gateway_client(config, upstream) as client:
        payload = client.get("/api/health").json()
    # The inference server's real state is reported unchanged...
    assert payload["ok"] is True
    assert payload["inference"]["ready"] is True
    # ...plus the fact that this API is a stateless gateway.
    gateway = payload["gateway"]
    assert gateway["enabled"] is True and gateway["stateless"] is True
    assert gateway["reachable"] is True
    assert gateway["inference_host"] == inference_host(base_url(upstream))


def test_gateway_loads_no_local_model(config, upstream) -> None:
    with gateway_client(config, upstream) as client:
        client.get("/api/health")
        # No AlphaRuntime is created: this deployment owns no weights and writes
        # nothing to disk.
        assert getattr(client.app.state, "runtime", None) is None


def test_gateway_serves_the_dashboard(config, upstream) -> None:
    with gateway_client(config, upstream) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "ALPHA" in page.text


def test_gateway_proxies_the_inference_server_api_docs(config, upstream) -> None:
    with gateway_client(config, upstream) as client:
        docs = client.get("/openapi.json")
    assert docs.status_code == 200
    assert docs.json()["info"]["title"] == "AlphaAI API"


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


def test_gateway_streams_sse_progressively(config, upstream) -> None:
    app = create_app(config, inference_url=base_url(upstream))
    with running_server(app) as origin:
        call = urlrequest.Request(
            f"{origin}/api/chat/stream",
            data=b'{"message": "hi"}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(call, timeout=20) as response:  # noqa: S310 - local test server
            assert response.headers["content-type"].startswith("text/event-stream")
            first = next(line for line in response if line.startswith(b"data: "))
            assert b'"one"' in first
            # Event one arrived while the upstream was still holding event two.
            upstream.first_event_read.set()  # type: ignore[attr-defined]
            second = next(line for line in response if b'"two"' in line)
            assert b'"two"' in second
    assert upstream.streaming_observed == [True]  # type: ignore[attr-defined]


def test_gateway_streams_ndjson_instead_of_buffering(config, upstream) -> None:
    config.api.inference_token = ""
    app = create_app(config, inference_url=base_url(upstream))
    with running_server(app) as origin:
        call = urlrequest.Request(
            f"{origin}/api/chat/stream?format=ndjson",
            data=b'{"message": "hi"}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(call, timeout=20) as response:  # noqa: S310 - local test server
            # The request reached the inference server with its query string.
            assert upstream.requests[-1]["path"] == "/api/chat/stream?format=ndjson"
            assert b"event: alphaai-done" in response.read()


# ---------------------------------------------------------------------------
# failures are reported, never faked
# ---------------------------------------------------------------------------


def test_gateway_passes_through_upstream_error_status(config, upstream) -> None:
    with gateway_client(config, upstream) as client:
        response = client.post("/api/tools/execute", json={"tool_id": "shell.run"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "tool_permission_denied"


def test_unreachable_inference_is_reported_not_answered(config) -> None:
    # A port nobody listens on: bind, then close.
    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()

    app = create_app(config, inference_url=f"http://127.0.0.1:{dead_port}")
    with TestClient(app) as client:
        for path in ("/api/health", "/api/chat"):
            response = client.post(path, json={"message": "hi"}) if path.endswith("chat") else client.get(path)
            assert response.status_code == 503
            error = response.json()["error"]
            assert error["code"] == "inference_unreachable"
            assert "ALPHAI_INFERENCE_URL" in error["remediation"]
            # No invented answer, no invented health.
            assert "text" not in response.json()
            assert "inference" not in response.json()


# ---------------------------------------------------------------------------
# enabling gateway mode from the environment
# ---------------------------------------------------------------------------


def test_inference_url_from_environment(tmp_path, upstream) -> None:
    root = tmp_path / "hosted"
    root.mkdir()
    config = load_config(
        project_root=str(root),
        env={"ALPHAI_INFERENCE_URL": base_url(upstream) + "/"},
    )
    assert config.api.inference_url == base_url(upstream) + "/"
    active, gateway_mode = resolve_mode(config)
    assert gateway_mode is True
    assert active.api.inference_url.startswith("http://")

    with TestClient(create_app(config)) as client:
        assert client.get("/api/health").json()["gateway"]["reachable"] is True
    # Gateway mode created no local state directories.
    assert not (root / ".alphaai").exists()
    assert not (root / "models").exists()


def test_alpha_inference_url_alias_is_accepted(tmp_path) -> None:
    root = tmp_path / "hosted"
    root.mkdir()
    config = load_config(
        project_root=str(root),
        env={"ALPHA_INFERENCE_URL": "https://inference.internal.example"},
    )
    assert config.api.inference_url == "https://inference.internal.example"
    assert resolve_mode(config)[1] is True


def test_lookup_order_prefers_the_alphaai_prefixed_variable(tmp_path) -> None:
    root = tmp_path / "hosted"
    root.mkdir()
    config = load_config(
        project_root=str(root),
        env={
            "ALPHAI_INFERENCE_URL": "https://prefixed.example",
            "ALPHA_INFERENCE_URL": "https://alias.example",
        },
    )
    assert config.api.inference_url == "https://prefixed.example"


def test_local_inference_is_default_without_an_inference_url(config) -> None:
    _, gateway_mode = resolve_mode(config)
    assert gateway_mode is False
    with TestClient(create_app(config)) as client:
        health = client.get("/api/health").json()
    # A self-hosted AlphaAI reports hardware and engines, never a gateway block.
    assert "gateway" not in health
    assert health["inference"]["ready"] is False  # no model registered in this fixture


def test_invalid_inference_url_is_rejected(tmp_path) -> None:
    root = tmp_path / "hosted"
    root.mkdir()
    with pytest.raises(Exception) as excinfo:
        load_config(project_root=str(root), env={"ALPHAI_INFERENCE_URL": "infer.example.com"})
    assert "inference_url" in str(excinfo.value)

"""Production-serving guarantees: health semantics, CORS, limits, bind resolution.

These cover the properties a real deployment depends on:

* ``/api/health`` must not claim inference readiness when no model can serve.
* CORS origins come from configuration/environment, never a silent wildcard.
* Request bodies larger than ``api.max_request_bytes`` are rejected early.
* ``alphaai serve`` honours the hosting platform's ``$PORT`` and log level.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alphaai.api import resolve_bind
from alphaai.api.app import create_app
from alphaai.config.loader import load_config
from alphaai.core.runtime import AlphaRuntime

from .conftest import FakeEngine, test_spec


def make_client(config, *, with_engine: bool = False) -> TestClient:
    runtime = AlphaRuntime.create(config, discover=False)
    if with_engine:
        runtime.registry.register(FakeEngine(test_spec(), config))
    return TestClient(create_app(config, runtime=runtime))


# ---------------------------------------------------------------------------
# health: server running vs inference ready
# ---------------------------------------------------------------------------


def test_health_reports_no_inference_without_a_usable_model(config) -> None:
    with make_client(config) as client:
        payload = client.get("/api/health").json()
    # The server is up...
    assert payload["ok"] is True
    # ...but inference is explicitly not ready, and that is distinguishable.
    inference = payload["inference"]
    assert inference["ready"] is False
    assert inference["model_loaded"] is False
    assert inference["models_registered"] == 0
    assert inference["usable_models"] == 0
    assert "cannot generate responses" in inference["detail"]


def test_health_reports_inference_ready_with_a_usable_model(config) -> None:
    with make_client(config, with_engine=True) as client:
        inference = client.get("/api/health").json()["inference"]
    assert inference["ready"] is True
    assert inference["usable_models"] == 1
    assert inference["model_unavailable"] is False


# ---------------------------------------------------------------------------
# request limits
# ---------------------------------------------------------------------------


def test_oversized_request_body_is_rejected(config) -> None:
    config.api.max_request_bytes = 256
    with make_client(config, with_engine=True) as client:
        response = client.post("/api/chat", json={"message": "x" * 2000})
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "request_too_large"
    assert "256" in error["message"]
    assert error["remediation"]


def test_normal_request_is_not_blocked_by_the_limit(config) -> None:
    config.api.max_request_bytes = 1_048_576
    with make_client(config, with_engine=True) as client:
        response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json()["ok"] is True


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_allows_the_configured_frontend_origin(config) -> None:
    config.api.cors_origins = ["https://alphaai.vercel.app"]
    with make_client(config) as client:
        response = client.get(
            "/api/health", headers={"Origin": "https://alphaai.vercel.app"}
        )
    assert response.headers["access-control-allow-origin"] == "https://alphaai.vercel.app"


def test_cors_preflight_allows_a_post_from_the_frontend(config) -> None:
    config.api.cors_origins = ["https://alphaai.vercel.app"]
    with make_client(config) as client:
        response = client.options(
            "/api/chat",
            headers={
                "Origin": "https://alphaai.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://alphaai.vercel.app"
    assert "POST" in response.headers["access-control-allow-methods"]


# ---------------------------------------------------------------------------
# environment configuration
# ---------------------------------------------------------------------------


def test_frontend_origin_and_limits_come_from_the_environment(tmp_path) -> None:
    env = {
        "ALPHAI_CORS_ORIGINS": "https://alphaai.vercel.app, https://alpha-ai.dev",
        "ALPHAI_MAX_REQUEST_BYTES": "2048",
    }
    loaded = load_config(project_root=str(tmp_path), env=env)
    assert loaded.api.cors_origins == ["https://alphaai.vercel.app", "https://alpha-ai.dev"]
    assert loaded.api.max_request_bytes == 2048


def test_wildcard_is_not_used_when_an_explicit_origin_is_configured(config) -> None:
    """The deployed origin must be the *only* allowed one, not a wildcard."""

    config.api.cors_origins = ["https://alphaai.vercel.app"]
    with make_client(config) as client:
        allowed = client.get(
            "/api/health", headers={"Origin": "https://alphaai.vercel.app"}
        ).headers["access-control-allow-origin"]
        other = client.get(
            "/api/health", headers={"Origin": "https://evil.example"}
        ).headers.get("access-control-allow-origin")
    assert allowed == "https://alphaai.vercel.app"
    assert other is None


# ---------------------------------------------------------------------------
# production bind resolution
# ---------------------------------------------------------------------------


def test_platform_port_and_log_level_are_honoured() -> None:
    host, port, level = resolve_bind(env={"PORT": "8080", "ALPHAI_LOG_LEVEL": "warning"})
    assert (host, port, level) == ("0.0.0.0", 8080, "warning")


def test_explicit_bind_arguments_win_over_the_environment() -> None:
    host, port, level = resolve_bind(
        "127.0.0.1", 9000, log_level="DEBUG", env={"PORT": "8080"}
    )
    assert (host, port, level) == ("127.0.0.1", 9000, "debug")


def test_configured_port_is_used_when_the_platform_sets_none() -> None:
    host, port, level = resolve_bind(fallback_port=8090, env={})
    assert (host, port, level) == ("0.0.0.0", 8090, "info")


def test_invalid_platform_port_is_reported_not_ignored() -> None:
    with pytest.raises(ValueError, match="Invalid PORT"):
        resolve_bind(env={"PORT": "not-a-port"})


# ---------------------------------------------------------------------------
# hosting the API on a read-only project root
# ---------------------------------------------------------------------------


def test_read_only_project_root_still_serves_json_health(tmp_path) -> None:
    """A host that cannot hold AlphaAI's state must still answer with JSON.

    Serverless hosting mounts the project read-only. Before this, the first
    request died creating the state directory and the platform replaced the
    answer with its own error page, so no client could read the API at all.
    """

    root = tmp_path / "readonly"
    root.mkdir()
    # A *file* where the state directory belongs: every mkdir below it fails.
    (root / ".alphaai").write_text("occupied", encoding="utf-8")
    config = load_config(project_root=str(root), env={})

    with TestClient(create_app(config)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["ok"] is True
    # Inference is reported as unavailable — never invented.
    assert payload["inference"]["ready"] is False
    assert "gateway" not in payload


def test_relocating_volatile_paths_leaves_code_and_weights_alone(tmp_path) -> None:
    from alphaai.config.loader import relocate_volatile_paths, resolve_paths

    config = load_config(project_root=str(tmp_path), env={})
    resolve_paths(config)
    before_models = config.paths.models_dir
    before_configs = config.paths.configs_dir

    moved = relocate_volatile_paths(config, base=tmp_path / "writable")

    assert moved["paths.state_dir"] == str(tmp_path / "writable" / "state")
    assert config.paths.log_dir == str(tmp_path / "writable" / "logs")
    assert config.paths.workspace_dir == str(tmp_path / "writable" / "workspace")
    assert config.memory.path == str(tmp_path / "writable" / "state" / "memory.sqlite3")
    assert config.tools.sandbox_root == str(tmp_path / "writable" / "workspace")
    # Code, configs and weights are read-only inputs: they never move.
    assert config.paths.models_dir == before_models
    assert config.paths.configs_dir == before_configs


def test_volatile_paths_are_created_after_relocation(tmp_path) -> None:
    from alphaai.config.loader import relocate_volatile_paths, resolve_paths

    config = load_config(project_root=str(tmp_path), env={})
    relocate_volatile_paths(config, base=tmp_path / "writable")
    resolve_paths(config, create=True)

    assert (tmp_path / "writable" / "state").is_dir()
    assert (tmp_path / "writable" / "logs").is_dir()
    assert (tmp_path / "writable" / "workspace").is_dir()


def test_unexpected_errors_are_reported_as_json(config, monkeypatch) -> None:
    """An unhandled failure must not become a hosting platform's error page."""

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    runtime = AlphaRuntime.create(config, discover=False)
    monkeypatch.setattr(runtime, "health", boom)

    with TestClient(create_app(config, runtime=runtime)) as client:
        response = client.get("/api/health")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    assert "boom" in error["message"]


# ---------------------------------------------------------------------------
# the hosting entrypoint (asgi:app)
# ---------------------------------------------------------------------------


def test_entrypoint_startup_failure_is_reported_as_json(monkeypatch) -> None:
    """A broken deployment must still answer in the API's JSON error shape."""

    import importlib

    import alphaai.api as api_module

    def boom(*args, **kwargs):
        raise RuntimeError("fastapi is not installed")

    monkeypatch.setattr(api_module, "create_app", boom)
    entrypoint = importlib.import_module("asgi")
    app_under_test = entrypoint.build_app()

    response = TestClient(app_under_test).get("/api/health")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["code"] == "startup_failed"
    assert "RuntimeError: fastapi is not installed" in error["message"]
    assert "traceback" in error["details"]

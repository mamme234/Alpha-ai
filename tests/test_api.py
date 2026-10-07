"""API tests: the nine required endpoints plus structured error behaviour."""

from __future__ import annotations

from fastapi.testclient import TestClient

from alphaai.api.app import create_app
from alphaai.core.runtime import AlphaRuntime

from .conftest import FakeEngine, test_spec


def make_client(config, *, with_engine: bool = False) -> TestClient:
    runtime = AlphaRuntime.create(config, discover=False)
    if with_engine:
        runtime.registry.register(FakeEngine(test_spec(), config))
    app = create_app(config, runtime=runtime)
    return TestClient(app)


def test_meta_endpoints(config) -> None:
    with make_client(config) as client:
        index = client.get("/api").json()
        assert index["ok"] and index["name"] == "AlphaAI"
        assert "GET /api/health" in index["endpoints"]
        assert "no external AI provider" in index["note"]

        health = client.get("/api/health").json()
        assert health["ok"] is True
        assert health["engines"]["registered"] == 0
        assert health["skills"]["registered"] == 12
        assert health["tools"]["registered"] == 15
        assert health["hardware_lines"]

        attribution = client.get("/api/attribution").json()
        assert "DeepSeek" in attribution["engine"]
        assert "LICENSE-MODEL" in attribution["preserved_files"]

        runtime_info = client.get("/api/runtime").json()
        assert runtime_info["ok"]
        assert "doctor" in runtime_info

        config_view = client.get("/api/config").json()
        assert config_view["config"]["paths"]["models_dir"].startswith("<local>/")


def test_models_endpoints(config) -> None:
    with make_client(config, with_engine=True) as client:
        payload = client.get("/api/models").json()
        assert payload["count"] == 1 and payload["usable"] == 1
        model = payload["models"][0]
        assert model["model_owner"] == "AlphaAI"
        assert model["engine_owner"] == "AlphaAI"

        single = client.get("/api/models/alphaai-test").json()
        assert single["ok"] and single["model"]["id"] == "alphaai-test"

        missing = client.get("/api/models/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "unknown_model"


def test_skills_and_tools_endpoints(config) -> None:
    with make_client(config) as client:
        skills = client.get("/api/skills").json()
        assert skills["count"] == 12
        assert all("input_schema" in item for item in skills["skills"])

        tools = client.get("/api/tools").json()
        assert tools["count"] == 15
        permitted = {item["id"] for item in tools["tools"] if item["permissions"]["allowed"]}
        assert "calculator.evaluate" in permitted
        assert "web.search" not in permitted


def test_tool_execution_endpoint(config) -> None:
    with make_client(config) as client:
        ok = client.post(
            "/api/tools/execute",
            json={"tool_id": "calculator.evaluate", "arguments": {"expression": "7*6"}},
        ).json()
        assert ok["ok"] is True and ok["result"]["output"]["value"] == 42

        denied = client.post(
            "/api/tools/execute", json={"tool_id": "web.search", "arguments": {"query": "alphaai"}}
        ).json()
        assert denied["ok"] is False
        assert denied["result"]["error"]["code"] == "tool_permission_denied"

        unknown = client.post(
            "/api/tools/execute", json={"tool_id": "nope.nope", "arguments": {}}
        ).json()
        assert unknown["ok"] is False and unknown["result"]["error"]["code"] == "tool_not_found"

        calls = client.get("/api/tools/calls").json()["log"]
        assert calls["calls"] == 3


def test_skill_execution_endpoint(config) -> None:
    with make_client(config) as client:
        ok = client.post(
            "/api/skills/execute",
            json={"skill_id": "skill.calculator", "inputs": {"expression": "(2+3)*4"}},
        ).json()
        assert ok["ok"] and ok["result"]["output"]["results"][-1]["value"] == 20

        unknown = client.post("/api/skills/execute", json={"skill_id": "skill.nope"}).json()
        assert unknown["ok"] is False and unknown["result"]["error"]["code"] == "skill_not_found"


def test_chat_endpoint_returns_structured_error_without_engine(config) -> None:
    with make_client(config) as client:
        response = client.post("/api/chat", json={"message": "hello"})
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "no_suitable_model"
        assert error["remediation"]
        assert "candidates" in error["details"]


def test_chat_error_is_valid_json_when_models_are_unusable(config) -> None:
    # An engine that is registered and enabled but unusable scores -inf. The API
    # must still return strict JSON (503 no_suitable_model), not crash the
    # encoder with a 500 (which is what happened before scores were sanitised).
    from alphaai.core.types import EngineStatus

    class UnavailableEngine(FakeEngine):
        def health(self, *, refresh: bool = False) -> EngineStatus:
            return EngineStatus(
                engine_id=self.id,
                state="unavailable",
                detail="weights are not present locally",
                loaded=False,
                device="cpu",
                dtype="float32",
                weights_present=False,
                hardware_ok=True,
            )

    runtime = AlphaRuntime.create(config, discover=False)
    runtime.registry.register(UnavailableEngine(test_spec(), config))
    try:
        with TestClient(create_app(config, runtime=runtime)) as client:
            response = client.post("/api/chat", json={"message": "hello"})
            assert response.status_code == 503
            body = response.json()
            assert body["ok"] is False
            assert body["error"]["code"] == "no_suitable_model"
            candidates = body["error"]["details"]["candidates"]
            assert candidates and candidates[0]["score"] is None
    finally:
        runtime.close()


def test_chat_endpoint_with_engine(config) -> None:
    with make_client(config, with_engine=True) as client:
        payload = client.post("/api/chat", json={"message": "hello", "max_tokens": 32}).json()
        assert payload["ok"] is True
        assert payload["text"].startswith("alphaai test reply")
        assert payload["engine_id"] == "alphaai-test"
        assert payload["session_id"]
        assert payload["usage"]["total_tokens"] == 15

        # continuing the session keeps history
        session = payload["session_id"]
        again = client.post("/api/chat", json={"message": "again", "session_id": session}).json()
        assert again["session_id"] == session


def test_chat_stream_endpoint_sse_and_ndjson(config) -> None:
    with make_client(config, with_engine=True) as client:
        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as response:
            body = "".join(response.iter_text())
        assert "data:" in body
        assert '"type": "route"' in body or '"type":"route"' in body
        assert "[DONE]" in body

        ndjson = client.post("/api/chat/stream?format=ndjson", json={"message": "hi"}).text
        assert '"type": "done"' in ndjson or '"type":"done"' in ndjson
        assert "[DONE]" not in ndjson


def test_chat_stream_reports_error_event(config) -> None:
    with make_client(config) as client:
        body = client.post("/api/chat/stream?format=ndjson", json={"message": "hi"}).text
        assert "no_suitable_model" in body


def test_orchestrate_endpoint(config) -> None:
    with make_client(config) as client:
        declared = client.post(
            "/api/orchestrate",
            json={
                "mode": "declared",
                "goal": "calc",
                "steps": [{"id": "one", "skill": "skill.calculator", "input": {"expression": "2+2"}}],
            },
        ).json()
        assert declared["ok"] is True
        assert declared["steps"][0]["action"] == "one"

        goal = client.post("/api/orchestrate", json={"mode": "goal", "goal": "hello"}).json()
        assert goal["ok"] is False
        assert goal["error"]["code"] == "no_suitable_model"


def test_dashboard_is_served(config) -> None:
    with make_client(config) as client:
        html = client.get("/").text
        assert "ALPHA" in html
        assert "Intelligence, built from the ground up." in html
        assert "/api/chat/stream" in html


def test_dashboard_can_be_disabled(config) -> None:
    config.api.enable_dashboard = False
    with make_client(config) as client:
        payload = client.get("/").json()
        assert payload["detail"].startswith("AlphaAI dashboard is disabled")

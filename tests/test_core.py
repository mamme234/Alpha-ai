"""AlphaAI Core tests: model metadata, registry discovery, router decisions."""

from __future__ import annotations

import pytest

from alphaai.branding import ATTRIBUTION_SHORT, DEEPSEEK_V3_ENGINE_NAME
from alphaai.core.engine import load_model_specs, spec_from_dict, unknown_model_error
from alphaai.core.errors import EngineUnavailableError, NoSuitableModelError, UnknownModelError
from alphaai.core.registry import ModelRegistry
from alphaai.core.router import ModelRouter
from alphaai.core.types import Capability, ChatMessage

from .conftest import REPO_ROOT, FakeEngine, test_spec


def test_shipped_model_catalog_is_valid() -> None:
    specs = load_model_specs(directory=REPO_ROOT / "configs" / "models")
    ids = {spec.model_id for spec in specs}
    assert "deepseek-v3" in ids
    assert "alphaai-x" in ids
    assert len(specs) >= 9

    deepseek = next(spec for spec in specs if spec.model_id == "deepseek-v3")
    assert deepseek.model_owner == "DeepSeek"
    assert deepseek.engine_owner == "AlphaAI"
    assert deepseek.engine == "deepseek"
    assert deepseek.license_file == "LICENSE-MODEL"
    assert deepseek.weights_published is True
    assert DEEPSEEK_V3_ENGINE_NAME in deepseek.attribution
    assert "DeepSeek" in deepseek.attribution

    alpha = next(spec for spec in specs if spec.model_id == "alphaai-x")
    assert alpha.status == "not_trained"
    assert alpha.is_alphaai_owned
    assert alpha.weights_published is False


def test_spec_requires_attribution_fields() -> None:
    with pytest.raises(ValueError):
        spec_from_dict({"id": "broken", "context_length": 10})
    with pytest.raises(ValueError):
        spec_from_dict(
            {
                "id": "broken",
                "engine": "transformers",
                "family": "f",
                "provider": "p",
                "model": "m",
                "model_owner": "o",
                "context_length": 10,
                "capabilities": ["not-a-capability"],
            }
        )


def test_registry_discovery_reports_unavailable_engines(config) -> None:
    registry = ModelRegistry(config, specs=load_model_specs(directory=REPO_ROOT / "configs" / "models"))
    report = registry.discover()
    assert report.skipped == []
    assert "deepseek-v3" in report.registered
    status = registry.status("deepseek-v3")
    assert status.usable is False
    assert "weights" in status.detail.lower() or "runtime" in status.detail.lower()
    assert status.remediation


def test_registry_skips_disabled_engine_keys(config) -> None:
    config.engines.enabled = ["qwen"]
    registry = ModelRegistry(
        config,
        specs=[test_spec(model_id="m1", engine="qwen"), test_spec(model_id="m2", engine="gemma")],
    )
    report = registry.discover()
    assert report.registered == ["m1"]
    assert report.skipped and "disabled" in report.skipped[0]["reason"]


def test_registry_registers_engines_with_factories(config) -> None:
    registry = ModelRegistry(config, specs=[test_spec()])
    report = registry.discover(engine_factories={"transformers": FakeEngine})
    assert report.registered == ["alphaai-test"]
    engine = registry.get("alphaai-test")
    assert engine.health().usable is True
    assert registry.capability_matrix()["alphaai-test"]


def test_registry_unknown_model_error_lists_known_ids(config) -> None:
    registry = ModelRegistry(config, specs=[test_spec()])
    registry.discover(engine_factories={"transformers": FakeEngine})
    with pytest.raises(UnknownModelError) as excinfo:
        registry.get("nope")
    assert "alphaai-test" in excinfo.value.remediation
    assert unknown_model_error("x", ["a"]).code == "unknown_model"


def test_router_classifies_and_requires_capabilities(config) -> None:
    registry = ModelRegistry(config, specs=[test_spec()])
    registry.discover(engine_factories={"transformers": FakeEngine})
    router = ModelRouter(registry, config)

    assert router.classify_task("write a python class with a docstring") == "coding"
    assert router.classify_task("calculate 12 * 4") == "mathematics"
    assert router.classify_task("translate this to french") == "translation"
    assert router.classify_task("summarise the key points") == "summarization"
    assert router.classify_task("hello there") == "general"
    assert router.classify_task("hi", tools=[object()]) == "tool_calling"
    assert router.classify_task("long" * 20000) == "long_context"

    decision = router.choose([ChatMessage(role="user", content="hello")])
    assert decision.engine_id == "alphaai-test"
    assert decision.explicit is False
    assert decision.candidates and decision.candidates[0].usable


def test_router_raises_no_suitable_model_with_reasons(config) -> None:
    registry = ModelRegistry(config, specs=[test_spec(capabilities=["chat"])])
    registry.discover(engine_factories={"transformers": FakeEngine})
    router = ModelRouter(registry, config)
    with pytest.raises(NoSuitableModelError) as excinfo:
        router.choose([ChatMessage(role="user", content="write code: ```py```")], task="coding")
    error = excinfo.value
    assert "coding" in error.message or "capabilities" in error.message
    assert error.details["required_capabilities"] == ["chat", "coding"]
    assert error.remediation


def test_router_explicit_engine_must_be_usable(config) -> None:
    registry = ModelRegistry(config, specs=[test_spec()])
    registry.discover(engine_factories={"transformers": FakeEngine})
    router = ModelRouter(registry, config)
    decision = router.choose([ChatMessage(role="user", content="hi")], engine_id="alphaai-test")
    assert decision.explicit is True
    engine = registry.get("alphaai-test")
    engine.enabled = False
    with pytest.raises(EngineUnavailableError):
        router.choose([ChatMessage(role="user", content="hi")], engine_id="alphaai-test")


def test_router_explain_reports_failure_as_data(config) -> None:
    registry = ModelRegistry(config, specs=[])
    router = ModelRouter(registry, config)
    payload = router.explain("hello")
    assert payload["ok"] is False
    assert payload["error"]["code"] == "no_suitable_model"


def test_capability_parse_and_attribution_short() -> None:
    assert Capability.parse("long-context") is Capability.LONG_CONTEXT
    assert Capability.parse("TOOL_CALLING") is Capability.TOOL_CALLING
    spec = test_spec()
    assert spec.attribution.startswith(spec.display_name)
    assert ATTRIBUTION_SHORT in ATTRIBUTION_SHORT  # constant is stable

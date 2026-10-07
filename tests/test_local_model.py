"""Tests for the real local lightweight model path.

These cover the whole chain the task requires::

    CLI / API -> AlphaAI Core -> router -> engine interface -> local model -> real text

Unit tests stub the runtime (a fake ``llama_cpp`` module, a stub engine) because
what they assert is *AlphaAI's* behaviour: metadata, routing, tool policy,
install checks. Tests that must prove real inference are marked with
:data:`tests.conftest.requires_local_model` — they load the actual installed GGUF
model and generate tokens. If the weights or the runtime are absent they skip
with the exact reason; they are never replaced by a mock.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import pytest

from alphaai.config.schema import AlphaAIConfig
from alphaai.core.conversation import MAX_TOOL_SCHEMAS_PER_REQUEST, tool_categories_for
from alphaai.core.engine import ModelSpec, load_model_specs, spec_from_dict
from alphaai.core.installer import ModelInstaller
from alphaai.core.runtime import AlphaRuntime
from alphaai.core.router import ModelRouter
from alphaai.engines.hardware import (
    HardwareReport,
    RuntimePresence,
    detect_hardware,
    hardware_block,
    recommend_model_size,
)
from alphaai.engines.llama_cpp import LlamaCppEngine

from .conftest import LOCAL_MODEL_ID, REPO_ROOT, requires_local_model, test_spec

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fake_hardware(
    *, ram_gb: float = 8.0, available_gb: float = 6.0, gpus=(), llama_cpp: bool = True
) -> HardwareReport:
    return HardwareReport(
        cpu_count=4,
        cpu_threads=4,
        cpu_name="Test CPU",
        total_ram_gb=ram_gb,
        available_ram_gb=available_gb,
        gpus=list(gpus),
        disk_free_gb={"models": 50.0},
        runtime=RuntimePresence(numpy=True, llama_cpp=llama_cpp),
        platform="Linux (x86_64)",
        python="3.10.0",
    )


def gguf_spec(**overrides: Any) -> ModelSpec:
    """A GGUF model spec shaped like the real installed one."""

    payload: dict[str, Any] = {
        "id": "unit-gguf",
        "display_name": "AlphaAI llama.cpp Engine (Unit-GGUF)",
        "engine": "llama_cpp",
        "family": "qwen",
        "provider": "alibaba",
        "model": "Unit-GGUF",
        "model_owner": "Alibaba",
        "engine_owner": "AlphaAI",
        "context_length": 4096,
        "capabilities": ["chat", "streaming", "tool_calling", "coding", "mathematics"],
        "params_total_b": 0.49,
        "weight_formats": ["q4_k_m"],
        "runtime_requirements": {"q4_k_m": {"runtime": "llama_cpp", "min_ram_gb": 1}},
        "local_paths": ["models/unit-gguf"],
        "weight_files": ["unit-q4_k_m.gguf"],
        "provenance": {
            "source": "https://example.invalid/unit",
            "download_url": "https://example.invalid/unit/resolve/main/unit-q4_k_m.gguf",
            "filename": "unit-q4_k_m.gguf",
            "quantization": "q4_k_m",
            "format": "gguf",
            "file_size_bytes": 491_400_032,
        },
        "weights_url": "https://example.invalid/unit",
    }
    payload.update(overrides)
    return spec_from_dict(payload)


class FakeLlama:
    """Minimal stand-in for ``llama_cpp.Llama`` (isolated unit tests only)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.n_ctx = kwargs.get("n_ctx", 4096)
        self.calls: list[dict[str, Any]] = []

    def tokenize(self, data: bytes, add_bos: bool = False) -> list[int]:
        return list(range(len(data) // 4 + 1))

    def create_chat_completion(self, *, messages, stream=False, **kwargs):
        self.calls.append({"messages": messages, "stream": stream, "kwargs": kwargs})
        if stream:
            def generator():
                for index, piece in enumerate(["Hel", "lo ", "AlphaAI"]):
                    yield {
                        "choices": [
                            {"delta": {"content": piece}, "finish_reason": None, "index": index}
                        ]
                    }
                yield {"choices": [{"delta": {}, "finish_reason": "stop", "index": 3}]}

            return generator()
        return {
            "choices": [{"message": {"content": "AlphaAI unit reply"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }


@pytest.fixture
def installed_runtime(monkeypatch) -> None:
    """Pretend the engine runtime is installed (isolated installer/report tests)."""

    monkeypatch.setattr(ModelInstaller, "runtime_present", lambda self, spec: True)


@pytest.fixture
def fake_llama_cpp(monkeypatch) -> Iterator[type[FakeLlama]]:
    module = types.ModuleType("llama_cpp")
    module.__spec__ = importlib.machinery.ModuleSpec("llama_cpp", loader=None)
    module.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    yield FakeLlama


def write_gguf(directory: Path, name: str = "unit-q4_k_m.gguf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return path


# ---------------------------------------------------------------------------
# Phase 1 — hardware detection
# ---------------------------------------------------------------------------


def test_detect_hardware_reports_real_machine() -> None:
    report = detect_hardware([REPO_ROOT])
    assert report.cpu_count >= 1
    assert report.cpu_threads >= 1
    assert report.total_ram_gb > 0
    assert report.platform and report.python
    payload = report.to_dict()
    assert {"cpu_count", "total_ram_gb", "available_ram_gb", "gpus", "runtime"} <= set(payload)


def test_hardware_block_has_the_documented_labels() -> None:
    block = hardware_block(fake_hardware(), storage_paths=["models"])
    for key in (
        "cpu",
        "ram",
        "gpu",
        "vram",
        "storage",
        "architecture",
        "recommended_model_size",
        # Everything the hardware report is required to detect:
        "os",
        "python",
        "cpu_architecture",
        "runtimes",
        "supported_runtimes",
        "gpu_vendor",
    ):
        assert block[key], key
    assert block["recommended_model_size"].startswith("up to")
    assert block["recommendation"]["quantization"] == "q4_k_m"
    assert block["cpu_architecture"]
    assert block["os"]
    assert block["python"]


def test_hardware_block_reports_gpu_vendor_and_vram() -> None:
    from alphaai.engines.hardware import GpuInfo, describe_hardware_block

    report = fake_hardware(gpus=[GpuInfo(index=0, name="NVIDIA A100-SXM4-40GB", vram_gb=40.0)])
    block = hardware_block(report)
    assert block["gpu_vendor"] == "NVIDIA"
    assert "vendor: NVIDIA" in block["gpu"]
    assert "40.00 GB VRAM" in block["gpu"]
    assert block["vram"] == "40.00 GB"
    assert block["gpu_details"][0] == {
        "index": 0,
        "name": "NVIDIA A100-SXM4-40GB",
        "vendor": "NVIDIA",
        "vram_gb": 40.0,
        "backend": "cuda",
    }

    lines = describe_hardware_block(report)
    assert "AlphaAI Hardware" in lines[0]
    assert any(line.startswith("GPU: NVIDIA A100-SXM4-40GB (vendor: NVIDIA") for line in lines)
    assert any(line.startswith("VRAM: 40.00 GB total") for line in lines)
    assert any(line.startswith("Supported inference runtimes:") for line in lines)
    assert any(line.startswith("Recommended model size:") for line in lines)


def test_gpu_vendor_detection_and_cpu_only_report() -> None:
    from alphaai.engines.hardware import GpuInfo, guess_gpu_vendor

    assert guess_gpu_vendor("AMD Radeon RX 7900", "rocm") == "AMD"
    assert guess_gpu_vendor("Apple Metal (MPS)", "mps") == "Apple"
    assert guess_gpu_vendor("Intel Arc A770", "vulkan") == "Intel"
    assert guess_gpu_vendor("Mystery Accelerator", "xpu") == "unknown"
    assert GpuInfo(index=0, name="GeForce RTX 4090", vram_gb=24.0).vendor == "NVIDIA"

    block = hardware_block(fake_hardware())
    assert block["gpu"] == "none detected (CPU inference only)"
    assert block["gpu_vendor"] == "none (CPU inference only)"
    assert block["gpu_details"] == []


def test_model_size_recommendation_scales_with_ram() -> None:
    small = recommend_model_size(fake_hardware(ram_gb=2.0, available_gb=1.5))
    large = recommend_model_size(fake_hardware(ram_gb=64.0, available_gb=60.0))
    assert small.cpu_only and large.cpu_only
    assert small.recommended_params_b <= 1.5
    assert large.recommended_params_b > small.recommended_params_b
    assert small.max_params_b < large.max_params_b


def test_model_size_recommendation_prefers_vram_when_present() -> None:
    from alphaai.engines.hardware import GpuInfo

    gpu = GpuInfo(index=0, name="Test GPU", vram_gb=24.0, backend="cuda")
    recommendation = recommend_model_size(fake_hardware(ram_gb=32.0, gpus=[gpu]))
    assert not recommendation.cpu_only
    assert recommendation.backend == "cuda"
    assert recommendation.recommended_params_b >= 7.0


def test_tiny_machine_recommends_no_practical_model() -> None:
    recommendation = recommend_model_size(fake_hardware(ram_gb=0.5, available_gb=0.2))
    assert recommendation.fits_nothing
    assert "no practical local model" in recommendation.summary()


# ---------------------------------------------------------------------------
# Phase 2 — model metadata and provenance
# ---------------------------------------------------------------------------


def test_local_model_metadata_is_registered_with_provenance() -> None:
    specs = {spec.model_id: spec for spec in load_model_specs()}
    spec = specs[LOCAL_MODEL_ID]
    assert spec.engine == "llama_cpp"
    assert spec.model_owner == "Alibaba"  # AlphaAI never claims the weights
    assert spec.engine_owner == "AlphaAI"
    assert spec.status == "published"
    assert spec.license == "Apache-2.0"
    assert spec.quantization == "q4_k_m"
    assert {"chat", "streaming", "tool_calling"} <= {cap.value for cap in spec.capabilities}

    provenance = spec.provenance
    assert provenance["repo"] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"  # authoritative source
    assert provenance["format"] == "gguf"
    assert provenance["revision"]
    assert provenance["sha256"] and len(provenance["sha256"]) == 64
    assert provenance["file_size_bytes"] > 0
    assert provenance["runtime"] == "llama_cpp"
    assert "created by Alibaba" in spec.attribution


def test_model_metadata_surfaces_provenance_and_quantization() -> None:
    payload = gguf_spec().to_dict()
    assert payload["quantization"] == "q4_k_m"
    assert payload["provenance"]["quantization"] == "q4_k_m"


def test_selection_rationale_surfaces_in_model_metadata() -> None:
    """A recorded selection is exposed through the spec, and absent by default."""

    assert gguf_spec().selection == {}
    assert gguf_spec().to_dict()["selection"] == {}
    spec = gguf_spec(selection={"phase": "select-model", "decision": "unit choice"})
    assert spec.selection["decision"] == "unit choice"
    assert spec.to_dict()["selection"]["phase"] == "select-model"


def test_local_model_records_the_selection_rationale() -> None:
    """The selected model records why it fits *this* machine, not a bare claim."""

    specs = {spec.model_id: spec for spec in load_model_specs()}
    spec = specs[LOCAL_MODEL_ID]
    selection = spec.selection
    assert selection, "the selected model must record its selection rationale"
    assert selection["phase"] == "select-model"

    criteria = selection["criteria"]
    assert criteria["inference_runtime"] == "llama_cpp"
    assert "q4_k_m" in criteria["quantization_preference"]
    low, high = criteria["parameter_range_b"]
    assert low <= spec.params_total_b <= high

    # The justification is tied to the measurements from `alphaai doctor`.
    basis = selection["hardware_basis"]
    assert basis["source"] == "alphaai doctor"
    assert basis["backend"] == "cpu"
    assert basis["total_ram_gb"] > 0
    assert spec.params_total_b <= basis["recommended_params_b"] + 0.01

    # Rejected alternatives and the official source are both on the record.
    assert selection["alternatives_considered"]
    assert selection["official_source"]["repo"] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert selection["official_source"]["license"] == spec.license
    assert selection["official_source"]["gated"] is False
    assert "created by Alibaba" in spec.attribution


def test_engine_exposes_metadata_and_capabilities(fake_llama_cpp, config: AlphaAIConfig) -> None:
    spec = gguf_spec()
    engine = LlamaCppEngine(spec, config, hardware=fake_hardware())
    metadata = engine.metadata()
    assert metadata["id"] == "unit-gguf"
    assert metadata["engine_name"] == "AlphaAI llama.cpp Engine"
    capabilities = engine.capabilities_dict()
    assert capabilities["supports_streaming"] and capabilities["supports_tool_calling"]
    assert capabilities["engine"] == "llama_cpp"


# ---------------------------------------------------------------------------
# registration + unavailable handling
# ---------------------------------------------------------------------------


def test_registry_discovers_the_local_model_from_repo_config() -> None:
    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        engine = runtime.registry.get(LOCAL_MODEL_ID)
        assert isinstance(engine, LlamaCppEngine)
        assert engine.spec.engine_owner == "AlphaAI"
        assert engine.spec.model_owner == "Alibaba"
        assert engine.health().state in {"available", "ready", "unavailable"}
    finally:
        runtime.close()


def test_missing_weights_are_reported_as_unavailable(tmp_path: Path, config: AlphaAIConfig) -> None:
    spec = gguf_spec(id="ghost-gguf", local_paths=["models/ghost-gguf"])
    engine = LlamaCppEngine(spec, config, hardware=fake_hardware())
    status = engine.health(refresh=True)
    assert status.state == "unavailable"
    assert status.usable is False
    assert status.weights_present is False
    assert status.remediation and "ghost-gguf" in status.remediation


def test_hardware_that_cannot_hold_the_model_is_unavailable(tmp_path: Path, config: AlphaAIConfig) -> None:
    write_gguf(Path(config.paths.models_dir) / "unit-gguf")
    spec = gguf_spec(params_total_b=70.0, runtime_requirements={"q4_k_m": {"runtime": "llama_cpp", "min_ram_gb": 48}})
    engine = LlamaCppEngine(spec, config, hardware=fake_hardware(ram_gb=4.0, available_gb=2.0))
    status = engine.health(refresh=True)
    assert status.state == "unavailable"
    assert status.hardware_ok is False
    assert "Insufficient local resources" in status.detail


def test_engine_without_runtime_reports_the_install_command(monkeypatch, config: AlphaAIConfig) -> None:
    monkeypatch.setattr("alphaai.engines.llama_cpp.module_present", lambda name: False)
    spec = gguf_spec()
    engine = LlamaCppEngine(spec, config, hardware=fake_hardware())
    status = engine.health(refresh=True)
    assert status.state == "unavailable"
    assert "not installed" in status.detail
    assert status.remediation and "pip install" in status.remediation


# ---------------------------------------------------------------------------
# Phase 9 — routing
# ---------------------------------------------------------------------------


def test_arithmetic_requests_route_to_the_math_tool_policy() -> None:
    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        router = ModelRouter(runtime.registry, runtime.config)
        for text in ("What is 1234 × 5678?", "What is 25 multiplied by 4?", "compute 2+2"):
            assert router.classify_task(text) == "mathematics", text
        assert tool_categories_for("mathematics") == ("math",)
        assert tool_categories_for("general") == ()
    finally:
        runtime.close()


def test_router_prefers_the_local_model_for_general_chat(engine_runtime) -> None:
    decision = engine_runtime.router.choose("hello there")
    assert decision.engine_id == "alphaai-test"
    assert decision.explicit is False


def test_router_ignores_engines_that_cannot_run(config: AlphaAIConfig) -> None:
    from alphaai.core.types import EngineStatus

    class Broken(LlamaCppEngine):
        def health(self, *, refresh: bool = False) -> EngineStatus:
            return EngineStatus(engine_id=self.id, state="unavailable", detail="no weights")

    runtime = AlphaRuntime.create(config, discover=False)
    runtime.registry.register(Broken(gguf_spec(), config, hardware=fake_hardware()))
    try:
        outcome = runtime.route("hello")
        assert outcome["ok"] is False
        assert outcome["error"]["code"] == "no_suitable_model"
    finally:
        runtime.close()


def test_model_spec_priority_config_prefers_the_local_model() -> None:
    from alphaai.config.loader import load_config

    config = load_config(project_root=str(REPO_ROOT), env={})
    assert config.routing.task_preferences["general"] == [LOCAL_MODEL_ID]
    options = config.engines.options[LOCAL_MODEL_ID]
    assert options["n_ctx"] >= 1024 and options["cpu_threads"] >= 0


# ---------------------------------------------------------------------------
# Phase 4/5 — the inference engine (isolated unit test with a stubbed runtime)
# ---------------------------------------------------------------------------


def test_llama_cpp_engine_generates_and_streams(fake_llama_cpp, config: AlphaAIConfig) -> None:
    from alphaai.core.types import ChatMessage, GenerationRequest, SamplingParams

    write_gguf(Path(config.paths.models_dir) / "unit-gguf")
    engine = LlamaCppEngine(gguf_spec(), config, hardware=fake_hardware())
    engine.load()
    assert engine.health().state == "ready"

    request = GenerationRequest(
        messages=[ChatMessage(role="user", content="hi")],
        sampling=SamplingParams(temperature=0.0, max_tokens=8),
    )
    result = engine.generate(request)
    assert result.text == "AlphaAI unit reply"
    assert result.engine_id == "unit-gguf" and result.model == "Unit-GGUF"
    assert result.usage.prompt_tokens == 7
    assert "llama_cpp" in result.runtime
    assert engine._llama.calls[0]["messages"][-1]["role"] == "user"

    chunks = [chunk for chunk in engine.stream(request) if not chunk.done]
    assert "".join(chunk.text for chunk in chunks) == "Hello AlphaAI"


def test_llama_cpp_engine_injects_tool_schemas_into_the_system_message(
    fake_llama_cpp, config: AlphaAIConfig
) -> None:
    from alphaai.core.types import ChatMessage, GenerationRequest, ToolSpecView

    write_gguf(Path(config.paths.models_dir) / "unit-gguf")
    engine = LlamaCppEngine(gguf_spec(), config, hardware=fake_hardware())
    engine.load()
    request = GenerationRequest(
        messages=[ChatMessage(role="system", content="You are AlphaAI."), ChatMessage(role="user", content="2+2?")],
        tools=[
            ToolSpecView(
                tool_id="calculator.evaluate",
                name="calculator.evaluate",
                description="Evaluate an expression exactly.",
                parameters={"type": "object"},
            )
        ],
    )
    engine.generate(request)
    system = engine._llama.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "calculator.evaluate" in system["content"]
    assert "You can call tools" in system["content"]


def test_tool_policy_scopes_schemas_to_relevant_categories(engine_runtime) -> None:
    conversation = engine_runtime.conversation
    math = conversation.request_tools(None, enabled=True, text="What is 1234 × 5678?")
    chat = conversation.request_tools(None, enabled=True, text="hello there")
    assert [schema["function"]["name"] for schema in math] == ["calculator_evaluate"]
    assert len(math) <= MAX_TOOL_SCHEMAS_PER_REQUEST
    assert chat == []


# ---------------------------------------------------------------------------
# Phase 13 — download safety and verification
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("installed_runtime")
def test_install_plan_reports_required_versus_available(tmp_path: Path, config: AlphaAIConfig) -> None:
    installer = ModelInstaller(config, hardware=fake_hardware(ram_gb=2.0, available_gb=1.5))
    plan = installer.plan(gguf_spec())
    payload = plan.to_dict()
    for key in (
        "model_id",
        "source",
        "quantization",
        "runtime",
        "required_ram_gb",
        "available_ram_gb",
        "required_storage_gb",
        "available_storage_gb",
        "target_path",
    ):
        assert key in payload
    assert plan.verdict == "ok" and plan.ok
    assert any(line.startswith("Available RAM:") for line in plan.report_lines())


def test_install_plan_stops_when_the_model_does_not_fit(config: AlphaAIConfig) -> None:
    installer = ModelInstaller(config, hardware=fake_hardware(ram_gb=0.6, available_gb=0.3))
    spec = gguf_spec(params_total_b=7.0, runtime_requirements={"q4_k_m": {"runtime": "llama_cpp", "min_ram_gb": 8}})
    plan = installer.plan(spec)
    assert plan.ok is False and plan.verdict == "insufficient"
    assert any("Not enough memory" in reason for reason in plan.reasons)
    assert plan.remediation and "smaller" in plan.remediation
    result = installer.install(spec)
    assert result["ok"] is False and result["status"] == "blocked"
    assert result["downloaded"] is False


def test_install_plan_refuses_a_model_without_an_authoritative_source(config: AlphaAIConfig) -> None:
    installer = ModelInstaller(config, hardware=fake_hardware())
    spec = gguf_spec(provenance={"filename": "x.gguf", "format": "gguf"}, weights_url=None)
    plan = installer.plan(spec)
    assert plan.ok is False
    assert any("no download URL" in reason for reason in plan.reasons)


def test_verify_detects_size_sha_and_magic_mismatches(tmp_path: Path, config: AlphaAIConfig) -> None:
    installer = ModelInstaller(config, hardware=fake_hardware())
    good = write_gguf(tmp_path, "good.gguf")
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    assert installer.verify(good, expected_sha256=digest, expected_size=good.stat().st_size)["ok"]

    bad_sha = installer.verify(good, expected_sha256="0" * 64)
    assert bad_sha["ok"] is False and "sha256 mismatch" in bad_sha["detail"]

    bad_size = installer.verify(good, expected_size=1)
    assert bad_size["ok"] is False and "size mismatch" in bad_size["detail"]

    not_gguf = tmp_path / "fake.gguf"
    not_gguf.write_bytes(b"NOPE" + b"\x00" * 16)
    checked = installer.verify(not_gguf)
    assert checked["ok"] is False and "not a GGUF" in checked["detail"]

    assert installer.verify(tmp_path / "missing.gguf")["ok"] is False


@pytest.mark.usefixtures("installed_runtime")
def test_install_downloads_verifies_records_and_load_tests(tmp_path: Path, monkeypatch, config: AlphaAIConfig) -> None:
    """Full install workflow against a stubbed transport and a stub engine."""

    payload = b"GGUF" + b"alphaai-unit-model" * 4
    digest = hashlib.sha256(payload).hexdigest()

    class FakeResponse(io.BytesIO):
        headers = {"content-length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=60: FakeResponse(payload))

    metadata = REPO_ROOT / "configs" / "models" / f"{LOCAL_MODEL_ID}.json"
    spec_payload = json.loads(metadata.read_text(encoding="utf-8"))
    spec_payload["id"] = "unit-install-gguf"
    spec_payload["local_paths"] = [f"models/unit-install-gguf"]
    spec_payload["provenance"] = dict(spec_payload["provenance"])
    spec_payload["provenance"].update(
        {"filename": "unit.gguf", "file_size_bytes": len(payload), "sha256": digest}
    )
    source = Path(config.paths.configs_dir) / "models"
    source.mkdir(parents=True, exist_ok=True)
    spec_file = source / "unit-install-gguf.json"
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")
    spec = spec_from_dict(spec_payload, source_path=str(spec_file))

    loaded: list[str] = []

    class StubEngine:
        def __init__(self, spec, config, **kwargs) -> None:
            self.spec = spec

        def load(self) -> None:
            loaded.append("load")

        def unload(self) -> None:
            loaded.append("unload")

        def generate(self, request):
            from alphaai.core.types import GenerationResult, TokenUsage

            return GenerationResult(
                text="Hello from the unit model.",
                engine_id=self.spec.model_id,
                model=self.spec.model,
                usage=TokenUsage(prompt_tokens=5, completion_tokens=6, total_tokens=11),
                runtime="llama_cpp/cpu",
                attribution=self.spec.attribution,
            )

    installer = ModelInstaller(
        config,
        hardware=fake_hardware(),
        engine_factory=lambda spec, cfg: StubEngine(spec, cfg),
    )
    result = installer.install(spec)
    assert result["ok"] is True, result
    assert result["downloaded"] is True
    assert result["verified"]["sha256"] == digest
    assert result["load_test"]["ok"] is True
    assert loaded == ["load", "unload"]

    target = Path(config.paths.models_dir) / "unit-install-gguf" / "unit.gguf"
    assert target.read_bytes() == payload
    recorded = json.loads(spec_file.read_text(encoding="utf-8"))["provenance"]
    assert recorded["sha256"] == digest and recorded["load_test"]["ok"] is True


@pytest.mark.usefixtures("installed_runtime")
def test_install_does_not_mark_available_when_the_load_test_fails(config: AlphaAIConfig, monkeypatch) -> None:
    payload = b"GGUF" + b"x" * 16

    class FakeResponse(io.BytesIO):
        headers = {"content-length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=60: FakeResponse(payload))

    class ExplodingEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load(self) -> None:
            raise RuntimeError("cannot allocate memory")

        def unload(self) -> None:
            pass

    installer = ModelInstaller(
        config,
        hardware=fake_hardware(),
        engine_factory=lambda spec, cfg: ExplodingEngine(),
    )
    result = installer.install(gguf_spec(id="unit-broken", provenance={
        "filename": "broken.gguf",
        "format": "gguf",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_size_bytes": len(payload),
        "source": "https://example.invalid/unit",
    }))
    assert result["ok"] is False
    assert result["status"] == "load_failed"
    assert result["error"]["code"] == "engine_load_failed"
    assert "did not load" in result["error"]["remediation"]


# ---------------------------------------------------------------------------
# Phase 14 — real inference through the real model
# ---------------------------------------------------------------------------


@requires_local_model
def test_real_model_generates_text_through_the_engine() -> None:
    """Integration: the installed model loads and writes real text."""

    from alphaai.core.types import ChatMessage, GenerationRequest, SamplingParams

    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        engine = runtime.registry.get(LOCAL_MODEL_ID)
        status = engine.health(refresh=True)
        assert status.usable, status.detail
        engine.load()
        assert engine.health().state == "ready"
        result = engine.generate(
            GenerationRequest(
                messages=[ChatMessage(role="user", content="Reply with the single word: ready")],
                sampling=SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8),
                engine_id=LOCAL_MODEL_ID,
            )
        )
        assert result.text.strip(), "the real model produced no text"
        assert result.usage.completion_tokens > 0
        assert result.engine_id == LOCAL_MODEL_ID
        assert "created by Alibaba" in result.attribution
        # The real local file matches the checksum recorded in the registry.
        digest = hashlib.sha256()
        with open(engine.gguf_path(), "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == engine.spec.provenance["sha256"]
    finally:
        runtime.close()


@requires_local_model
def test_real_model_chat_uses_core_and_answers_with_real_text() -> None:
    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        outcome = runtime.chat("What is 25 multiplied by 4?", max_new_tokens=24)
        assert outcome.engine_id == LOCAL_MODEL_ID
        assert outcome.model.startswith("Qwen2.5-0.5B-Instruct")
        assert outcome.text.strip()
        assert outcome.usage.completion_tokens > 0
        assert "llama_cpp" in outcome.runtime
    finally:
        runtime.close()


@requires_local_model
def test_real_model_tool_calling_uses_the_real_calculator() -> None:
    """The router picks the calculator, the model asks for it, AlphaAI executes it."""

    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        outcome = runtime.chat("What is 1234 × 5678?", use_tools=True, max_new_tokens=40)
        assert [result.tool_id for result in outcome.tool_results] == ["calculator.evaluate"]
        result = outcome.tool_results[0]
        assert result.ok and result.output["value"] == 1234 * 5678
        assert str(1234 * 5678) in outcome.text.replace(",", "")
    finally:
        runtime.close()


@requires_local_model
def test_real_model_streams_progressively() -> None:
    import time

    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        session = runtime.create_session()
        started = time.perf_counter()
        arrivals: list[float] = []
        done = None
        for event in runtime.stream("Count from one to five.", session=session, max_new_tokens=24):
            if event["type"] == "delta" and event["text"]:
                arrivals.append(time.perf_counter() - started)
            elif event["type"] == "done":
                done = event
        assert arrivals, "no streamed tokens arrived"
        assert done is not None and done["model"].startswith("Qwen2.5-0.5B-Instruct")
        # Tokens are produced while decoding, not returned in one final batch.
        assert arrivals[0] <= arrivals[-1]
        assert done["latency_ms"] >= arrivals[0] * 1000
    finally:
        runtime.close()


@requires_local_model
def test_real_model_remembers_earlier_turns() -> None:
    """Multi-turn context is real: the answer comes from the conversation."""

    runtime = AlphaRuntime.create(project_root=str(REPO_ROOT), create_dirs=False)
    try:
        session = runtime.create_session()
        runtime.chat("My name is Alex.", session=session, max_new_tokens=16)
        answer = runtime.chat(
            "What is my name? Reply with exactly: Your name is <name>.", session=session, max_new_tokens=24
        )
        assert "Alex" in answer.text, answer.text
    finally:
        runtime.close()


@requires_local_model
def test_api_chat_and_stream_use_the_real_model(config: AlphaAIConfig) -> None:
    from fastapi.testclient import TestClient

    from alphaai.api.app import create_app

    app = create_app(project_root=str(REPO_ROOT))
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["engines"]["usable"] >= 1

        model = client.get(f"/api/models/{LOCAL_MODEL_ID}").json()["model"]
        assert model["status"]["usable"] is True
        assert model["provenance"]["sha256"]

        payload = client.post("/api/chat", json={"message": "What is 25 multiplied by 4?", "max_tokens": 24}).json()
        assert payload["ok"] is True
        assert payload["engine_id"] == LOCAL_MODEL_ID
        assert payload["text"].strip()
        assert payload["usage"]["completion_tokens"] > 0

        deltas = []
        with client.stream(
            "POST", "/api/chat/stream?format=ndjson", json={"message": "Say hello.", "max_tokens": 24}
        ) as response:
            for line in response.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event["type"] == "delta":
                    deltas.append(event["text"])
                if event["type"] == "done":
                    assert event["model"].startswith("Qwen2.5-0.5B-Instruct")
        assert deltas and "".join(deltas).strip()

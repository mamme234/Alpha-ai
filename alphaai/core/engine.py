"""The AlphaAI ``ModelEngine`` interface and model metadata.

Every engine — the DeepSeek engine, the Hugging Face ``transformers`` engine,
the GGUF ``llama.cpp`` engine and any future AlphaAI-trained engine — implements
this interface. AlphaAI Core only ever talks to this interface.

A ``ModelSpec`` is *metadata about a model* (who owns it, what it can do, what
it needs). It is deliberately separate from the engine that runs it, which is
what lets AlphaAI plug an AlphaAI-trained model into the same core later.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..branding import engine_attribution
from ..config.schema import AlphaAIConfig
from ..engines.hardware import (
    AvailabilityAssessment,
    HardwareReport,
    assess_model,
    detect_hardware,
)
from .errors import EngineLoadError, EngineUnavailableError, UnknownModelError
from .types import (
    Capability,
    ChatMessage,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
    TokenUsage,
)

#: How long a health snapshot stays fresh before AlphaAI re-probes.
STATUS_TTL_SECONDS = 5.0


# ---------------------------------------------------------------------------
# model metadata
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ModelSpec:
    """Publisher-declared metadata for one model.

    Attribution fields (``model_owner``/``engine_owner``/``base_model``) are
    mandatory: AlphaAI must always be able to say who created the weights and who
    created the engine that runs them.
    """

    model_id: str
    display_name: str
    engine: str
    family: str
    provider: str
    model: str
    model_owner: str
    engine_owner: str
    context_length: int
    capabilities: tuple[Capability, ...]
    params_total_b: float = 0.0
    params_active_b: float = 0.0
    base_model: str | None = None
    license: str = "unknown"
    license_url: str | None = None
    license_file: str | None = None
    weights_url: str | None = None
    weight_formats: tuple[str, ...] = ()
    runtime_requirements: Mapping[str, Any] = field(default_factory=dict)
    kv_bytes_per_token: float | None = None
    strengths: tuple[str, ...] = ()
    router_priority: int = 50
    status: str = "published"  # published | not_trained | deprecated
    local_paths: tuple[str, ...] = ()
    weight_files: tuple[str, ...] = ()
    tokenizer_path: str | None = None
    chat_template: str | None = None
    notes: str = ""
    source_path: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def provenance(self) -> dict[str, Any]:
        """Where these exact weights came from and how they were installed.

        Recorded by ``alphaai models install`` (source repo, revision, format,
        quantisation, file size, checksum, runtime and the real load test), so an
        installed model is always traceable to an authoritative source.
        """

        payload = self.raw.get("provenance")
        return dict(payload) if isinstance(payload, Mapping) else {}

    @property
    def selection(self) -> dict[str, Any]:
        """Why this model was chosen for this machine (recorded in the registry).

        Written when a model is selected from the ``alphaai doctor`` hardware
        report: the criteria, the measured hardware basis, the decision and the
        alternatives that were rejected. Empty for models registered by hand.
        """

        payload = self.raw.get("selection")
        return dict(payload) if isinstance(payload, Mapping) else {}

    @property
    def quantization(self) -> str | None:
        """Declared quantisation (from provenance, then weight formats)."""

        quant = self.provenance.get("quantization")
        if quant:
            return str(quant)
        if self.weight_formats:
            return str(self.weight_formats[-1])
        return None

    @property
    def weights_published(self) -> bool:
        return self.status == "published"

    @property
    def attribution(self) -> str:
        return engine_attribution(self.display_name, self.model, self.model_owner)

    @property
    def is_alphaai_owned(self) -> bool:
        return self.model_owner.strip().lower() == "alphaai"

    def to_dict(self, *, redact_paths: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.model_id,
            "display_name": self.display_name,
            "engine": self.engine,
            "family": self.family,
            "provider": self.provider,
            "model": self.model,
            "model_owner": self.model_owner,
            "engine_owner": self.engine_owner,
            "base_model": self.base_model,
            "params_total_b": self.params_total_b,
            "params_active_b": self.params_active_b,
            "context_length": self.context_length,
            "capabilities": [cap.value for cap in self.capabilities],
            "strengths": list(self.strengths),
            "license": self.license,
            "license_url": self.license_url,
            "license_file": self.license_file,
            "weights_url": self.weights_url,
            "weight_formats": list(self.weight_formats),
            "runtime_requirements": dict(self.runtime_requirements),
            "status": self.status,
            "quantization": self.quantization,
            "provenance": self.provenance,
            "selection": self.selection,
            "attribution": self.attribution,
            "notes": self.notes,
        }
        if redact_paths:
            payload["local_paths"] = [f"<local>/{Path(p).name}" for p in self.local_paths]
        else:
            payload["local_paths"] = list(self.local_paths)
        return payload


_REQUIRED_SPEC_FIELDS = ("id", "engine", "family", "provider", "model", "model_owner", "context_length")


def spec_from_dict(data: Mapping[str, Any], *, source_path: str | None = None) -> ModelSpec:
    """Build a :class:`ModelSpec` from a ``configs/models/*.json`` payload."""

    missing = [key for key in _REQUIRED_SPEC_FIELDS if not data.get(key)]
    if missing:
        raise ValueError(f"Model metadata missing required fields: {', '.join(missing)}")
    try:
        capabilities = tuple(Capability.parse(item) for item in data.get("capabilities", ["chat"]))
    except ValueError as exc:
        raise ValueError(f"Model '{data.get('id')}' has an invalid capability: {exc}") from exc

    return ModelSpec(
        model_id=str(data["id"]),
        display_name=str(data.get("display_name") or data["id"]),
        engine=str(data["engine"]),
        family=str(data["family"]),
        provider=str(data["provider"]),
        model=str(data["model"]),
        model_owner=str(data["model_owner"]),
        engine_owner=str(data.get("engine_owner") or "AlphaAI"),
        context_length=int(data["context_length"]),
        capabilities=capabilities,
        params_total_b=float(data.get("params_total_b") or 0.0),
        params_active_b=float(data.get("params_active_b") or 0.0),
        base_model=data.get("base_model"),
        license=str(data.get("license") or "unknown"),
        license_url=data.get("license_url"),
        license_file=data.get("license_file"),
        weights_url=data.get("weights_url"),
        weight_formats=tuple(data.get("weight_formats") or ()),
        runtime_requirements=data.get("runtime_requirements") or {},
        kv_bytes_per_token=data.get("kv_bytes_per_token"),
        strengths=tuple(data.get("strengths") or ()),
        router_priority=int(data.get("router_priority", 50)),
        status=str(data.get("status") or "published"),
        local_paths=tuple(data.get("local_paths") or ()),
        weight_files=tuple(data.get("weight_files") or ()),
        tokenizer_path=data.get("tokenizer_path"),
        chat_template=data.get("chat_template"),
        notes=data.get("notes") or "",
        source_path=source_path,
        # ``raw`` keeps the full JSON document, so provenance (written by
        # ``alphaai models install``) and any future fields stay reachable.
        raw=dict(data),
    )


#: Model-metadata directory relative to the repository root. The literal path is
#: deliberate: deployment bundlers (Vercel's Python build) decide which data files
#: to ship with a function by tracing path references in the source, so a purely
#: constructed path would leave a deployed API with no model list at all.
MODEL_SPECS_DIR = "configs/models"


def load_model_specs(config: AlphaAIConfig | None = None, directory: str | Path | None = None) -> list[ModelSpec]:
    """Load every ``configs/models/*.json`` file (sorted, deterministic)."""

    if directory is None:
        specs_dir = Path(MODEL_SPECS_DIR)
        if config is not None:
            specs_dir = Path(config.paths.configs_dir) / specs_dir.name
        directory = specs_dir
    directory = Path(directory)
    if not directory.exists():
        return []
    specs: list[ModelSpec] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid model metadata file {path}: {exc}") from exc
        specs.append(spec_from_dict(payload, source_path=str(path)))
    return specs


# ---------------------------------------------------------------------------
# the engine interface
# ---------------------------------------------------------------------------
class ModelEngine(ABC):
    """A real, locally executed model.

    Subclasses implement :meth:`load`, :meth:`generate` and :meth:`stream`, plus
    :meth:`probe_runtime` / :meth:`find_weights` so that AlphaAI can report an
    accurate status *without* importing heavy runtimes or loading weights.
    """

    #: Engine runtime key, matching ``ModelSpec.engine``.
    engine_key: str = "base"
    #: Human-readable engine name shown in AlphaAI surfaces.
    engine_name: str = "AlphaAI Engine"

    def __init__(
        self,
        spec: ModelSpec,
        config: AlphaAIConfig,
        *,
        enabled: bool = True,
        options: Mapping[str, Any] | None = None,
        hardware: HardwareReport | None = None,
    ) -> None:
        self.spec = spec
        self.config = config
        self.options: dict[str, Any] = dict(options or {})
        self.enabled = enabled
        self.hardware = hardware or detect_hardware(
            [config.paths.models_dir, config.paths.state_dir]
        )
        self._loaded = False
        self._status: EngineStatus | None = None
        self._status_at: float = 0.0
        self._last_error: str | None = None
        self._assessment: AvailabilityAssessment | None = None

    # -- identity ---------------------------------------------------------
    @property
    def id(self) -> str:
        return self.spec.model_id

    @property
    def name(self) -> str:
        return self.spec.display_name

    @property
    def provider(self) -> str:
        return self.spec.provider

    @property
    def model(self) -> str:
        return self.spec.model

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(self.spec.capabilities)

    @property
    def context_length(self) -> int:
        return self.spec.context_length

    @property
    def runtime(self) -> str:
        device = self._assessment.device if self._assessment and self._assessment.device else self.config.runtime.device
        return f"{self.engine_key}/{device}"

    @property
    def status(self) -> EngineStatus:
        return self.health()

    @property
    def attribution(self) -> str:
        return self.spec.attribution

    def supports(self, capability: Capability | str) -> bool:
        try:
            return Capability.parse(capability) in self.capabilities
        except ValueError:
            return False

    # -- runtime probing --------------------------------------------------
    def probe_runtime(self) -> tuple[bool, str, str | None]:
        """Return ``(present, detail, remediation)`` for this engine's runtime."""

        return True, f"{self.engine_name} runtime available", None

    def find_weights(self) -> Path | None:
        """Return the local path holding this model's weights, or ``None``."""

        candidates: list[Path] = []
        for raw in self.spec.local_paths:
            path = Path(raw)
            candidates.append(path if path.is_absolute() else Path(self.config.paths.project_root) / path)
        models_dir = Path(self.config.paths.models_dir)
        candidates.extend([models_dir / self.spec.model_id, models_dir / self.spec.model])
        override = self.options.get("model_path")
        if override:
            candidates.insert(0, Path(str(override)))
        for candidate in candidates:
            if candidate.exists() and self._has_weights(candidate):
                return candidate
        return None

    def _has_weights(self, path: Path) -> bool:
        """True when ``path`` contains files that look like this model's weights."""

        if path.is_file():
            return path.suffix in {".gguf", ".safetensors", ".pt", ".bin"}
        try:
            entries = list(path.iterdir())
        except OSError:
            return False
        names = {entry.name for entry in entries}
        if self.spec.weight_files and any(pattern in names for pattern in self.spec.weight_files):
            return True
        return any(
            name.endswith((".gguf", ".safetensors", ".bin", ".pt"))
            or name in {"config.json", "tokenizer.json"}
            for name in names
        )

    # -- health -----------------------------------------------------------
    def refresh_status(self) -> EngineStatus:
        self._status = None
        self._status_at = 0.0
        return self.health()

    def health(self, *, refresh: bool = False) -> EngineStatus:
        """Cheap availability snapshot (cached for :data:`STATUS_TTL_SECONDS`)."""

        if not refresh and self._status is not None and (time.time() - self._status_at) < STATUS_TTL_SECONDS:
            return self._status

        if not self.enabled:
            status = EngineStatus(
                engine_id=self.id,
                state="disabled",
                detail="Engine disabled by configuration.",
                remediation=f"Enable it in engines.enabled (or remove it from engines.disabled).",
                loaded=False,
                extras={"display_name": self.name, "attribution": self.attribution},
            )
            self._status, self._status_at = status, time.time()
            return status

        if not self.spec.weights_published:
            status = EngineStatus(
                engine_id=self.id,
                state="unavailable",
                detail=(
                    f"{self.name} is declared with status '{self.spec.status}': no weights exist yet, "
                    "so there is nothing to load."
                ),
                remediation="Training/tokenizer pipeline: `alphaai train validate`; see docs/TRAINING.md.",
                loaded=False,
                weights_present=False,
                extras={"display_name": self.name, "attribution": self.attribution, "status": self.spec.status},
            )
            self._status, self._status_at = status, time.time()
            return status

        runtime_ok, runtime_detail, runtime_fix = self.probe_runtime()
        weights_path = self.find_weights()
        assessment = None
        if runtime_ok and weights_path is not None:
            assessment = self.assess_hardware(weights_path)
        self._assessment = assessment or self._assessment

        device = None
        dtype = None
        hardware_ok: bool | None = None
        if assessment is not None:
            hardware_ok = assessment.ok
            device = assessment.device
            dtype = (assessment.estimate or {}).get("dtype")

        reasons: list[str] = []
        remediation: str | None = None
        if not runtime_ok:
            reasons.append(runtime_detail)
            remediation = runtime_fix
        if weights_path is None:
            reasons.append(self.missing_weights_detail())
            remediation = remediation or self.missing_weights_remediation()
        if assessment is not None and not assessment.ok and reasons == []:
            reasons.append("Insufficient local resources for this model.")
            remediation = assessment.remediation

        if reasons:
            state = "error" if self._last_error else "unavailable"
            status = EngineStatus(
                engine_id=self.id,
                state=state,
                detail=" ".join(reasons),
                remediation=remediation,
                loaded=self._loaded,
                device=device,
                dtype=dtype,
                weights_present=weights_path is not None,
                hardware_ok=hardware_ok,
                last_error=self._last_error,
                extras={
                    "display_name": self.name,
                    "attribution": self.attribution,
                    "model": self.model,
                    "hardware": (assessment.to_dict() if assessment else None),
                },
            )
        else:
            status = EngineStatus(
                engine_id=self.id,
                state="ready" if self._loaded else "available",
                detail=(
                    "Model loaded in memory."
                    if self._loaded
                    else "Weights and runtime present; model loads on first request."
                ),
                loaded=self._loaded,
                device=device or self.config.runtime.device,
                dtype=dtype,
                weights_present=True,
                hardware_ok=True,
                last_error=None,
                extras={
                    "display_name": self.name,
                    "attribution": self.attribution,
                    "model": self.model,
                    "weights_path": str(weights_path),
                    "hardware": (assessment.to_dict() if assessment else None),
                },
            )
        self._status, self._status_at = status, time.time()
        return status

    def assess_hardware(self, weights_path: Path | None = None) -> AvailabilityAssessment:
        """Compare declared model requirements against this machine."""

        return assess_model(
            params_total_b=self.spec.params_total_b,
            runtime_requirements=self.spec.runtime_requirements,
            context_length=self.context_length,
            hardware=self.hardware,
            kv_bytes_per_token=self.spec.kv_bytes_per_token,
            requested_device=self.config.runtime.device,
            quant=self.options.get("quant") or self._preferred_quant(),
            disk_path=self.config.paths.models_dir,
        )

    def _preferred_quant(self) -> str | None:
        """The quantisation AlphaAI assesses this model with.

        The *installed* quantisation (recorded by ``alphaai models install``)
        wins, then the smallest declared GGUF quantisation — that is the build a
        CPU-only machine would actually run.
        """

        installed = self.spec.provenance.get("quantization")
        if installed:
            return str(installed)
        pref = [fmt for fmt in self.spec.weight_formats if fmt in {"q4_k_m", "q5_k_m", "q6_k", "q8_0"}]
        if pref:
            return pref[0]
        return None

    def missing_weights_detail(self) -> str:
        location = self.config.paths.models_dir
        return (
            f"{self.name} weights not found locally (looked in {Path(location).name}/ and declared "
            f"local_paths)."
        )

    def missing_weights_remediation(self) -> str:
        return (
            f"Place the {self.model} weights in {Path(self.config.paths.models_dir).name}/{self.id}/ "
            f"(or set engines.options.{self.id}.model_path), then run `alphaai models check {self.id}`. "
            "AlphaAI never downloads weights implicitly — see docs/ENGINES.md for the exact download "
            "commands and the license you must accept first."
        )

    def ensure_ready(self) -> None:
        """Load the model if needed, or raise a precise availability error."""

        status = self.health()
        if not status.usable:
            raise EngineUnavailableError(
                f"{self.name} cannot run: {status.detail}",
                remediation=status.remediation,
                details={"engine_id": self.id, "state": status.state, "attribution": self.attribution},
            )
        if not self._loaded:
            try:
                self.load()
            except EngineUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as EngineLoadError
                self._last_error = f"{type(exc).__name__}: {exc}"
                self.refresh_status()
                raise EngineLoadError(
                    f"{self.name} failed to load weights: {self._last_error}",
                    details={"engine_id": self.id},
                ) from exc

    # -- interface --------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights into memory (idempotent)."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run real inference and return the full result."""

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Run real inference and yield incremental chunks."""

    def unload(self) -> None:
        """Release model memory. Engines override when they hold resources."""

        self._loaded = False
        self.refresh_status()

    # -- helpers for subclasses ------------------------------------------
    def count_tokens(self, text: str) -> int | None:
        """Exact token count when the engine has a tokenizer, else ``None``."""

        return None

    def estimate_tokens(self, text: str) -> int:
        """Deterministic fallback estimate (~4 chars/token). Labelled as such."""

        return max(1, int(len(text) / 4) + 1)

    def build_usage(self, prompt: str, completion: str, *, exact_prompt: int | None, exact_completion: int | None) -> TokenUsage:
        if exact_prompt is not None and exact_completion is not None:
            return TokenUsage(
                prompt_tokens=exact_prompt,
                completion_tokens=exact_completion,
                total_tokens=exact_prompt + exact_completion,
                source="tokenizer",
            )
        prompt_tokens = exact_prompt if exact_prompt is not None else self.estimate_tokens(prompt)
        completion_tokens = exact_completion if exact_completion is not None else self.estimate_tokens(completion)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            source="engine" if exact_prompt is not None or exact_completion is not None else "estimate",
        )

    def default_sampling(self) -> SamplingParams:
        sampling = self.config.sampling
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            max_tokens=sampling.max_tokens,
            repetition_penalty=sampling.repetition_penalty,
            seed=sampling.seed,
            stop=tuple(sampling.stop),
        )

    def render_chat_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Render messages into a prompt.

        Engines that own a tokenizer chat template override this; the fallback is
        a documented, deterministic ChatML-style rendering.
        """

        parts: list[str] = []
        for message in messages:
            if message.role == "tool":
                parts.append(f"<|tool|>\n{message.content}\n<|/tool|>")
            elif message.role == "system":
                parts.append(f"<|system|>\n{message.content}")
            elif message.role == "assistant":
                parts.append(f"<|assistant|>\n{message.content}")
            else:
                parts.append(f"<|user|>\n{message.content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def info(self, *, redact_paths: bool = False) -> dict[str, Any]:
        status = self.health()
        payload = self.spec.to_dict(redact_paths=redact_paths)
        payload.update(
            {
                "engine_id": self.id,
                "engine_name": self.engine_name,
                "runtime": self.runtime,
                "status": status.to_dict(),
                "enabled": self.enabled,
                "supports_streaming": self.supports(Capability.STREAMING),
            }
        )
        return payload

    # -- explicit interface aliases ---------------------------------------
    def metadata(self, *, redact_paths: bool = False) -> dict[str, Any]:
        """Model + engine metadata (who owns the weights, what can it do)."""

        return self.info(redact_paths=redact_paths)

    def capabilities_dict(self) -> dict[str, Any]:
        """Declared capabilities and the concrete runtime this engine uses."""

        return {
            "capabilities": sorted(cap.value for cap in self.capabilities),
            "supports_streaming": self.supports(Capability.STREAMING),
            "supports_tool_calling": self.supports(Capability.TOOL_CALLING),
            "engine": self.engine_key,
            "runtime": self.runtime,
            "model": self.model,
            "model_owner": self.spec.model_owner,
            "engine_owner": self.spec.engine_owner,
            "attribution": self.attribution,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.id!r} model={self.model!r}>"


def engine_id_from_spec(spec: ModelSpec) -> str:
    return spec.model_id


def unknown_model_error(model_id: str, known: Sequence[str]) -> UnknownModelError:
    return UnknownModelError(
        f"Unknown engine/model id '{model_id}'.",
        remediation=f"Known ids: {', '.join(sorted(known)) or '(none registered)'}. Run `alphaai models list`.",
    )

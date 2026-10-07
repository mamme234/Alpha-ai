"""AlphaAI model registry.

The registry owns the set of *real* engines AlphaAI may route to. It can:

* register / remove engines
* enable and disable engines
* detect which engines are actually available on this machine
* expose model capabilities and live status

Discovery is explicit: the registry only creates engines for model specs whose
``engine`` key maps to a known engine class, and it reports every rejected spec
with a reason instead of silently ignoring it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..config.schema import AlphaAIConfig
from ..engines.hardware import HardwareReport, detect_hardware
from .engine import ModelEngine, ModelSpec, load_model_specs, unknown_model_error
from .errors import AlphaAIError
from .types import Capability, EngineStatus

#: engine key -> "module:ClassName". Resolved lazily so that importing the
#: registry never imports torch / llama.cpp / transformers. The single source of
#: truth lives in :mod:`alphaai.engines`.
from ..engines import ENGINE_CLASSES  # noqa: E402  (import is dependency-light)


@dataclass(slots=True)
class DiscoveryReport:
    """Outcome of a discovery pass over ``configs/models``."""

    registered: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    statuses: dict[str, EngineStatus] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "registered": list(self.registered),
            "skipped": list(self.skipped),
            "statuses": {key: status.to_dict() for key, status in self.statuses.items()},
        }


class ModelRegistry:
    """Holds the engines AlphaAI can route to."""

    def __init__(
        self,
        config: AlphaAIConfig,
        *,
        hardware: HardwareReport | None = None,
        specs: Sequence[ModelSpec] | None = None,
    ) -> None:
        self.config = config
        self.hardware = hardware or detect_hardware([config.paths.models_dir, config.paths.state_dir])
        self._engines: dict[str, ModelEngine] = {}
        self._specs: dict[str, ModelSpec] = {spec.model_id: spec for spec in (specs or load_model_specs(config))}
        self.discovery = DiscoveryReport()

    # -- registration -----------------------------------------------------
    def register(self, engine: ModelEngine, *, replace: bool = False) -> ModelEngine:
        if not isinstance(engine, ModelEngine):
            raise TypeError(f"register() expects a ModelEngine, got {type(engine)!r}")
        if engine.id in self._engines and not replace:
            raise AlphaAIError(
                f"Engine id '{engine.id}' is already registered.",
                remediation="Use register(engine, replace=True) to override an existing engine.",
            )
        self._engines[engine.id] = engine
        return engine

    def unregister(self, engine_id: str) -> bool:
        engine = self._engines.pop(engine_id, None)
        if engine is None:
            return False
        try:
            engine.unload()
        except Exception:  # noqa: BLE001 - unload must never break removal
            pass
        return True

    def get(self, engine_id: str) -> ModelEngine:
        engine = self._engines.get(engine_id)
        if engine is None:
            raise unknown_model_error(engine_id, list(self._engines))
        return engine

    def has(self, engine_id: str) -> bool:
        return engine_id in self._engines

    def __iter__(self) -> Iterator[ModelEngine]:
        return iter(self._engines.values())

    def __len__(self) -> int:
        return len(self._engines)

    # -- enable / disable -------------------------------------------------
    def enable(self, engine_id: str, enabled: bool = True) -> EngineStatus:
        engine = self.get(engine_id)
        engine.enabled = enabled
        return engine.refresh_status()

    @property
    def disabled_engines(self) -> list[ModelEngine]:
        return [engine for engine in self._engines.values() if not engine.enabled]

    def engines(self, *, enabled_only: bool = False) -> list[ModelEngine]:
        engines = list(self._engines.values())
        if enabled_only:
            engines = [engine for engine in engines if engine.enabled]
        return sorted(engines, key=lambda engine: engine.id)

    def enabled_engines(self) -> list[ModelEngine]:
        return self.engines(enabled_only=True)

    # -- capabilities & status -------------------------------------------
    def capabilities(self, engine_id: str) -> frozenset[Capability]:
        return self.get(engine_id).capabilities

    def capability_matrix(self) -> dict[str, list[str]]:
        return {
            engine.id: sorted(cap.value for cap in engine.capabilities)
            for engine in self.engines()
        }

    def status(self, engine_id: str | None = None) -> EngineStatus | dict[str, EngineStatus]:
        if engine_id is not None:
            return self.get(engine_id).health()
        return {engine.id: engine.health() for engine in self.engines()}

    def refresh(self) -> dict[str, EngineStatus]:
        return {engine.id: engine.refresh_status() for engine in self.engines()}

    def usable(self, *, capability: Capability | str | None = None) -> list[ModelEngine]:
        engines = [engine for engine in self.enabled_engines() if engine.health().usable]
        if capability is not None:
            engines = [engine for engine in engines if engine.supports(capability)]
        return engines

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise unknown_model_error(model_id, list(self._specs) + list(self._engines)) from exc

    def summary(self) -> dict[str, Any]:
        engines = self.engines()
        states: dict[str, int] = {}
        for engine in engines:
            state = engine.health().state
            states[state] = states.get(state, 0) + 1
        return {
            "registered": len(engines),
            "enabled": len([e for e in engines if e.enabled]),
            "usable": len([e for e in engines if e.health().usable]),
            "states": states,
            "hardware": self.hardware.to_dict(),
            "engines": [engine.id for engine in engines],
        }

    # -- discovery --------------------------------------------------------
    def discover(
        self,
        *,
        specs: Sequence[ModelSpec] | None = None,
        engine_factories: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> DiscoveryReport:
        """Instantiate engines for every enabled, known model spec."""

        report = DiscoveryReport()
        candidate_specs = list(specs) if specs is not None else list(self._specs.values())
        for spec in candidate_specs:
            self._specs[spec.model_id] = spec
            if spec.model_id in self._engines and not replace:
                report.registered.append(spec.model_id)
                report.statuses[spec.model_id] = self._engines[spec.model_id].health()
                continue
            if not self._engine_allowed(spec):
                report.skipped.append({"id": spec.model_id, "reason": "engine disabled by configuration"})
                continue
            try:
                engine_cls = self._resolve_engine_class(spec.engine, engine_factories)
            except AlphaAIError as exc:
                report.skipped.append({"id": spec.model_id, "reason": exc.message})
                continue
            options = self.config.engines.options.get(spec.model_id, {})
            engine = engine_cls(spec, self.config, options=options, hardware=self.hardware)
            self.register(engine, replace=replace)
            report.registered.append(spec.model_id)
            report.statuses[spec.model_id] = engine.health()
        self.discovery = report
        return report

    def _engine_allowed(self, spec: ModelSpec) -> bool:
        enabled = self.config.engines.enabled
        disabled = self.config.engines.disabled
        if spec.engine in disabled or spec.model_id in disabled:
            return False
        if "*" in enabled:
            return True
        return spec.engine in enabled or spec.model_id in enabled

    def _resolve_engine_class(self, key: str, factories: Mapping[str, Any] | None) -> type[ModelEngine]:
        if factories and key in factories:
            return factories[key]
        target = ENGINE_CLASSES.get(key)
        if target is None:
            raise AlphaAIError(
                f"No AlphaAI engine class is registered for engine key '{key}'.",
                remediation=f"Known engine keys: {', '.join(sorted(ENGINE_CLASSES))}. "
                "Register custom engines via registry.discover(engine_factories={...}).",
            )
        module_name, _, class_name = target.partition(":")
        try:
            from importlib import import_module

            module = import_module(module_name)
            engine_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise AlphaAIError(
                f"AlphaAI engine adapter '{target}' could not be imported: {exc}",
                remediation="Verify the AlphaAI installation is complete (pip install -e .).",
            ) from exc
        return engine_cls

    # -- introspection ----------------------------------------------------
    def to_dict(self, *, redact_paths: bool = False) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "models": [engine.info(redact_paths=redact_paths) for engine in self.engines()],
            "discovery": self.discovery.to_dict(),
        }

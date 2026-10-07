"""AlphaAI skill system — base types.

A *skill* is a higher-level, named capability built on top of AlphaAI Core: real
computation, tools, files, network access or an available model engine. Every
skill declares:

* a unique ``skill_id`` and display name
* a description and the capabilities it needs from a model engine
* an input schema and an output schema (JSON-schema subset)
* the tools it uses and the permissions those tools imply
* a real execution handler
* error handling (``SkillResult`` with structured errors)

Nothing in AlphaAI produces a simulated skill result: a skill either performs the
operation or reports a structured failure.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ...config.schema import AlphaAIConfig
from ..errors import AlphaAIError, SkillExecutionError
from ..types import Capability, GenerationResult

#: Signature of the engine bridge handed to skills that can use a live model.
EngineGenerate = Callable[[str, Mapping[str, Any] | None], GenerationResult]


@dataclass(slots=True)
class SkillContext:
    """Everything a skill is allowed to use, injected by the AlphaAI runtime."""

    config: AlphaAIConfig
    tools: Any = None  # alphaai.core.tools.ToolExecutor
    router: Any = None  # alphaai.core.router.ModelRouter
    registry: Any = None  # alphaai.core.registry.ModelRegistry
    memory: Any = None  # alphaai.core.memory.MemoryStore
    engine_generate: EngineGenerate | None = None
    run_id: str = ""
    timeout_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def tool(self, tool_id: str, arguments: Mapping[str, Any] | None = None, *, timeout_s: float | None = None) -> Any:
        """Run a tool through the executor (raises typed AlphaAI errors)."""

        if self.tools is None:
            raise SkillExecutionError(
                "This skill needs the AlphaAI tool executor, but none was provided.",
                remediation="Run skills through the AlphaAI runtime (alphaai.core.runtime) or the API.",
            )
        return self.tools.execute(
            tool_id,
            dict(arguments or {}),
            run_id=self.run_id or "skill",
            context_metadata={"skill_manager": self.metadata.get("skill_manager")},
            timeout_s=timeout_s,
        )

    def tool_output(self, tool_id: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Run a tool and return its output, raising on failure."""

        result = self.tool(tool_id, arguments)
        if not result.ok:
            error = result.error or {}
            raise SkillExecutionError(
                f"Tool '{tool_id}' failed: {error.get('message', 'unknown error')}",
                details={"tool_id": tool_id, "tool_error": error},
            )
        return result.output

    def require_engine(self, prompt: str, *, task: str | None = None, engine_id: str | None = None, **options: Any) -> GenerationResult:
        """Run real inference through the AlphaAI router (or report why not)."""

        if self.engine_generate is None:
            raise SkillExecutionError(
                "This skill needs a live AlphaAI model engine, but no engine bridge is attached.",
                remediation=(
                    "Make a model available locally (see docs/ENGINES.md) and run the skill through "
                    "the AlphaAI runtime or API."
                ),
            )
        return self.engine_generate(prompt, {"task": task, "engine_id": engine_id, **options})


@dataclass(slots=True)
class SkillResult:
    """The real outcome of a skill execution."""

    skill_id: str
    ok: bool
    output: Any = None
    error: dict[str, Any] | None = None
    duration_ms: float = 0.0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    engine_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "skill_id": self.skill_id,
            "ok": self.ok,
            "output": self.output,
            "duration_ms": round(self.duration_ms, 3),
            "metadata": self.metadata,
        }
        if self.error:
            payload["error"] = self.error
        if self.artifacts:
            payload["artifacts"] = self.artifacts
        if self.engine_id:
            payload["engine_id"] = self.engine_id
        return payload


class Skill(ABC):
    """Base class for every AlphaAI skill."""

    skill_id: str = "skill.base"
    name: str = "Base skill"
    description: str = ""
    category: str = "general"
    version: str = "1.0"
    #: Model capabilities this skill needs if it uses an engine.
    required_capabilities: tuple[Capability, ...] = ()
    #: Tools this skill calls (used for permission reporting and availability).
    required_tools: tuple[str, ...] = ()
    #: True when the skill can only run against a live model engine.
    requires_engine: bool = False
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    output_schema: dict[str, Any] = {"type": "object"}
    timeout_s: float = 30.0
    tags: tuple[str, ...] = ()

    # -- contract ---------------------------------------------------------
    @abstractmethod
    def run(self, inputs: dict[str, Any], context: SkillContext) -> Any:
        """Perform the real operation and return its output."""

    def validate_input(self, inputs: Any) -> dict[str, Any]:
        from ..errors import SkillValidationError, ToolValidationError
        from ..tools.registry import validate_schema

        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            raise SkillValidationError(f"Skill '{self.skill_id}' input must be an object.")
        defaults = {
            key: schema.get("default")
            for key, schema in (self.input_schema.get("properties") or {}).items()
            if isinstance(schema, dict) and "default" in schema
        }
        merged = {**defaults, **inputs}
        try:
            validate_schema(merged, self.input_schema, path=self.skill_id)
        except ToolValidationError as exc:
            # Skills and tools share the schema validator, but a bad skill input is
            # a skill error: re-tag it so callers see ``skill_invalid_input``.
            raise SkillValidationError(exc.message, remediation=exc.remediation, details=exc.details) from exc
        return merged

    def validate_output(self, output: Any) -> Any:
        """Validate the handler's output against the declared schema."""

        try:
            from ..tools.registry import validate_schema

            validate_schema(output, self.output_schema, path=f"{self.skill_id}.output")
        except AlphaAIError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface a broken skill contract
            raise SkillExecutionError(
                f"Skill '{self.skill_id}' produced output that does not match its schema: {exc}",
            ) from exc
        return output

    # -- availability -----------------------------------------------------
    def availability(self, context: SkillContext | None = None) -> tuple[bool, str]:
        """Return ``(available, reason)`` for the current configuration."""

        if context is None:
            return True, "No context supplied; runtime policy applies."
        for tool_id in self.required_tools:
            if context.tools is not None:
                decision = context.tools.permissions(tool_id)
                if not decision.allowed:
                    return False, f"Required tool '{tool_id}' is not permitted: {decision.reason}"
        if self.requires_engine:
            if context.engine_generate is None:
                return False, "No live model engine bridge is available (skill requires real inference)."
            registry = context.registry
            if registry is not None and not registry.usable():
                return False, (
                    "No usable AlphaAI model engine is registered, and this skill needs real "
                    "inference. Put weights under models/<id>/ and run `alphaai models check` "
                    "(see docs/ENGINES.md)."
                )
        return True, "Ready."

    # -- introspection ----------------------------------------------------
    def describe(self, availability: tuple[bool, str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "required_capabilities": [cap.value for cap in self.required_capabilities],
            "required_tools": list(self.required_tools),
            "requires_engine": self.requires_engine,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_s": self.timeout_s,
            "tags": list(self.tags),
        }
        if availability is not None:
            payload["available"] = availability[0]
            payload["availability_reason"] = availability[1]
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self.describe()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.skill_id!r}>"


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, duration_ms)``."""

    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000


__all__ = ["EngineGenerate", "Skill", "SkillContext", "SkillResult", "timed", "Sequence"]

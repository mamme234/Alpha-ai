"""AlphaAI skill manager.

The manager owns skill registration, enable/disable policy and execution. Every
execution is validated, timed and error-wrapped, so callers (the API, the
orchestrator, the CLI) always receive a ``SkillResult`` instead of an exception —
unless they explicitly ask for ``run_or_raise``.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Iterable, Sequence

from ...config.schema import AlphaAIConfig
from ..errors import AlphaAIError, SkillNotFoundError, SkillPermissionError, SkillTimeoutError
from .base import Skill, SkillContext, SkillResult

logger = logging.getLogger("alphaai.skills")


class SkillManager:
    """Registers, lists and executes AlphaAI skills."""

    def __init__(self, config: AlphaAIConfig, *, skills: Iterable[Skill] = ()) -> None:
        self.config = config
        self._skills: dict[str, Skill] = {}
        self._disabled: set[str] = set(config.skills.disabled)
        for skill in skills:
            self.register(skill)

    # -- registration -----------------------------------------------------
    def register(self, skill: Skill, *, replace: bool = False) -> Skill:
        if not isinstance(skill, Skill):
            raise TypeError(f"register() expects a Skill, got {type(skill)!r}")
        if skill.skill_id in self._skills and not replace:
            raise ValueError(f"Skill '{skill.skill_id}' is already registered.")
        self._skills[skill.skill_id] = skill
        return skill

    def unregister(self, skill_id: str) -> bool:
        self._disabled.discard(skill_id)
        return self._skills.pop(skill_id, None) is not None

    def get(self, skill_id: str) -> Skill:
        skill = self._skills.get(skill_id)
        if skill is None:
            raise SkillNotFoundError(
                f"Unknown AlphaAI skill '{skill_id}'.",
                remediation=f"Known skills: {', '.join(sorted(self._skills)) or '(none registered)'}.",
            )
        return skill

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self.list())

    # -- policy -----------------------------------------------------------
    def enabled_in_config(self, skill_id: str) -> bool:
        if skill_id in self.config.skills.disabled:
            return False
        enabled = self.config.skills.enabled
        if "*" in enabled:
            return True
        return skill_id in enabled or skill_id.split(".")[-1] in enabled

    def enable(self, skill_id: str, enabled: bool = True) -> bool:
        self.get(skill_id)
        if enabled:
            self._disabled.discard(skill_id)
        else:
            self._disabled.add(skill_id)
        return enabled

    def is_enabled(self, skill_id: str) -> bool:
        return skill_id not in self._disabled and self.enabled_in_config(skill_id)

    def list(self, *, category: str | None = None, enabled_only: bool = False) -> list[Skill]:
        skills = sorted(self._skills.values(), key=lambda skill: skill.skill_id)
        if category:
            skills = [skill for skill in skills if skill.category == category]
        if enabled_only:
            skills = [skill for skill in skills if self.is_enabled(skill.skill_id)]
        return skills

    def describe_all(self, context: SkillContext | None = None) -> list[dict[str, Any]]:
        payload = []
        for skill in self.list():
            availability = skill.availability(context)
            entry = skill.describe(availability)
            entry["enabled"] = self.is_enabled(skill.skill_id)
            payload.append(entry)
        return payload

    # -- execution --------------------------------------------------------
    def run_result(
        self,
        skill_id: str,
        inputs: Any = None,
        *,
        context: SkillContext | None = None,
        run_id: str = "",
    ) -> SkillResult:
        """Execute a skill, always returning a ``SkillResult``."""

        started = time.perf_counter()
        try:
            output, engine_id = self._execute(skill_id, inputs, context=context, run_id=run_id)
        except AlphaAIError as exc:
            duration = (time.perf_counter() - started) * 1000
            logger.warning("skill %s failed: %s (%s)", skill_id, exc.message, exc.code)
            return SkillResult(skill_id=skill_id, ok=False, error=exc.to_dict(), duration_ms=duration)
        except Exception as exc:  # noqa: BLE001 - never leak unexpected exceptions
            duration = (time.perf_counter() - started) * 1000
            wrapped = AlphaAIError(f"Skill '{skill_id}' crashed: {type(exc).__name__}: {exc}")
            logger.exception("skill %s crashed", skill_id)
            return SkillResult(skill_id=skill_id, ok=False, error=wrapped.to_dict(), duration_ms=duration)
        duration = (time.perf_counter() - started) * 1000
        return SkillResult(
            skill_id=skill_id,
            ok=True,
            output=output,
            duration_ms=duration,
            engine_id=engine_id,
            metadata={"name": self._skills[skill_id].name, "category": self._skills[skill_id].category},
        )

    def run_or_raise(
        self,
        skill_id: str,
        inputs: Any = None,
        *,
        context: SkillContext | None = None,
        run_id: str = "",
    ) -> Any:
        """Execute a skill and return its output, raising typed AlphaAI errors."""

        output, _ = self._execute(skill_id, inputs, context=context, run_id=run_id)
        return output

    def _execute(
        self,
        skill_id: str,
        inputs: Any,
        *,
        context: SkillContext | None,
        run_id: str,
    ) -> tuple[Any, str | None]:
        skill = self.get(skill_id)
        if not self.is_enabled(skill_id):
            raise SkillPermissionError(
                f"Skill '{skill_id}' is disabled by configuration.",
                remediation="Enable it in configs/alphaai.toml (skills.enabled) or via `alphaai skills enable`.",
            )
        if context is None:
            raise SkillPermissionError(
                f"Skill '{skill_id}' requires a SkillContext (config, tools, engines).",
                remediation="Execute skills through the AlphaAI runtime or API.",
            )
        available, reason = skill.availability(context)
        if not available:
            raise SkillPermissionError(
                f"Skill '{skill_id}' is unavailable: {reason}",
                remediation=(
                    "Make the skill's requirements available (a usable model engine or the tools "
                    "it needs) and retry. See `alphaai skills list` for the exact reason."
                ),
            )

        validated = skill.validate_input(inputs)
        run_id = run_id or f"skill_{uuid.uuid4().hex[:10]}"
        call_context = context
        call_context.run_id = run_id
        call_context.metadata.setdefault("skill_manager", self)

        # The context timeout is a *default*, not an override: a skill that declares a
        # tighter budget than the runtime default must still be enforced.
        timeouts = [value for value in (call_context.timeout_s, skill.timeout_s) if value]
        timeout = min(timeouts) if timeouts else None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"alphaai-{skill.skill_id}")
        try:
            future = pool.submit(skill.run, validated, call_context)
            try:
                output = future.result(timeout=timeout)
            except FuturesTimeout as exc:
                future.cancel()
                raise SkillTimeoutError(
                    f"Skill '{skill_id}' exceeded its {timeout:g}s timeout.",
                    remediation="Increase skills.<id>.timeout or reduce the input size.",
                ) from exc
        finally:
            pool.shutdown(wait=False)

        skill.validate_output(output)
        engine_id = None
        if isinstance(output, dict):
            engine_id = output.get("engine_id") or (output.get("engine") or {}).get("engine_id") if isinstance(output.get("engine"), dict) else output.get("engine_id")
        return output, engine_id

    # -- discovery helpers ------------------------------------------------
    def by_category(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for skill in self.list():
            grouped.setdefault(skill.category, []).append(skill.skill_id)
        return grouped

    def required_tools(self) -> dict[str, list[str]]:
        return {skill.skill_id: list(skill.required_tools) for skill in self.list() if skill.required_tools}

    def summary(self) -> dict[str, Any]:
        return {
            "registered": len(self._skills),
            "enabled": len(self.list(enabled_only=True)),
            "categories": self.by_category(),
            "engine_backed": [skill.skill_id for skill in self.list() if skill.requires_engine],
        }


def build_default_manager(config: AlphaAIConfig, context: SkillContext | None = None) -> SkillManager:
    """Build the manager with every AlphaAI built-in skill registered."""

    from . import analytic, files, system, text

    skills: Sequence[Skill] = [
        *analytic.build_skills(),
        *text.build_skills(),
        *files.build_skills(),
        *system.build_skills(),
    ]
    return SkillManager(config, skills=skills)

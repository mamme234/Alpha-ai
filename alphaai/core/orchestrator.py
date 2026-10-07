"""AlphaAI agent orchestrator.

There is no separate "planner model" and no scripted agent. The orchestrator runs
one of two *real* modes:

* **Declared plan** – the caller supplies steps (``skill`` / ``tool`` / ``prompt``)
  with dependencies; AlphaAI validates them, orders them topologically and runs
  them through the skill manager and tool executor. This is the
  ``skill.agent_workflow`` engine, exposed as an API-level primitive.
* **Goal loop** – AlphaAI hands the goal to the conversation engine, which offers
  the model its available tools and skills; the model decides each step, tools are
  executed under policy, and the loop stops on an answer or on the configured step
  budget. Every step is recorded in the returned report.

Both modes report exactly what ran, what failed and why.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config.schema import AlphaAIConfig
from .conversation import ConversationEngine, ConversationSession
from .errors import AlphaAIError, SkillValidationError
from .registry import ModelRegistry
from .skills.manager import SkillManager

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are AlphaAI running an autonomous multi-step task. Work toward the goal "
    "given by the user. Call the available tools whenever a step needs computation, "
    "files or the web; do not guess. When you have the answer, reply with a concise "
    "final answer and no tool call."
)


@dataclass(slots=True)
class OrchestrationStep:
    """One recorded step of an orchestration run."""

    index: int
    kind: str  # declared | goal
    action: str
    ok: bool
    duration_ms: float
    output: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "kind": self.kind,
            "action": self.action,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.output is not None:
            payload["output"] = self.output
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(slots=True)
class OrchestrationReport:
    """The complete, auditable record of an orchestration run."""

    run_id: str
    goal: str
    mode: str
    ok: bool
    steps: list[OrchestrationStep] = field(default_factory=list)
    result: Any = None
    engine_id: str | None = None
    error: dict[str, Any] | None = None
    duration_ms: float = 0.0

    @property
    def succeeded(self) -> int:
        return len([step for step in self.steps if step.ok])

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "mode": self.mode,
            "ok": self.ok,
            "steps": [step.to_dict() for step in self.steps],
            "succeeded": self.succeeded,
            "failed": len(self.steps) - self.succeeded,
            "result": self.result,
            "engine_id": self.engine_id,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
        }


class AgentOrchestrator:
    """Runs declared plans and goal loops on top of AlphaAI Core."""

    def __init__(
        self,
        config: AlphaAIConfig,
        *,
        registry: ModelRegistry,
        conversation: ConversationEngine,
        skills: SkillManager | None = None,
        skill_context_factory=None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.conversation = conversation
        self.skills = skills
        #: Callable returning a fully wired ``SkillContext`` (set by the runtime).
        self.skill_context_factory = skill_context_factory

    # -- declared plans ---------------------------------------------------
    def run_plan(
        self,
        steps: Sequence[Mapping[str, Any]],
        *,
        goal: str = "",
        run_id: str = "",
    ) -> OrchestrationReport:
        """Execute a declared step plan through ``skill.agent_workflow``."""

        run_id = run_id or f"orch_{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        report = OrchestrationReport(run_id=run_id, goal=goal or "(declared plan)", mode="declared", ok=False)
        if self.skills is None:
            report.error = {
                "code": "skill_unavailable",
                "message": "No skill manager is attached to this AlphaAI runtime.",
            }
            report.duration_ms = (time.perf_counter() - started) * 1000
            return report
        if not steps:
            raise SkillValidationError("A declared orchestrator plan needs at least one step.")

        result = self.skills.run_result(
            "skill.agent_workflow",
            {"goal": goal, "steps": [self._normalise_step(step, index) for index, step in enumerate(steps)], "run_id": run_id},
            context=self._skill_context(),
            run_id=run_id,
        )
        if not result.ok:
            report.error = result.error
            report.duration_ms = (time.perf_counter() - started) * 1000
            return report

        output = result.output or {}
        for index, step in enumerate(output.get("steps") or []):
            report.steps.append(
                OrchestrationStep(
                    index=index,
                    kind="declared",
                    action=str(step.get("id") or step.get("action") or f"step{index}"),
                    ok=bool(step.get("ok", True)),
                    duration_ms=float(step.get("duration_ms") or 0.0),
                    output=step.get("output"),
                    error=step.get("error"),
                )
            )
        report.ok = all(step.ok for step in report.steps)
        report.result = output
        report.duration_ms = (time.perf_counter() - started) * 1000
        return report

    # -- goal loops -------------------------------------------------------
    def run_goal(
        self,
        goal: str,
        *,
        max_steps: int | None = None,
        engine_id: str | None = None,
        session: ConversationSession | None = None,
        use_tools: bool = True,
    ) -> OrchestrationReport:
        """Run a goal through the conversation engine's real tool loop."""

        run_id = f"orch_{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        report = OrchestrationReport(run_id=run_id, goal=goal, mode="goal", ok=False)
        budget = int(max_steps or self.config.conversation.tool_loop_limit)

        session = session or self.conversation.create_session(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            metadata={"orchestrator": run_id, "goal": goal},
            engine_id=engine_id,
        )
        transcript: list[dict[str, Any]] = []
        try:
            for step in range(1, budget + 1):
                instruction = (
                    goal
                    if step == 1
                    else (
                        "Continue toward the goal: " + goal + ". If you already have the answer, "
                        "reply with the final answer only."
                    )
                )
                step_started = time.perf_counter()
                outcome = self.conversation.chat(
                    instruction,
                    session=session,
                    engine_id=engine_id,
                    use_tools=use_tools,
                )
                report.engine_id = outcome.engine_id
                transcript.append(outcome.to_dict())
                report.steps.append(
                    OrchestrationStep(
                        index=step,
                        kind="goal",
                        action=f"model turn {step}",
                        ok=True,
                        duration_ms=(time.perf_counter() - step_started) * 1000,
                        output={
                            "text": outcome.text,
                            "tool_calls": len(outcome.tool_results),
                            "iterations": outcome.iterations,
                        },
                    )
                )
                if not outcome.tool_results:
                    report.ok = True
                    break
            report.result = {
                "final_answer": transcript[-1]["text"] if transcript else "",
                "transcript": transcript,
                "steps_used": len(report.steps),
                "budget": budget,
            }
        except AlphaAIError as exc:
            report.error = exc.to_dict()
        report.duration_ms = (time.perf_counter() - started) * 1000
        return report

    # -- plan normalisation ------------------------------------------------
    @staticmethod
    def _normalise_step(step: Mapping[str, Any], index: int) -> dict[str, Any]:
        """Accept both the explicit (``kind``/``target``) and shorthand (``skill``/``tool``) forms."""

        if not isinstance(step, Mapping):
            raise SkillValidationError(f"Plan step {index} must be an object.")
        entry: dict[str, Any] = dict(step)
        entry.setdefault("id", str(entry.get("name") or f"step{index + 1}"))
        for key, kind in (("skill", "skill"), ("tool", "tool")):
            if entry.get(key):
                entry.setdefault("kind", kind)
                entry.setdefault("target", str(entry[key]))
                break
        kind = entry.get("kind")
        if kind not in {"tool", "skill"} or not entry.get("target"):
            raise SkillValidationError(
                f"Plan step {index} needs 'kind' ('tool' or 'skill') and 'target', "
                f"or the shorthand 'skill'/'tool' key. Model-driven steps belong to goal mode.",
                details={"step": entry},
            )
        entry["depends_on"] = [str(dependency) for dependency in (entry.get("depends_on") or [])]
        return entry

    # -- introspection ----------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        engines = [engine.id for engine in self.registry.enabled_engines()]
        return {
            "modes": ["declared", "goal"],
            "engines": engines,
            "skills": sorted(skill.skill_id for skill in self.skills.list(enabled_only=True)) if self.skills else [],
            "max_goal_steps": self.config.conversation.tool_loop_limit,
            "note": "AlphaAI has no separate planner model: plans are declared or driven by the model + tools.",
        }

    def _skill_context(self):
        if self.skill_context_factory is not None:
            return self.skill_context_factory()
        from .skills.base import SkillContext

        return SkillContext(
            config=self.config,
            router=self.conversation.router,
            registry=self.registry,
            tools=self.conversation.tools,
            memory=self.conversation.memory,
            metadata={"skill_manager": self.skills},
        )


__all__ = [
    "AgentOrchestrator",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "OrchestrationReport",
    "OrchestrationStep",
]

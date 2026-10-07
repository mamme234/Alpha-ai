"""AlphaAI model router.

The router maps a *task* to a *real, available engine*. It never invents a
model: candidates come from the registry, engines that cannot serve the request
are excluded, and if nothing fits the router raises ``NoSuitableModelError`` with
the exact reason and remediation.

Task detection is deterministic and explainable (no separate model is required
to route). Callers may always override it with an explicit task or engine id.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config.schema import AlphaAIConfig
from .engine import ModelEngine
from .errors import EngineUnavailableError, NoSuitableModelError
from .registry import ModelRegistry
from .types import Capability, ChatMessage, TASK_REQUIREMENTS, TaskKind

#: Deterministic keyword signals per task. Kept intentionally small and legible;
#: explicit overrides always win.
_TASK_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "coding": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"```",
            r"\b(def|class|import|function|const|let|var|SELECT|FROM)\b",
            r"\b(python|javascript|typescript|rust|golang|sql|bash|html|css|react|docker|git)\b",
            r"\b(stack ?trace|traceback|compile|debug|refactor|unit test|regex)\b",
            r"\.(py|ts|tsx|js|jsx|go|rs|java|c|cpp|h|sql|sh|yaml|toml)\b",
        )
    ),
    "mathematics": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(calculate|compute|solve|integral|derivative|equation|matrix|probability)\b",
            r"\b(sum|product|average|mean|median|sigma|log|sqrt|factorial|percent(age)?)\b",
            r"\d\s*[\+\-\*/^%]\s*\d",
            r"\d\s*[×÷]\s*\d",
            # Colloquial multiplication: "what is 1234 x 5678?".
            r"\d\s*[xX]\s*\d",
            # Spelled-out arithmetic: "what is 25 multiplied by 4?".
            r"\b(multiplied by|multiply|times|divided by|divide|plus|minus|squared|cubed|square root)\b",
            r"\b[A-Za-z]\s*=\s*-?\d",
        )
    ),
    "reasoning": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(why|prove|derive|deduce|infer|analyse|analyze|compare|trade-?offs?)\b",
            r"\b(step by step|chain of thought|reasoning|logic|paradox|puzzle)\b",
        )
    ),
    "summarization": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (r"\b(summar(y|ise|ize)|tl;?dr|key points|condense)\b",)
    ),
    "translation": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (r"\b(translate|translation|in (french|german|spanish|chinese|japanese|korean|hindi|arabic))\b",)
    ),
}

_CODE_FENCE = re.compile(r"```")


@dataclass(slots=True)
class RouteCandidate:
    """A scored routing candidate, retained for explainability."""

    engine_id: str
    model: str
    score: float
    usable: bool
    missing_capabilities: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        # Unusable candidates carry a -inf score; strict JSON (and Starlette's
        # encoder) rejects non-finite floats, so report null instead.
        score = round(self.score, 3) if math.isfinite(self.score) else None
        return {
            "engine_id": self.engine_id,
            "model": self.model,
            "score": score,
            "usable": self.usable,
            "missing_capabilities": list(self.missing_capabilities),
            "reason": self.reason,
        }


@dataclass(slots=True)
class RouteDecision:
    """The router's decision plus the audit trail behind it."""

    task: str
    engine_id: str
    model: str
    required: tuple[str, ...]
    reason: str
    candidates: list[RouteCandidate] = field(default_factory=list)
    explicit: bool = False

    def engine(self, registry: ModelRegistry) -> ModelEngine:
        return registry.get(self.engine_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "engine_id": self.engine_id,
            "model": self.model,
            "required_capabilities": list(self.required),
            "reason": self.reason,
            "explicit": self.explicit,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class ModelRouter:
    """Routes AlphaAI requests to available model engines."""

    def __init__(self, registry: ModelRegistry, config: AlphaAIConfig | None = None) -> None:
        self.registry = registry
        self.config = config or registry.config

    # -- task classification ---------------------------------------------
    def classify_task(
        self,
        messages: Sequence[ChatMessage] | str | None = None,
        *,
        tools: Sequence[Any] | None = None,
        has_images: bool = False,
        explicit: str | None = None,
        estimated_tokens: int | None = None,
    ) -> str:
        """Classify a request into a :data:`~alphaai.core.types.TaskKind`."""

        if explicit:
            return _normalise_task(explicit)
        if has_images:
            return "vision"
        if tools:
            return "tool_calling"

        text = _text_of(messages)
        tokens = estimated_tokens if estimated_tokens is not None else _approx_tokens(text)
        if tokens >= self.config.routing.long_context_threshold_tokens:
            return "long_context"
        if _CODE_FENCE.search(text) and _TASK_PATTERNS["coding"][0].search(text):
            return "coding"
        for task in ("translation", "summarization", "mathematics", "coding", "reasoning"):
            if any(pattern.search(text) for pattern in _TASK_PATTERNS[task]):
                return task
        return "general"

    def requirements_for(self, task: str) -> tuple[Capability, ...]:
        return TASK_REQUIREMENTS.get(task, TASK_REQUIREMENTS["general"])

    # -- candidate scoring ------------------------------------------------
    def candidates(
        self,
        task: str,
        *,
        required: Iterable[Capability | str] | None = None,
        min_context: int | None = None,
        include_unavailable: bool = False,
    ) -> list[RouteCandidate]:
        required_caps = _parse_caps(required if required is not None else self.requirements_for(task))
        candidates: list[RouteCandidate] = []
        for engine in self.registry.enabled_engines():
            status = engine.health()
            caps = engine.capabilities
            missing = sorted(cap.value for cap in required_caps - caps)
            fits_context = min_context is None or engine.context_length >= min_context
            usable = status.usable and not missing and fits_context
            score = self._score(engine, task, usable=usable, min_context=min_context)
            if missing:
                reason = f"missing capabilities: {', '.join(missing)}"
            elif not fits_context:
                reason = f"context window {engine.context_length} < required {min_context}"
            elif not status.usable:
                reason = status.detail or status.state
            else:
                reason = "usable"
            if usable or include_unavailable:
                candidates.append(
                    RouteCandidate(
                        engine_id=engine.id,
                        model=engine.model,
                        score=score,
                        usable=usable,
                        missing_capabilities=missing,
                        reason=reason,
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.engine_id))
        return candidates

    def _score(self, engine: ModelEngine, task: str, *, usable: bool, min_context: int | None) -> float:
        if not usable:
            return float("-inf")
        score = float(engine.spec.router_priority)
        if task in engine.spec.strengths:
            score += 25.0
        if engine.supports(Capability.REASONING) and task == "reasoning":
            score += 8.0
        if engine.supports(Capability.CODING) and task == "coding":
            score += 8.0
        if min_context:
            # Prefer the tightest window that still fits (cheaper models).
            score += max(0.0, 8.0 - (engine.context_length / max(min_context, 1)))
        preferences = self.config.routing.task_preferences.get(task) or []
        if engine.id in preferences:
            score += 30.0 - preferences.index(engine.id) * 2.0
        if engine.health().loaded:
            score += 5.0  # avoid unloading/reloading a hot model
        return score

    # -- routing ----------------------------------------------------------
    def choose(
        self,
        messages: Sequence[ChatMessage] | str | None = None,
        *,
        task: str | None = None,
        engine_id: str | None = None,
        required: Iterable[Capability | str] | None = None,
        min_context: int | None = None,
        tools: Sequence[Any] | None = None,
        has_images: bool = False,
        estimated_tokens: int | None = None,
    ) -> RouteDecision:
        """Pick an engine for this request, or raise ``NoSuitableModelError``."""

        if engine_id:
            engine = self.registry.get(engine_id)  # raises UnknownModelError for unknown ids
            status = engine.health()
            if not status.usable:
                raise EngineUnavailableError(
                    f"Engine '{engine.id}' was requested explicitly but cannot run: {status.detail}",
                    remediation=status.remediation,
                    details={"engine_id": engine.id},
                )
            return RouteDecision(
                task=task or "explicit",
                engine_id=engine.id,
                model=engine.model,
                required=(),
                reason="Explicit engine_id requested by the caller.",
                explicit=True,
            )

        resolved_task = self.classify_task(
            messages, tools=tools, has_images=has_images, explicit=task, estimated_tokens=estimated_tokens
        )
        required_caps = _parse_caps(required) if required is not None else self.requirements_for(resolved_task)
        candidates = self.candidates(resolved_task, required=required_caps, min_context=min_context)

        if not candidates:
            all_reasons = self.candidates(
                resolved_task, required=required_caps, min_context=min_context, include_unavailable=True
            )
            detail = "; ".join(f"{item.engine_id}: {item.reason}" for item in all_reasons) or "no engines registered"
            raise NoSuitableModelError(
                f"No available AlphaAI engine can handle a '{resolved_task}' request "
                f"(required: {', '.join(sorted(cap.value for cap in required_caps))}). {detail}",
                remediation=(
                    "Make a model local: put weights under models/<id>/ and run `alphaai models check`, "
                    "or install the runtime for an existing engine (`pip install -e '.[llama]'`). "
                    "See docs/ENGINES.md."
                ),
                details={
                    "task": resolved_task,
                    "required_capabilities": sorted(cap.value for cap in required_caps),
                    "candidates": [item.to_dict() for item in all_reasons],
                },
            )

        winner = candidates[0]
        engine = self.registry.get(winner.engine_id)
        return RouteDecision(
            task=resolved_task,
            engine_id=engine.id,
            model=engine.model,
            required=tuple(sorted(cap.value for cap in required_caps)),
            reason=(
                f"Best match for task '{resolved_task}': {engine.name} "
                f"(priority {engine.spec.router_priority}, "
                f"{len([c for c in candidates if c.usable])} usable candidate(s))."
            ),
            candidates=candidates,
        )

    def explain(
        self,
        messages: Sequence[ChatMessage] | str | None = None,
        *,
        task: str | None = None,
        tools: Sequence[Any] | None = None,
        has_images: bool = False,
    ) -> dict[str, Any]:
        """Return the routing decision or the precise reason nothing fits."""

        try:
            decision = self.choose(messages, task=task, tools=tools, has_images=has_images)
        except (NoSuitableModelError, EngineUnavailableError) as exc:
            return {"ok": False, "error": exc.to_dict()}
        return {"ok": True, "decision": decision.to_dict()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _text_of(messages: Sequence[ChatMessage] | str | None) -> str:
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    return "\n".join(message.content or "" for message in messages)


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4) + 1)


def _parse_caps(values: Iterable[Capability | str]) -> frozenset[Capability]:
    return frozenset(Capability.parse(value) for value in values)


def _normalise_task(task: str) -> str:
    normalised = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    if normalised not in TASK_REQUIREMENTS:
        raise NoSuitableModelError(
            f"Unknown task kind '{task}'.",
            remediation=f"Supported tasks: {', '.join(sorted(TASK_REQUIREMENTS))}.",
        )
    return normalised

"""AlphaAI core value types shared by engines, router, tools and skills."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Literal, Sequence

Role = Literal["system", "user", "assistant", "tool"]


class Capability(str, Enum):
    """Capabilities an engine can advertise and the router can require."""

    CHAT = "chat"
    STREAMING = "streaming"
    CODING = "coding"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    LONG_CONTEXT = "long_context"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    MULTILINGUAL = "multilingual"
    STRUCTURED_OUTPUT = "structured_output"

    @classmethod
    def parse(cls, value: str | "Capability") -> "Capability":
        if isinstance(value, Capability):
            return value
        normalised = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalised:
                return member
        raise ValueError(f"Unknown capability '{value}'")


#: Task kinds the router understands (see Phase 2 routing requirements).
TaskKind = Literal[
    "general",
    "coding",
    "reasoning",
    "mathematics",
    "long_context",
    "tool_calling",
    "vision",
    "summarization",
    "translation",
]

TASK_REQUIREMENTS: dict[str, tuple[Capability, ...]] = {
    "general": (Capability.CHAT,),
    "coding": (Capability.CHAT, Capability.CODING),
    "reasoning": (Capability.CHAT, Capability.REASONING),
    "mathematics": (Capability.CHAT, Capability.MATHEMATICS),
    "long_context": (Capability.CHAT, Capability.LONG_CONTEXT),
    "tool_calling": (Capability.CHAT, Capability.TOOL_CALLING),
    "vision": (Capability.CHAT, Capability.VISION),
    "summarization": (Capability.CHAT,),
    "translation": (Capability.CHAT, Capability.MULTILINGUAL),
}

EngineState = Literal["ready", "available", "unavailable", "loading", "error", "disabled"]


@dataclass(slots=True)
class EngineStatus:
    """Health/availability snapshot for one engine.

    ``state`` semantics:

    * ``ready``       – runtime + weights present, model loaded in memory
    * ``available``   – runtime + weights present, not loaded yet (lazy)
    * ``unavailable`` – cannot serve requests; ``detail``/``remediation`` say why
    * ``error``       – a load or inference attempt failed; see ``last_error``
    * ``disabled``    – administratively turned off in configuration
    """

    engine_id: str
    state: EngineState
    detail: str = ""
    loaded: bool = False
    device: str | None = None
    dtype: str | None = None
    weights_present: bool = False
    hardware_ok: bool | None = None
    last_error: str | None = None
    remediation: str | None = None
    checked_at: float = field(default_factory=time.time)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.state in {"ready", "available"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "state": self.state,
            "usable": self.usable,
            "detail": self.detail,
            "loaded": self.loaded,
            "device": self.device,
            "dtype": self.dtype,
            "weights_present": self.weights_present,
            "hardware_ok": self.hardware_ok,
            "last_error": self.last_error,
            "remediation": self.remediation,
            "checked_at": self.checked_at,
            "extras": self.extras,
        }


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        role = str(data.get("role", "user"))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role '{role}'")
        return cls(
            role=role,  # type: ignore[arg-type]
            content=str(data.get("content", "") or ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls") or []],
        )


@dataclass(slots=True)
class ToolCall:
    """A structured tool request produced by a model or a client."""

    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.call_id, "tool_id": self.tool_id, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Accept every tool-call shape AlphaAI documents.

        AlphaAI tells models to emit ``{"tool": "<id>", "arguments": {...}}``
        (see the transformers engine's tool prompt), and also accepts
        ``tool_id``/``name`` keys and OpenAI-style ``function`` objects, so one
        parser understands every model that follows any AlphaAI instruction.
        """

        function = data.get("function") if isinstance(data.get("function"), dict) else {}
        arguments = data.get("arguments")
        if arguments is None:
            arguments = function.get("arguments")
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return cls(
            tool_id=str(
                data.get("tool_id")
                or data.get("name")
                or data.get("tool")
                or function.get("name")
                or ""
            ),
            arguments=dict(arguments or {}) if isinstance(arguments, dict) else {},
            call_id=str(data.get("id") or data.get("call_id") or ""),
            raw=data.get("raw"),
        )


@dataclass(slots=True)
class ToolResult:
    """The real outcome of a tool execution."""

    tool_id: str
    call_id: str = ""
    ok: bool = True
    output: Any = None
    error: dict[str, Any] | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_id": self.tool_id,
            "call_id": self.call_id,
            "ok": self.ok,
            "output": self.output,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            payload["error"] = self.error
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def as_message(self) -> ChatMessage:
        """Render the result as a ``tool`` chat message for the model."""

        import json

        if self.ok:
            body = self.output if isinstance(self.output, str) else json.dumps(self.output, ensure_ascii=False, default=str)
        else:
            body = f"ERROR: {(self.error or {}).get('message', 'tool failed')}"
        return ChatMessage(role="tool", content=body, name=self.tool_id, tool_call_id=self.call_id)


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = "engine"  # "tokenizer" | "engine" | "estimate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }


@dataclass(slots=True)
class ToolSpecView:
    """Tool schema as presented to a model (JSON-schema style)."""

    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    permissions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "permissions": self.permissions,
        }


@dataclass(slots=True)
class SamplingParams:
    """Sampling parameters used for one generation request."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 0
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: Sequence[str] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@dataclass(slots=True)
class GenerationRequest:
    """Everything an engine needs to perform one real generation."""

    messages: list[ChatMessage]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    tools: list[ToolSpecView] = field(default_factory=list)
    engine_id: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages if m.content)


@dataclass(slots=True)
class StreamChunk:
    """One incremental unit of a real streaming generation."""

    text: str = ""
    index: int = 0
    done: bool = False
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    engine_id: str = ""
    model: str = ""


@dataclass(slots=True)
class GenerationResult:
    """The real result of a generation request."""

    text: str
    engine_id: str
    model: str
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    runtime: str = ""
    attribution: str = ""
    provider: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "engine_id": self.engine_id,
            "provider": self.provider,
            "model": self.model,
            "runtime": self.runtime,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "latency_ms": round(self.latency_ms, 3),
            "attribution": self.attribution,
            "extras": self.extras,
        }


def iterate_stream(chunks: Iterator[StreamChunk]) -> Iterator[StreamChunk]:
    """Small helper used by tests and the API to normalise engine iterators."""

    for chunk in chunks:
        if not isinstance(chunk, StreamChunk):  # pragma: no cover - defensive
            raise TypeError(f"Engine yielded {type(chunk)!r} instead of StreamChunk")
        yield chunk

"""AlphaAI conversation engine.

The conversation engine is the part of AlphaAI Core that turns a user message into
a real model response *and* runs the tool loop around it:

1. retrieve relevant memory (optional)
2. fit the conversation into the engine's context window (context manager)
3. route the request to a real, available engine (model router)
4. generate — streaming or not — with tools offered when policy allows
5. if the model asked for tools, execute them through the permission-checked
   executor and feed the results back
6. repeat until the model answers or the tool-loop limit is reached

If no engine can serve the request the engine raises, with the router's exact
reason and remediation. AlphaAI never answers from a canned reply.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from ..config.schema import AlphaAIConfig
from .context import ContextManager, ContextPlan, build_memory_block
from .engine import ModelEngine
from .errors import AlphaAIError, ConversationError, NoSuitableModelError
from .memory import MemoryStore
from .registry import ModelRegistry
from .router import ModelRouter, RouteDecision
from .skills.manager import SkillManager
from .skills.system import _extract_tool_payloads
from .tools.executor import ToolExecutor
from .types import (
    ChatMessage,
    GenerationRequest,
    SamplingParams,
    StreamChunk,
    ToolCall,
    ToolResult,
    TokenUsage,
)

#: Which tool *categories* are relevant to each task kind the router detects.
#:
#: The tool policy offers a model only these schemas instead of every registered
#: tool. That matters for real inference: a full tool prompt costs ~1200 prompt
#: tokens, which on a one-core CPU-only machine is ~30 s of prefill per turn,
#: and it makes a small model much more likely to reach for the wrong tool. The
#: mapping is deterministic, legible, and only ever *narrows* the permission-
#: checked tool set. Tasks that need no tool (general chat) offer none.
TASK_TOOL_CATEGORIES: dict[str, tuple[str, ...]] = {
    "mathematics": ("math",),
    "coding": ("code", "data", "filesystem", "text"),
    "reasoning": ("math", "data", "text"),
    "summarization": ("text", "data", "filesystem"),
    "translation": ("text",),
    "tool_calling": ("math", "code", "data", "text", "filesystem", "network", "system"),
    "general": (),
}

#: Upper bound on how many tool schemas one request may offer a model.
MAX_TOOL_SCHEMAS_PER_REQUEST = 4


def tool_categories_for(task: str) -> tuple[str, ...]:
    """Tool categories the policy allows for a task kind (empty = no tools)."""

    return TASK_TOOL_CATEGORIES.get(task, ())


@dataclass(slots=True)
class ConversationSession:
    """One conversation and the messages in it."""

    session_id: str
    system_prompt: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    engine_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    turns: int = 0

    def to_dict(self, *, include_messages: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "engine_id": self.engine_id,
            "turns": self.turns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "messages": len(self.messages),
        }
        if include_messages:
            payload["history"] = [message.to_dict() for message in self.messages]
        return payload


@dataclass(slots=True)
class ChatOutcome:
    """The full, auditable result of one conversation turn."""

    session_id: str
    text: str
    engine_id: str
    model: str
    provider: str = ""
    finish_reason: str = "stop"
    iterations: int = 1
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_results: list[ToolResult] = field(default_factory=list)
    routing: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    attribution: str = ""
    runtime: str = ""
    latency_ms: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "text": self.text,
            "engine_id": self.engine_id,
            "model": self.model,
            "provider": self.provider,
            "runtime": self.runtime,
            "finish_reason": self.finish_reason,
            "iterations": self.iterations,
            "usage": self.usage.to_dict(),
            "tool_results": [result.to_dict() for result in self.tool_results],
            "routing": self.routing,
            "context": self.context,
            "attribution": self.attribution,
            "latency_ms": round(self.latency_ms, 3),
        }


class ConversationEngine:
    """Runs conversations against the AlphaAI model registry."""

    def __init__(
        self,
        config: AlphaAIConfig,
        *,
        registry: ModelRegistry,
        router: ModelRouter | None = None,
        tools: ToolExecutor | None = None,
        skills: SkillManager | None = None,
        memory: MemoryStore | None = None,
        context: ContextManager | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.router = router or ModelRouter(registry, config)
        self.tools = tools
        self.skills = skills
        self.memory = memory
        self.context = context or ContextManager(config)
        self._sessions: dict[str, ConversationSession] = {}

    # -- sessions ---------------------------------------------------------
    def create_session(
        self,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        engine_id: str | None = None,
    ) -> ConversationSession:
        session = ConversationSession(
            session_id=session_id or f"sess_{uuid.uuid4().hex[:12]}",
            system_prompt=system_prompt or self.config.conversation.system_prompt,
            engine_id=engine_id,
            metadata=dict(metadata or {}),
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ConversationError(
                f"Unknown conversation session '{session_id}'.",
                remediation="Create a session with POST /api/chat/stream or start a new chat.",
            )
        return session

    def sessions(self) -> list[dict[str, Any]]:
        return [session.to_dict(include_messages=False) for session in self._sessions.values()]

    def drop_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    # -- tool policy ------------------------------------------------------
    def tool_categories(self, text: str | None) -> tuple[str, ...]:
        """Tool categories this message is actually about (deterministic)."""

        if not text:
            return tuple(TASK_TOOL_CATEGORIES["tool_calling"])
        return tool_categories_for(self.router.classify_task(text))

    def routable_tools(self, *, enabled: bool, text: str | None = None) -> list[Any]:
        """Tool descriptors the router may use to classify this request.

        Passing the permitted tools to the router is what turns a mathematics or
        research-looking message into a *tool-calling* request when the available
        engine can call tools but does not claim to solve that domain natively.
        """

        if not enabled or self.tools is None:
            return []
        return list(self.tools.available_tools(categories=self.tool_categories(text)))

    def request_tools(
        self, engine: ModelEngine | None, *, enabled: bool, text: str | None = None
    ) -> list[dict[str, Any]]:
        """Tool schemas offered to the model for this request (may be empty)."""

        if not enabled or self.tools is None:
            return []
        if engine is not None and not engine.supports("tool_calling"):
            return []
        return self.tools.model_tool_schemas(
            categories=self.tool_categories(text), limit=MAX_TOOL_SCHEMAS_PER_REQUEST
        )

    def execute_tool_calls(
        self, calls: Sequence[ToolCall], *, run_id: str
    ) -> list[ToolResult]:
        if self.tools is None:
            raise ConversationError(
                "This AlphaAI runtime has no tool executor attached.",
                remediation="Build the runtime through alphaai.core.runtime.AlphaRuntime.",
            )
        results: list[ToolResult] = []
        for call in calls:
            results.append(
                self.tools.run(
                    call.tool_id,
                    call.arguments,
                    call_id=call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                    run_id=run_id,
                    context_metadata={"skill_manager": self.skills},
                )
            )
        return results

    # -- routing ----------------------------------------------------------
    def route(
        self,
        messages: Sequence[ChatMessage],
        *,
        engine_id: str | None = None,
        task: str | None = None,
        tools: Sequence[Any] | None = None,
    ) -> RouteDecision:
        """Route the request, downgrading gracefully if tools cannot be served."""

        try:
            return self.router.choose(messages, task=task, engine_id=engine_id, tools=tools or None)
        except NoSuitableModelError:
            if not tools:
                raise
            return self.router.choose(messages, task=task, engine_id=engine_id)

    # -- memory -----------------------------------------------------------
    def _memory_block(self, text: str, session: ConversationSession) -> str | None:
        """Retrieved memory for this turn, without duplicating live history.

        The memory store mirrors every turn, so a session-namespace lookup returns
        turns the context manager already sends as chat messages. Repeating them
        inside the system message wastes prompt tokens (expensive on CPU
        inference) and confuses small models, so entries already present in the
        live history are filtered out. What remains is genuine recall of turns
        that have scrolled out of the context window, plus long-term memory.
        """

        if self.memory is None:
            return None
        seen = {message.content.strip() for message in session.messages if message.content}
        seen.add(text.strip())

        def _fresh(namespace: str) -> list[Any]:
            return [
                entry
                for entry in self.memory.search(
                    text, namespace=namespace, limit=self.config.memory.retrieval_limit
                )
                if str(getattr(entry, "value", "")).strip() not in seen
            ]

        entries = _fresh(session.session_id)
        if not entries:
            entries = _fresh(self.config.memory.default_namespace)
        return build_memory_block(entries)

    def _remember(self, session: ConversationSession, role: str, content: str) -> None:
        if self.memory is None:
            return
        self.memory.remember(
            f"turn{len(session.messages)}:{role}",
            content,
            namespace=session.session_id,
            kind="conversation",
            tags=[role, session.session_id],
        )

    # -- history ----------------------------------------------------------
    def history(self, session: ConversationSession) -> list[ChatMessage]:
        limit = max(1, self.config.conversation.max_history_turns) * 2
        return list(session.messages)[-limit:]

    def plan(
        self,
        session: ConversationSession,
        message: str,
        *,
        engine: ModelEngine,
        max_new_tokens: int | None = None,
    ) -> ContextPlan:
        history = self.history(session) + [ChatMessage(role="user", content=message)]
        return self.context.plan(
            history,
            engine=engine,
            system_prompt=session.system_prompt,
            max_new_tokens=max_new_tokens or self.config.sampling.max_tokens,
            memory_block=self._memory_block(message, session),
        )

    def sampling(self, overrides: Mapping[str, Any] | None = None) -> SamplingParams:
        base = self.config.sampling
        params = SamplingParams(
            temperature=base.temperature,
            top_p=base.top_p,
            top_k=base.top_k,
            max_tokens=base.max_tokens,
            repetition_penalty=base.repetition_penalty,
            seed=base.seed,
            stop=tuple(base.stop),
        )
        for key, value in (overrides or {}).items():
            if value is None or not hasattr(params, key):
                continue
            setattr(params, key, tuple(value) if key == "stop" else value)
        return params

    # -- chat (non-streaming) ---------------------------------------------
    def chat(
        self,
        message: str,
        *,
        session: ConversationSession | None = None,
        session_id: str | None = None,
        engine_id: str | None = None,
        task: str | None = None,
        use_memory: bool = True,
        use_tools: bool = True,
        max_new_tokens: int | None = None,
        sampling: Mapping[str, Any] | None = None,
    ) -> ChatOutcome:
        session = self._resolve_session(session, session_id)
        started = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:10]}"

        decision = self.route(
            self.history(session) + [ChatMessage(role="user", content=message)],
            engine_id=engine_id,
            task=task,
            tools=self.routable_tools(enabled=use_tools, text=message),
        )
        engine = decision.engine(self.registry)
        params = self.sampling(sampling)
        if max_new_tokens:
            params.max_tokens = int(max_new_tokens)

        session.messages.append(ChatMessage(role="user", content=message))
        self._remember(session, "user", message)

        events: list[dict[str, Any]] = []
        tool_results: list[ToolResult] = []
        usage = TokenUsage()
        tool_schemas = self.request_tools(engine, enabled=use_tools, text=message)
        plan = self._plan_with_history(
            session,
            engine,
            params.max_tokens,
            memory_block=(self._memory_block(message, session) if use_memory else None),
        )

        text = ""
        finish_reason = "stop"
        iterations = 0
        limit = max(1, self.config.conversation.tool_loop_limit)

        while iterations < limit:
            iterations += 1
            request = GenerationRequest(
                messages=plan.messages,
                sampling=params,
                tools=[
                    _tool_view(schema) for schema in tool_schemas
                ] if tool_schemas else [],
                engine_id=engine.id,
                model_id=engine.model,
            )
            result = engine.generate(request)
            usage.prompt_tokens += result.usage.prompt_tokens
            usage.completion_tokens += result.usage.completion_tokens
            usage.total_tokens += result.usage.total_tokens
            usage.source = result.usage.source if result.usage.source == "tokenizer" else usage.source
            text = result.text
            finish_reason = result.finish_reason

            calls = _tool_calls_from_text(text) if tool_schemas else []
            if not calls:
                break
            session.messages.append(
                ChatMessage(role="assistant", content=text, tool_calls=calls)
            )
            events.append({"type": "tool_call", "iteration": iterations, "calls": [call.to_dict() for call in calls]})
            results = self.execute_tool_calls(calls, run_id=run_id)
            tool_results.extend(results)
            for tool_result in results:
                events.append({"type": "tool_result", "result": tool_result.to_dict()})
                session.messages.append(tool_result.as_message())
            if iterations >= limit:
                events.append(
                    {
                        "type": "tool_loop_limit",
                        "detail": f"Tool loop limit ({limit}) reached; returning the model's last message.",
                    }
                )
                break
            plan = self._plan_with_history(
                session,
                engine,
                params.max_tokens,
                memory_block=(self._memory_block(message, session) if use_memory else None),
            )

        session.messages.append(ChatMessage(role="assistant", content=text))
        session.turns += 1
        session.updated_at = time.time()
        session.engine_id = engine.id
        self._remember(session, "assistant", text)

        return ChatOutcome(
            session_id=session.session_id,
            text=text,
            engine_id=engine.id,
            model=engine.model,
            provider=engine.provider,
            finish_reason=finish_reason,
            iterations=iterations,
            usage=usage,
            tool_results=tool_results,
            routing=decision.to_dict(),
            context=plan.to_dict(),
            attribution=engine.attribution,
            runtime=engine.runtime,
            latency_ms=(time.perf_counter() - started) * 1000,
            events=events,
        )

    # -- chat (streaming) -------------------------------------------------
    def stream(
        self,
        message: str,
        *,
        session: ConversationSession | None = None,
        session_id: str | None = None,
        engine_id: str | None = None,
        task: str | None = None,
        use_memory: bool = True,
        use_tools: bool = True,
        max_new_tokens: int | None = None,
        sampling: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield real-time conversation events (delta / tool / done / error)."""

        session = self._resolve_session(session, session_id)
        started = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        params = self.sampling(sampling)
        if max_new_tokens:
            params.max_tokens = int(max_new_tokens)

        try:
            decision = self.route(
                self.history(session) + [ChatMessage(role="user", content=message)],
                engine_id=engine_id,
                task=task,
                tools=self.routable_tools(enabled=use_tools, text=message),
            )
            engine = decision.engine(self.registry)
        except AlphaAIError as exc:
            yield {"type": "error", "error": exc.to_dict()}
            return

        yield {
            "type": "route",
            "session_id": session.session_id,
            "routing": decision.to_dict(),
            "attribution": engine.attribution,
        }

        session.messages.append(ChatMessage(role="user", content=message))
        self._remember(session, "user", message)
        tool_schemas = self.request_tools(engine, enabled=use_tools, text=message)
        limit = max(1, self.config.conversation.tool_loop_limit)
        total_text = ""
        iterations = 0
        finish_reason = "stop"

        while iterations < limit:
            iterations += 1
            plan = self._plan_with_history(
                session,
                engine,
                params.max_tokens,
                memory_block=(
                    self._memory_block(message, session)
                    if use_memory and iterations == 1
                    else None
                ),
            )
            request = GenerationRequest(
                messages=plan.messages,
                sampling=params,
                tools=[_tool_view(schema) for schema in tool_schemas] if tool_schemas else [],
                engine_id=engine.id,
                model_id=engine.model,
            )
            buffered = ""
            try:
                for chunk in engine.stream(request):
                    if chunk.done:
                        finish_reason = chunk.finish_reason or finish_reason
                        continue
                    if chunk.text:
                        buffered += chunk.text
                        yield {"type": "delta", "text": chunk.text}
            except AlphaAIError as exc:
                yield {"type": "error", "error": exc.to_dict()}
                return
            total_text = buffered

            calls = _tool_calls_from_text(buffered) if tool_schemas else []
            if not calls:
                break
            session.messages.append(ChatMessage(role="assistant", content=buffered, tool_calls=calls))
            yield {
                "type": "tool_call",
                "iteration": iterations,
                "calls": [call.to_dict() for call in calls],
            }
            results = self.execute_tool_calls(calls, run_id=run_id)
            for tool_result in results:
                yield {"type": "tool_result", "result": tool_result.to_dict()}
                session.messages.append(tool_result.as_message())
            if iterations >= limit:
                yield {
                    "type": "tool_loop_limit",
                    "detail": f"Tool loop limit ({limit}) reached.",
                }
                break

        session.messages.append(ChatMessage(role="assistant", content=total_text))
        session.turns += 1
        session.updated_at = time.time()
        session.engine_id = engine.id
        self._remember(session, "assistant", total_text)

        tokens, source = self.context.count_messages(session.messages, engine)
        yield {
            "type": "done",
            "session_id": session.session_id,
            "engine_id": engine.id,
            "model": engine.model,
            "provider": engine.provider,
            "runtime": engine.runtime,
            "attribution": engine.attribution,
            "finish_reason": finish_reason,
            "iterations": iterations,
            "approximate_context_tokens": tokens,
            "token_source": source,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    # -- internals --------------------------------------------------------
    def _resolve_session(
        self, session: ConversationSession | None, session_id: str | None
    ) -> ConversationSession:
        if session is not None:
            return session
        if session_id:
            return self.get_session(session_id)
        return self.create_session()

    def _plan_with_history(
        self,
        session: ConversationSession,
        engine: ModelEngine,
        max_new_tokens: int,
        *,
        memory_block: str | None = None,
    ) -> ContextPlan:
        return self.context.plan(
            self.history(session),
            engine=engine,
            system_prompt=session.system_prompt,
            max_new_tokens=max_new_tokens,
            memory_block=memory_block,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tool_view(schema: Mapping[str, Any]) -> Any:
    from .types import ToolSpecView

    function = schema.get("function") or {}
    return ToolSpecView(
        tool_id=str(function.get("name", "")),
        name=str(function.get("name", "")),
        description=str(function.get("description", "")),
        parameters=function.get("parameters") or {},
    )


def _tool_calls_from_text(text: str) -> list[ToolCall]:
    """Parse tool calls a model emitted as text.

    AlphaAI reuses one parser everywhere (``alphaai.core.skills.system``), so the
    tool-calling skill and the conversation engine accept identical syntax.
    """

    if not text or "{" not in text:
        return []
    calls: list[ToolCall] = []
    for index, payload in enumerate(_extract_tool_payloads(text)):
        tool_id = str(payload.get("tool_id") or payload.get("name") or payload.get("tool") or "")
        if not tool_id and isinstance(payload.get("function"), dict):
            tool_id = str(payload["function"].get("name") or "")
        if not tool_id:
            continue
        arguments = payload.get("arguments")
        if arguments is None and isinstance(payload.get("function"), dict):
            arguments = payload["function"].get("arguments")
        if arguments is None and isinstance(payload.get("parameters"), dict):
            arguments = payload["parameters"]
        calls.append(
            ToolCall(
                tool_id=tool_id,
                arguments=arguments if isinstance(arguments, dict) else {},
                call_id=str(payload.get("id") or payload.get("call_id") or f"call_{index}"),
                raw=text[:400],
            )
        )
    return calls


__all__ = [
    "ChatOutcome",
    "ConversationEngine",
    "ConversationSession",
    "MAX_TOOL_SCHEMAS_PER_REQUEST",
    "TASK_TOOL_CATEGORIES",
    "tool_categories_for",
]

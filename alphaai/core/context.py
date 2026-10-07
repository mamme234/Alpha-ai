"""AlphaAI context manager.

The context manager decides what a model actually sees. It uses the engine's real
tokenizer when one is loaded, otherwise a clearly-labelled character estimate, and
it *drops* history rather than inventing a summary for it: the returned plan says
exactly which turns were dropped and why.

Rules
-----
* the system prompt is always kept (it is the smallest, most important message)
* the newest turns are kept first
* the budget is ``context_length - reserve_tokens - max_tokens``
* retrieval from AlphaAI memory can add a small, attributed context block
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config.schema import AlphaAIConfig
from .engine import ModelEngine
from .types import ChatMessage

#: How much of the estimated token budget the memory block may consume.
MEMORY_BUDGET_SHARE = 0.10
MEMORY_BUDGET_MAX_TOKENS = 512


@dataclass(slots=True)
class ContextPlan:
    """The exact prompt AlphaAI will send, plus the audit trail."""

    messages: list[ChatMessage]
    system_prompt: str | None = None
    budget_tokens: int = 0
    used_tokens: int = 0
    max_new_tokens: int = 0
    context_length: int = 0
    token_source: str = "estimate"
    dropped_messages: int = 0
    dropped_turns: int = 0
    truncated: bool = False
    memory_block: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "max_new_tokens": self.max_new_tokens,
            "context_length": self.context_length,
            "token_source": self.token_source,
            "messages": [message.to_dict() for message in self.messages],
            "system_prompt": self.system_prompt,
            "dropped_messages": self.dropped_messages,
            "dropped_turns": self.dropped_turns,
            "truncated": self.truncated,
            "memory_block": self.memory_block,
            "notes": list(self.notes),
        }


class ContextManager:
    """Builds the message list handed to an engine."""

    def __init__(self, config: AlphaAIConfig) -> None:
        self.config = config

    # -- token accounting -------------------------------------------------
    def count_tokens(self, text: str, engine: ModelEngine | None = None) -> tuple[int, str]:
        """Return ``(tokens, source)`` using the engine tokenizer when possible."""

        if engine is not None:
            exact = engine.count_tokens(text)
            if exact is not None:
                return exact, "tokenizer"
        return max(1, int(len(text) / 4) + 1), "estimate"

    def count_messages(
        self, messages: Sequence[ChatMessage], engine: ModelEngine | None = None
    ) -> tuple[int, str]:
        total = 0
        source = "estimate"
        for message in messages:
            # ~4 tokens of role/framing overhead per message, plus the content.
            tokens, message_source = self.count_tokens(message.content or "", engine)
            total += tokens + 4
            if message_source == "tokenizer":
                source = "tokenizer"
        return total, source

    # -- planning ---------------------------------------------------------
    def plan(
        self,
        messages: Sequence[ChatMessage],
        *,
        engine: ModelEngine | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        memory_block: str | None = None,
        reserve_tokens: int | None = None,
    ) -> ContextPlan:
        """Fit ``messages`` into the engine's context window."""

        conversation = self.config.conversation
        context_length = engine.context_length if engine is not None else 32768
        reserve = conversation.context_reserve_tokens if reserve_tokens is None else reserve_tokens
        max_new = int(max_new_tokens or self.config.sampling.max_tokens)
        budget = max(256, context_length - reserve - max_new)

        plan = ContextPlan(
            messages=[],
            system_prompt=system_prompt,
            budget_tokens=budget,
            max_new_tokens=max_new,
            context_length=context_length,
        )

        working: list[ChatMessage] = []
        if system_prompt:
            working.append(ChatMessage(role="system", content=system_prompt))

        block = self._trim_memory_block(memory_block, budget)
        if block:
            plan.memory_block = block
            target = working[0] if working and working[0].role == "system" else None
            if target is not None:
                working[0] = ChatMessage(role="system", content=f"{target.content}\n\n{block}")
            else:
                working.insert(0, ChatMessage(role="system", content=block))

        candidate_messages = [message for message in messages if message.role != "system"]
        history_budget = budget - (self.count_messages(working, engine)[0] if working else 0)

        # Keep the newest turns that fit; report exactly what was dropped.
        kept: list[ChatMessage] = []
        used = 0
        for message in reversed(candidate_messages):
            tokens, source = self.count_tokens(message.content or "", engine)
            if source == "tokenizer":
                plan.token_source = "tokenizer"
            if used + tokens + 4 > history_budget:
                plan.truncated = True
                break
            kept.append(message)
            used += tokens + 4
        kept.reverse()
        plan.dropped_messages = len(candidate_messages) - len(kept)
        plan.dropped_turns = plan.dropped_messages

        if plan.dropped_messages:
            plan.notes.append(
                f"{plan.dropped_messages} older message(s) were dropped to fit the context window; "
                "AlphaAI does not fabricate a summary for them."
            )

        plan.messages = working + kept
        total, source = self.count_messages(plan.messages, engine)
        plan.used_tokens = total
        if source == "tokenizer":
            plan.token_source = "tokenizer"
        if total > budget:
            # The system prompt alone can exceed the budget on very small models.
            plan.notes.append(
                f"Prompt uses {total} tokens against a {budget}-token budget "
                f"(context_length={context_length}); the engine may reject or truncate it."
            )
            plan.truncated = True
        return plan

    # -- memory -----------------------------------------------------------
    def _trim_memory_block(self, block: str | None, budget: int) -> str | None:
        if not block:
            return None
        limit = min(MEMORY_BUDGET_MAX_TOKENS, int(budget * MEMORY_BUDGET_SHARE))
        if limit <= 0:
            return None
        text = block.strip()
        # ~4 chars per token: keep the block inside its share of the budget.
        max_chars = limit * 4
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n[memory block truncated by AlphaAI]"
        return text


def build_memory_block(entries: Iterable[Any], *, limit: int = 5) -> str | None:
    """Render retrieved memory entries as an attributed context block."""

    lines: list[str] = []
    for entry in list(entries)[:limit]:
        value = getattr(entry, "value", None)
        key = getattr(entry, "key", "")
        if value is None:
            continue
        lines.append(f"- {key}: {value}")
    if not lines:
        return None
    return "AlphaAI memory (retrieved by key overlap, most relevant first):\n" + "\n".join(lines)


__all__ = ["ContextManager", "ContextPlan", "build_memory_block"]

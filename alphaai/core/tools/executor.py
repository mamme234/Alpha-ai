"""AlphaAI tool executor.

The executor is the *only* component that invokes tool handlers, and it always
applies, in order:

1. registry lookup (unknown tool -> ``ToolNotFoundError``)
2. permission resolution (denied -> ``ToolPermissionError``)
3. argument validation against the tool's JSON schema (-> ``ToolValidationError``)
4. per-run call budget
5. execution with a hard timeout (-> ``ToolTimeoutError``)
6. output size limiting and structured logging

``run()`` converts any of those failures into a ``ToolResult`` with ``ok=False``,
which is what the conversation engine feeds back to the model. ``execute()``
raises, which is what the HTTP API uses to pick a status code.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ...config.schema import AlphaAIConfig
from ..errors import (
    AlphaAIError,
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolTimeoutError,
)
from ..types import ToolCall, ToolResult
from .permissions import PermissionDecision, resolve_permissions
from .registry import Tool, ToolContext, ToolRegistry

logger = logging.getLogger("alphaai.tools.executor")


@dataclass(slots=True)
class ToolCallRecord:
    """Structured audit record for one tool call."""

    tool_id: str
    call_id: str
    run_id: str
    ok: bool
    duration_ms: float
    started_at: float
    error_code: str | None = None
    error_message: str | None = None
    output_bytes: int = 0
    truncated: bool = False
    permission: dict[str, Any] | None = None
    arguments_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "call_id": self.call_id,
            "run_id": self.run_id,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 3),
            "started_at": self.started_at,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "output_bytes": self.output_bytes,
            "truncated": self.truncated,
            "permission": self.permission,
            "arguments_preview": self.arguments_preview,
        }


@dataclass(slots=True)
class ToolExecutionLog:
    """In-memory tool call log (also mirrored to the ``alphaai.tools`` logger)."""

    records: list[ToolCallRecord] = field(default_factory=list)

    def add(self, record: ToolCallRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": len(self.records),
            "failures": len([record for record in self.records if not record.ok]),
            "records": [record.to_dict() for record in self.records],
        }

    def summary(self) -> list[str]:
        return [
            f"{record.tool_id}: {'ok' if record.ok else 'failed'} in {record.duration_ms:.1f} ms"
            for record in self.records
        ]

    def clear(self) -> None:
        self.records.clear()


class ToolExecutor:
    """Executes AlphaAI tools under policy control."""

    def __init__(
        self,
        registry: ToolRegistry,
        config: AlphaAIConfig,
        *,
        log: ToolExecutionLog | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.log = log or ToolExecutionLog()
        self._budgets: dict[str, int] = {}

    # -- policy -----------------------------------------------------------
    def permissions(self, tool_id: str) -> PermissionDecision:
        tool = self.registry.get(tool_id)
        return resolve_permissions(
            self.config,
            tool.tool_id,
            requires_network=tool.requires_network,
            requires_write=tool.requires_write,
            requires_code_execution=tool.requires_code_execution,
            default_timeout_s=tool.timeout_s,
        )

    def available_tools(self, *, categories: Sequence[str] | None = None) -> list[Tool]:
        """Tools allowed by the current policy (used for model tool schemas).

        ``categories`` narrows the result to the tool categories a request is
        actually about (see
        :data:`alphaai.core.conversation.TASK_TOOL_CATEGORIES`). Offering a model
        only the relevant schemas keeps the prompt small — which dominates
        latency on CPU-only machines — and makes a small model far more likely to
        pick the right tool.
        """

        wanted = set(categories) if categories is not None else None
        available: list[Tool] = []
        for tool in self.registry.list():
            if wanted is not None and tool.category not in wanted:
                continue
            decision = self.permissions(tool.tool_id)
            if decision.allowed and tool.model_callable:
                available.append(tool)
        return available

    def model_tool_schemas(
        self, *, categories: Sequence[str] | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Tool schemas handed to a model, optionally scoped and capped."""

        tools = self.available_tools(categories=categories)
        if limit is not None:
            tools = tools[:limit]
        return [tool.model_schema() for tool in tools]

    # -- execution --------------------------------------------------------
    def execute(
        self,
        tool_id: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        call_id: str = "",
        run_id: str = "",
        context_metadata: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ToolResult:
        """Execute a tool or raise a typed AlphaAI error."""

        tool = self.registry.get(tool_id)
        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        run_id = run_id or call_id
        started = time.time()

        decision = self.permissions(tool.tool_id)
        if not decision.allowed:
            raise ToolPermissionError(
                f"Tool '{tool.tool_id}' is not permitted: {decision.reason}",
                remediation=(
                    "Enable it in configs/alphaai.toml (tools.enabled / tools.permissions) — "
                    "AlphaAI tool permissions are deny-by-default."
                ),
                details={"tool_id": tool.tool_id, "permission": decision.to_dict()},
            )

        if timeout_s is not None:
            decision.timeout_s = float(timeout_s)

        self._check_budget(run_id, decision)

        arguments = tool.validate(arguments)
        context = ToolContext(
            config=self.config,
            call_id=call_id,
            run_id=run_id,
            sandbox_root=decision.sandbox_root,
            allow_network=decision.network,
            allow_write=decision.write,
            timeout_s=decision.timeout_s,
            max_output_bytes=self.config.tools.max_output_bytes,
            metadata=dict(context_metadata or {}),
        )

        logger.info(
            "tool call %s tool=%s run=%s timeout=%.1fs",
            call_id,
            tool.tool_id,
            run_id,
            decision.timeout_s,
        )
        result = self._invoke(tool, arguments, context, call_id, run_id, started, decision)
        return result

    def run(
        self,
        tool_id: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        call_id: str = "",
        run_id: str = "",
        context_metadata: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ToolResult:
        """Execute a tool, converting any failure into a ``ToolResult``."""

        started = time.time()
        try:
            return self.execute(
                tool_id,
                arguments,
                call_id=call_id,
                run_id=run_id,
                context_metadata=context_metadata,
                timeout_s=timeout_s,
            )
        except AlphaAIError as exc:
            duration = (time.time() - started) * 1000
            record = ToolCallRecord(
                tool_id=tool_id,
                call_id=call_id or "unknown",
                run_id=run_id or call_id or "unknown",
                ok=False,
                duration_ms=duration,
                started_at=started,
                error_code=exc.code,
                error_message=exc.message,
            )
            self.log.add(record)
            logger.warning("tool %s failed: %s (%s)", tool_id, exc.message, exc.code)
            return ToolResult(
                tool_id=tool_id,
                call_id=call_id,
                ok=False,
                error=exc.to_dict(),
                duration_ms=duration,
            )
        except Exception as exc:  # noqa: BLE001 - unexpected handler failure
            duration = (time.time() - started) * 1000
            wrapped = ToolExecutionError(f"Tool '{tool_id}' crashed: {type(exc).__name__}: {exc}")
            self.log.add(
                ToolCallRecord(
                    tool_id=tool_id,
                    call_id=call_id or "unknown",
                    run_id=run_id or call_id or "unknown",
                    ok=False,
                    duration_ms=duration,
                    started_at=started,
                    error_code=wrapped.code,
                    error_message=wrapped.message,
                )
            )
            return ToolResult(tool_id=tool_id, call_id=call_id, ok=False, error=wrapped.to_dict(), duration_ms=duration)

    def execute_calls(
        self,
        calls: Sequence[ToolCall],
        *,
        run_id: str = "",
        context_metadata: dict[str, Any] | None = None,
    ) -> list[ToolResult]:
        """Execute several tool calls sequentially, preserving order."""

        return [
            self.run(
                call.tool_id,
                call.arguments,
                call_id=call.call_id,
                run_id=run_id,
                context_metadata=context_metadata,
            )
            for call in calls
        ]

    # -- internals --------------------------------------------------------
    def _check_budget(self, run_id: str, decision: PermissionDecision) -> None:
        used = self._budgets.get(run_id, 0)
        if used >= decision.max_calls_per_run:
            raise ToolPermissionError(
                f"Tool call budget exhausted for run '{run_id}' ({used}/{decision.max_calls_per_run}).",
                remediation="Increase tools.permissions.<tool>.max_calls_per_run or split the workflow.",
            )
        self._budgets[run_id] = used + 1

    def _invoke(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolContext,
        call_id: str,
        run_id: str,
        started: float,
        decision: PermissionDecision,
    ) -> ToolResult:
        output: Any = None
        timeout = decision.timeout_s
        if tool.handler is None:
            raise ToolExecutionError(
                f"Tool '{tool.tool_id}' has no handler bound and cannot execute.",
                remediation="This is an AlphaAI installation problem; reinstall (`pip install -e .`).",
            )
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"alphaai-tool-{tool.tool_id}")
        try:
            future = pool.submit(tool.handler, arguments, context)
            try:
                output = future.result(timeout=timeout)
            except FuturesTimeout as exc:
                future.cancel()
                raise ToolTimeoutError(
                    f"Tool '{tool.tool_id}' exceeded its {timeout}s timeout.",
                    remediation="Raise tools.permissions.<tool>.timeout_s or reduce the workload.",
                    details={"tool_id": tool.tool_id, "timeout_s": timeout},
                ) from exc
        except ToolError:
            raise
        except PermissionError as exc:
            raise ToolPermissionError(str(exc)) from exc
        except AlphaAIError:
            raise
        except Exception as exc:  # noqa: BLE001 - handler failure surface
            raise ToolExecutionError(
                f"Tool '{tool.tool_id}' failed: {type(exc).__name__}: {exc}",
                details={"tool_id": tool.tool_id},
            ) from exc
        finally:
            pool.shutdown(wait=False)

        payload, truncated, size = self._limit_output(output)
        duration = (time.time() - started) * 1000
        record = ToolCallRecord(
            tool_id=tool.tool_id,
            call_id=call_id,
            run_id=run_id,
            ok=True,
            duration_ms=duration,
            started_at=started,
            output_bytes=size,
            truncated=truncated,
            permission=decision.to_dict(),
            arguments_preview=_preview(arguments),
        )
        self.log.add(record)
        logger.info("tool %s ok in %.1f ms (%d bytes)", tool.tool_id, duration, size)
        return ToolResult(
            tool_id=tool.tool_id,
            call_id=call_id,
            ok=True,
            output=payload,
            duration_ms=duration,
            metadata={
                "truncated": truncated,
                "output_bytes": size,
                "permitted_via": decision.reason,
                "sandbox_root": str(decision.sandbox_root) if decision.sandbox_root else None,
            },
        )

    def _limit_output(self, output: Any) -> tuple[Any, bool, int]:
        limit = self.config.tools.max_output_bytes
        if isinstance(output, str):
            encoded = output.encode("utf-8")
            if len(encoded) > limit:
                clipped = encoded[:limit].decode("utf-8", errors="ignore")
                return (
                    clipped + f"\n[AlphaAI truncated tool output at {limit} bytes]",
                    True,
                    len(encoded),
                )
            return output, False, len(encoded)
        try:
            serialised = json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            serialised = str(output)
        encoded = serialised.encode("utf-8")
        if len(encoded) > limit:
            return (
                {
                    "truncated": True,
                    "output_bytes": len(encoded),
                    "text": encoded[:limit].decode("utf-8", errors="ignore"),
                    "note": f"AlphaAI truncated this tool result at {limit} bytes.",
                },
                True,
                len(encoded),
            )
        return output, False, len(encoded)


def _preview(arguments: dict[str, Any], length: int = 200) -> str:
    try:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(arguments)
    return text[:length]


def build_executor(
    config: AlphaAIConfig,
    *,
    registry: ToolRegistry | None = None,
    extra_tools: Iterable[Tool] = (),
) -> ToolExecutor:
    """Build a default executor (optionally with extra tools registered)."""

    if registry is None:
        from .registry import build_default_registry

        registry = build_default_registry()
    for tool in extra_tools:
        registry.register(tool, replace=True)
    return ToolExecutor(registry, config)

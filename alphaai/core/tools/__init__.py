"""AlphaAI tool engine: registry, permissions, executor and built-in tools.

Tool call flow (Phase 4):

    Model -> Tool Call -> AlphaAI Tool Executor -> real operation
          -> Tool Result -> Model -> Final Answer

Every tool declares its permission requirements and JSON-schema parameters. The
executor is the only place that runs a handler, and it enforces permissions,
timeouts, output limits and call budgets before anything executes.
"""

from __future__ import annotations

from .executor import ToolCallRecord, ToolExecutionLog, ToolExecutor
from .permissions import PermissionDecision, resolve_permissions
from .registry import Tool, ToolContext, ToolRegistry, ToolSpecView, build_default_registry

__all__ = [
    "PermissionDecision",
    "Tool",
    "ToolCallRecord",
    "ToolContext",
    "ToolExecutionLog",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSpecView",
    "build_default_registry",
    "resolve_permissions",
]

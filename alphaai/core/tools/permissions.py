"""Tool permission resolution.

Permissions are deny-by-default. A tool runs only when:

1. the tool is enabled by the ``tools.enabled`` allow-list,
2. the global switch it needs is on (network / writes / code execution), and
3. its per-tool override (``tools.permissions.<tool>.allow``) permits it.

The resolved decision carries the effective sandbox root, timeout and output
limit, which the executor then enforces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...config.schema import AlphaAIConfig, ToolPermissions


@dataclass(slots=True)
class PermissionDecision:
    """Effective permission state for one tool call."""

    allowed: bool
    tool_id: str
    reason: str = ""
    sandbox_root: Path | None = None
    network: bool = False
    write: bool = False
    timeout_s: float = 10.0
    max_calls_per_run: int = 32
    require_approval: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "tool_id": self.tool_id,
            "reason": self.reason,
            "network": self.network,
            "write": self.write,
            "timeout_s": self.timeout_s,
            "max_calls_per_run": self.max_calls_per_run,
            "sandbox_root": str(self.sandbox_root) if self.sandbox_root else None,
        }


def tool_enabled(config: AlphaAIConfig, tool_id: str) -> bool:
    enabled = config.tools.enabled
    if tool_id in config.tools.disabled:
        return False
    if "*" in enabled:
        return True
    return tool_id in enabled or any(tool_id.startswith(prefix) for prefix in enabled if prefix.endswith(".*"))


def resolve_permissions(
    config: AlphaAIConfig,
    tool_id: str,
    *,
    requires_network: bool = False,
    requires_write: bool = False,
    requires_code_execution: bool = False,
    default_timeout_s: float | None = None,
) -> PermissionDecision:
    """Compute the effective permission decision for ``tool_id``."""

    override: ToolPermissions | None = config.tools.permissions.get(tool_id)
    root = Path(config.tools.sandbox_root)

    if not tool_enabled(config, tool_id):
        return PermissionDecision(
            allowed=False,
            tool_id=tool_id,
            reason=f"Tool '{tool_id}' is not enabled in tools.enabled.",
            sandbox_root=root,
        )

    if override is not None and not override.allow:
        return PermissionDecision(
            allowed=False,
            tool_id=tool_id,
            reason=f"Tool '{tool_id}' is disabled by tools.permissions.{tool_id}.allow = false.",
            sandbox_root=Path(override.root) if override.root else root,
        )

    if requires_code_execution and not config.tools.allow_code_execution:
        return PermissionDecision(
            allowed=False,
            tool_id=tool_id,
            reason="Code execution is disabled (tools.allow_code_execution = false).",
            sandbox_root=root,
        )

    network = bool(config.tools.allow_network or (override and override.network))
    if requires_network and not network:
        return PermissionDecision(
            allowed=False,
            tool_id=tool_id,
            reason="Network access is disabled (tools.allow_network = false).",
            sandbox_root=root,
        )

    write = bool(config.tools.allow_writes or (override and override.write))
    if requires_write and not write:
        return PermissionDecision(
            allowed=False,
            tool_id=tool_id,
            reason="Filesystem writes are disabled (tools.allow_writes = false).",
            sandbox_root=root,
        )

    sandbox_root = Path(override.root) if override and override.root else root
    timeout = (
        override.timeout_s
        if override is not None and override.timeout_s
        else (default_timeout_s if default_timeout_s is not None else config.tools.default_timeout_s)
    )
    return PermissionDecision(
        allowed=True,
        tool_id=tool_id,
        reason="Allowed by AlphaAI tool policy.",
        sandbox_root=sandbox_root,
        network=network,
        write=write,
        timeout_s=timeout,
        max_calls_per_run=override.max_calls_per_run if override else 32,
    )


def resolve_sandbox_path(root: Path | None, raw_path: str) -> Path:
    """Resolve ``raw_path`` inside ``root``, refusing escapes.

    ``~`` is expanded, symlinks are resolved, and any path that ends up outside
    the sandbox root raises ``PermissionError`` (mapped to a tool permission
    error by the executor).
    """

    if root is None:
        raise PermissionError("No sandbox root is configured for this tool.")
    candidate = Path(os.path.expanduser(raw_path))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PermissionError(f"Path '{raw_path}' escapes the AlphaAI sandbox root ({root_resolved}).")
    return resolved

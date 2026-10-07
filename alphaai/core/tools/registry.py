"""Tool registry, tool definition and JSON-schema validation.

A ``Tool`` is the AlphaAI primitive: a small, real operation with a typed input
schema, declared permission needs and a handler. Skills compose tools (and are
themselves registered in the skill manager).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ...config.schema import AlphaAIConfig
from ..errors import ToolExecutionError, ToolNotFoundError, ToolValidationError
from ..types import ToolSpecView

logger = logging.getLogger("alphaai.tools")

ToolHandler = Callable[[dict[str, Any], "ToolContext"], Any]


@dataclass(slots=True)
class ToolContext:
    """Everything a tool handler may use. No global state, no ambient config."""

    config: AlphaAIConfig
    call_id: str = ""
    run_id: str = ""
    sandbox_root: Path | None = None
    allow_network: bool = False
    allow_write: bool = False
    timeout_s: float = 10.0
    max_output_bytes: int = 65536
    metadata: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logger)

    def child(self, **overrides: Any) -> "ToolContext":
        data = {
            "config": self.config,
            "call_id": self.call_id,
            "run_id": self.run_id,
            "sandbox_root": self.sandbox_root,
            "allow_network": self.allow_network,
            "allow_write": self.allow_write,
            "timeout_s": self.timeout_s,
            "max_output_bytes": self.max_output_bytes,
            "metadata": dict(self.metadata),
            "logger": self.logger,
        }
        data.update(overrides)
        return ToolContext(**data)


@dataclass(slots=True)
class Tool:
    """A single real operation AlphaAI can execute."""

    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    category: str = "general"
    requires_network: bool = False
    requires_write: bool = False
    requires_code_execution: bool = False
    timeout_s: float | None = None
    output_schema: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    #: False for tools a model must never call on its own (privileged setup).
    model_callable: bool = True
    #: The real handler. Normally passed inline; ``alphaai.core.tools.builtin``
    #: binds handlers after all of its handler functions are defined, and a tool
    #: without a handler refuses to execute rather than pretending to work.
    handler: ToolHandler | None = None

    # -- schema -----------------------------------------------------------
    def spec(self, permissions: Mapping[str, Any] | None = None) -> ToolSpecView:
        return ToolSpecView(
            tool_id=self.tool_id,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            permissions=dict(permissions or {}),
        )

    def model_schema(self) -> dict[str, Any]:
        """OpenAI-style function schema handed to models that support tools."""

        return {
            "type": "function",
            "function": {
                "name": self.tool_id.replace(".", "_"),
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # -- validation -------------------------------------------------------
    def validate(self, arguments: Any) -> dict[str, Any]:
        """Validate arguments against the declared JSON schema subset."""

        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ToolValidationError(f"Arguments for '{self.tool_id}' are not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ToolValidationError(
                f"Arguments for '{self.tool_id}' must be an object, got {type(arguments).__name__}."
            )
        _validate_object(arguments, self.parameters, path=self.tool_id)
        return arguments

    def to_dict(self, permissions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "output_schema": self.output_schema,
            "requires_network": self.requires_network,
            "requires_write": self.requires_write,
            "requires_code_execution": self.requires_code_execution,
            "timeout_s": self.timeout_s,
            "tags": list(self.tags),
            "model_callable": self.model_callable,
            "permissions": dict(permissions or {}),
        }

    def __call__(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        if self.handler is None:
            raise ToolExecutionError(
                f"Tool '{self.tool_id}' has no handler bound and cannot execute.",
                remediation="This is an AlphaAI installation problem; reinstall (`pip install -e .`).",
            )
        return self.handler(arguments, context)


class ToolRegistry:
    """Holds every tool AlphaAI knows about."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(f"register() expects a Tool, got {type(tool)!r}")
        if tool.tool_id in self._tools and not replace:
            raise ValueError(f"Tool '{tool.tool_id}' is already registered.")
        self._tools[tool.tool_id] = tool
        return tool

    def unregister(self, tool_id: str) -> bool:
        return self._tools.pop(tool_id, None) is not None

    def get(self, tool_id: str) -> Tool:
        tool = self._tools.get(tool_id)
        if tool is None:
            normalised = tool_id.replace("_", ".") if tool_id not in self._tools else tool_id
            tool = self._tools.get(normalised)
        if tool is None:
            raise ToolNotFoundError(
                f"Unknown AlphaAI tool '{tool_id}'.",
                remediation=f"Known tools: {', '.join(sorted(self._tools)) or '(none)'}.",
            )
        return tool

    def has(self, tool_id: str) -> bool:
        try:
            self.get(tool_id)
            return True
        except ToolNotFoundError:
            return False

    def __iter__(self):
        return iter(sorted(self._tools.values(), key=lambda tool: tool.tool_id))

    def __len__(self) -> int:
        return len(self._tools)

    def list(self, *, category: str | None = None, model_callable_only: bool = False) -> list[Tool]:
        tools = list(self)
        if category:
            tools = [tool for tool in tools if tool.category == category]
        if model_callable_only:
            tools = [tool for tool in tools if tool.model_callable]
        return tools

    def model_schemas(self, tool_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        tools = [self.get(tool_id) for tool_id in tool_ids] if tool_ids else self.list(model_callable_only=True)
        return [tool.model_schema() for tool in tools if tool.model_callable]


# ---------------------------------------------------------------------------
# JSON-schema subset validation (real checks, no third-party dependency)
# ---------------------------------------------------------------------------
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
    "null": (type(None),),
}


def validate_schema(value: Any, schema: Mapping[str, Any], *, path: str = "input") -> None:
    """Public entry point for AlphaAI's JSON-schema-subset validation.

    Used by tools *and* skills so both layers reject bad input identically.
    """

    if schema.get("type") == "object" and isinstance(value, dict):
        _validate_object(value, schema, path=path)
        return
    _validate_value(value, schema, path=path)


def _validate_object(value: dict[str, Any], schema: Mapping[str, Any], *, path: str) -> None:
    properties: dict[str, Any] = schema.get("properties") or {}
    required: Sequence[str] = schema.get("required") or ()
    additional = schema.get("additionalProperties", True)

    for key in required:
        if key not in value:
            raise ToolValidationError(f"Missing required argument '{key}' for {path}.")
    for key, raw in value.items():
        if key not in properties:
            if additional is False:
                raise ToolValidationError(
                    f"Unexpected argument '{key}' for {path}. Allowed: {', '.join(sorted(properties)) or '(none)'}."
                )
            continue
        _validate_value(raw, properties[key], path=f"{path}.{key}")


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if expected:
        candidates = expected if isinstance(expected, list) else [expected]
        allowed_types = tuple(t for name in candidates for t in _TYPES.get(name, ()))
        if not isinstance(value, bool) and allowed_types and not isinstance(value, allowed_types):
            raise ToolValidationError(
                f"Argument '{path}' must be of type {'|'.join(candidates)}, got {type(value).__name__}."
            )
        if isinstance(value, bool) and "boolean" not in candidates and "integer" not in candidates:
            raise ToolValidationError(f"Argument '{path}' must be of type {'|'.join(candidates)}, got boolean.")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"Argument '{path}' must be one of {schema['enum']}, got {value!r}.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"Argument '{path}' must be >= {schema['minimum']}.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"Argument '{path}' must be <= {schema['maximum']}.")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"Argument '{path}' exceeds maxLength {schema['maxLength']}.")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ToolValidationError(f"Argument '{path}' does not match pattern {schema['pattern']!r}.")
    if isinstance(value, dict) and schema.get("type") == "object":
        _validate_object(value, schema, path=path)
    if isinstance(value, (list, tuple)) and schema.get("type") == "array":
        items = schema.get("items")
        if items:
            for index, item in enumerate(value):
                _validate_value(item, items, path=f"{path}[{index}]")


# ---------------------------------------------------------------------------
# default registry
# ---------------------------------------------------------------------------
def build_default_registry() -> ToolRegistry:
    """Build the registry of AlphaAI built-in tools."""

    from . import builtin

    return ToolRegistry(builtin.build_tools())

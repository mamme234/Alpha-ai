"""Tool system tests: registry, permissions, validation, execution, built-ins."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaai.core.errors import (
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)
from alphaai.core.tools.executor import build_executor
from alphaai.core.tools.registry import build_default_registry, validate_schema

EXPECTED_TOOLS = {
    "calculator.evaluate",
    "text.transform",
    "text.regex",
    "json.path",
    "file.read",
    "file.list",
    "file.write",
    "file.hash",
    "file.stat",
    "python.sandbox",
    "http.fetch",
    "web.search",
    "system.info",
    "clock.now",
    "skill.run",
}


def test_default_registry_has_handlers(config) -> None:
    registry = build_default_registry()
    assert {tool.tool_id for tool in registry.list()} == EXPECTED_TOOLS
    assert all(tool.handler is not None for tool in registry.list())
    with pytest.raises(ToolNotFoundError):
        registry.get("does.not.exist")
    assert registry.has("calculator.evaluate")
    assert registry.get("calculator_evaluate").tool_id == "calculator.evaluate"


def test_deny_by_default_policy(config) -> None:
    executor = build_executor(config)
    permitted = {tool.tool_id for tool in executor.available_tools()}
    assert "calculator.evaluate" in permitted
    assert "web.search" not in permitted  # network off by default
    assert "python.sandbox" not in permitted  # code execution off by default
    assert "file.write" not in permitted  # writes off by default

    with pytest.raises(ToolPermissionError) as excinfo:
        executor.execute("web.search", {"query": "alphaai"})
    assert "Network access is disabled" in excinfo.value.message
    assert excinfo.value.remediation

    result = executor.run("web.search", {"query": "alphaai"})
    assert result.ok is False
    assert result.error["code"] == "tool_permission_denied"


def test_network_and_write_gates_can_be_enabled(config) -> None:
    config.tools.allow_network = True
    config.tools.allow_writes = True
    executor = build_executor(config)
    assert "web.search" in {tool.tool_id for tool in executor.available_tools()}
    assert "file.write" in {tool.tool_id for tool in executor.available_tools()}


def test_argument_validation(config) -> None:
    executor = build_executor(config)
    with pytest.raises(ToolValidationError):
        executor.execute("calculator.evaluate", {})
    with pytest.raises(ToolValidationError):
        executor.execute("calculator.evaluate", {"expression": 5})
    with pytest.raises(ToolValidationError):
        executor.execute("calculator.evaluate", {"expression": "2+2", "extra": 1})
    # JSON strings are accepted (models emit JSON strings)
    result = executor.run("calculator.evaluate", '{"expression": "6*7"}')
    assert result.ok and result.output["value"] == 42


def test_calculator_and_text_tools(config) -> None:
    executor = build_executor(config)
    calc = executor.run("calculator.evaluate", {"expression": "(3.5 ** 2 + 41) / 7"})
    assert calc.ok and abs(calc.output["value"] - 7.607142857) < 1e-6
    bad = executor.run("calculator.evaluate", {"expression": "__import__('os')"})
    assert bad.ok is False

    text = executor.run("text.transform", {"text": "Hello Alpha  World", "operation": "slugify"})
    assert text.ok and text.output["result"] == "hello-alpha-world"

    words = executor.run("text.transform", {"text": "a b a", "operation": "frequency", "limit": 5})
    assert words.ok
    json_tool = executor.run("json.path", {"data": {"a": {"b": [1, 2]}}, "path": "a.b[1]"})
    assert json_tool.ok


def test_file_tools_stay_inside_sandbox(config) -> None:
    executor = build_executor(config)
    config.tools.allow_writes = True
    (Path(config.tools.sandbox_root) / "note.txt").write_text("hello alphaai", encoding="utf-8")

    read = executor.run("file.read", {"path": "note.txt"})
    assert read.ok and "hello alphaai" in read.output["content"]

    listed = executor.run("file.list", {"path": "."})
    assert listed.ok
    assert "note.txt" in [entry["name"] for entry in listed.output["entries"]]
    assert all(not entry["path"].startswith("/") for entry in listed.output["entries"])

    escape = executor.run("file.read", {"path": "../../etc/passwd"})
    assert escape.ok is False
    assert escape.error["code"] in {"tool_permission_denied", "tool_execution_failed"}

    digest = executor.run("file.hash", {"path": "note.txt", "algorithm": "sha256"})
    assert digest.ok and len(digest.output["digest"]) == 64

    written = executor.run("file.write", {"path": "made.txt", "content": "written by alphaai"})
    assert written.ok and (Path(config.tools.sandbox_root) / "made.txt").exists()


def test_timeout_and_call_budget(config) -> None:
    from alphaai.core.tools.registry import Tool, ToolRegistry

    registry = build_default_registry()
    registry.register(
        Tool(
            tool_id="test.sleep",
            name="Sleep",
            description="test only",
            parameters={"type": "object", "properties": {"seconds": {"type": "number"}}},
            category="test",
            handler=lambda arguments, context: __import__("time").sleep(arguments.get("seconds", 1)),
            timeout_s=0.1,
        ),
        replace=True,
    )
    executor = build_executor(config, registry=registry)
    with pytest.raises(ToolTimeoutError):
        executor.execute("test.sleep", {"seconds": 1.0})
    assert executor.run("test.sleep", {"seconds": 1.0}).error["code"] == "tool_timeout"

    config.tools.permissions["calculator.evaluate"] = type(config.tools.permissions) and __import__(
        "alphaai.config.schema", fromlist=["ToolPermissions"]
    ).ToolPermissions(allow=True, max_calls_per_run=1)
    budgeted = build_executor(config)
    assert budgeted.run("calculator.evaluate", {"expression": "1+1"}, run_id="cli").ok is True
    second = budgeted.run("calculator.evaluate", {"expression": "1+1"}, run_id="cli")
    assert second.ok is False  # the budget is per run_id and already spent
    assert second.error["code"] == "tool_permission_denied"


def test_executor_logs_every_call(config) -> None:
    executor = build_executor(config)
    executor.run("calculator.evaluate", {"expression": "1+1"})
    executor.run("web.search", {"query": "blocked by policy"})
    log = executor.log.to_dict()
    assert log["calls"] == 2
    assert log["failures"] == 1
    assert all(record["run_id"] for record in log["records"])


def test_schema_validation_helper() -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["n"],
        "additionalProperties": False,
    }
    validate_schema({"n": 3}, schema)
    with pytest.raises(ToolValidationError):
        validate_schema({"n": 9}, schema)
    with pytest.raises(ToolValidationError):
        validate_schema({"other": 1}, schema)


def test_output_truncation(config) -> None:
    config.tools.max_output_bytes = 64
    executor = build_executor(config)
    result = executor.run("text.transform", {"text": "x" * 500, "operation": "upper"})
    assert result.ok
    assert result.metadata["truncated"] is True

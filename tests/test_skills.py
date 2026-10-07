"""Skill system tests: 12 real skills through the manager and the runtime."""

from __future__ import annotations

import pytest

EXPECTED_SKILLS = {
    "skill.calculator",
    "skill.reasoning",
    "skill.data_analysis",
    "skill.text_transform",
    "skill.summarization",
    "skill.translation",
    "skill.file_analysis",
    "skill.document_processing",
    "skill.code_analysis",
    "skill.tool_calling",
    "skill.web_research",
    "skill.agent_workflow",
}


def test_twelve_skills_registered(runtime) -> None:
    assert {skill.skill_id for skill in runtime.skills.list()} == EXPECTED_SKILLS
    assert len(runtime.skills.list()) == 12
    summary = runtime.skills.summary()
    assert summary["registered"] == 12
    assert summary["engine_backed"] == ["skill.translation"]


def test_skills_describe_includes_schemas(runtime) -> None:
    described = runtime.skills_view()
    for skill in described:
        assert skill["input_schema"]["type"] == "object"
        assert "available" in skill
        assert "requires_engine" in skill
    calculator = next(skill for skill in described if skill["id"] == "skill.calculator")
    assert calculator["available"] is True
    # Exactly the skills whose requirements are missing here: no usable model engine
    # (translation) and no network permission (web research).
    unavailable = {skill["id"] for skill in described if not skill["available"]}
    assert unavailable == {"skill.translation", "skill.web_research"}


def test_calculator_skill(runtime) -> None:
    result = runtime.execute_skill("skill.calculator", {"expression": "2 * (3 + 4)"})
    assert result.ok
    assert result.output["results"][0]["value"] == 14
    assert result.output["last"] == "14"
    invalid = runtime.execute_skill("skill.calculator", {"expression": "import os"})
    assert invalid.ok is False
    assert invalid.error["code"] in {"skill_execution_failed", "skill_invalid_input"}


def test_reasoning_skill_modes(runtime) -> None:
    sequence = runtime.execute_skill(
        "skill.reasoning", {"mode": "sequence", "values": [2, 4, 6, 8], "predict": 2}
    )
    assert sequence.ok
    assert sequence.output["kind"] == "arithmetic"
    assert sequence.output["next_values"] == [10.0, 12.0]

    linear = runtime.execute_skill(
        "skill.reasoning",
        {"mode": "linear_system", "matrix": [[2, 1], [1, -1]], "vector": [5, 1]},
    )
    assert linear.ok
    assert linear.output["solution"] == {"x1": "2", "x2": "1"}

    steps = runtime.execute_skill(
        "skill.reasoning",
        {
            "mode": "evaluate_steps",
            "steps": [
                {"name": "base", "expression": "3 + 4"},
                {"name": "doubled", "expression": "base * 2"},
            ],
        },
    )
    assert steps.ok and steps.output["final_value"] == "14"
    assert steps.output["variables"]["doubled"] == 14

    options = runtime.execute_skill(
        "skill.reasoning",
        {
            "mode": "compare_options",
            "options": [{"name": "a", "values": {"speed": 5}}, {"name": "b", "values": {"speed": 3}}],
            "weights": {"speed": 1},
        },
    )
    assert options.ok and options.output["winner"] == "a"


def test_data_analysis_skill(runtime) -> None:
    csv = "city,pop\nA,10\nB,20\nA,30\n"
    result = runtime.execute_skill(
        "skill.data_analysis",
        {
            "data": csv,
            "format": "csv",
            "operations": ["describe", "group_by"],
            "group_by": "city",
            "value_column": "pop",
            "aggregate": "sum",
        },
    )
    assert result.ok
    assert result.output["rows"] == 3
    assert result.output["columns"] == ["city", "pop"]
    assert result.output["describe"]["pop"]["max"] == 30
    assert result.output["describe"]["pop"]["sum"] == 60.0
    assert result.output["group_by"] == [
        {"group": "A", "count": 2, "value": 40.0, "aggregate": "sum"},
        {"group": "B", "count": 1, "value": 20.0, "aggregate": "sum"},
    ]


def test_text_and_summarization_skills(runtime) -> None:
    transformed = runtime.execute_skill(
        "skill.text_transform",
        {"text": "  Alpha   AI  ", "steps": [{"op": "collapse_whitespace"}, {"op": "strip"}]},
    )
    assert transformed.ok and transformed.output["text"] == "Alpha AI"
    assert transformed.output["words"] == 2

    document = (
        "AlphaAI runs local models. The AlphaAI router picks an engine. "
        "The context manager trims history. AlphaAI tools are permission checked. "
        "The AlphaAI training foundation trains a tokenizer."
    )
    summary = runtime.execute_skill(
        "skill.summarization", {"text": document, "max_sentences": 2, "mode": "extractive"}
    )
    assert summary.ok
    assert summary.output["selected_sentences"]
    assert "AlphaAI" in summary.output["summary"]


def test_translation_skill_reports_missing_engine(runtime) -> None:
    result = runtime.execute_skill("skill.translation", {"text": "hello", "target_language": "french"})
    assert result.ok is False
    assert result.error["code"] in {
        "skill_permission_denied",
        "skill_execution_failed",
        "no_suitable_model",
    }
    assert result.error.get("remediation")


def test_file_document_and_code_skills(runtime, config) -> None:
    from pathlib import Path

    root = Path(config.tools.sandbox_root)
    (root / "data.json").write_text('{"a": [1, 2, 3], "b": {"c": "d"}}', encoding="utf-8")
    (root / "app.py").write_text(
        "import os\n\n\ndef run(value):\n    if value:\n        return os.path.join('a', 'b')\n    return None\n",
        encoding="utf-8",
    )

    analysis = runtime.execute_skill("skill.file_analysis", {"path": "data.json"})
    assert analysis.ok
    entry = analysis.output["files"][0]
    assert len(entry["sha256"]) == 64
    assert entry["language"] == "json"

    document = runtime.execute_skill("skill.document_processing", {"path": "data.json"})
    assert document.ok
    assert document.output["chunks"][0]["text"]
    assert document.output["format"] == "json"

    code = runtime.execute_skill("skill.code_analysis", {"path": "app.py"})
    assert code.ok
    assert code.output["language"] == "python"
    metrics = code.output.get("metrics") or {}
    assert metrics.get("lines", 0) > 0
    assert metrics.get("functions", 0) >= 1


def test_tool_calling_skill_parses_and_executes(runtime) -> None:
    text = 'Use a tool:\n```json\n{"tool": "calculator.evaluate", "arguments": {"expression": "6*7"}}\n```'
    parsed = runtime.execute_skill("skill.tool_calling", {"mode": "parse", "text": text})
    assert parsed.ok
    assert parsed.output["calls"][0]["tool_id"] == "calculator.evaluate"

    executed = runtime.execute_skill("skill.tool_calling", {"mode": "execute", "text": text})
    assert executed.ok
    assert executed.output["succeeded"] == 1
    assert executed.output["results"][0]["output"]["value"] == 42


def test_web_research_skill_is_denied_without_network(runtime) -> None:
    result = runtime.execute_skill("skill.web_research", {"query": "alphaai"})
    assert result.ok is False
    assert "network" in (result.error.get("message") or "").lower() or result.error["code"].endswith(
        "permission_denied"
    )


def test_agent_workflow_skill_runs_declared_plan(runtime) -> None:
    result = runtime.execute_skill(
        "skill.agent_workflow",
        {
            "steps": [
                {"id": "first", "kind": "skill", "target": "skill.calculator", "input": {"expression": "21 * 2"}},
                {
                    "id": "second",
                    "kind": "tool",
                    "target": "calculator.evaluate",
                    "input": {"expression": "6 * 7"},
                    "depends_on": ["first"],
                },
            ],
        },
    )
    assert result.ok
    assert result.output["completed"] == 2
    assert result.output["order"] == ["first", "second"]


def test_unknown_skill_and_validation_errors(runtime) -> None:
    unknown = runtime.execute_skill("skill.nope", {})
    assert unknown.ok is False and unknown.error["code"] == "skill_not_found"
    invalid = runtime.execute_skill("skill.calculator", {"expression": 42})
    assert invalid.ok is False and invalid.error["code"] == "skill_invalid_input"
    unreadable = runtime.execute_skill("skill.calculator", {"expression": "2 +"})
    assert unreadable.ok is False and unreadable.error["code"] == "skill_execution_failed"


def test_skill_timeout_reports_structured_error(runtime, config) -> None:
    from alphaai.core.skills.base import Skill, SkillResult
    from alphaai.core.skills.manager import SkillManager

    class SlowSkill(Skill):
        skill_id = "skill.slow"
        name = "Slow"
        description = "sleeps"
        timeout_s = 0.05
        input_schema = {"type": "object", "properties": {}}

        def run(self, inputs, context):
            import time

            time.sleep(0.5)
            return {}

    manager = SkillManager(config, skills=[SlowSkill()])
    result = manager.run_result("skill.slow", {}, context=runtime._fresh_skill_context())
    assert result.ok is False
    assert result.error["code"] == "skill_timeout"
    assert "0.05" in result.error["message"]

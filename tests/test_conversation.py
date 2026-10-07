"""Conversation engine, streaming and orchestrator tests (test engine only)."""

from __future__ import annotations

from alphaai.core.errors import NoSuitableModelError


def test_chat_returns_real_outcome(engine_runtime) -> None:
    outcome = engine_runtime.chat("hello there")
    payload = outcome.to_dict()
    # ChatOutcome carries data only: the API adds the "ok" envelope.
    assert "ok" not in payload
    assert outcome.text.startswith("alphaai test reply")
    assert outcome.engine_id == "alphaai-test"
    assert outcome.usage.total_tokens == 15
    assert outcome.routing["engine_id"] == "alphaai-test"
    assert outcome.context["messages"]
    assert payload["attribution"]


def test_chat_records_history_and_sessions(engine_runtime) -> None:
    session = engine_runtime.create_session(system_prompt="You are AlphaAI.")
    engine_runtime.chat("first", session=session)
    engine_runtime.chat("second", session=session)
    assert session.turns == 2
    roles = [message.role for message in session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert session.session_id in {entry["session_id"] for entry in engine_runtime.conversation.sessions()}


def test_chat_memory_is_stored_and_retrieved(engine_runtime) -> None:
    session = engine_runtime.create_session()
    engine_runtime.chat("my favourite engine is the llama cpp engine", session=session)
    entries = engine_runtime.memory.entries(namespace=session.session_id)
    assert any("llama" in entry.value for entry in entries)
    hits = engine_runtime.memory.search("llama", namespace=session.session_id)
    assert hits


def test_tool_loop_executes_real_tools(tool_call_engine) -> None:
    outcome = tool_call_engine.chat("what is 2+2?", use_tools=True)
    assert outcome.iterations == 2
    assert len(outcome.tool_results) == 1
    result = outcome.tool_results[0]
    assert result.tool_id == "calculator.evaluate"
    assert result.ok and result.output["value"] == 4
    assert [event["type"] for event in outcome.events] == ["tool_call", "tool_result"]
    assert outcome.text == "final answer after tools"


def test_tool_loop_can_be_disabled(tool_call_engine) -> None:
    outcome = tool_call_engine.chat("what is 2+2?", use_tools=False)
    assert outcome.iterations == 1
    assert outcome.tool_results == []


def test_stream_emits_route_deltas_and_done(engine_runtime) -> None:
    events = list(engine_runtime.stream("hello"))
    kinds = [event["type"] for event in events]
    assert kinds[0] == "route"
    assert kinds[-1] == "done"
    text = "".join(event["text"] for event in events if event["type"] == "delta")
    assert text.startswith("alphaai test reply")
    assert events[-1]["token_source"] in {"tokenizer", "estimate"}


def test_stream_reports_no_engine_as_event(runtime) -> None:
    events = list(runtime.stream("hello"))
    assert events[0]["type"] == "error"
    assert events[0]["error"]["code"] == "no_suitable_model"


def test_chat_without_engine_raises_structured_error(runtime) -> None:
    import pytest

    with pytest.raises(NoSuitableModelError) as excinfo:
        runtime.chat("hello")
    assert excinfo.value.remediation
    assert "candidates" in excinfo.value.details


def test_generate_bridge_routes_prompt(runtime, config) -> None:
    from .conftest import FakeEngine, test_spec

    runtime.registry.register(FakeEngine(test_spec(), config))
    result = runtime.generate("say something", max_tokens=16)
    assert result.engine_id == "alphaai-test"
    assert result.attribution


def test_goal_orchestration_runs_model_turns(tool_call_engine) -> None:
    report = tool_call_engine.orchestrator.run_goal("compute 2+2", max_steps=3)
    assert report.mode == "goal"
    assert report.ok is True
    assert report.steps
    assert report.result["final_answer"]


def test_declared_plan_orchestration(runtime) -> None:
    report = runtime.orchestrator.run_plan(
        [
            {"id": "double", "skill": "skill.calculator", "input": {"expression": "6*7"}},
            {"id": "add", "skill": "skill.calculator", "input": {"expression": "21*2"}},
        ],
        goal="two calculations",
    )
    assert report.mode == "declared"
    assert report.ok is True
    assert len(report.steps) == 2
    assert report.to_dict()["succeeded"] == 2


def test_orchestrator_capabilities(runtime) -> None:
    capabilities = runtime.orchestrator.capabilities()
    assert capabilities["modes"] == ["declared", "goal"]
    assert "no separate planner" in capabilities["note"]

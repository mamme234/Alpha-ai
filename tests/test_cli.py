"""CLI tests: every command runs and reports truthfully."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaai.cli import build_parser, main

from .conftest import LOCAL_MODEL_ID, REPO_ROOT, requires_local_model

#: Model id AlphaAI installs on CPU-only machines (configs/models/*.json).
LOCAL_MODEL_ID = "qwen2.5-0.5b-instruct-gguf"


def run_cli(capsys, *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_parser_has_expected_commands() -> None:
    parser = build_parser()
    actions = parser._subparsers._group_actions[0].choices
    for command in ("version", "doctor", "models", "skills", "tools", "chat", "infer", "route", "orchestrate", "serve", "train", "config"):
        assert command in actions


def test_version_and_attribution(capsys) -> None:
    code, out = run_cli(capsys, "version")
    assert code == 0
    assert "ALPHA AI" in out
    assert "Intelligence, built from the ground up." in out
    assert "DeepSeek-V3" in out

    code, out = run_cli(capsys, "attribution")
    assert code == 0
    assert "LICENSE-MODEL" in out
    assert "created by DeepSeek" in out


@pytest.fixture
def empty_project(tmp_path: Path) -> str:
    """A project root with no model metadata: the honest "no engine" case."""

    root = tmp_path / "empty-project"
    root.mkdir()
    return str(root)


def test_models_list_and_show(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "models", "list")
    assert code == 0
    assert "deepseek-v3" in out
    assert "AlphaAI DeepSeek Engine" in out
    # Models whose weights/runtime are absent are reported honestly (never as ok).
    assert "UNAVAILABLE" in out

    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "models", "list", "--json")
    payload = json.loads(out)
    assert payload["count"] >= 9
    assert any(model["model_owner"] == "DeepSeek" for model in payload["models"])


def test_models_show_records_the_selected_model_rationale(capsys) -> None:
    """`models show` exposes why the local model was selected for this machine."""

    code, out = run_cli(
        capsys, "--project-root", str(REPO_ROOT), "models", "show", "qwen2.5-0.5b-instruct-gguf"
    )
    assert code == 0
    payload = json.loads(out)
    selection = payload["selection"]
    assert selection["phase"] == "select-model"
    assert selection["criteria"]["inference_runtime"] == "llama_cpp"
    assert selection["official_source"]["repo"] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert payload["model_owner"] == "Alibaba"  # AlphaAI never claims the weights
    assert payload["quantization"] == "q4_k_m"


def test_models_check_reports_remediation(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "models", "check", "deepseek-v3", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["deepseek-v3"]["usable"] is False
    assert payload["deepseek-v3"]["remediation"]


def test_skills_and_tools(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "skills", "list", "--json")
    assert code == 0
    assert json.loads(out)["count"] == 12

    code, out = run_cli(
        capsys,
        "--project-root",
        str(REPO_ROOT),
        "skills",
        "run",
        "skill.calculator",
        "--input",
        '{"expression": "1+1"}',
    )
    payload = json.loads(out)
    assert payload["ok"] is True and payload["output"]["results"][-1]["value"] == 2

    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "tools", "run", "calculator.evaluate", "--input", '{"expression": "9*9"}')
    assert json.loads(out)["output"]["value"] == 81

    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "tools", "run", "web.search", "--input", '{"query": "x"}')
    payload = json.loads(out)
    assert payload["ok"] is False and payload["error"]["code"] == "tool_permission_denied"


def test_route_explains_failure(capsys, empty_project: str) -> None:
    code, out = run_cli(capsys, "--project-root", empty_project, "route", "write a python function")
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "no_suitable_model"


def test_chat_reports_no_engine_instead_of_faking(capsys, empty_project: str) -> None:
    """With no engines registered the CLI refuses with a structured error."""

    code, out = run_cli(capsys, "--project-root", empty_project, "chat", "hello", "--no-stream")
    assert code == 1
    assert "no_suitable_model" in out
    assert "remediation" in out


def test_doctor_prints_hardware_block(capsys) -> None:
    """`alphaai doctor` reports the measured hardware, then the JSON report."""

    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "doctor")
    for label in (
        "AlphaAI Hardware",
        "OS:",
        "CPU:",
        "RAM:",
        "GPU:",
        "VRAM:",
        "Storage:",
        "Architecture:",
        "Runtimes:",
        "Supported inference runtimes:",
        "Recommended model size:",
    ):
        assert label in out
    payload = json.loads(out[out.index("{") :])
    assert payload["config_source"]
    assert any("CUDA" in item["message"] for item in payload["findings"])


def test_doctor_json_reports_recommendation(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "doctor", "--json")
    payload = json.loads(out)
    block = payload["hardware_block"]
    assert block["cpu"] and block["ram"] and block["architecture"]
    assert block["recommendation"]["recommended_params_b"] >= 0.0
    assert "recommended_model_size" in payload


def test_doctor_and_config(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "doctor", "--json")
    payload = json.loads(out)
    assert payload["config_source"]
    assert any("CUDA" in item["message"] for item in payload["findings"])

    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "config", "validate")
    assert json.loads(out)["ok"] is True

    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "config", "paths")
    assert "models_dir" in json.loads(out)


def test_train_status(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "train", "status")
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["datasets"]
    assert payload["datasets"][0]["name"] == "alphaai-sample"


def test_train_validate(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "train", "validate")
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["datasets"][0]["manifest_ok"] is True


def test_models_install_dry_run_prints_safety_report(capsys) -> None:
    """`models install --dry-run` shows the safety report and downloads nothing."""

    code, out = run_cli(
        capsys, "--project-root", str(REPO_ROOT), "models", "install", LOCAL_MODEL_ID, "--dry-run"
    )
    for label in (
        "Model:",
        "Size:",
        "Parameters:",
        "Quantization:",
        "Runtime:",
        "Required RAM:",
        "Available RAM:",
        "Required storage:",
        "Available storage:",
    ):
        assert label in out
    assert "Downloaded" not in out


def test_models_install_unknown_model(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "models", "show", "nope-nope")
    assert code == 1
    assert "unknown_model" in out


@requires_local_model
def test_models_list_shows_the_local_model_available(capsys) -> None:
    code, out = run_cli(capsys, "--project-root", str(REPO_ROOT), "models", "list")
    assert code == 0
    line = next(line for line in out.splitlines() if line.startswith(LOCAL_MODEL_ID))
    assert "AVAILABLE" in line
    assert "AlphaAI llama.cpp Engine" in line


@requires_local_model
def test_cli_chat_produces_real_model_output(capsys) -> None:
    """CLI chat returns real generated text and names the model that wrote it."""

    code, out = run_cli(
        capsys,
        "--project-root",
        str(REPO_ROOT),
        "chat",
        "--no-stream",
        "--max-tokens",
        "32",
        "Hello AlphaAI. Tell me what you are.",
    )
    assert code == 0, out
    assert "no_suitable_model" not in out
    assert "Model: Qwen2.5-0.5B-Instruct" in out  # the real model, never hidden
    assert "Mode: Local" in out
    assert "created by Alibaba" in out  # honest attribution

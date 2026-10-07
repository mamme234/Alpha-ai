"""Configuration system tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaai.config.loader import ConfigurationError, load_config, public_config_view, resolve_config_path
from alphaai.config.schema import AlphaAIConfig


def test_defaults_are_usable() -> None:
    config = load_config(project_root=".", env={})
    assert config.name == "AlphaAI"
    assert config.tagline == "Intelligence, built from the ground up."
    assert config.validate() == []
    assert config.tools.allow_network is False
    assert config.tools.permissions == {}
    assert config.api.port == 8090


def test_toml_config_round_trip(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "alphaai.toml").write_text(
        """
name = "AlphaAI Test"
[api]
port = 8111
[tools]
allow_network = true
allow_writes = false
""",
        encoding="utf-8",
    )
    config = load_config(project_root=str(tmp_path), env={}, create_dirs=True)
    assert config.name == "AlphaAI Test"
    assert config.api.port == 8111
    assert config.tools.allow_network is True
    assert resolve_config_path(project_root=str(tmp_path)).name == "alphaai.toml"
    assert Path(config.paths.models_dir).is_absolute()


def test_json_config_and_namespacing(tmp_path: Path) -> None:
    (tmp_path / "alphaai.json").write_text(
        json.dumps({"alphaai": {"sampling": {"temperature": 1.25}, "name": "AlphaAI JSON"}}),
        encoding="utf-8",
    )
    config = load_config(project_root=str(tmp_path), env={})
    assert config.name == "AlphaAI JSON"
    assert config.sampling.temperature == 1.25


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "alphaai.toml").write_text("[sampling]\nnot_a_setting = 3\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(project_root=str(tmp_path), env={})


def test_env_overrides_and_validation(tmp_path: Path) -> None:
    env = {"ALPHAI_DEVICE": "cpu", "ALPHAI_MAX_TOKENS": "32", "ALPHAI_TOOL_NETWORK": "yes"}
    config = load_config(project_root=str(tmp_path), env=env)
    assert config.runtime.device == "cpu"
    assert config.sampling.max_tokens == 32
    assert config.tools.allow_network is True
    assert "+env" in config.source

    # An invalid environment override is rejected at load time...
    with pytest.raises(ConfigurationError):
        load_config(project_root=str(tmp_path), env={"ALPHAI_TEMPERATURE": "9"})
    # ...and validate() catches the same problem when it reaches the object directly.
    bad = load_config(project_root=str(tmp_path), env={})
    bad.sampling.temperature = 9.0
    assert any("temperature" in problem for problem in bad.validate())


def test_public_view_redacts_paths(tmp_path: Path) -> None:
    config = load_config(project_root=str(tmp_path), env={}, create_dirs=True)
    view = public_config_view(config)
    assert view["paths"]["models_dir"].startswith("<local>/")
    assert str(tmp_path) not in json.dumps(view)


def test_memory_and_tool_permission_objects() -> None:
    config = AlphaAIConfig()
    config.tools.permissions = {"web.search": __import__("alphaai.config.schema", fromlist=["ToolPermissions"]).ToolPermissions(allow=True, network=True, timeout_s=5.0)}
    assert config.validate() == []
    config.tools.permissions["broken"] = __import__("alphaai.config.schema", fromlist=["ToolPermissions"]).ToolPermissions(timeout_s=0)
    assert any("timeout_s" in problem for problem in config.validate())

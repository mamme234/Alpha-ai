"""Branding, attribution and legal-hygiene tests.

These guard the rule that AlphaAI branding must be separated from DeepSeek
attribution: AlphaAI owns the system layer, DeepSeek owns DeepSeek-V3, and the
licence/notice files must stay in the repository.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from alphaai import branding
from alphaai.core.engine import load_model_specs

from .conftest import REPO_ROOT


def test_identity_constants() -> None:
    assert branding.NAME == "AlphaAI"
    assert branding.DISPLAY_NAME == "ALPHA AI"
    assert branding.TAGLINE == "Intelligence, built from the ground up."
    assert branding.DEEPSEEK_V3_ENGINE_NAME == "AlphaAI DeepSeek Engine"
    assert branding.ATTRIBUTION_SHORT == "AlphaAI powered by DeepSeek-V3"
    assert branding.ENGINE_OWNER_DEFAULT == "AlphaAI"


def test_banner_and_attribution_lines() -> None:
    banner = branding.banner("hello")
    assert "ALPHA AI" in banner and branding.TAGLINE in banner and "hello" in banner
    lines = branding.attribution_lines()
    assert any("DeepSeek-V3" in line for line in lines)
    assert any("LICENSE-CODE" in line or "MIT" in line for line in lines)
    assert "not an AlphaAI-trained model" in branding.ATTRIBUTION_ENGINE
    assert "No AlphaAI weights exist yet" in branding.ATTRIBUTION_LONG


def test_engine_attribution_never_claims_ownership() -> None:
    line = branding.engine_attribution("AlphaAI DeepSeek Engine", "DeepSeek-V3", "DeepSeek")
    assert "created by DeepSeek" in line
    assert "owned by AlphaAI" not in line
    own = branding.engine_attribution("AlphaAI Core Engine", "AlphaAI-X", "AlphaAI")
    assert "owned by AlphaAI" in own


def test_preserved_license_files_exist() -> None:
    for name in branding.PRESERVED_NOTICES:
        assert (REPO_ROOT / name).exists(), f"{name} must remain in the repository"
    model_license = (REPO_ROOT / "LICENSE-MODEL").read_text(encoding="utf-8")
    assert "DeepSeek" in model_license
    code_license = (REPO_ROOT / "LICENSE-CODE").read_text(encoding="utf-8")
    assert "MIT" in code_license and "DeepSeek" in code_license


def test_vendored_inference_keeps_deepseek_copyright() -> None:
    model_source = (REPO_ROOT / "inference" / "model.py").read_text(encoding="utf-8")
    assert "DeepSeek" in model_source
    assert "Copyright" in model_source


def test_model_metadata_separates_owner_from_engine() -> None:
    specs = {spec.model_id: spec for spec in load_model_specs(directory=REPO_ROOT / "configs" / "models")}
    assert specs["deepseek-v3"].model_owner == "DeepSeek"
    assert specs["deepseek-v3"].engine_owner == "AlphaAI"
    assert specs["qwen2.5-7b-instruct"].model_owner == "Alibaba"
    assert specs["llama-3.1-8b-instruct"].model_owner == "Meta"
    assert specs["gemma-2-9b-it"].model_owner == "Google"
    assert specs["kimi-k2-instruct"].model_owner == "Moonshot AI"
    assert specs["alphaai-x"].model_owner == "AlphaAI"
    assert specs["alphaai-x"].base_model in {"from scratch", "actual"}


def test_api_and_cli_surfaces_carry_the_attribution(config) -> None:
    from fastapi.testclient import TestClient

    from alphaai.api.app import create_app
    from alphaai.core.runtime import AlphaRuntime

    runtime = AlphaRuntime.create(config, discover=False)
    try:
        with TestClient(create_app(config, runtime=runtime)) as client:
            body = client.get("/api/attribution").json()
            assert body["short"] == branding.ATTRIBUTION_SHORT
            assert "DeepSeek" in body["engine"]
    finally:
        runtime.close()


def test_readme_documents_attribution_and_is_rebranded() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.lstrip().startswith("<!-- AlphaAI")
    assert "ALPHA AI" in readme
    assert "Intelligence, built from the ground up." in readme
    assert "DeepSeek-V3" in readme
    assert "LICENSE-MODEL" in readme and "LICENSE-CODE" in readme
    assert "not created by AlphaAI" in readme or "created by DeepSeek" in readme


def test_notice_and_attribution_documents_exist() -> None:
    for name in ("NOTICE", "ATTRIBUTION.md", "docs/ARCHITECTURE.md", "docs/ENGINES.md", "docs/TRAINING.md"):
        assert (REPO_ROOT / name).exists(), f"{name} is part of the AlphaAI deliverables"
    attribution = (REPO_ROOT / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "Alex6712B" not in attribution
    assert "arXiv:2412.19437" in attribution
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "DeepSeek" in notice

"""Tests for the image-build weight prefetcher (deploy/prefetch_model.py).

Everything here runs without a network: the download is exercised against a
local file, because what must be proven is the provenance handling — the URL
and checksum come from the model spec, a bad file is refused, and a good one is
left alone rather than fetched twice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "prefetch_model.py"
_SPEC = importlib.util.spec_from_file_location("alphaai_prefetch_model", _MODULE_PATH)
prefetch_model = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prefetch_model)

WEIGHTS = b"not really a GGUF, but byte-exact for the test"
SHA256 = hashlib.sha256(WEIGHTS).hexdigest()


def write_spec(tmp_path: Path, **overrides) -> Path:
    """A model spec under ``configs/models`` with the test's provenance."""

    spec_dir = tmp_path / "configs"
    (spec_dir / "models").mkdir(parents=True, exist_ok=True)
    provenance = {
        "download_url": "https://example.invalid/qwen.gguf",
        "filename": "qwen.gguf",
        "file_size_bytes": len(WEIGHTS),
        "sha256": SHA256,
    }
    provenance.update(overrides)
    target = spec_dir / "models" / "qwen.json"
    target.write_text(json.dumps({"id": "qwen", "provenance": provenance}), encoding="utf-8")
    return spec_dir


def fake_download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(WEIGHTS)


def test_prefetch_downloads_and_verifies(tmp_path, monkeypatch):
    spec_dir = write_spec(tmp_path)
    monkeypatch.setattr(prefetch_model, "download", fake_download)

    result = prefetch_model.prefetch("qwen", spec_dir, tmp_path / "cache")

    assert result.read_bytes() == WEIGHTS
    assert result.parent.name == "qwen"


def test_prefetch_leaves_a_verified_file_alone(tmp_path, monkeypatch):
    spec_dir = write_spec(tmp_path)
    dest = tmp_path / "cache" / "qwen" / "qwen.gguf"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(WEIGHTS)

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a verified file must not be downloaded again")

    monkeypatch.setattr(prefetch_model, "download", explode)
    assert prefetch_model.prefetch("qwen", spec_dir, tmp_path / "cache") == dest


def test_prefetch_replaces_a_corrupt_partial_file(tmp_path, monkeypatch):
    spec_dir = write_spec(tmp_path)
    dest = tmp_path / "cache" / "qwen" / "qwen.gguf"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"truncated")  # wrong size: not treated as installed

    monkeypatch.setattr(prefetch_model, "download", fake_download)
    prefetch_model.prefetch("qwen", spec_dir, tmp_path / "cache")

    assert dest.read_bytes() == WEIGHTS


def test_prefetch_refuses_a_download_that_fails_its_checksum(tmp_path, monkeypatch):
    spec_dir = write_spec(tmp_path, sha256="0" * 64)
    monkeypatch.setattr(prefetch_model, "download", fake_download)

    with pytest.raises(prefetch_model.PrefetchError, match="SHA-256"):
        prefetch_model.prefetch("qwen", spec_dir, tmp_path / "cache")
    # The bad bytes are removed rather than left to be loaded later.
    assert not (tmp_path / "cache" / "qwen" / "qwen.gguf").exists()


def test_prefetch_refuses_a_short_download(tmp_path, monkeypatch):
    spec_dir = write_spec(tmp_path, file_size_bytes=len(WEIGHTS) + 10)

    def short_download(_url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(WEIGHTS)

    monkeypatch.setattr(prefetch_model, "download", short_download)

    with pytest.raises(prefetch_model.PrefetchError, match="incomplete"):
        prefetch_model.prefetch("qwen", spec_dir, tmp_path / "cache")


def test_prefetch_requires_provenance_and_spec(tmp_path):
    missing = tmp_path / "configs"
    with pytest.raises(prefetch_model.PrefetchError, match="not found"):
        prefetch_model.prefetch("nope", missing, tmp_path / "cache")

    spec_dir = write_spec(tmp_path)
    spec_path = spec_dir / "models" / "qwen.json"
    spec_path.write_text(json.dumps({"id": "qwen", "provenance": {"filename": "qwen.gguf"}}))
    with pytest.raises(prefetch_model.PrefetchError, match="download_url"):
        prefetch_model.prefetch("qwen", spec_dir, tmp_path / "cache")


def test_real_qwen_spec_declares_a_verifiable_download():
    """The committed spec the image build actually uses must be complete."""

    spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "models"
            / "qwen2.5-0.5b-instruct-gguf.json"
        ).read_text(encoding="utf-8")
    )
    provenance = spec["provenance"]
    assert provenance["download_url"].startswith("https://huggingface.co/Qwen/")
    assert provenance["filename"].endswith(".gguf")
    assert provenance["file_size_bytes"] == 491400032
    assert len(provenance["sha256"]) == 64


def test_main_reports_a_failure_with_a_nonzero_status(tmp_path, capsys):
    status = prefetch_model.main(
        ["qwen", "--spec-dir", str(tmp_path / "configs"), "--dest", str(tmp_path / "cache")]
    )
    assert status == 1
    assert "not found" in capsys.readouterr().err

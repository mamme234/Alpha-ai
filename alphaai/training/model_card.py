"""AlphaAI model metadata and model cards.

AlphaAI must always be able to answer three questions about a model:

1. **who trained the weights** (``trained_by`` / ``model_owner``)
2. **what it was trained on** (dataset names *and* versions, with licenses)
3. **which engine runs it** (always AlphaAI)

The card builder writes both a machine-readable ``model_card.json`` — shaped so it
can be dropped into ``configs/models/`` — and a human-readable ``MODEL_CARD.md``.
Nothing here ever claims AlphaAI trained a model it did not train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..branding import ENGINE_OWNER_DEFAULT, MODEL_OWNER_FUTURE, NAME

FRAMEWORK = "AlphaAI training foundation (reference)"


@dataclass(slots=True)
class ModelMetadata:
    """Everything AlphaAI records about one set of weights."""

    model_name: str
    model_owner: str
    engine: str = "alphaai"
    engine_owner: str = ENGINE_OWNER_DEFAULT
    base_model: str | None = None
    family: str = "alphaai"
    provider: str = "alphaai"
    context_length: int = 32768
    vocab_size: int = 0
    parameters: int = 0
    params_total_b: float = 0.0
    capabilities: tuple[str, ...] = ("chat", "streaming")
    trained_by: str = NAME
    trained_on: tuple[str, ...] = ()
    dataset_versions: Mapping[str, str] = field(default_factory=dict)
    datasets: tuple[Mapping[str, Any], ...] = ()
    tokenizer: str = "tokenizer"
    tokenizer_version: str = "1.0.0"
    checkpoint: str | None = None
    license: str = "AlphaAI Model License (planned)"
    license_file: str | None = None
    status: str = "reference"
    hparams: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    framework: str = FRAMEWORK
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_alphaai_owned(self) -> bool:
        return self.model_owner.strip().lower() == MODEL_OWNER_FUTURE.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_name.lower().replace(" ", "-"),
            "display_name": f"AlphaAI Core Engine ({self.model_name})",
            "engine": self.engine,
            "family": self.family,
            "provider": self.provider,
            "model": self.model_name,
            "model_owner": self.model_owner,
            "engine_owner": self.engine_owner,
            "base_model": self.base_model,
            "status": self.status,
            "context_length": self.context_length,
            "params_total_b": self.params_total_b,
            "params_active_b": self.params_total_b,
            "capabilities": list(self.capabilities),
            "weight_formats": ["bf16"],
            "license": self.license,
            "license_file": self.license_file,
            "vocab_size": self.vocab_size,
            "parameters": self.parameters,
            "trained_by": self.trained_by,
            "trained_on": list(self.trained_on),
            "dataset_versions": dict(self.dataset_versions),
            "datasets": [dict(entry) for entry in self.datasets],
            "tokenizer": {"name": self.tokenizer, "version": self.tokenizer_version},
            "checkpoint": self.checkpoint,
            "framework": self.framework,
            "hparams": dict(self.hparams),
            "metrics": dict(self.metrics),
            "notes": self.notes,
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.model_name}",
            "",
            f"- **Weights created by:** {self.model_owner}",
            f"- **Engine:** {self.engine} (engine by {self.engine_owner})",
            f"- **Base model:** {self.base_model or 'from scratch'}",
            f"- **Status:** {self.status}",
            f"- **License:** {self.license}",
            f"- **Framework:** {self.framework}",
            f"- **Parameters:** {self.parameters:,} ({self.params_total_b} B)",
            f"- **Context length:** {self.context_length:,} tokens",
            f"- **Tokenizer:** {self.tokenizer} v{self.tokenizer_version} (vocab {self.vocab_size})",
            f"- **Checkpoint:** {self.checkpoint or '(none yet)'}",
            "",
            "## Training data",
        ]
        if self.datasets:
            for entry in self.datasets:
                lines.append(
                    f"- `{entry.get('name')}` v{entry.get('version')} — {entry.get('records')} records, "
                    f"license {entry.get('license')} (source: {entry.get('source')})"
                )
        else:
            lines.append("- (no dataset recorded)")
        lines += ["", "## Hyperparameters"]
        for key, value in (self.hparams or {}).items():
            lines.append(f"- `{key}` = {value}")
        lines += ["", "## Measured metrics"]
        for key, value in (self.metrics or {}).items():
            lines.append(f"- `{key}` = {value}")
        lines += [
            "",
            "## Attribution",
            self.notes
            or (
                "If the weights above were not trained by AlphaAI, AlphaAI provides only the engine layer. "
                "See ATTRIBUTION.md and NOTICE for the licenses that apply to vendored components."
            ),
            "",
        ]
        return "\n".join(lines)


def build_metadata(
    *,
    model_name: str,
    model_owner: str,
    checkpoint: str | None,
    dataset_dir: str | Path | None,
    tokenizer_dir: str | Path | None,
    hparams: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    status: str = "reference",
    notes: str = "",
) -> ModelMetadata:
    """Collect real facts from the dataset/tokenizer/checkpoint on disk."""

    datasets: list[dict[str, Any]] = []
    versions: dict[str, str] = {}
    if dataset_dir:
        from . import dataset as dataset_mod

        try:
            spec = dataset_mod.load_spec(dataset_dir)
            report = dataset_mod.validate_dataset(dataset_dir)
            entry = {
                "name": spec.name,
                "version": spec.version,
                "license": spec.license,
                "source": spec.source,
                "language": spec.language,
                "records": report.records,
                "directory": str(dataset_dir),
                "sha256": {
                    name: info["sha256"] for name, info in report.splits.items()
                },
            }
            datasets.append(entry)
            versions[spec.name] = spec.version
        except Exception:  # noqa: BLE001 - the card must still be written
            pass

    vocab_size = 0
    tokenizer_name = "tokenizer"
    tokenizer_version = "1.0.0"
    if tokenizer_dir:
        from . import tokenizer as tokenizer_mod

        try:
            artifact = tokenizer_mod.load_tokenizer(tokenizer_dir)
            vocab_size = artifact.vocab_size
            tokenizer_name = artifact.config.name
            tokenizer_version = artifact.config.version
        except Exception:  # noqa: BLE001
            pass

    return ModelMetadata(
        model_name=model_name,
        model_owner=model_owner,
        base_model=(hparams or {}).get("base_model"),
        context_length=int((hparams or {}).get("block_size", 32768)),
        vocab_size=vocab_size,
        parameters=int((metrics or {}).get("parameters", 0)),
        params_total_b=round(int((metrics or {}).get("parameters", 0)) / 1e9, 6),
        trained_on=tuple(entry["name"] for entry in datasets) or ("(none recorded)",),
        dataset_versions=versions,
        datasets=tuple(datasets),
        tokenizer=tokenizer_name,
        tokenizer_version=tokenizer_version,
        checkpoint=checkpoint,
        hparams=dict(hparams or {}),
        metrics=dict(metrics or {}),
        status=status,
        notes=notes,
    )


def write_model_card(metadata: ModelMetadata, directory: str | Path) -> dict[str, str]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "model_card.json"
    markdown_path = target / "MODEL_CARD.md"
    json_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(metadata.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def candidate_model_entry(metadata: ModelMetadata) -> dict[str, Any]:
    """Return a ``configs/models/*.json``-shaped entry for these weights."""

    entry = metadata.to_dict()
    entry["local_paths"] = [f"models/{metadata.model_name.lower()}"]
    entry["weight_files"] = ["model.safetensors", "config.json"]
    return entry


__all__ = ["ModelMetadata", "build_metadata", "candidate_model_entry", "write_model_card"]

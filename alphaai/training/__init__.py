"""AlphaAI training foundation.

Layout produced and consumed here (all paths come from ``alphaai.config``):

    datasets/<name>/          dataset.json, splits, manifest.json
    tokenizer/                tokenizer.json, manifest.json
    checkpoints/<experiment>/ step-00010/{model.pt,checkpoint.json,options.json}
    training/experiments/<experiment>/runs/<run_id>/
    evaluation/reports/<timestamp>.json

`alphaai train <action>` and every script in ``scripts/`` call into this package,
so the CLI and the scripts cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.errors import AlphaAIError, DatasetError, TrainingError
from . import checkpoints as checkpoints_mod
from . import dataset as dataset_mod
from . import evaluation as evaluation_mod
from . import experiment as experiment_mod
from . import model_card as model_card_mod
from . import pipeline as pipeline_mod
from . import tokenizer as tokenizer_mod

__all__ = [
    "checkpoints_mod",
    "dataset_mod",
    "evaluation_mod",
    "experiment_mod",
    "model_card_mod",
    "pipeline_mod",
    "tokenizer_mod",
    "cli",
    "status",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_toml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise TrainingError(
                "Reading TOML training configs on Python < 3.11 needs the 'tomli' package.",
                remediation="pip install tomli",
            ) from exc
    return tomllib.loads(text)


def _config(args: argparse.Namespace) -> Any:
    """Load the AlphaAI config (never the training config file).

    ``alphaai train --config <file>`` points at a *training* config, so this
    deliberately ignores it and uses ALPHAI_CONFIG / configs/alphaai.toml.
    """

    from ..config.loader import load_config

    return load_config(
        None,
        project_root=getattr(args, "project_root", None),
        create_dirs=True,
    )


def _dataset_dir(config: Any, args: argparse.Namespace) -> Path:
    name = getattr(args, "dataset", None) or config.training.default_dataset
    candidate = Path(config.paths.datasets_dir) / name
    if candidate.exists():
        return candidate
    available = dataset_mod.find_datasets(config.paths.datasets_dir)
    if len(available) == 1:
        return available[0]
    raise DatasetError(
        f"Dataset '{name}' not found under {Path(config.paths.datasets_dir).name}/.",
        remediation="Pass --dataset <name>; see `alphaai train status` for what exists.",
    )


def _tokenizer_dir(config: Any, args: argparse.Namespace) -> Path:
    name = getattr(args, "tokenizer", None) or config.training.default_tokenizer
    path = Path(name)
    if path.is_absolute():
        return path
    if name in {config.training.default_tokenizer, "tokenizer"}:
        return Path(config.paths.tokenizer_dir)
    root = Path(config.paths.project_root)
    return path if path.is_absolute() else root / path


def _tokenizer_config(config: Any) -> tokenizer_mod.TokenizerConfig:
    for candidate in (Path(config.paths.configs_dir) / "training" / "alpha-1-tokenizer.toml",):
        payload = _load_toml(candidate)
        if payload:
            return tokenizer_mod.TokenizerConfig.from_dict(payload)
    return tokenizer_mod.TokenizerConfig()


def _experiment_config(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    explicit = getattr(args, "config", None)
    if explicit and Path(str(explicit)).exists():
        payload = _load_toml(explicit)
        if payload:
            return payload
    for candidate in (
        Path(config.paths.configs_dir) / "training" / "alpha-1-sft.toml",
        Path(config.paths.configs_dir) / "training" / "alpha-1-pretrain.toml",
    ):
        payload = _load_toml(candidate)
        if payload:
            return payload
    return {}


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def status(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    datasets = []
    for path in dataset_mod.find_datasets(config.paths.datasets_dir):
        try:
            datasets.append(dataset_mod.dataset_summary(path))
        except AlphaAIError as exc:
            datasets.append({"name": path.name, "valid": False, "problems": [exc.message]})
    runs = experiment_mod.list_runs(config.paths.training_dir, getattr(args, "experiment", None))
    checkpoint_root = Path(config.paths.checkpoints_dir)
    checkpoints = {
        entry.name: len(checkpoints_mod.CheckpointManager(entry, keep=config.training.checkpoint_keep).list())
        for entry in sorted(checkpoint_root.iterdir())
        if entry.is_dir()
    } if checkpoint_root.exists() else {}
    reports_dir = Path(config.paths.evaluation_dir) / "reports"
    return {
        "ok": True,
        "config_source": config.source,
        "paths": {
            "datasets": config.paths.datasets_dir,
            "tokenizer": config.paths.tokenizer_dir,
            "checkpoints": config.paths.checkpoints_dir,
            "training": config.paths.training_dir,
            "evaluation": config.paths.evaluation_dir,
        },
        "datasets": datasets,
        "tokenizer": tokenizer_mod.tokenizer_summary(config.paths.tokenizer_dir),
        "checkpoints": checkpoints,
        "experiments": runs[:10],
        "evaluation_reports": sorted(path.name for path in reports_dir.glob("*.json"))[-5:] if reports_dir.exists() else [],
    }


def validate(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    reports = []
    ok = True
    for path in dataset_mod.find_datasets(config.paths.datasets_dir):
        report = dataset_mod.validate_dataset(path).to_dict()
        reports.append(report)
        ok = ok and bool(report["ok"]) and (report["manifest_ok"] is not False)
    tokenizer_report = tokenizer_mod.tokenizer_summary(config.paths.tokenizer_dir)
    checkpoint_root = Path(config.paths.checkpoints_dir)
    checkpoint_checks: dict[str, Any] = {}
    if checkpoint_root.exists():
        for entry in sorted(checkpoint_root.iterdir()):
            if entry.is_dir():
                checkpoint_checks[entry.name] = checkpoints_mod.CheckpointManager(
                    entry, keep=config.training.checkpoint_keep
                ).verify()
                ok = ok and bool(checkpoint_checks[entry.name]["ok"])
    return {
        "ok": ok,
        "datasets": reports,
        "tokenizer": tokenizer_report,
        "checkpoints": checkpoint_checks,
    }


def prepare(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    target = _dataset_dir(config, args)
    spec = dataset_mod.load_spec(target)
    collected: list[dataset_mod.Record] = []
    for split in ("train", "valid", "test"):
        if split not in spec.splits:
            continue
        collected.extend(dataset_mod.load_records(dataset_mod.resolve_split_path(spec, split)))
    if not collected:
        raise DatasetError(f"Dataset '{spec.name}' has no records to prepare.")
    deduped, removed = dataset_mod.deduplicate(collected)
    splits = dataset_mod.split_records(
        deduped, valid_ratio=0.2, test_ratio=0.0, seed=spec.seed
    )
    written = {
        split: dataset_mod.write_jsonl(target / f"{split}.jsonl", records)
        for split, records in splits.items()
    }
    spec.splits = {split: f"{split}.jsonl" for split in written}
    (target / "dataset.json").write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = dataset_mod.build_manifest(target, seed=spec.seed)
    return {
        "ok": True,
        "dataset": spec.name,
        "version": spec.version,
        "records_in": len(collected),
        "duplicates_removed": removed,
        "records_out": len(deduped),
        "splits": written,
        "manifest": manifest,
    }


def tokenize(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    datasets = dataset_mod.find_datasets(config.paths.datasets_dir)
    if not datasets:
        raise DatasetError(
            "No datasets found; run `alphaai train prepare` after adding one.",
            remediation="See datasets/alphaai-sample for the expected layout.",
        )
    documents: list[str] = []
    for path in datasets:
        spec = dataset_mod.load_spec(path)
        split_path = dataset_mod.resolve_split_path(spec, "train")
        if split_path.exists():
            documents.extend(record.training_text for record in dataset_mod.load_records(split_path))
    tokenizer_settings = _tokenizer_config(config)
    started = time.perf_counter()
    artifact = tokenizer_mod.train_tokenizer(documents, tokenizer_settings)
    target = _tokenizer_dir(config, args)
    manifest = tokenizer_mod.save_tokenizer(artifact, target)
    report = tokenizer_mod.roundtrip_report(artifact, documents[:16])
    return {
        "ok": True,
        "tokenizer_dir": str(target),
        "documents": len(documents),
        "duration_s": round(time.perf_counter() - started, 3),
        "manifest": manifest,
        "roundtrip": report,
    }


def finetune(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    settings = _experiment_config(config, args)
    experiment = getattr(args, "experiment", None) or config.training.default_experiment
    model_settings = settings.get("model") or settings.get("reference_model") or {}
    training_settings = settings.get("training") or {}
    merged = {
        **training_settings,
        "name": settings.get("name", "alphaai-reference"),
        "model_name": model_settings.get("model_name", "AlphaAI-X-reference"),
        "dim": model_settings.get("dim", 96),
        "n_layers": model_settings.get("n_layers", 2),
        "n_heads": model_settings.get("n_heads", 4),
        "block_size": model_settings.get("block_size", 64),
        "max_steps": training_settings.get("max_steps", config.training.max_steps),
        "batch_size": training_settings.get("batch_size", config.training.device_batch_size),
        "gradient_accumulation_steps": training_settings.get(
            "gradient_accumulation_steps", config.training.gradient_accumulation_steps
        ),
        "learning_rate": training_settings.get("learning_rate", config.training.learning_rate),
        "seed": training_settings.get("seed", config.training.seed),
        "max_records": training_settings.get("max_records", 200),
    }
    model_config = pipeline_mod.ReferenceModelConfig.from_dict(merged)
    dataset_dir = _dataset_dir(config, args)
    tokenizer_dir = _tokenizer_dir(config, args)

    tracker = experiment_mod.ExperimentTracker.start(
        config.paths.training_dir, experiment, config={"alphaai": config.to_dict(), "run": merged}
    )
    checkpoints = checkpoints_mod.CheckpointManager(
        Path(config.paths.checkpoints_dir) / experiment, keep=config.training.checkpoint_keep
    )
    report = pipeline_mod.train_reference(
        config=model_config,
        dataset_dir=dataset_dir,
        tokenizer_dir=tokenizer_dir,
        checkpoints=checkpoints,
        tracker=tracker,
    )

    metadata = model_card_mod.build_metadata(
        model_name=model_config.model_name,
        model_owner="AlphaAI",
        checkpoint=report["checkpoints"][-1]["path"] if report["checkpoints"] else None,
        dataset_dir=dataset_dir,
        tokenizer_dir=tokenizer_dir,
        hparams=model_config.to_dict(),
        metrics={
            "parameters": report["parameters"],
            "first_loss": report["first_loss"],
            "final_loss": report["final_loss"],
            "steps": len(report["steps"]),
            "tokens": report["corpus"]["tokens"],
            "documents": report["corpus"]["documents"],
        },
        status="reference",
        notes=(
            "Reference AlphaAI training run: this checkpoint is produced by the AlphaAI training "
            "foundation to prove the pipeline end to end. It is NOT AlphaAI-X and must not be "
            "presented as a production AlphaAI model. Any part of the corpus that came from a "
            "third-party model keeps that model's license."
        ),
    )
    card_paths = model_card_mod.write_model_card(metadata, tracker.directory / "artifacts")
    candidate = model_card_mod.candidate_model_entry(metadata)
    (tracker.directory / "artifacts" / "candidate-model-entry.json").write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = tracker.finish(
        status="completed",
        summary={
            "final_loss": report["final_loss"],
            "steps": len(report["steps"]),
            "parameters": report["parameters"],
        },
    )
    return {
        "ok": True,
        "experiment": experiment,
        "run": tracker.to_dict(),
        "result": result,
        "model_card": card_paths,
        "training": report,
    }


def evaluate(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = _dataset_dir(config, args)
    tokenizer_dir = _tokenizer_dir(config, args)
    experiment = getattr(args, "experiment", None) or config.training.default_experiment
    checkpoint_root = Path(config.paths.checkpoints_dir) / experiment
    checkpoint = None
    if checkpoint_root.exists():
        latest = checkpoints_mod.CheckpointManager(
            checkpoint_root, keep=config.training.checkpoint_keep
        ).latest()
        checkpoint = str(latest.path) if latest else None

    prompts: list[str] = []
    prompts_path = Path(config.paths.evaluation_dir) / "prompts.jsonl"
    if prompts_path.exists():
        prompts = [
            record.training_text
            for record in dataset_mod.load_records(prompts_path)
            if record.training_text
        ]

    runtime = None
    engine_errors: list[str] = []
    try:
        from ..core.runtime import AlphaRuntime

        runtime = AlphaRuntime.create(config)
        if not runtime.registry.usable():
            engine_errors.append(
                "No usable engine: " + "; ".join(
                    f"{engine.id}: {engine.health().detail}" for engine in runtime.registry.engines()
                )
            )
    except AlphaAIError as exc:
        engine_errors.append(exc.message)

    context = {
        "dataset_dir": str(dataset_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "checkpoint": checkpoint,
        "runtime": runtime,
        "prompts": prompts[:5],
        "max_tokens": 32,
        "block_size": 64,
    }
    if not prompts:
        context["prompts"] = ["Explain what a mixture-of-experts layer does in one sentence."]
    suite = evaluation_mod.default_suite()
    report = suite.run(context)
    if engine_errors:
        report["engine_notes"] = engine_errors
    report["checkpoint"] = checkpoint
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = evaluation_mod.write_report(
        report, Path(config.paths.evaluation_dir) / "reports" / f"{timestamp}.json"
    )
    report["report_path"] = str(path)
    if runtime is not None:
        runtime.close()
    return report


ACTIONS = {
    "status": status,
    "validate": validate,
    "prepare": prepare,
    "tokenize": tokenize,
    "finetune": finetune,
    "evaluate": evaluate,
}


def cli(action: str, args: argparse.Namespace) -> int:
    """Entry point used by ``alphaai train <action>`` and the training scripts."""

    handler = ACTIONS.get(action)
    if handler is None:
        raise TrainingError(f"Unknown training action '{action}'. Known: {', '.join(sorted(ACTIONS))}")
    config = _config(args)
    payload = handler(config, args)
    json.dump(payload, __import__("sys").stdout, indent=2, ensure_ascii=False, default=str)
    print()
    return 0 if payload.get("ok", False) else 1

"""AlphaAI experiment tracking.

Every training or evaluation run gets its own directory:

    training/experiments/<experiment>/runs/<run_id>/
        run.json        ← immutable facts (started_at, config hash, git commit)
        config.json     ← the effective configuration
        metrics.jsonl   ← one JSON object per logged step (append-only)
        events.log      ← human-readable log
        artifacts/      ← checkpoints, reports, model cards

Nothing is uploaded anywhere: tracking is local files, hashed so two runs are
comparable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core.errors import TrainingError


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    commit = result.stdout.strip()
    return commit or None


@dataclass(slots=True)
class ExperimentTracker:
    """One tracked run."""

    directory: Path
    experiment: str
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    steps_logged: int = 0
    status: str = "running"

    @classmethod
    def start(
        cls,
        experiments_dir: str | Path,
        experiment: str,
        *,
        config: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> "ExperimentTracker":
        if not experiment or "/" in experiment:
            raise TrainingError(f"Invalid experiment name '{experiment}'.")
        run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        directory = Path(experiments_dir) / experiment / "runs" / run_id
        (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        tracker = cls(directory=directory, experiment=experiment, run_id=run_id, config=dict(config or {}))
        (directory / "config.json").write_text(
            json.dumps(tracker.config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "experiment": experiment,
                    "run_id": run_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "config_hash": config_hash(tracker.config),
                    "git_commit": git_commit(),
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "framework": "AlphaAI training foundation",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tracker.log_event(f"run {run_id} started (config hash {config_hash(tracker.config)})")
        return tracker

    # -- logging ----------------------------------------------------------
    def log_metrics(self, step: int | float, metrics: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "step": step,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **{key: _jsonable(value) for key, value in metrics.items()},
        }
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.steps_logged += 1
        return record

    def log_event(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        with (self.directory / "events.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")

    def artifact_path(self, name: str) -> Path:
        target = self.directory / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def finish(self, *, status: str = "completed", summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.status = status
        payload = {
            "status": status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.time() - self.started_at, 3),
            "steps_logged": self.steps_logged,
            "summary": _jsonable(dict(summary or {})),
        }
        (self.directory / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self.log_event(f"run finished: {status}")
        return payload

    def metrics(self) -> list[dict[str, Any]]:
        path = self.directory / "metrics.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "directory": str(self.directory),
            "config_hash": config_hash(self.config),
            "steps_logged": self.steps_logged,
            "status": self.status,
        }


def list_runs(experiments_dir: str | Path, experiment: str | None = None) -> list[dict[str, Any]]:
    """List tracked runs (newest first)."""

    root = Path(experiments_dir)
    pattern = f"{experiment}/runs/*/run.json" if experiment else "*/runs/*/run.json"
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result_path = path.parent / "result.json"
        payload["directory"] = str(path.parent)
        payload["result"] = (
            json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        )
        runs.append(payload)
    runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return runs


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = ["ExperimentTracker", "config_hash", "git_commit", "list_runs"]

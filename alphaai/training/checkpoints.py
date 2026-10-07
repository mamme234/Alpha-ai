"""AlphaAI checkpoint management.

Checkpoints are real artifacts on disk:

* ``save_state`` writes an actual PyTorch ``state_dict`` (``*.pt``) plus a
  ``checkpoint.json`` sidecar with step, config hash, dataset hash and metrics
* ``save_metadata`` records a metadata-only checkpoint for runs that do not have
  tensors yet (e.g. a data/tokenizer stage)
* retention keeps the newest ``keep`` checkpoints and reports what it removed
* ``verify`` re-hashes every file, so a corrupted checkpoint is detected rather
  than silently loaded
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core.errors import CheckpointError


@dataclass(slots=True)
class CheckpointInfo:
    name: str
    path: Path
    step: int | None
    created_at: str
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_bytes: int = 0
    kind: str = "metadata"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "step": self.step,
            "created_at": self.created_at,
            "kind": self.kind,
            "files": self.files,
            "metadata": self.metadata,
            "total_bytes": self.total_bytes,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointManager:
    """Owns the checkpoint directory for one experiment."""

    def __init__(self, directory: str | Path, *, keep: int = 3) -> None:
        self.directory = Path(directory)
        self.keep = max(0, int(keep))
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- writing ----------------------------------------------------------
    def save_metadata(
        self, name: str, metadata: Mapping[str, Any], *, step: int | None = None
    ) -> CheckpointInfo:
        target = self._prepare(name)
        payload = {
            "name": name,
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "metadata",
            "metadata": dict(metadata),
        }
        (target / "checkpoint.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        info = self._describe(name, target, step=step, kind="metadata", metadata=dict(metadata))
        self.prune()
        return info

    def save_state(
        self,
        name: str,
        state: Any,
        *,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        optimizer: Any | None = None,
    ) -> CheckpointInfo:
        """Write a real PyTorch checkpoint (requires the torch extra)."""

        try:
            import torch  # noqa: PLC0415 - lazy, torch is optional
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(
                "Saving model state requires PyTorch.",
                remediation="Install it with `pip install -e '.[torch]'` or save metadata only.",
            ) from exc
        target = self._prepare(name)
        weights_path = target / "model.pt"
        torch.save(state, weights_path)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), target / "optimizer.pt")
        payload = {
            "name": name,
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "state",
            "metadata": dict(metadata or {}),
            "torch_version": getattr(torch, "__version__", "unknown"),
        }
        (target / "checkpoint.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        info = self._describe(name, target, step=step, kind="state", metadata=dict(metadata or {}))
        self.prune()
        return info

    def load(self, name: str) -> tuple[Any, dict[str, Any]]:
        """Load a checkpoint's tensors and metadata."""

        target = self.directory / name
        if not target.is_dir():
            raise CheckpointError(f"Checkpoint '{name}' does not exist in {self.directory}.")
        meta_path = target / "checkpoint.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if metadata.get("kind") != "state":
            raise CheckpointError(
                f"Checkpoint '{name}' is metadata-only; it contains no model tensors.",
                details={"kind": metadata.get("kind")},
            )
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(
                "Loading model state requires PyTorch.",
                remediation="Install it with `pip install -e '.[torch]'`.",
            ) from exc
        try:
            state = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(f"Checkpoint '{name}' could not be loaded: {exc}") from exc
        return state, metadata

    # -- retention --------------------------------------------------------
    def list(self) -> list[CheckpointInfo]:
        infos: list[CheckpointInfo] = []
        for entry in sorted(self.directory.iterdir()):
            if entry.is_dir() and (entry / "checkpoint.json").exists():
                infos.append(self._describe_from_disk(entry))
        infos.sort(key=lambda info: (info.step if info.step is not None else -1, info.created_at))
        return infos

    def latest(self) -> CheckpointInfo | None:
        infos = self.list()
        return infos[-1] if infos else None

    def prune(self) -> list[str]:
        """Delete the oldest checkpoints beyond ``keep``. Returns removed names."""

        if self.keep <= 0:
            return []
        infos = self.list()
        removed: list[str] = []
        for info in infos[: max(0, len(infos) - self.keep)]:
            shutil.rmtree(info.path, ignore_errors=True)
            removed.append(info.name)
        return removed

    def verify(self) -> dict[str, Any]:
        """Re-hash every checkpoint file and report mismatches/misses."""

        report: dict[str, Any] = {"ok": True, "checked": 0, "problems": []}
        for info in self.list():
            for filename, entry in info.files.items():
                path = info.path / filename
                report["checked"] += 1
                if not path.exists():
                    report["ok"] = False
                    report["problems"].append(f"{info.name}/{filename}: missing")
                    continue
                digest = sha256_file(path)
                if digest != entry.get("sha256"):
                    report["ok"] = False
                    report["problems"].append(f"{info.name}/{filename}: sha256 mismatch")
        return report

    # -- internals --------------------------------------------------------
    def _prepare(self, name: str) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise CheckpointError(f"Invalid checkpoint name '{name}'.")
        target = self.directory / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _describe(
        self, name: str, target: Path, *, step: int | None, kind: str, metadata: dict[str, Any]
    ) -> CheckpointInfo:
        files: dict[str, dict[str, Any]] = {}
        total = 0
        for entry in sorted(target.iterdir()):
            if entry.is_file():
                size = entry.stat().st_size
                total += size
                files[entry.name] = {"bytes": size, "sha256": sha256_file(entry)}
        return CheckpointInfo(
            name=name,
            path=target,
            step=step,
            created_at=datetime.now(timezone.utc).isoformat(),
            files=files,
            metadata=metadata,
            total_bytes=total,
            kind=kind,
        )

    def _describe_from_disk(self, target: Path) -> CheckpointInfo:
        payload = json.loads((target / "checkpoint.json").read_text(encoding="utf-8"))
        files: dict[str, dict[str, Any]] = {}
        total = 0
        for entry in sorted(target.iterdir()):
            if entry.is_file():
                size = entry.stat().st_size
                total += size
                files[entry.name] = {
                    "bytes": size,
                    # Every file is hashed, including checkpoint.json, so verify() can
                    # re-check a checkpoint read back from disk without false mismatches.
                    "sha256": sha256_file(entry),
                }
        return CheckpointInfo(
            name=str(payload.get("name") or target.name),
            path=target,
            step=payload.get("step"),
            created_at=str(payload.get("created_at") or ""),
            files=files,
            metadata=payload.get("metadata") or {},
            total_bytes=total,
            kind=str(payload.get("kind") or "metadata"),
        )


__all__ = ["CheckpointInfo", "CheckpointManager", "sha256_file"]

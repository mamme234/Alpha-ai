"""AlphaAI dataset management.

Real dataset handling for the AlphaAI training foundation:

* load JSONL / JSON / CSV / plain text into normalised records
* validate records against a declared schema (and report *every* problem)
* deduplicate, split deterministically, and write new split files
* version datasets: a manifest with sizes, sha256 hashes and record counts

Nothing here invents data: a dataset that fails validation is reported as invalid
along with the offending record indexes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..core.errors import DatasetError, DatasetValidationError

SUPPORTED_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".txt"}
SPLITS = ("train", "valid", "test")


@dataclass(slots=True)
class Record:
    """One training record: an instruction/response pair or plain text."""

    text: str = ""
    instruction: str = ""
    response: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        if self.messages:
            return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in self.messages)
        if self.instruction:
            return self.instruction
        return self.text

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.messages:
            payload["messages"] = self.messages
        if self.instruction:
            payload["instruction"] = self.instruction
            payload["response"] = self.response
        if self.text and not self.instruction:
            payload["text"] = self.text
        if self.meta:
            payload["meta"] = self.meta
        return payload

    @property
    def training_text(self) -> str:
        if self.messages:
            return "\n".join(str(m.get("content", "")) for m in self.messages)
        if self.instruction:
            return f"{self.instruction}\n{self.response}".strip()
        return self.text


@dataclass(slots=True)
class DatasetSpec:
    """Declared metadata for a dataset (``datasets/<name>/dataset.json``)."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    license: str = "unspecified"
    source: str = "unspecified"
    language: str = "en"
    format: str = "jsonl"
    splits: dict[str, str] = field(default_factory=dict)
    created_by: str = "AlphaAI"
    seed: int = 1337
    max_record_chars: int = 200_000
    min_record_chars: int = 1
    schema: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: Path | None = None) -> "DatasetSpec":
        if not data.get("name"):
            raise DatasetValidationError("Dataset metadata needs a 'name'.")
        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "0.0.0")),
            description=str(data.get("description", "")),
            license=str(data.get("license", "unspecified")),
            source=str(data.get("source", "unspecified")),
            language=str(data.get("language", "en")),
            format=str(data.get("format", "jsonl")),
            splits={str(k): str(v) for k, v in (data.get("splits") or {}).items()},
            created_by=str(data.get("created_by", "AlphaAI")),
            seed=int(data.get("seed", 1337)),
            max_record_chars=int(data.get("max_record_chars", 200_000)),
            min_record_chars=int(data.get("min_record_chars", 1)),
            schema=data.get("schema") or {},
            notes=str(data.get("notes", "")),
            path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "license": self.license,
            "source": self.source,
            "language": self.language,
            "format": self.format,
            "splits": self.splits,
            "created_by": self.created_by,
            "seed": self.seed,
            "max_record_chars": self.max_record_chars,
            "min_record_chars": self.min_record_chars,
            "schema": self.schema,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating a dataset directory."""

    dataset: str
    ok: bool
    records: int = 0
    problems: list[dict[str, Any]] = field(default_factory=list)
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifest_ok: bool | None = None
    manifest_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "ok": self.ok,
            "records": self.records,
            "problems": self.problems,
            "splits": self.splits,
            "manifest_ok": self.manifest_ok,
            "manifest_problems": self.manifest_problems,
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_spec(dataset_dir: str | Path) -> DatasetSpec:
    directory = Path(dataset_dir)
    meta_path = directory / "dataset.json"
    if not meta_path.exists():
        raise DatasetError(
            f"Dataset metadata not found: {meta_path}",
            remediation="Create datasets/<name>/dataset.json (see datasets/alphaai-sample/dataset.json).",
        )
    return DatasetSpec.from_dict(json.loads(meta_path.read_text(encoding="utf-8")), path=meta_path)


def resolve_split_path(spec: DatasetSpec, split: str) -> Path:
    directory = spec.path.parent if spec.path else Path(".")
    declared = spec.splits.get(split)
    if declared:
        candidate = Path(declared)
        if not candidate.is_absolute():
            candidate = directory / candidate
        return candidate
    return directory / f"{split}.{spec.format}"


def load_records(path: str | Path) -> list[Record]:
    """Load a split file into normalised records."""

    target = Path(path)
    if not target.exists():
        raise DatasetError(f"Split file not found: {target}")
    suffix = target.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DatasetValidationError(
            f"Unsupported dataset format '{suffix}' (supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )
    if suffix == ".jsonl":
        return [_record_from_obj(json.loads(line), index) for index, line in _jsonl_lines(target)]
    if suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records") or payload.get("data") or []
        if not isinstance(payload, list):
            raise DatasetValidationError(f"{target.name}: JSON dataset must be a list of records.")
        return [_record_from_obj(item, index) for index, item in enumerate(payload)]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with target.open("r", encoding="utf-8", newline="") as handle:
            return [_record_from_obj(dict(row), index) for index, row in enumerate(csv.DictReader(handle, delimiter=delimiter))]
    text = target.read_text(encoding="utf-8")
    return [Record(text=block.strip()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield line_number, stripped


def _record_from_obj(obj: Any, index: int) -> Record:
    if isinstance(obj, str):
        return Record(text=obj)
    if not isinstance(obj, dict):
        raise DatasetValidationError(f"Record {index} is {type(obj).__name__}, expected an object.")
    messages = obj.get("messages")
    meta = {key: value for key, value in obj.items() if key not in {"text", "instruction", "response", "messages"}}
    return Record(
        text=str(obj.get("text") or ""),
        instruction=str(obj.get("instruction") or obj.get("prompt") or ""),
        response=str(obj.get("response") or obj.get("completion") or obj.get("output") or ""),
        messages=[
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in messages or []
            if isinstance(item, dict)
        ],
        meta=meta,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_dataset(dataset_dir: str | Path, *, check_manifest: bool = True) -> ValidationReport:
    """Validate every declared split and report all real problems."""

    directory = Path(dataset_dir)
    spec = load_spec(directory)
    report = ValidationReport(dataset=spec.name, ok=True)
    seen: set[str] = set()

    for split in sorted(spec.splits or {s: f"{s}.{spec.format}" for s in ("train", "valid")}):
        path = resolve_split_path(spec, split)
        try:
            records = load_records(path)
        except (DatasetError, DatasetValidationError) as exc:
            report.ok = False
            report.problems.append({"split": split, "index": None, "problem": exc.message})
            continue
        digest = sha256_file(path)
        report.splits[split] = {
            "path": path.name,
            "records": len(records),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        report.records += len(records)
        for index, record in enumerate(records):
            text = record.training_text
            if not text.strip():
                report.ok = False
                report.problems.append({"split": split, "index": index, "problem": "empty record"})
                continue
            if len(text) < spec.min_record_chars:
                report.ok = False
                report.problems.append(
                    {"split": split, "index": index, "problem": f"record shorter than {spec.min_record_chars} chars"}
                )
            if len(text) > spec.max_record_chars:
                report.ok = False
                report.problems.append(
                    {"split": split, "index": index, "problem": f"record longer than {spec.max_record_chars} chars"}
                )
            digest_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest_key in seen:
                report.problems.append({"split": split, "index": index, "problem": "duplicate record (warning)"})
            seen.add(digest_key)
        if spec.format == "jsonl" and len(records) == 0:
            report.ok = False
            report.problems.append({"split": split, "index": None, "problem": "split has no records"})

    if check_manifest:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            report.manifest_ok = False
            report.manifest_problems.append("manifest.json is missing (run `alphaai train prepare`).")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report.manifest_ok = True
            for split, entry in (manifest.get("splits") or {}).items():
                current = report.splits.get(split)
                if current is None:
                    report.manifest_ok = False
                    report.manifest_problems.append(f"manifest lists unknown split '{split}'")
                    continue
                if current["sha256"] != entry.get("sha256"):
                    report.manifest_ok = False
                    report.manifest_problems.append(
                        f"split '{split}' changed: expected {str(entry.get('sha256'))[:12]}, "
                        f"found {current['sha256'][:12]}"
                    )
    return report


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def deduplicate(records: Sequence[Record]) -> tuple[list[Record], int]:
    seen: set[str] = set()
    kept: list[Record] = []
    for record in records:
        key = hashlib.sha256(record.training_text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept, len(records) - len(kept)


def split_records(
    records: Sequence[Record], *, valid_ratio: float = 0.1, test_ratio: float = 0.0, seed: int = 1337
) -> dict[str, list[Record]]:
    """Deterministic split: same seed, same order, same result."""

    if not 0 <= valid_ratio < 1 or not 0 <= test_ratio < 1 or valid_ratio + test_ratio >= 1:
        raise DatasetValidationError("Split ratios must be >= 0 and sum to less than 1.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    valid_count = int(total * valid_ratio)
    test_count = int(total * test_ratio)
    train = shuffled[: total - valid_count - test_count]
    valid = shuffled[total - valid_count - test_count : total - test_count]
    test = shuffled[total - test_count :] if test_count else []
    result = {"train": train, "valid": valid}
    if test_count:
        result["test"] = test
    return result


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(dataset_dir: str | Path, *, seed: int | None = None) -> dict[str, Any]:
    """(Re)write ``manifest.json`` with real hashes and record counts."""

    directory = Path(dataset_dir)
    spec = load_spec(directory)
    splits: dict[str, Any] = {}
    for split in sorted(spec.splits):
        path = resolve_split_path(spec, split)
        records = load_records(path)
        texts = [record.training_text for record in records]
        splits[split] = {
            "path": path.name,
            "records": len(records),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "characters": sum(len(text) for text in texts),
            "characters_mean": round(sum(len(text) for text in texts) / max(len(texts), 1), 1),
        }
    manifest = {
        "dataset": spec.name,
        "version": spec.version,
        "license": spec.license,
        "source": spec.source,
        "language": spec.language,
        "seed": spec.seed if seed is None else seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "alphaai.training.dataset",
        "splits": splits,
        "records_total": sum(entry["records"] for entry in splits.values()),
        "characters_total": sum(entry["characters"] for entry in splits.values()),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def dataset_summary(dataset_dir: str | Path) -> dict[str, Any]:
    """Human/CLI-friendly summary of a dataset directory."""

    directory = Path(dataset_dir)
    spec = load_spec(directory)
    report = validate_dataset(directory)
    return {
        "name": spec.name,
        "version": spec.version,
        "license": spec.license,
        "source": spec.source,
        "language": spec.language,
        "directory": str(directory),
        "splits": report.splits,
        "records": report.records,
        "valid": report.ok,
        "problems": report.problems[:20],
        "problem_count": len(report.problems),
        "manifest_ok": report.manifest_ok,
        "manifest_problems": report.manifest_problems,
    }


def find_datasets(datasets_dir: str | Path) -> list[Path]:
    """List dataset directories (anything with a ``dataset.json``)."""

    root = Path(datasets_dir)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob("*/dataset.json"))


__all__ = [
    "DatasetSpec",
    "Record",
    "SPLITS",
    "ValidationReport",
    "build_manifest",
    "dataset_summary",
    "deduplicate",
    "find_datasets",
    "load_records",
    "load_spec",
    "resolve_split_path",
    "sha256_file",
    "split_records",
    "validate_dataset",
    "write_jsonl",
]

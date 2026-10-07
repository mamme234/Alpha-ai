"""AlphaAI evaluation framework.

Every evaluator returns *measured* numbers. When an evaluator cannot run — no
weights, no torch, no usable engine — it returns ``ok = False`` with the exact
reason instead of a placeholder score, and the suite reports it as SKIPPED.

Built-in evaluators
-------------------
* ``dataset``      – record counts, sizes, duplicate rate, split hashes
* ``tokenizer``    – exact round-trip rate, characters per token, vocab size
* ``language_model`` – real generation latency/throughput and token usage through
  a live AlphaAI engine (skipped when no engine can run)
* ``perplexity``   – real cross-entropy / perplexity of a torch checkpoint on a
  held-out split (skipped without torch + a checkpoint)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..core.errors import EvaluationError
from . import dataset as dataset_mod
from . import tokenizer as tokenizer_mod

Evaluator = Callable[[dict[str, Any]], "EvaluationResult"]


@dataclass(slots=True)
class EvaluationResult:
    name: str
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    skipped_reason: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "metrics": self.metrics,
            "detail": self.detail,
            "skipped_reason": self.skipped_reason,
            "duration_ms": round(self.duration_ms, 3),
        }


class EvaluationSuite:
    """Runs evaluators over one context and aggregates the report."""

    def __init__(self, name: str = "alphaai-evaluation") -> None:
        self.name = name
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, name: str, evaluator: Evaluator) -> None:
        self._evaluators[name] = evaluator

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        results: list[EvaluationResult] = []
        for name, evaluator in self._evaluators.items():
            started = time.perf_counter()
            try:
                result = evaluator(context)
            except EvaluationError as exc:
                result = EvaluationResult(name=name, ok=False, detail=exc.message, skipped_reason=exc.message)
            except Exception as exc:  # noqa: BLE001 - an evaluator must not crash the suite
                result = EvaluationResult(
                    name=name, ok=False, detail=f"evaluator crashed: {type(exc).__name__}: {exc}"
                )
            result.duration_ms = (time.perf_counter() - started) * 1000
            results.append(result)
        passed = [item for item in results if item.ok]
        skipped = [item for item in results if item.skipped_reason]
        return {
            "suite": self.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": len(passed) == len(results) - len(skipped),
            "evaluated": len(passed),
            "skipped": len(skipped),
            "results": [item.to_dict() for item in results],
        }


# ---------------------------------------------------------------------------
# evaluators
# ---------------------------------------------------------------------------
def evaluate_dataset(context: dict[str, Any]) -> EvaluationResult:
    directory = context.get("dataset_dir")
    if not directory:
        return EvaluationResult("dataset", False, skipped_reason="no dataset_dir supplied")
    summary = dataset_mod.dataset_summary(directory)
    return EvaluationResult(
        name="dataset",
        ok=bool(summary["valid"]),
        metrics={
            "records": summary["records"],
            "splits": {name: entry["records"] for name, entry in summary["splits"].items()},
            "problem_count": summary["problem_count"],
            "manifest_ok": summary["manifest_ok"],
            "version": summary["version"],
            "license": summary["license"],
        },
        detail=f"{summary['records']} records across {len(summary['splits'])} split(s)",
    )


def evaluate_tokenizer(context: dict[str, Any]) -> EvaluationResult:
    directory = context.get("tokenizer_dir")
    if not directory:
        return EvaluationResult("tokenizer", False, skipped_reason="no tokenizer_dir supplied")
    try:
        artifact = tokenizer_mod.load_tokenizer(directory)
    except Exception as exc:  # noqa: BLE001 - reported as skipped
        return EvaluationResult("tokenizer", False, skipped_reason=f"tokenizer unavailable: {exc}")
    samples = list(context.get("samples") or [])
    if not samples:
        return EvaluationResult(
            "tokenizer",
            True,
            metrics={"vocab_size": artifact.vocab_size, "merges": len(artifact.merges)},
            detail="no samples supplied; reported structural metrics only",
        )
    report = tokenizer_mod.roundtrip_report(artifact, samples)
    return EvaluationResult(
        name="tokenizer",
        ok=bool(report.get("ok")) and report.get("roundtrip_rate") == 1.0,
        metrics=report,
        detail=f"round-trip {report.get('roundtrip_exact')}/{report.get('documents')} documents",
    )


def evaluate_language_model(context: dict[str, Any]) -> EvaluationResult:
    """Measure a live AlphaAI engine on the eval prompts (real inference)."""

    runtime = context.get("runtime")
    prompts: Iterable[str] = context.get("prompts") or []
    if runtime is None:
        return EvaluationResult("language_model", False, skipped_reason="no AlphaAI runtime supplied")
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        return EvaluationResult("language_model", False, skipped_reason="no evaluation prompts supplied")

    latencies: list[float] = []
    tokens = 0
    engine_id = None
    failures: list[str] = []
    for prompt in prompts:
        try:
            started = time.perf_counter()
            result = runtime.generate(prompt, max_tokens=context.get("max_tokens", 64))
        except Exception as exc:  # noqa: BLE001 - unavailable engines are expected
            failures.append(f"{type(exc).__name__}: {exc}")
            continue
        latencies.append(time.perf_counter() - started)
        tokens += result.usage.total_tokens
        engine_id = result.engine_id
    if not latencies:
        return EvaluationResult(
            "language_model",
            False,
            skipped_reason="no AlphaAI engine could run: " + ("; ".join(failures[:2]) or "unknown"),
        )
    return EvaluationResult(
        name="language_model",
        ok=True,
        metrics={
            "engine_id": engine_id,
            "prompts": len(latencies),
            "tokens": tokens,
            "latency_ms_mean": round(sum(latencies) / len(latencies) * 1000, 2),
            "latency_ms_max": round(max(latencies) * 1000, 2),
            "tokens_per_second": round(tokens / max(sum(latencies), 1e-9), 2),
            "failures": len(failures),
        },
        detail=f"{len(latencies)} prompt(s) served by {engine_id}",
    )


def evaluate_perplexity(context: dict[str, Any]) -> EvaluationResult:
    """Real perplexity of a torch checkpoint over a held-out split."""

    checkpoint = context.get("checkpoint")
    tokenizer_dir = context.get("tokenizer_dir")
    dataset_dir = context.get("dataset_dir")
    if not checkpoint:
        return EvaluationResult("perplexity", False, skipped_reason="no checkpoint supplied")
    if not (tokenizer_dir and dataset_dir):
        return EvaluationResult("perplexity", False, skipped_reason="tokenizer_dir and dataset_dir are required")
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult("perplexity", False, skipped_reason=f"torch unavailable: {exc}")
    try:
        from .pipeline import load_reference_model  # noqa: PLC0415 - avoids a cycle

        artifact = tokenizer_mod.load_tokenizer(tokenizer_dir)
        spec = dataset_mod.load_spec(dataset_dir)
        records = dataset_mod.load_records(dataset_mod.resolve_split_path(spec, "valid"))
        model, _ = load_reference_model(checkpoint, tokenizer_dir)
    except Exception as exc:  # noqa: BLE001 - reported as skipped
        return EvaluationResult("perplexity", False, skipped_reason=f"perplexity could not run: {exc}")

    block = int(context.get("block_size", 64))
    total_loss = 0.0
    windows = 0
    with torch.inference_mode():
        for record in records:
            ids = artifact.encode(record.training_text)
            if len(ids) < 2:
                continue
            ids = ids[:block]
            tensor = torch.tensor([ids[:-1]], dtype=torch.long)
            targets = torch.tensor([ids[1:]], dtype=torch.long)
            logits, _ = model(tensor)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            total_loss += float(loss)
            windows += 1
    if windows == 0:
        return EvaluationResult("perplexity", False, skipped_reason="no usable validation windows")
    mean_loss = total_loss / windows
    return EvaluationResult(
        name="perplexity",
        ok=True,
        metrics={
            "checkpoint": Path(checkpoint).name,
            "windows": windows,
            "cross_entropy": round(mean_loss, 4),
            "perplexity": round(math.exp(min(mean_loss, 50)), 4),
        },
        detail=f"cross-entropy {mean_loss:.3f} over {windows} window(s)",
    )


def small_sample(text: str, limit: int = 600) -> str:
    """Take a deterministic sample of a long document (used by the tokenizer evaluator)."""

    if len(text) <= limit:
        return text
    step = len(text) / limit
    return "".join(text[int(index * step)] for index in range(limit))


def default_suite() -> EvaluationSuite:
    suite = EvaluationSuite()
    suite.register("dataset", evaluate_dataset)
    suite.register("tokenizer", evaluate_tokenizer)
    suite.register("language_model", evaluate_language_model)
    suite.register("perplexity", evaluate_perplexity)
    return suite


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "EvaluationResult",
    "EvaluationSuite",
    "default_suite",
    "evaluate_dataset",
    "evaluate_language_model",
    "evaluate_perplexity",
    "evaluate_tokenizer",
    "small_sample",
    "write_report",
]

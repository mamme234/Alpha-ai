"""AlphaAI reference training pipeline.

This is a **real** training loop, sized so it runs on a CPU-only machine: a small
decoder-only Transformer (the same architecture family AlphaAI-X will use, just
much smaller) is trained with AdamW on the tokenized dataset AlphaAI prepared, and
the resulting checkpoint contains real tensors with measured loss values.

It exists for three honest reasons:

1. the training foundation must be executable end-to-end, not a stub;
2. every artifact (tokenizer, dataset, checkpoint, metrics, model card) is
   produced by real code and can be verified;
3. it gives `alphaai train finetune` something true to report on any machine —
   while making clear that it is a *reference* run, not AlphaAI-X.

A production AlphaAI-X run swaps ``ReferenceModelConfig`` for the full
architecture and the same pipeline (data → tokens → steps → eval → checkpoint →
model card) applies.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.errors import CheckpointError, TrainingError
from . import dataset as dataset_mod
from . import tokenizer as tokenizer_mod
from .checkpoints import CheckpointManager
from .experiment import ExperimentTracker


@dataclass(slots=True)
class ReferenceModelConfig:
    """Architecture + optimisation hyperparameters for one training run."""

    name: str = "alphaai-reference"
    model_name: str = "AlphaAI-X-reference"
    vocab_size: int = 1024
    dim: int = 96
    n_layers: int = 2
    n_heads: int = 4
    block_size: int = 64
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 30
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    eval_every: int = 10
    checkpoint_every: int = 10
    warmup_steps: int = 2
    seed: int = 1337
    max_records: int = 200

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceModelConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {key: value for key, value in data.items() if key in known and value is not None}
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_reference_model(config: ReferenceModelConfig):
    """Construct the reference decoder-only Transformer (requires torch)."""

    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(
            "Training requires PyTorch.",
            remediation="Install it with `pip install -e '.[torch]'`.",
        ) from exc

    class Block(nn.Module):
        def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
            normed = self.norm1(x)
            attended, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
            x = x + self.dropout(attended)
            return x + self.dropout(self.mlp(self.norm2(x)))

    class ReferenceModel(nn.Module):
        def __init__(self, config: ReferenceModelConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
            self.position_embedding = nn.Embedding(config.block_size, config.dim)
            self.blocks = nn.ModuleList(
                [Block(config.dim, config.n_heads, config.dropout) for _ in range(config.n_layers)]
            )
            self.norm = nn.LayerNorm(config.dim)
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        def forward(self, tokens):
            batch, length = tokens.shape
            if length > self.config.block_size:
                tokens = tokens[:, -self.config.block_size :]
                length = self.config.block_size
            positions = torch.arange(length, device=tokens.device)
            x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
            mask = torch.triu(
                torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1
            )
            for block in self.blocks:
                x = block(x, mask)
            logits = self.head(self.norm(x))
            return logits, None

        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

    return ReferenceModel(config)


def load_reference_model(checkpoint_dir: str | Path, tokenizer_dir: str | Path):
    """Load a reference checkpoint back into a model (real weights)."""

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError("Loading a checkpoint requires PyTorch.") from exc
    directory = Path(checkpoint_dir)
    options_path = directory / "options.json"
    if not options_path.exists():
        raise CheckpointError(
            f"Checkpoint '{directory.name}' has no options.json describing its architecture.",
            remediation="Use a checkpoint written by the AlphaAI training pipeline.",
        )
    # The checkpoint's own options.json is authoritative for the architecture
    # (including vocab_size); the tokenizer only has to exist and be loadable.
    config = ReferenceModelConfig.from_dict(json.loads(options_path.read_text(encoding="utf-8")))
    tokenizer_mod.load_tokenizer(tokenizer_dir)
    model = build_reference_model(config)
    state, _ = CheckpointManager(directory.parent).load(directory.name)
    model.load_state_dict(state)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TokenizedCorpus:
    """Token streams for training and validation."""

    train: list[list[int]] = field(default_factory=list)
    valid: list[list[int]] = field(default_factory=list)
    documents: int = 0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "tokens": self.tokens,
            "train_documents": len(self.train),
            "valid_documents": len(self.valid),
        }


def tokenize_dataset(
    artifact: tokenizer_mod.TokenizerArtifact, dataset_dir: str | Path, *, max_records: int = 200
) -> TokenizedCorpus:
    """Tokenize the dataset splits with the AlphaAI tokenizer (real encoding)."""

    spec = dataset_mod.load_spec(dataset_dir)
    corpus = TokenizedCorpus()
    for split in ("train", "valid", "test"):
        if split not in spec.splits:
            continue
        path = dataset_mod.resolve_split_path(spec, split)
        records = dataset_mod.load_records(path)
        target = corpus.train if split == "train" else corpus.valid
        for record in records[:max_records]:
            ids = artifact.encode(record.training_text)
            if len(ids) < 4:
                continue
            target.append(ids)
            corpus.documents += 1
            corpus.tokens += len(ids)
    if not corpus.train:
        raise TrainingError(
            "No training documents were tokenized.",
            remediation="Check the dataset split files and run `alphaai train validate`.",
        )
    return corpus


def iter_batches(
    corpus: Sequence[Sequence[int]], *, batch_size: int, block_size: int, seed: int = 0
) -> Iterable[tuple[list[list[int]], list[list[int]]]]:
    """Deterministic windowed batches for next-token prediction."""

    windows: list[list[int]] = []
    for ids in corpus:
        for start in range(0, max(1, len(ids) - 1), block_size):
            window = list(ids[start : start + block_size + 1])
            if len(window) >= 3:
                windows.append(window)
    if not windows:
        raise TrainingError(
            "Tokenized documents are shorter than the model block size.",
            remediation="Add longer records or reduce the configured block_size.",
        )
    step = 0
    while True:
        batch: list[list[int]] = []
        for offset in range(batch_size):
            index = (seed + step * batch_size + offset) % len(windows)
            batch.append(windows[index])
        length = min(len(window) for window in batch)
        inputs = [window[: length - 1] for window in batch]
        labels = [window[1:length] for window in batch]
        yield inputs, labels
        step += 1


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_reference(
    *,
    config: ReferenceModelConfig,
    dataset_dir: str | Path,
    tokenizer_dir: str | Path,
    checkpoints: CheckpointManager,
    tracker: ExperimentTracker | None = None,
) -> dict[str, Any]:
    """Run a real reference training loop and return its measured report."""

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(
            "Training requires PyTorch.",
            remediation="Install it with `pip install -e '.[torch]'`.",
        ) from exc

    artifact = tokenizer_mod.load_tokenizer(tokenizer_dir)
    config.vocab_size = artifact.vocab_size
    corpus = tokenize_dataset(artifact, dataset_dir, max_records=config.max_records)
    torch.manual_seed(config.seed)
    model = build_reference_model(config)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_iterator = iter_batches(
        corpus.train, batch_size=config.batch_size, block_size=config.block_size, seed=config.seed
    )
    validation_iterator = iter_batches(
        corpus.valid or corpus.train,
        batch_size=config.batch_size,
        block_size=config.block_size,
        seed=config.seed + 1,
    )

    report: dict[str, Any] = {
        "model_name": config.model_name,
        "parameters": model.parameter_count(),
        "corpus": corpus.to_dict(),
        "config": config.to_dict(),
        "vocab_size": config.vocab_size,
        "steps": [],
        "checkpoints": [],
    }
    started = time.perf_counter()
    history: list[dict[str, Any]] = []

    for step in range(1, config.max_steps + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(max(1, config.gradient_accumulation_steps)):
            inputs, labels = next(batch_iterator)
            tensor_inputs = torch.tensor(inputs, dtype=torch.long)
            tensor_labels = torch.tensor(labels, dtype=torch.long)
            logits, _ = model(tensor_inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tensor_labels.reshape(-1)
            )
            (loss / max(1, config.gradient_accumulation_steps)).backward()
            step_loss += float(loss.detach())
        lr_scale = min(1.0, step / max(1, config.warmup_steps))
        for group in optimiser.param_groups:
            group["lr"] = config.learning_rate * lr_scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        mean_loss = step_loss / max(1, config.gradient_accumulation_steps)
        record = {"step": step, "loss": round(mean_loss, 4), "lr": round(config.learning_rate * lr_scale, 8)}
        history.append(record)
        if tracker is not None:
            tracker.log_metrics(step, record)

        if step % max(1, config.eval_every) == 0 or step == config.max_steps:
            model.eval()
            with torch.inference_mode():
                inputs, labels = next(validation_iterator)
                logits, _ = model(torch.tensor(inputs, dtype=torch.long))
                validation_loss = float(
                    torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        torch.tensor(labels, dtype=torch.long).reshape(-1),
                    )
                )
            record["validation_loss"] = round(validation_loss, 4)
            record["perplexity"] = round(math.exp(min(validation_loss, 50)), 4)
            if tracker is not None:
                tracker.log_metrics(step, {"validation_loss": round(validation_loss, 4)})

        if step % max(1, config.checkpoint_every) == 0 or step == config.max_steps:
            info = checkpoints.save_state(
                f"step-{step:05d}",
                model.state_dict(),
                step=step,
                metadata={
                    "model_name": config.model_name,
                    "loss": record["loss"],
                    "dataset": Path(dataset_dir).name,
                    "tokenizer": Path(tokenizer_dir).name,
                    "vocab_size": config.vocab_size,
                    "note": "AlphaAI reference training run (not AlphaAI-X)",
                },
            )
            # The architecture description travels with the tensors so the
            # checkpoint can be loaded by `load_reference_model`.
            (info.path / "options.json").write_text(
                json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            report["checkpoints"].append(info.to_dict())

    report["steps"] = history
    report["duration_s"] = round(time.perf_counter() - started, 3)
    report["final_loss"] = history[-1]["loss"] if history else None
    report["first_loss"] = history[0]["loss"] if history else None
    report["checkpoint_dir"] = str(checkpoints.directory)
    report["tokenizer"] = {
        "name": artifact.config.name,
        "vocab_size": artifact.vocab_size,
        "merges": len(artifact.merges),
    }
    return report


__all__ = [
    "ReferenceModelConfig",
    "TokenizedCorpus",
    "build_reference_model",
    "iter_batches",
    "load_reference_model",
    "tokenize_dataset",
    "train_reference",
]

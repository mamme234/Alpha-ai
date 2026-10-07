"""AlphaAI tokenizer foundation.

A **real** byte-level BPE tokenizer, implemented from scratch in pure Python so
AlphaAI can train its own tokenizer for AlphaAI-owned weights without borrowing
another model's vocabulary:

* training counts byte pairs over a corpus and merges the most frequent ones
  (incremental BPE with a lazy max-heap — no rescans of the whole corpus)
* encoding applies the learned merges by rank, byte-exactly
* decoding is a lossless byte concatenation, so any UTF-8 input round-trips
* artifacts are plain JSON (``tokenizer.json``) with a manifest and hashes

The reference trainer is intentionally conservative (small corpus cap, modest
vocab) so it runs on a laptop. Raise ``max_chars``/``vocab_size`` for real
pretraining corpora; the algorithm is the same.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.errors import DatasetValidationError

#: Byte values occupy ids 0..255 in every AlphaAI byte-level tokenizer.
BYTE_VOCAB_SIZE = 256

DEFAULT_SPECIAL_TOKENS: dict[str, str] = {
    "<|endoftext|>": "document separator / padding",
    "<|system|>": "system message",
    "<|user|>": "user message",
    "<|assistant|>": "assistant message",
    "<|tool|>": "tool result",
}


@dataclass(slots=True)
class TokenizerConfig:
    """Declared tokenizer hyperparameters (``configs/training/*.toml``)."""

    name: str = "alphaai-bpe"
    version: str = "1.0.0"
    vocab_size: int = 1024
    min_frequency: int = 2
    max_chars: int = 50_000
    special_tokens: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SPECIAL_TOKENS))

    @property
    def target_merges(self) -> int:
        return max(0, self.vocab_size - BYTE_VOCAB_SIZE - len(self.special_tokens))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerConfig":
        specials = data.get("special_tokens")
        if isinstance(specials, list):
            specials = {str(token): "" for token in specials}
        return cls(
            name=str(data.get("name", "alphaai-bpe")),
            version=str(data.get("version", "1.0.0")),
            vocab_size=int(data.get("vocab_size", 1024)),
            min_frequency=int(data.get("min_frequency", 2)),
            max_chars=int(data.get("max_chars", 50_000)),
            special_tokens={str(k): str(v) for k, v in (specials or DEFAULT_SPECIAL_TOKENS).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "max_chars": self.max_chars,
            "special_tokens": dict(self.special_tokens),
        }


@dataclass(slots=True)
class TokenizerArtifact:
    """A trained tokenizer plus the facts about its training run."""

    config: TokenizerConfig
    merges: list[tuple[int, int]]
    special_ids: dict[str, int]
    corpus_chars: int
    corpus_documents: int
    vocab: dict[int, bytes]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.special_ids.get("<|endoftext|>", 0))
        ids.extend(self._encode_bytes(text.encode("utf-8")))
        return ids

    def _encode_bytes(self, data: bytes) -> list[int]:
        ids = list(data)
        if not self.merges:
            return ids
        ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        while len(ids) > 1:
            best_rank = None
            best_index = -1
            for index in range(len(ids) - 1):
                rank = ranks.get((ids[index], ids[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_rank is None:
                break
            merged_id = BYTE_VOCAB_SIZE + best_rank
            ids[best_index : best_index + 2] = [merged_id]
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        special_ids = set(self.special_ids.values())
        chunks: list[bytes] = []
        for token_id in ids:
            if token_id in special_ids:
                if not skip_special_tokens:
                    name = next((n for n, i in self.special_ids.items() if i == token_id), None)
                    chunks.append(f"<|{name}|>".encode("utf-8"))
                continue
            piece = self.vocab.get(int(token_id))
            if piece is not None:
                chunks.append(piece)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes | None:
        return self.vocab.get(int(token_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "type": "byte_level_bpe",
                "byte_vocab_size": BYTE_VOCAB_SIZE,
                "vocab_size": self.vocab_size,
                "merges": [[left, right] for left, right in self.merges],
            },
            "special_tokens": dict(self.special_ids),
            "config": self.config.to_dict(),
            "training": {
                "corpus_documents": self.corpus_documents,
                "corpus_chars": self.corpus_chars,
                "merges": len(self.merges),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "implementation": "alphaai.training.tokenizer (reference byte-level BPE)",
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerArtifact":
        model = data.get("model") or {}
        merges = [tuple(int(part) for part in pair) for pair in model.get("merges", [])]
        special_ids = {str(k): int(v) for k, v in (data.get("special_tokens") or {}).items()}
        vocab: dict[int, bytes] = {index: bytes([index]) for index in range(BYTE_VOCAB_SIZE)}
        for rank, (left, right) in enumerate(merges):
            vocab[BYTE_VOCAB_SIZE + rank] = vocab[left] + vocab[right]
        # Special-token ids are stored explicitly; restore their byte forms so
        # ``vocab_size`` stays identical to the trained artifact.
        for token, token_id in special_ids.items():
            vocab.setdefault(int(token_id), token.encode("utf-8"))
        corpus = data.get("training") or {}
        return cls(
            config=TokenizerConfig.from_dict(data.get("config") or {}),
            merges=merges,
            special_ids=special_ids,
            corpus_chars=int(corpus.get("corpus_chars", 0)),
            corpus_documents=int(corpus.get("corpus_documents", 0)),
            vocab=vocab,
        )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_tokenizer(
    documents: Iterable[str],
    config: TokenizerConfig | None = None,
) -> TokenizerArtifact:
    """Train a byte-level BPE tokenizer on ``documents``."""

    active = config or TokenizerConfig()
    corpus: list[list[int]] = []
    corpus_chars = 0
    for document in documents:
        text = document if isinstance(document, str) else str(document)
        if not text:
            continue
        budget = max(0, active.max_chars - corpus_chars)
        if budget <= 0:
            break
        chunk = text[:budget]
        corpus.append(list(chunk.encode("utf-8")))
        corpus_chars += len(chunk)
    if not corpus:
        raise DatasetValidationError(
            "The tokenizer training corpus is empty.",
            remediation="Add dataset records (run `alphaai train prepare` first).",
        )

    merges = _learn_merges(corpus, active.target_merges, active.min_frequency)
    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(BYTE_VOCAB_SIZE)}
    for rank, (left, right) in enumerate(merges):
        vocab[BYTE_VOCAB_SIZE + rank] = vocab[left] + vocab[right]
    next_id = BYTE_VOCAB_SIZE + len(merges)
    special_ids: dict[str, int] = {}
    for token in active.special_tokens:
        special_ids[token] = next_id
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    return TokenizerArtifact(
        config=active,
        merges=merges,
        special_ids=special_ids,
        corpus_chars=corpus_chars,
        corpus_documents=len(corpus),
        vocab=vocab,
    )


def _learn_merges(corpus: list[list[int]], target_merges: int, min_frequency: int) -> list[tuple[int, int]]:
    """Incremental byte-pair merges driven by a lazy max-heap."""

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_items: dict[tuple[int, int], set[int]] = defaultdict(set)
    heap: list[tuple[int, int, int]] = []

    def index_item(item_index: int) -> None:
        ids = corpus[item_index]
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] += 1
            pair_items[pair].add(item_index)
            heapq.heappush(heap, (-pair_counts[pair], pair[0], pair[1]))

    for item_index in range(len(corpus)):
        index_item(item_index)

    merges: list[tuple[int, int]] = []
    while len(merges) < target_merges and heap:
        negative, left, right = heapq.heappop(heap)
        pair = (left, right)
        count = pair_counts.get(pair, 0)
        if count <= 0 or -negative != count:
            continue  # stale entry
        if count < min_frequency:
            break
        new_id = BYTE_VOCAB_SIZE + len(merges)
        for item_index in list(pair_items.get(pair, ())):
            ids = corpus[item_index]
            for existing in zip(ids, ids[1:]):
                pair_counts[existing] -= 1
                if pair_counts[existing] <= 0:
                    pair_counts.pop(existing, None)
                pair_items[existing].discard(item_index)
            merged: list[int] = []
            position = 0
            while position < len(ids):
                if position < len(ids) - 1 and ids[position] == left and ids[position + 1] == right:
                    merged.append(new_id)
                    position += 2
                else:
                    merged.append(ids[position])
                    position += 1
            corpus[item_index] = merged
            index_item(item_index)
        pair_items.pop(pair, None)
        merges.append(pair)
    return merges


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def save_tokenizer(artifact: TokenizerArtifact, directory: str | Path) -> dict[str, Any]:
    """Write ``tokenizer.json`` + ``manifest.json`` and return the manifest."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer_path = target / "tokenizer.json"
    payload = json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n"
    tokenizer_path.write_text(payload, encoding="utf-8")
    manifest = {
        "name": artifact.config.name,
        "version": artifact.config.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "alphaai.training.tokenizer",
        "vocab_size": artifact.vocab_size,
        "merges": len(artifact.merges),
        "special_tokens": artifact.special_ids,
        "corpus_documents": artifact.corpus_documents,
        "corpus_chars": artifact.corpus_chars,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "file": tokenizer_path.name,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_tokenizer(directory: str | Path) -> TokenizerArtifact:
    path = Path(directory)
    tokenizer_path = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_path.exists():
        raise DatasetValidationError(
            f"AlphaAI tokenizer not found: {tokenizer_path}",
            remediation="Train one first: `alphaai train tokenize` (see docs/TRAINING.md).",
        )
    return TokenizerArtifact.from_dict(json.loads(tokenizer_path.read_text(encoding="utf-8")))


def tokenizer_summary(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["present"] = True
        return manifest
    return {"present": False, "directory": str(path), "detail": "no tokenizer manifest"}


def roundtrip_report(artifact: TokenizerArtifact, samples: Iterable[str]) -> dict[str, Any]:
    """Real, measured tokenizer quality numbers for given samples."""

    documents = [text for text in samples if text]
    if not documents:
        return {"ok": False, "detail": "no samples provided"}
    exact = 0
    tokens = 0
    characters = 0
    for text in documents:
        ids = artifact.encode(text)
        if artifact.decode(ids) == text:
            exact += 1
        tokens += len(ids)
        characters += len(text)
    return {
        "ok": True,
        "documents": len(documents),
        "roundtrip_exact": exact,
        "roundtrip_rate": round(exact / len(documents), 4),
        "tokens": tokens,
        "characters": characters,
        "characters_per_token": round(characters / max(tokens, 1), 4),
        "vocab_size": artifact.vocab_size,
    }


__all__ = [
    "BYTE_VOCAB_SIZE",
    "DEFAULT_SPECIAL_TOKENS",
    "TokenizerArtifact",
    "TokenizerConfig",
    "load_tokenizer",
    "roundtrip_report",
    "save_tokenizer",
    "tokenizer_summary",
    "train_tokenizer",
]

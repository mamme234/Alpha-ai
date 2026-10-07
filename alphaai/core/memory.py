"""AlphaAI memory system.

Persistent, namespaced key/value memory with tags, stored in a real SQLite
database (``sqlite3`` from the standard library — no extra dependency). Retrieval
is lexical: AlphaAI scores entries by token overlap plus recency and returns the
top matches with the score it actually computed, so a caller can tell a strong
match from a weak one.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import MemoryError_

_WORD = re.compile(r"[a-z0-9_]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'note',
    tags TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace, updated_at DESC);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    action TEXT NOT NULL,
    key TEXT,
    created_at REAL NOT NULL,
    detail TEXT
);
"""


@dataclass(slots=True)
class MemoryEntry:
    namespace: str
    key: str
    value: str
    kind: str = "note"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "key": self.key,
            "value": self.value,
            "kind": self.kind,
            "tags": list(self.tags),
            "importance": round(self.importance, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "score": round(self.score, 4),
        }


class MemoryStore:
    """Namespaced persistent memory for conversations, skills and agents."""

    def __init__(self, path: str | Path, *, max_entries_per_namespace: int = 5000) -> None:
        self.path = Path(path)
        self.max_entries_per_namespace = max_entries_per_namespace
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - filesystem/permissions
            raise MemoryError_(
                f"AlphaAI memory database could not be opened at {self.path.name}: {exc}",
                remediation="Set memory.path to a writable location or disable memory (memory.enabled=false).",
            ) from exc
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- writes -----------------------------------------------------------
    def remember(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = "default",
        kind: str = "note",
        tags: Sequence[str] | None = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Insert or update a memory entry."""

        if not key:
            raise MemoryError_("Memory keys must not be empty.")
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        now = time.time()
        payload = json.dumps(list(tags or []), ensure_ascii=False)
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created_at FROM memories WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO memories (namespace, key, value, kind, tags, importance, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (namespace, key, text, kind, payload, float(importance), now, now),
                )
                action = "insert"
            else:
                self._conn.execute(
                    "UPDATE memories SET value = ?, kind = ?, tags = ?, importance = ?, updated_at = ?"
                    " WHERE id = ?",
                    (text, kind, payload, float(importance), now, row["id"]),
                )
                action = "update"
            self._conn.execute(
                "INSERT INTO memory_events (namespace, action, key, created_at, detail)"
                " VALUES (?, ?, ?, ?, ?)",
                (namespace, action, key, now, f"{len(text)} chars"),
            )
            self._conn.commit()
            self._enforce_limit(namespace)
        entry = MemoryEntry(
            namespace=namespace,
            key=key,
            value=text,
            kind=kind,
            tags=list(tags or []),
            importance=float(importance),
            created_at=now,
            updated_at=now,
        )
        return entry

    def forget(self, key: str, *, namespace: str = "default") -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE namespace = ? AND key = ?", (namespace, key)
            )
            self._conn.execute(
                "INSERT INTO memory_events (namespace, action, key, created_at, detail)"
                " VALUES (?, 'forget', ?, ?, ?)",
                (namespace, key, time.time(), "explicit delete"),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def forget_namespace(self, namespace: str) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
            self._conn.commit()
            return cursor.rowcount

    # -- reads ------------------------------------------------------------
    def get(self, key: str, *, namespace: str = "default") -> MemoryEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE namespace = ? AND key = ?", (namespace, key)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (row["id"],)
            )
            self._conn.commit()
        return _row_to_entry(row)

    def entries(self, *, namespace: str = "default", limit: int = 100) -> list[MemoryEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE namespace = ? ORDER BY updated_at DESC LIMIT ?",
                (namespace, int(limit)),
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        namespace: str | None = None,
        limit: int = 5,
        tags: Sequence[str] | None = None,
    ) -> list[MemoryEntry]:
        """Lexical retrieval with an explicit, computed score."""

        terms = _terms(query)
        with self._lock:
            if namespace:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE namespace = ? ORDER BY updated_at DESC LIMIT 2000",
                    (namespace,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT 2000"
                ).fetchall()
        now = time.time()
        wanted = {tag.lower() for tag in (tags or [])}
        scored: list[MemoryEntry] = []
        for row in rows:
            entry = _row_to_entry(row)
            if wanted and not wanted.issubset({tag.lower() for tag in entry.tags}):
                continue
            entry.score = _score(entry, terms, now)
            if entry.score > 0:
                scored.append(entry)
        scored.sort(key=lambda item: (-item.score, -item.updated_at))
        return scored[: max(1, int(limit))]

    def namespaces(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace, COUNT(*) AS entries, MAX(updated_at) AS updated_at"
                " FROM memories GROUP BY namespace ORDER BY namespace"
            ).fetchall()
        return [
            {"namespace": row["namespace"], "entries": row["entries"], "updated_at": row["updated_at"]}
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
            events = self._conn.execute("SELECT COUNT(*) AS c FROM memory_events").fetchone()["c"]
        return {
            "path": str(self.path),
            "entries": int(total),
            "events": int(events),
            "namespaces": self.namespaces(),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- internals --------------------------------------------------------
    def _enforce_limit(self, namespace: str) -> None:
        if self.max_entries_per_namespace <= 0:
            return
        count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE namespace = ?", (namespace,)
        ).fetchone()["c"]
        excess = int(count) - self.max_entries_per_namespace
        if excess <= 0:
            return
        self._conn.execute(
            "DELETE FROM memories WHERE id IN ("
            " SELECT id FROM memories WHERE namespace = ?"
            " ORDER BY importance ASC, updated_at ASC LIMIT ?)",
            (namespace, excess),
        )
        self._conn.execute(
            "INSERT INTO memory_events (namespace, action, key, created_at, detail)"
            " VALUES (?, 'evict', NULL, ?, ?)",
            (namespace, time.time(), f"evicted {excess} low-importance entries"),
        )
        self._conn.commit()


def memory_from_config(config) -> "MemoryStore | None":
    """Build the memory store if ``memory.enabled`` is true."""

    if not config.memory.enabled:
        return None
    return MemoryStore(
        config.memory.path,
        max_entries_per_namespace=config.memory.max_entries_per_namespace,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _terms(text: str) -> list[str]:
    return [match.group(0) for match in _WORD.finditer((text or "").lower())]


def _score(entry: MemoryEntry, terms: Iterable[str], now: float) -> float:
    if not terms:
        return 0.0
    haystack = f"{entry.key} {entry.value} {' '.join(entry.tags)}".lower()
    if not haystack:
        return 0.0
    hits = sum(1 for term in terms if term in haystack)
    if hits == 0:
        return 0.0
    coverage = hits / len(set(terms))
    # Recency decays over ~30 days; importance nudges the order.
    age_days = max(0.0, (now - entry.updated_at) / 86400.0)
    recency = 1.0 / (1.0 + age_days / 30.0)
    return round(coverage * 0.7 + recency * 0.2 + entry.importance * 0.1, 6)


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    try:
        tags = json.loads(row["tags"] or "[]")
    except json.JSONDecodeError:  # pragma: no cover - corrupted row
        tags = []
    return MemoryEntry(
        namespace=row["namespace"],
        key=row["key"],
        value=row["value"],
        kind=row["kind"],
        tags=[str(tag) for tag in tags],
        importance=float(row["importance"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        access_count=int(row["access_count"]),
    )


__all__ = ["MemoryEntry", "MemoryStore", "memory_from_config", "SCHEMA"]

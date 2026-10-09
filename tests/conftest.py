"""Shared AlphaAI test fixtures.

The tests use a *test engine* (a real ``ModelEngine`` implementation that returns
fixed text) so the conversation engine, router, API and CLI can be exercised
without weights. Nothing in ``alphaai`` itself ever falls back to a fake engine:
the CLI and API only ever use engines discovered from ``configs/models``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pytest

from alphaai.config.loader import load_config
from alphaai.config.schema import AlphaAIConfig
from alphaai.core.engine import ModelEngine, ModelSpec, spec_from_dict
from alphaai.core.errors import DatabaseNotFoundError
from alphaai.core.runtime import AlphaRuntime
from alphaai.core.types import (
    Capability,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    StreamChunk,
    TokenUsage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_TOOL_CALL_TEXT = 'I will use a tool.\n```json\n{"tool": "calculator.evaluate", "arguments": {"expression": "2+2"}}\n```'

#: The real CPU-friendly open-weight model AlphaAI runs (see configs/models).
LOCAL_MODEL_ID = "qwen2.5-0.5b-instruct-gguf"
LOCAL_MODEL_DIR = REPO_ROOT / "models" / LOCAL_MODEL_ID


def local_model_status() -> tuple[bool, str]:
    """Whether the repository's real local model can actually run here."""

    from alphaai.engines.helpers import module_present

    if not module_present("llama_cpp"):
        return False, "llama-cpp-python is not installed (pip install -e '.[llama]')"
    if not (REPO_ROOT / "configs" / "models" / f"{LOCAL_MODEL_ID}.json").exists():
        return False, f"configs/models/{LOCAL_MODEL_ID}.json is missing"
    for candidate in sorted(LOCAL_MODEL_DIR.glob("*.gguf")):
        if candidate.stat().st_size > 0:
            return True, str(candidate)
    return False, f"{LOCAL_MODEL_ID} weights are not installed (run `alphaai models install {LOCAL_MODEL_ID}`)"


LOCAL_MODEL_READY, LOCAL_MODEL_REASON = local_model_status()

#: Apply to tests that must exercise the *real* installed model. They are skipped
#: (with the exact reason) on a machine without the runtime or the weights —
#: never replaced by a mock.
requires_local_model = pytest.mark.skipif(not LOCAL_MODEL_READY, reason=LOCAL_MODEL_REASON)


# ---------------------------------------------------------------------------
# PostgreSQL (Supabase) test support
# ---------------------------------------------------------------------------
#: Point this at any empty PostgreSQL database (a throwaway Supabase project is
#: fine) to run the persistence tests against a real server.
DATABASE_URL_ENV = "ALPHAI_TEST_DATABASE_URL"

REPO_MIGRATIONS_DIR = REPO_ROOT / "supabase" / "migrations"


def _database_url() -> tuple[str, str]:
    """A real PostgreSQL URL for the persistence tests, or the reason there is none.

    Follows the local-model convention: real when it can be, skipped with the
    exact reason when it cannot, never silently replaced by a fake. An explicit
    ``ALPHAI_TEST_DATABASE_URL`` wins; otherwise a private temporary PostgreSQL
    is started with ``pgserver`` if that package is installed.
    """

    try:
        import psycopg  # noqa: F401 - the driver the store uses
    except ImportError:
        return (
            "",
            "no PostgreSQL driver: `pip install 'psycopg[binary]'` (or `pip install -e '.[db]'`) "
            "to run the persistence tests against a real database",
        )
    explicit = os.environ.get(DATABASE_URL_ENV)
    if explicit:
        return explicit, ""
    try:
        import pgserver  # noqa: F401 - optional developer dependency
    except ImportError:
        return (
            "",
            f"no PostgreSQL available: set {DATABASE_URL_ENV} or `pip install pgserver` to "
            "run the persistence tests against a real database",
        )
    try:
        server = pgserver.get_server(tempfile.mkdtemp(prefix="alphaai-test-pg-"))
        return server.get_uri(), ""
    except Exception as exc:  # noqa: BLE001 - any failure means "no database here"
        return "", f"no PostgreSQL available: pgserver could not start one here ({exc})"


DATABASE_READY, DATABASE_REASON = _database_url()

#: Apply to tests that need a real PostgreSQL server. They are skipped (with the
#: exact reason) when there is none — never weakened into a fake.
requires_database = pytest.mark.skipif(not DATABASE_READY, reason=DATABASE_REASON)


class MemoryStore:
    """An in-memory :class:`~alphaai.db.base.ConversationStore`, for tests only.

    It exists so the API's persistence wiring (routes, chat persistence, the
    websocket-free streaming event) can be exercised without a database. It is
    never used by ``alphaai`` itself, and the SQL behaviour of the real store is
    verified against a real PostgreSQL in ``tests/test_database.py``.
    """

    configured = True

    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.messages: list[dict[str, Any]] = []
        self.preferences: dict[str, dict[str, Any]] = {}
        self.usage: list[dict[str, Any]] = []
        self.counter = 0

    # -- health ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "reachable": True,
            "mode": "memory",
            "host": "memory",
            "migrations": {"applied": 0, "pending": []},
        }

    # -- conversations ---------------------------------------------------
    def list_conversations(
        self, *, client_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        items = [item for item in self.conversations.values() if item["client_id"] == client_id]
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return [dict(item) for item in items[offset : offset + limit]]

    def create_conversation(self, **kwargs: Any) -> dict[str, Any]:
        self.counter += 1
        identifier = kwargs.get("conversation_id") or f"00000000-0000-4000-8000-{self.counter:012d}"
        item = {
            "id": identifier,
            "client_id": kwargs["client_id"],
            "title": kwargs.get("title") or "New conversation",
            "session_id": kwargs.get("session_id"),
            "engine_id": kwargs.get("engine_id"),
            "model": kwargs.get("model"),
            "metadata": kwargs.get("metadata") or {},
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "message_count": 0,
            "preview": "",
        }
        self.conversations[identifier] = item
        return dict(item)

    def get_conversation(
        self,
        conversation_id: str,
        *,
        client_id: str | None = None,
        include_messages: bool = True,
    ) -> dict[str, Any] | None:
        item = self.conversations.get(conversation_id)
        if item is None or (client_id and item["client_id"] != client_id):
            return None
        payload = dict(item)
        if include_messages:
            payload["messages"] = [
                dict(message)
                for message in self.messages
                if message["conversation_id"] == conversation_id
            ]
        return payload

    def delete_conversation(self, conversation_id: str, *, client_id: str | None = None) -> bool:
        item = self.conversations.get(conversation_id)
        if item is None or (client_id and item["client_id"] != client_id):
            return False
        del self.conversations[conversation_id]
        self.messages = [m for m in self.messages if m["conversation_id"] != conversation_id]
        return True

    def append_message(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        item = self.conversations.get(conversation_id)
        owner = kwargs.get("client_id")
        if item is None or (owner and item["client_id"] != owner):
            raise DatabaseNotFoundError(
                "No conversation with that id exists for this client."
            )
        self.counter += 1
        message = {
            "id": self.counter,
            "conversation_id": conversation_id,
            "role": kwargs["role"],
            "content": kwargs["content"],
            "engine_id": kwargs.get("engine_id"),
            "model": kwargs.get("model"),
            "finish_reason": kwargs.get("finish_reason"),
            "latency_ms": kwargs.get("latency_ms"),
            "usage": kwargs.get("usage"),
            "metadata": kwargs.get("metadata") or {},
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        self.messages.append(message)
        item["message_count"] = item["message_count"] + 1
        if item["title"] == "New conversation" and message["role"] == "user":
            item["title"] = message["content"][:120]
        item["preview"] = message["content"][:160]
        return dict(message)

    # -- preferences / usage ---------------------------------------------
    def get_preferences(self, *, client_id: str) -> dict[str, Any]:
        stored = self.preferences.get(client_id)
        return {
            "client_id": client_id,
            "preferences": dict(stored or {}),
            "stored": stored is not None,
        }

    def set_preferences(self, *, client_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        self.preferences[client_id] = dict(preferences)
        return {"client_id": client_id, "preferences": dict(preferences), "stored": True}

    def record_usage(self, **kwargs: Any) -> dict[str, Any]:
        self.counter += 1
        record = {"id": self.counter, **kwargs}
        self.usage.append(record)
        return record

    def usage_summary(self, *, client_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        rows = [row for row in self.usage if not client_id or row.get("client_id") == client_id]
        return {
            "totals": {"generations": len(rows), "total_tokens": 0},
            "recent": rows[:limit],
        }

    def migrate(self, *, directory: str | None = None) -> dict[str, Any]:
        return {"ok": True, "applied": [], "already_applied": [], "drift": [], "total": 0}


class FakeEngine(ModelEngine):
    """Deterministic engine used only by tests."""

    engine_key = "transformers"
    engine_name = "AlphaAI Test Engine"

    def __init__(self, *args: Any, reply: str = "alphaai test reply", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reply = reply
        self.loads = 0
        self.requests: list[GenerationRequest] = []

    def probe_runtime(self) -> tuple[bool, str, str | None]:
        return True, "test runtime", None

    def find_weights(self) -> Path | None:
        return Path(self.config.paths.models_dir) / self.id

    def health(self, *, refresh: bool = False) -> EngineStatus:
        return EngineStatus(
            engine_id=self.id,
            state="disabled" if not self.enabled else ("ready" if self._loaded else "available"),
            detail="AlphaAI test engine is ready.",
            loaded=self._loaded,
            device="cpu",
            dtype="float32",
            weights_present=True,
            hardware_ok=True,
            extras={"display_name": self.name, "attribution": self.attribution},
        )

    def load(self) -> None:
        self.loads += 1
        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        text = self.reply if len(self.requests) == 1 or "tool" not in self.reply else "final answer after tools"
        return GenerationResult(
            text=text,
            engine_id=self.id,
            model=self.model,
            provider=self.provider,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, source="engine"),
            runtime="transformers/cpu",
            attribution=self.attribution,
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        self.requests.append(request)
        for index, word in enumerate(self.reply.split(" ")):
            yield StreamChunk(text=(word if index == 0 else " " + word), index=index, engine_id=self.id, model=self.model)
        yield StreamChunk(
            index=999,
            done=True,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, source="engine"),
            engine_id=self.id,
            model=self.model,
        )


def test_spec(model_id: str = "alphaai-test", **overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "id": model_id,
        "display_name": "AlphaAI Test Engine (Test-Model)",
        "engine": "transformers",
        "family": "alphaai",
        "provider": "alphaai",
        "model": "Test-Model",
        "model_owner": "AlphaAI",
        "engine_owner": "AlphaAI",
        "context_length": 8192,
        "capabilities": ["chat", "streaming", "tool_calling", "coding", "mathematics"],
        "params_total_b": 0.1,
        "strengths": ["general", "coding"],
    }
    payload.update(overrides)
    return spec_from_dict(payload)


# ``test_spec`` is a helper, not a test: stop pytest from collecting it when test
# modules import it by name.
test_spec.__test__ = False  # type: ignore[attr-defined]


@pytest.fixture
def config(tmp_path: Path) -> AlphaAIConfig:
    root = tmp_path / "project"
    (root / "configs" / "models").mkdir(parents=True)
    (root / "workspace").mkdir(parents=True)
    (root / "datasets").mkdir(parents=True)
    active = load_config(project_root=str(root), env={}, create_dirs=True)
    active.paths.project_root = str(root)
    active.tools.sandbox_root = str(root / "workspace")
    active.memory.path = str(root / "state" / "memory.sqlite3")
    active.api.port = 8099
    return active


@pytest.fixture
def runtime(config: AlphaAIConfig) -> Iterator[AlphaRuntime]:
    active = AlphaRuntime.create(config, discover=False)
    try:
        yield active
    finally:
        active.close()


@pytest.fixture
def engine_runtime(config: AlphaAIConfig) -> Iterator[AlphaRuntime]:
    """Runtime with one usable test engine registered."""

    active = AlphaRuntime.create(config, discover=False)
    active.registry.register(FakeEngine(test_spec(), config))
    try:
        yield active
    finally:
        active.close()


@pytest.fixture
def tool_call_engine(config: AlphaAIConfig) -> Iterator[AlphaRuntime]:
    """Runtime whose test engine emits one real tool call before answering."""

    active = AlphaRuntime.create(config, discover=False)
    active.registry.register(FakeEngine(test_spec(), config, reply=TEST_TOOL_CALL_TEXT))
    try:
        yield active
    finally:
        active.close()


@pytest.fixture
def echo_dataset(tmp_path: Path) -> Path:
    """A tiny, valid dataset directory for training tests."""

    directory = tmp_path / "datasets" / "unit-sample"
    directory.mkdir(parents=True)
    records = [
        {"instruction": f"question {index}", "response": f"answer {index} " + "detail " * 5}
        for index in range(12)
    ]
    (directory / "dataset.json").write_text(
        json.dumps(
            {
                "name": "unit-sample",
                "version": "1.0.0",
                "license": "Apache-2.0",
                "source": "test fixture",
                "format": "jsonl",
                "splits": {"train": "train.jsonl", "valid": "valid.jsonl"},
                "min_record_chars": 5,
            }
        ),
        encoding="utf-8",
    )
    for split, chunk in (("train", records[:9]), ("valid", records[9:])):
        (directory / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in chunk) + "\n", encoding="utf-8"
        )
    return directory


__all__ = [
    "Capability",
    "DATABASE_READY",
    "DATABASE_REASON",
    "DATABASE_URL_ENV",
    "FakeEngine",
    "LOCAL_MODEL_ID",
    "LOCAL_MODEL_READY",
    "LOCAL_MODEL_REASON",
    "MemoryStore",
    "REPO_MIGRATIONS_DIR",
    "REPO_ROOT",
    "TEST_TOOL_CALL_TEXT",
    "echo_dataset",
    "local_model_status",
    "requires_database",
    "requires_local_model",
    "test_spec",
]

"""Context manager and memory system tests."""

from __future__ import annotations

from alphaai.core.context import ContextManager, build_memory_block
from alphaai.core.memory import MemoryStore
from alphaai.core.types import ChatMessage


def test_context_keeps_system_prompt_and_drops_oldest(config) -> None:
    manager = ContextManager(config)
    messages = [ChatMessage(role="user", content=f"message {index} " + "x" * 200) for index in range(40)]
    plan = manager.plan(
        messages,
        engine=None,
        system_prompt="You are AlphaAI.",
        max_new_tokens=64,
        reserve_tokens=0,
    )
    # No engine: default 32k window, so everything fits.
    assert plan.truncated is False
    assert plan.messages[0].role == "system"
    assert plan.messages[0].content == "You are AlphaAI."


def test_context_truncates_with_small_window(config) -> None:
    from alphaai.core.engine import ModelEngine

    manager = ContextManager(config)
    messages = [ChatMessage(role="user", content=f"turn {index}: " + "wide " * 60) for index in range(30)]

    class Small:
        context_length = 600

        def count_tokens(self, text: str) -> int:
            return len(text) // 4

    plan = manager.plan(messages, engine=Small(), system_prompt="sys", max_new_tokens=64, reserve_tokens=0)
    assert plan.truncated is True
    assert plan.dropped_messages > 0
    assert plan.used_tokens <= plan.budget_tokens
    assert any("dropped" in note for note in plan.notes)
    # newest message is kept
    assert "turn 29" in plan.messages[-1].content


def test_context_tokenizer_source_is_reported(config) -> None:
    manager = ContextManager(config)

    class Counter:
        context_length = 4096

        def count_tokens(self, text: str) -> int:
            return 7

    plan = manager.plan([ChatMessage(role="user", content="hi")], engine=Counter(), system_prompt=None)
    assert plan.token_source == "tokenizer"
    assert plan.used_tokens == 7 + 4


def test_memory_roundtrip_and_search(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3", max_entries_per_namespace=50)
    store.remember("router", "AlphaAI routes coding requests to coding engines", tags=["routing"])
    store.remember("tokenizer", "AlphaAI trains a byte-level BPE tokenizer", tags=["training"])
    assert store.get("router") is not None

    hits = store.search("tokenizer bpe", limit=3)
    assert hits and hits[0].key == "tokenizer"
    assert hits[0].score > 0

    tagged = store.search("alphaai", tags=["routing"], limit=5)
    assert [entry.key for entry in tagged] == ["router"]

    assert store.forget("router") is True
    assert store.get("router") is None
    stats = store.stats()
    assert stats["entries"] == 1 and stats["events"] >= 3
    store.close()


def test_memory_upsert_and_namespace_limit(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3", max_entries_per_namespace=2)
    store.remember("a", "one", namespace="ns")
    store.remember("a", "two", namespace="ns")
    assert store.get("a", namespace="ns").value == "two"
    store.remember("b", "bee", namespace="ns", importance=0.9)
    store.remember("c", "cee", namespace="ns", importance=0.1)
    keys = {entry.key for entry in store.entries(namespace="ns")}
    assert len(keys) <= 2  # the limit evicts the lowest-importance entry
    namespaces = {entry["namespace"] for entry in store.namespaces()}
    assert namespaces == {"ns"}
    assert store.forget_namespace("ns") >= 1
    store.close()


def test_memory_block_rendering() -> None:
    from alphaai.core.memory import MemoryEntry

    entries = [MemoryEntry(namespace="n", key="k1", value="v1"), MemoryEntry(namespace="n", key="k2", value="v2")]
    block = build_memory_block(entries)
    assert block and "k1: v1" in block and "memory" in block.lower()
    assert build_memory_block([]) is None


def test_runtime_memory_is_wired(runtime) -> None:
    assert runtime.memory is not None
    runtime.memory.remember("fact", "AlphaAI runs local models", namespace="default")
    hits = runtime.memory.search("local models", namespace="default")
    assert hits and hits[0].value.startswith("AlphaAI runs")

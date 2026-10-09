"""PostgreSQL (Supabase) persistence tests.

Three layers, in order of how much they need:

1. **Migrations and schema** — the ``.sql`` files are the schema. They are read
   here and checked for the things AlphaAI's code depends on (tables, columns,
   constraints, indexes, row level security). Real files, no database needed.
2. **The API contract** — with no database configured every persistence request
   must answer ``database_not_configured`` (never an empty history), health must
   keep working, and a chat turn must state whether it was saved. These use
   :class:`tests.conftest.MemoryStore` for the store and the real API for
   everything else.
3. **A real PostgreSQL server** — migrations, CRUD, client isolation, RLS and the
   full chat → persist → retrieve path, against a genuine server. Skipped with
   the exact reason when there is none (see ``ALPHAI_TEST_DATABASE_URL`` /
   ``pgserver`` in ``tests/conftest.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphaai.api.app import create_app
from alphaai.cli import main as cli_main
from alphaai.config.loader import load_config, public_config_view
from alphaai.core.errors import DatabaseNotFoundError
from alphaai.core.runtime import AlphaRuntime
from alphaai.db import open_database
from alphaai.db.base import UnconfiguredStore, connection_summary
from alphaai.db.migrate import (
    Migration,
    checksum_sql,
    discover_migrations,
    plan_migrations,
    run_migrations,
    split_sql_statements,
)
from alphaai.db.postgres import (
    DELETE_CONVERSATION,
    INSERT_CONVERSATION,
    INSERT_MESSAGE,
    INSERT_USAGE,
    SELECT_CONVERSATIONS,
    SELECT_CONVERSATION,
    SELECT_MESSAGES,
    UPSERT_PREFERENCES,
)

from .conftest import (
    DATABASE_READY,
    DATABASE_REASON,
    REPO_MIGRATIONS_DIR,
    FakeEngine,
    MemoryStore,
    requires_database,
    test_spec,
)

DATABASE_URL = DATABASE_READY
CLIENT = "client-0123456789abcdef"
OTHER_CLIENT = "client-ffffffffffffffff"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def migration_sql(name: str) -> str:
    return (REPO_MIGRATIONS_DIR / name).read_text(encoding="utf-8")


def all_migration_sql() -> str:
    return "\n".join(item.sql for item in discover_migrations(REPO_MIGRATIONS_DIR))


class RecordingExecutor:
    """A migration executor that only records what it was asked to run."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.statements: list[str] = []
        self.params: list[tuple] = []
        self.commits = 0
        self.tables: dict[str, dict[str, str]] = {}
        self.fail_on = fail_on

    def run(self, sql: str, params=None) -> list[dict]:
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("simulated failure")
        self.statements.append(sql)
        if params:
            self.params.append(tuple(params))
            if sql.strip().startswith("insert into public.alphaai_migrations"):
                self.tables.setdefault("applied", {})[params[0]] = params[1]
        if sql.strip().startswith("select name, checksum from"):
            return [
                {"name": name, "checksum": checksum}
                for name, checksum in self.tables.get("applied", {}).items()
            ]
        return []

    def commit(self) -> None:
        self.commits += 1


def memory_client(config, *, store: MemoryStore | None = None, with_engine: bool = True):
    """A TestClient whose persistence is an in-memory store (API contract tests)."""

    runtime = AlphaRuntime.create(config, discover=False)
    if with_engine:
        runtime.registry.register(FakeEngine(test_spec(), config))
    app = create_app(config, runtime=runtime)
    app.state.database = store if store is not None else MemoryStore()
    return TestClient(app)


def no_database_client(config, *, with_engine: bool = False):
    """A TestClient for a deployment with no DATABASE_URL at all."""

    runtime = AlphaRuntime.create(config, discover=False)
    if with_engine:
        runtime.registry.register(FakeEngine(test_spec(), config))
    return TestClient(create_app(config, runtime=runtime))


# ---------------------------------------------------------------------------
# 1. migrations and schema (always run: no database needed)
# ---------------------------------------------------------------------------
def test_migrations_are_versioned_files_in_order() -> None:
    migrations = discover_migrations(REPO_MIGRATIONS_DIR)
    assert migrations, f"no migrations found in {REPO_MIGRATIONS_DIR}"
    names = [item.name for item in migrations]
    assert names == sorted(names), "migrations must apply in filename order"
    assert len(names) == len(set(names))
    for item in migrations:
        assert item.path.suffix == ".sql"
        assert item.name.split("_", 1)[0].isdigit()
        assert item.checksum == checksum_sql(item.sql)
        assert item.sql.strip(), f"{item.name} is empty"

    # Every .sql file in the directory is a versioned migration: a file the
    # runner would silently ignore is a bug, not a convenience.
    on_disk = {path.name for path in REPO_MIGRATIONS_DIR.glob("*.sql")}
    assert on_disk == set(names)


def test_schema_covers_the_alphaai_data_model() -> None:
    sql = all_migration_sql()
    lowered = sql.lower()

    # Tables: the four AlphaAI owns, and nothing else.
    for table in ("conversations", "messages", "user_preferences", "usage_records"):
        assert f"create table if not exists public.{table}" in lowered
    assert lowered.count("create table if not exists") == 4

    # Messages carry role + content + timestamps + conversation id.
    assert "conversation_id uuid not null references public.conversations (id) on delete cascade" in lowered
    assert "role text not null" in lowered
    assert "content text not null" in lowered
    assert "created_at timestamptz not null default now()" in lowered
    assert "constraint messages_role_valid check (role in ('system', 'user', 'assistant', 'tool'))" in lowered

    # A conversation belongs to a client and can be listed newest-first.
    assert "client_id text not null" in lowered
    assert "conversations_client_updated_idx" in lowered
    assert "messages_conversation_created_idx" in lowered

    # Extensible where it should be, bounded where it must be.
    assert "metadata jsonb not null default '{}'::jsonb" in lowered
    assert "usage jsonb" in lowered
    assert "char_length(title) <= 200" in lowered

    # No weights, no binaries, no credentials in the statements themselves (the
    # comments are allowed to *discuss* secrets - the SQL is not allowed to carry
    # any).
    statements_only = re.sub(r"--[^\n]*", "", sql).lower()
    assert "bytea" not in statements_only
    assert "lo_import" not in statements_only
    for secret in ("postgresql://", "postgres://", "service_role", "supabase_anon_key", "sk-"):
        assert secret not in statements_only

    # Row level security on every table, and no policy that grants `anon`.
    for table in ("conversations", "messages", "user_preferences", "usage_records"):
        assert f"alter table public.{table} enable row level security" in lowered
    assert "to anon" not in lowered
    assert lowered.count("to authenticated") >= 8


def test_statement_splitter_respects_postgres_quoting() -> None:
    script = """
    -- a comment with a ; semicolon
    create table public.t (note text);  /* block ; comment */
    insert into public.t values ('a ; b');
    insert into public.t values ('it''s fine');
    create function public.f() returns int language sql as $$
        select 1; -- inside a dollar-quoted body
    $$;
    """
    statements = split_sql_statements(script)
    assert len(statements) == 4, statements
    assert statements[0].startswith("-- a comment with a ; semicolon")
    assert "'a ; b'" in statements[1]
    assert "it''s fine" in statements[2]
    assert "$$" in statements[3] and statements[3].count("$$") == 2

    # The shipped RLS migration is one guarded block, not 22 fragile statements.
    rls = split_sql_statements(migration_sql("20261009120200_alphaai_row_level_security.sql"))
    assert len(rls) == 5
    assert any(statement.startswith("do $$") for statement in rls)


def test_plan_reports_pending_and_drift() -> None:
    migrations = discover_migrations(REPO_MIGRATIONS_DIR)
    first = migrations[0]

    pending, drift = plan_migrations(migrations, {})
    assert pending == list(migrations) and drift == []

    applied = {item.name: item.checksum for item in migrations}
    pending, drift = plan_migrations(migrations, applied)
    assert pending == [] and drift == []

    stale = dict(applied, **{first.name: "0" * 64})
    pending, drift = plan_migrations(migrations, stale)
    assert pending == []
    assert drift and drift[0]["name"] == first.name


def test_runner_applies_files_records_checksums_and_is_idempotent() -> None:
    migrations = discover_migrations(REPO_MIGRATIONS_DIR)
    executor = RecordingExecutor()

    report = run_migrations(executor, migrations)
    assert report["ok"] is True
    assert report["applied"] == [item.name for item in migrations]
    assert report["drift"] == []
    assert "create table if not exists public.alphaai_migrations" in executor.statements[0]
    # One durable point per file (plus the ledger table itself): a failure never
    # leaves a file applied but unrecorded.
    assert executor.commits == 1 + len(migrations)
    # The ledger was written for every file, with the file's checksum.
    recorded = {params[0]: params[1] for params in executor.params if len(params) == 2 and params[0].endswith(".sql")}
    assert recorded == {item.name: item.checksum for item in migrations}

    # Second run: nothing to apply, nothing re-executed.
    again = RecordingExecutor()
    second = run_migrations(again, migrations, applied=recorded)
    assert second["applied"] == []
    assert second["already_applied"] == [item.name for item in migrations]
    assert again.commits == 1, "only the ledger's own durable point"

    # Applied-then-edited file is reported as drift, never re-run.
    edited = [
        Migration(name=item.name, path=item.path, sql=item.sql + "\n-- changed\n", checksum=checksum_sql(item.sql + "\n-- changed\n"))
        for item in migrations[:1]
    ]
    third = run_migrations(RecordingExecutor(), edited, applied=recorded)
    assert third["ok"] is False
    assert third["drift"][0]["name"] == migrations[0].name


def test_runner_names_the_failing_statement_and_file() -> None:
    migrations = discover_migrations(REPO_MIGRATIONS_DIR)
    executor = RecordingExecutor(fail_on="create index if not exists conversations_client_updated_idx")
    with pytest.raises(Exception) as caught:
        run_migrations(executor, migrations)
    error = caught.value
    assert getattr(error, "code", "") == "database_unavailable"
    assert migrations[0].name in str(error)
    assert "statement" in str(error.details)


def test_store_sql_matches_the_schema() -> None:
    """Pin the queries to the columns the migrations actually create."""

    assert "from public.conversations c" in SELECT_CONVERSATIONS
    assert "order by c.updated_at desc" in SELECT_CONVERSATIONS
    assert "where c.id = %s::uuid" in SELECT_CONVERSATION
    assert "from public.messages m" in SELECT_MESSAGES
    assert "insert into public.conversations" in INSERT_CONVERSATION
    assert "gen_random_uuid()" in INSERT_CONVERSATION
    assert "insert into public.messages" in INSERT_MESSAGE
    for column in ("conversation_id", "role", "content", "engine_id", "model", "finish_reason", "latency_ms", "usage", "metadata"):
        assert column in INSERT_MESSAGE
    assert "delete from public.conversations" in DELETE_CONVERSATION
    assert "insert into public.user_preferences" in UPSERT_PREFERENCES
    assert "insert into public.usage_records" in INSERT_USAGE

    # Parameters, never interpolation: no query embeds a value.
    for statement in (SELECT_CONVERSATIONS, INSERT_MESSAGE, INSERT_USAGE, UPSERT_PREFERENCES):
        assert "%s" in statement


# ---------------------------------------------------------------------------
# 2a. configuration
# ---------------------------------------------------------------------------
def test_database_url_is_read_from_the_environment(tmp_path: Path) -> None:
    url = "postgresql://alphaai:secret@db.example.supabase.co:5432/postgres?sslmode=require"
    config = load_config(project_root=str(tmp_path), env={"DATABASE_URL": url})
    assert config.database.url == url

    # ALPHAI_DATABASE_URL wins when both are present (two databases, one env).
    both = load_config(
        project_root=str(tmp_path),
        env={
            "DATABASE_URL": "postgresql://other@host/db",
            "ALPHAI_DATABASE_URL": url,
            "ALPHAI_DB_STATEMENT_TIMEOUT_MS": "9000",
        },
    )
    assert both.database.url == url
    assert both.database.statement_timeout_ms == 9000


def test_a_non_postgres_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception) as caught:
        load_config(project_root=str(tmp_path), env={"DATABASE_URL": "sqlite:///alphaai.db"})
    assert "PostgreSQL connection string" in str(caught.value)


def test_public_config_never_exposes_the_password(tmp_path: Path) -> None:
    url = "postgresql://alphaai:super-secret-password@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"
    config = load_config(project_root=str(tmp_path), env={"DATABASE_URL": url})
    view = public_config_view(config)["database"]
    serialized = json.dumps(view)
    assert "super-secret-password" not in serialized
    assert "pooler.supabase.com" in serialized
    assert view["configured"] is True
    assert view["mode"] == "transaction-pooler"
    assert set(view).isdisjoint({"url", "password", "dsn", "database_url"})


def test_connection_summary_identifies_the_connection_method() -> None:
    direct = connection_summary("postgresql://postgres:pw@db.abcdefg.supabase.co:5432/postgres")
    assert direct["mode"] == "direct" and direct["host"] == "db.abcdefg.supabase.co"
    assert direct["sslmode"] == "require" and "pw" not in json.dumps(direct)

    transaction = connection_summary(
        "postgresql://postgres.abcdefg:pw@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
    )
    assert transaction["mode"] == "transaction-pooler"
    assert transaction["user"] == "postgres.abcdefg"

    session = connection_summary("postgresql://postgres:pw@db.x.supabase.co:5432/postgres")
    assert session["mode"] == "direct"

    session_pooler = connection_summary("postgresql://postgres.x:pw@aws-0.pooler.supabase.com:5432/postgres")
    assert session_pooler["mode"] == "session-pooler"

    unset = connection_summary("")
    assert unset["configured"] is False and unset["host"] == ""


# ---------------------------------------------------------------------------
# 2b. no database configured: honest failures, working health
# ---------------------------------------------------------------------------
def test_unconfigured_store_reports_and_refuses_clearly() -> None:
    store = UnconfiguredStore()
    assert store.configured is False
    status = store.status()
    assert status["configured"] is False
    assert "DATABASE_URL" in status["remediation"]

    for call in (
        lambda: store.list_conversations(client_id=CLIENT),
        lambda: store.create_conversation(client_id=CLIENT),
        lambda: store.get_conversation("id"),
        lambda: store.delete_conversation("id"),
        lambda: store.append_message("id", role="user", content="hi"),
        lambda: store.get_preferences(client_id=CLIENT),
        lambda: store.set_preferences(client_id=CLIENT, preferences={}),
        lambda: store.record_usage(),
        lambda: store.usage_summary(),
        lambda: store.migrate(),
    ):
        with pytest.raises(Exception) as caught:
            call()
        assert getattr(caught.value, "code", "") == "database_not_configured"


def test_api_without_a_database_keeps_health_and_refuses_persistence(config) -> None:
    with no_database_client(config) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["ok"] is True, "health must survive an unconfigured database"
        assert payload["database"]["configured"] is False
        assert payload["database"]["reachable"] is False

        listed = client.get("/api/conversations", params={"client_id": CLIENT})
        assert listed.status_code == 503
        assert listed.json()["error"]["code"] == "database_not_configured"

        created = client.post("/api/conversations", json={"client_id": CLIENT})
        assert created.status_code == 503
        assert created.json()["error"]["code"] == "database_not_configured"

        status = client.get("/api/database").json()
        assert status["ok"] is True
        assert status["database"]["configured"] is False

        migrated = client.post("/api/database/migrate", json={})
        assert migrated.status_code == 503
        assert migrated.json()["error"]["code"] == "database_not_configured"

        # No fabricated history anywhere: every read path fails the same way.
        for path, kwargs in (
            ("/api/conversations", {"params": {"client_id": CLIENT}}),
            ("/api/preferences", {"params": {"client_id": CLIENT}}),
            ("/api/usage", {"params": {"client_id": CLIENT}}),
        ):
            response = client.get(path, **kwargs)
            assert response.status_code == 503, path
            assert response.json()["error"]["code"] == "database_not_configured", path


def test_chat_states_whether_it_was_persisted(config) -> None:
    with no_database_client(config, with_engine=True) as client:
        # No client_id: answered, explicitly not stored.
        plain = client.post("/api/chat", json={"message": "hello"}).json()
        assert plain["ok"] is True and plain["text"]
        assert plain["persistence"]["persisted"] is False
        assert "client_id" in plain["persistence"]["detail"]

        # client_id but no database: the answer stands, persistence is refused.
        with_id = client.post(
            "/api/chat", json={"message": "hello", "client_id": CLIENT}
        ).json()
        assert with_id["ok"] is True and with_id["text"] == plain["text"]
        assert with_id["persistence"]["requested"] is True
        assert with_id["persistence"]["persisted"] is False
        assert with_id["persistence"]["error"]["code"] == "database_not_configured"

        # persist=false is honoured too.
        opted_out = client.post(
            "/api/chat",
            json={"message": "hello", "client_id": CLIENT, "persist": False},
        ).json()
        assert opted_out["persistence"] == {
            "requested": False,
            "persisted": False,
            "detail": "persist=false",
        }


def test_persistence_routes_reject_a_bad_client_id(config) -> None:
    with memory_client(config) as client:
        response = client.get("/api/conversations", params={"client_id": "bad id!"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# 2c. API contract with a store (in-memory)
# ---------------------------------------------------------------------------
def test_chat_persists_a_turn_and_history_is_retrievable(config) -> None:
    store = MemoryStore()
    with memory_client(config, store=store) as client:
        first = client.post("/api/chat", json={"message": "what are you?", "client_id": CLIENT}).json()
        assert first["persistence"]["persisted"] is True
        conversation_id = first["persistence"]["conversation_id"]

        # The same thread continues when the id is sent back.
        second = client.post(
            "/api/chat",
            json={"message": "again", "client_id": CLIENT, "conversation_id": conversation_id},
        ).json()
        assert second["persistence"]["conversation_id"] == conversation_id

        listed = client.get("/api/conversations", params={"client_id": CLIENT}).json()
        assert listed["count"] == 1
        assert listed["conversations"][0]["id"] == conversation_id
        assert listed["conversations"][0]["message_count"] == 4
        assert listed["persistence"]["configured"] is True

        detail = client.get(f"/api/conversations/{conversation_id}", params={"client_id": CLIENT}).json()
        roles = [message["role"] for message in detail["conversation"]["messages"]]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert detail["conversation"]["title"] == "what are you?"
        assistant = detail["conversation"]["messages"][1]
        assert assistant["content"] == "alphaai test reply"
        assert assistant["engine_id"] == "alphaai-test"
        assert assistant["usage"]["total_tokens"] == 15

        # Usage was recorded from the engine's own measurement.
        usage = client.get("/api/usage", params={"client_id": CLIENT}).json()
        assert usage["totals"]["generations"] == 2

        # Another client sees nothing, and cannot read this thread.
        assert client.get("/api/conversations", params={"client_id": OTHER_CLIENT}).json()["count"] == 0
        assert client.get(
            f"/api/conversations/{conversation_id}", params={"client_id": OTHER_CLIENT}
        ).status_code == 404

        # Deleting removes it for good.
        assert client.delete(
            f"/api/conversations/{conversation_id}", params={"client_id": CLIENT}
        ).json()["ok"] is True
        assert client.get(
            f"/api/conversations/{conversation_id}", params={"client_id": CLIENT}
        ).status_code == 404


def test_unknown_conversation_id_is_reported_not_ignored(config) -> None:
    store = MemoryStore()
    with memory_client(config, store=store) as client:
        body = client.post(
            "/api/chat",
            json={"message": "hi", "client_id": CLIENT, "conversation_id": "00000000-0000-4000-8000-000000000999"},
        ).json()
    assert body["ok"] is True, "the answer still arrives"
    assert body["persistence"]["persisted"] is False
    assert body["persistence"]["error"]["code"] == "conversation_not_found"


def test_streaming_chat_emits_a_persisted_event(config) -> None:
    store = MemoryStore()
    with memory_client(config, store=store) as client:
        response = client.post(
            "/api/chat/stream", json={"message": "stream please", "client_id": CLIENT}
        )
        assert response.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    kinds = [event.get("type") for event in events]
    assert "delta" in kinds and kinds[-1] == "persisted"
    persisted = events[-1]
    assert persisted["persisted"] is True
    assert persisted["conversation_id"]
    # The streamed text is what was stored (no invented completion).
    stored = store.messages[-1]
    assert stored["role"] == "assistant"
    assert stored["content"] == "alphaai test reply"
    # A streamed turn records what the engine actually reported, and never an
    # estimated prompt/completion token count.
    assert stored["usage"]["source"] == "stream"
    assert "prompt_tokens" not in stored["usage"]
    assert "completion_tokens" not in stored["usage"]


def test_store_status_and_migrate_round_trip(config) -> None:
    store = MemoryStore()
    with memory_client(config, store=store) as client:
        status = client.get("/api/database").json()
        assert status["database"]["mode"] == "memory"
        assert status["config"]["configured"] is False  # config has no URL: the store was injected
        migration = client.post("/api/database/migrate", json={}).json()
        assert migration["ok"] is True


def test_preferences_round_trip_through_the_api(config) -> None:
    store = MemoryStore()
    with memory_client(config, store=store) as client:
        fresh = client.get("/api/preferences", params={"client_id": CLIENT}).json()
        assert fresh["preferences"] == {} and fresh["stored"] is False

        saved = client.put(
            "/api/preferences",
            json={"client_id": CLIENT, "preferences": {"engine_id": "alphaai-test", "stream": True}},
        ).json()
        assert saved["stored"] is True

        read = client.get("/api/preferences", params={"client_id": CLIENT}).json()
        assert read["preferences"] == {"engine_id": "alphaai-test", "stream": True}


# ---------------------------------------------------------------------------
# 2d. gateway deployments
# ---------------------------------------------------------------------------
def test_gateway_without_a_database_forwards_history_to_the_inference_host(config) -> None:
    app = create_app(config, inference_url="https://inference.invalid")
    with TestClient(app) as client:
        response = client.get("/api/conversations", params={"client_id": CLIENT})
    # Not registered locally: the catch-all proxies it, and the unreachable host is
    # reported honestly rather than as an empty history.
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "inference_unreachable"


def test_gateway_with_a_database_serves_history_itself(config) -> None:
    config.database.url = "postgresql://alphaai:pw@aws-0.pooler.supabase.com:6543/postgres"
    app = create_app(config, inference_url="https://inference.invalid")
    # Never connects in these tests: the routes answer from an injected store.
    app.state.database = MemoryStore()
    with TestClient(app) as client:
        created = client.post(
            "/api/conversations", json={"client_id": CLIENT, "title": "gateway thread"}
        ).json()
        assert created["ok"] is True
        listed = client.get("/api/conversations", params={"client_id": CLIENT}).json()
    assert listed["count"] == 1
    assert listed["conversations"][0]["title"] == "gateway thread"


def test_gateway_health_names_both_databases(config) -> None:
    """The gateway's own database block never collides with the host's."""

    from alphaai.api.app import database_health_decorator

    decorate = database_health_decorator(MemoryStore())
    payload = decorate({"ok": True, "database": {"configured": True, "host": "inference-host"}})
    assert payload["inference_database"]["host"] == "inference-host"
    assert payload["database"]["mode"] == "memory"


# ---------------------------------------------------------------------------
# 2e. the CLI
# ---------------------------------------------------------------------------
def test_cli_db_status_explains_an_unconfigured_database(tmp_path: Path, capsys) -> None:
    exit_code = cli_main(["--project-root", str(tmp_path), "db", "status"])
    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "configured: False" in printed
    assert "DATABASE_URL" in printed


def test_cli_db_plan_lists_every_migration(tmp_path: Path, capsys) -> None:
    exit_code = cli_main(
        ["--project-root", str(tmp_path), "db", "plan", "--directory", str(REPO_MIGRATIONS_DIR)]
    )
    printed = capsys.readouterr().out
    assert exit_code == 0
    for item in discover_migrations(REPO_MIGRATIONS_DIR):
        assert item.name in printed
    assert "pending" in printed


def test_cli_db_migrate_without_a_database_fails_loudly(tmp_path: Path, capsys) -> None:
    exit_code = cli_main(
        ["--project-root", str(tmp_path), "db", "migrate", "--directory", str(REPO_MIGRATIONS_DIR)]
    )
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "database_not_configured" in output


# ---------------------------------------------------------------------------
# 3. a real PostgreSQL server
# ---------------------------------------------------------------------------
@pytest.fixture
def db_store(config):
    """A migrated store on a real server, with the AlphaAI tables emptied."""

    if not DATABASE_READY:  # pragma: no cover - guarded by the marker below
        pytest.skip(DATABASE_REASON)
    import psycopg

    config.database.url = DATABASE_URL
    config.paths.migrations_dir = str(REPO_MIGRATIONS_DIR)
    store = open_database(config)
    report = store.migrate()
    assert report["ok"] is True, report
    with psycopg.connect(DATABASE_URL) as connection, connection.cursor() as cursor:
        cursor.execute(
            "truncate public.usage_records, public.user_preferences, public.messages, "
            "public.conversations restart identity"
        )
        connection.commit()
    return store


def real_db_client(config):
    runtime = AlphaRuntime.create(config, discover=False)
    runtime.registry.register(FakeEngine(test_spec(), config))
    return TestClient(create_app(config, runtime=runtime))


@requires_database
def test_real_database_reports_its_connection_and_schema(db_store) -> None:
    status = db_store.status()
    assert status["reachable"] is True
    assert status["migrations"]["ledger"] is True
    assert status["migrations"]["applied"] == 3
    assert status["migrations"]["pending"] == []
    assert status["migrations"]["drift"] == []
    assert status["database"]  # a real database name

    # Idempotent: a second run applies nothing and reports no drift.
    second = db_store.migrate()
    assert second["applied"] == []
    assert second["drift"] == []


@requires_database
def test_real_database_conversation_round_trip(db_store) -> None:
    conversation = db_store.create_conversation(client_id=CLIENT, engine_id="alphaai-test")
    assert conversation["id"] and conversation["title"] == "New conversation"

    user = db_store.append_message(
        conversation["id"], role="user", content="Hello AlphaAI", client_id=CLIENT
    )
    assistant = db_store.append_message(
        conversation["id"],
        role="assistant",
        content="Real stored answer",
        client_id=CLIENT,
        engine_id="alphaai-test",
        model="Test-Model",
        finish_reason="stop",
        latency_ms=42.5,
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "source": "engine"},
    )
    assert user["id"] != assistant["id"]

    fetched = db_store.get_conversation(conversation["id"], client_id=CLIENT)
    assert fetched is not None
    assert [message["role"] for message in fetched["messages"]] == ["user", "assistant"]
    assert fetched["messages"][1]["usage"]["total_tokens"] == 7
    assert fetched["messages"][1]["latency_ms"] == pytest.approx(42.5)
    assert fetched["title"] == "Hello AlphaAI", "the thread titles itself from the first question"
    assert fetched["messages"][0]["created_at"]  # timestamps are real

    listed = db_store.list_conversations(client_id=CLIENT)
    assert [item["id"] for item in listed] == [conversation["id"]]
    assert listed[0]["message_count"] == 2
    assert listed[0]["preview"] == "Hello AlphaAI"

    db_store.record_usage(
        client_id=CLIENT,
        conversation_id=conversation["id"],
        engine_id="alphaai-test",
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        latency_ms=42.5,
    )
    summary = db_store.usage_summary(client_id=CLIENT)
    assert summary["totals"]["generations"] == 1
    assert summary["totals"]["total_tokens"] == 7
    assert summary["recent"][0]["model"] is None

    db_store.set_preferences(client_id=CLIENT, preferences={"theme": "black-gold"})
    stored = db_store.get_preferences(client_id=CLIENT)
    assert stored["preferences"] == {"theme": "black-gold"} and stored["stored"] is True

    assert db_store.delete_conversation(conversation["id"], client_id=CLIENT) is True
    assert db_store.get_conversation(conversation["id"], client_id=CLIENT) is None
    assert db_store.list_conversations(client_id=CLIENT) == []


@requires_database
def test_real_database_isolates_clients(db_store) -> None:
    conversation = db_store.create_conversation(client_id=CLIENT, title="private")
    db_store.append_message(conversation["id"], role="user", content="secret", client_id=CLIENT)

    assert db_store.get_conversation(conversation["id"], client_id=OTHER_CLIENT) is None
    assert db_store.list_conversations(client_id=OTHER_CLIENT) == []
    assert db_store.delete_conversation(conversation["id"], client_id=OTHER_CLIENT) is False
    with pytest.raises(DatabaseNotFoundError):
        db_store.append_message(
            conversation["id"], role="user", content="hijack", client_id=OTHER_CLIENT
        )
    # An anonymous read cannot even ask for it without a client id.
    assert db_store.list_conversations(client_id="client-other-2") == []


@requires_database
def test_real_database_checks_ids_roles_and_statements(db_store) -> None:
    from alphaai.core.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError):
        db_store.get_conversation("not-a-uuid")
    with pytest.raises(InvalidRequestError):
        db_store.list_conversations(client_id="bad id!")

    conversation = db_store.create_conversation(client_id=CLIENT)
    with pytest.raises(InvalidRequestError):
        db_store.append_message(conversation["id"], role="root", content="hi", client_id=CLIENT)

    # Migrating without migration files is reported, not silently "done".
    with pytest.raises(Exception) as caught:
        db_store.migrate(directory="")
    assert getattr(caught.value, "code", "") == "database_unavailable"
    assert "No migration files found" in str(caught.value)


@requires_database
def test_real_database_row_level_security_default_denies(db_store) -> None:
    """RLS is on for every table, and a non-owner role sees nothing."""

    import psycopg

    conversation = db_store.create_conversation(client_id=CLIENT, title="private")
    db_store.append_message(conversation["id"], role="user", content="private", client_id=CLIENT)

    with psycopg.connect(DATABASE_URL) as connection, connection.cursor() as cursor:
        cursor.execute(
            "select c.relname from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where n.nspname = 'public' and c.relrowsecurity"
        )
        protected = {row[0] for row in cursor.fetchall()}
        assert {"conversations", "messages", "user_preferences", "usage_records"} <= protected

        # A role with table grants but no policy must still see nothing (the
        # Supabase `anon` situation: default deny under row level security).
        cursor.execute("do $$ begin if not exists (select 1 from pg_roles where rolname = 'alphaai_test_anon') then create role alphaai_test_anon nologin; end if; end $$;")
        cursor.execute("grant usage on schema public to alphaai_test_anon")
        cursor.execute("grant select on public.conversations, public.messages to alphaai_test_anon")
        connection.commit()

        cursor.execute("select count(*) from public.conversations")
        assert cursor.fetchone()[0] >= 1, "the owner sees its rows (owner bypasses RLS)"
        cursor.execute("set role alphaai_test_anon")
        cursor.execute("select count(*) from public.conversations")
        assert cursor.fetchone()[0] == 0
        cursor.execute("select count(*) from public.messages")
        assert cursor.fetchone()[0] == 0
        cursor.execute("reset role")


@requires_database
def test_real_database_chat_is_persisted_and_readable(config) -> None:
    """The whole path: HTTP chat -> PostgreSQL -> history over the API."""

    import psycopg

    config.database.url = DATABASE_URL
    config.paths.migrations_dir = str(REPO_MIGRATIONS_DIR)
    open_database(config).migrate()

    with real_db_client(config) as client:
        with psycopg.connect(DATABASE_URL) as connection, connection.cursor() as cursor:
            cursor.execute(
                "truncate public.usage_records, public.user_preferences, public.messages, "
                "public.conversations restart identity"
            )
            connection.commit()

        health = client.get("/api/health").json()
        assert health["database"]["configured"] is True
        assert health["database"]["reachable"] is True
        assert health["database"]["migrations"]["applied"] == 3

        answer = client.post("/api/chat", json={"message": "persist me", "client_id": CLIENT}).json()
        assert answer["persistence"]["persisted"] is True
        conversation_id = answer["persistence"]["conversation_id"]

        streamed = client.post(
            "/api/chat/stream",
            json={"message": "and me", "client_id": CLIENT, "conversation_id": conversation_id},
        )
        events = [
            json.loads(line[len("data: ") :])
            for line in streamed.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert events[-1]["type"] == "persisted"
        assert events[-1]["persisted"] is True
        assert events[-1]["conversation_id"] == conversation_id

        detail = client.get(
            f"/api/conversations/{conversation_id}", params={"client_id": CLIENT}
        ).json()["conversation"]
        assert [message["role"] for message in detail["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert detail["messages"][1]["content"] == "alphaai test reply"

        # A client with no id gets an answer and no stored thread.
        anonymous = client.post("/api/chat", json={"message": "no client id"}).json()
        assert anonymous["persistence"]["persisted"] is False
        assert client.get("/api/conversations", params={"client_id": CLIENT}).json()["count"] == 1

        # Deleting the thread removes it from the database for good.
        assert client.delete(
            f"/api/conversations/{conversation_id}", params={"client_id": CLIENT}
        ).json()["ok"] is True
        assert client.get("/api/conversations", params={"client_id": CLIENT}).json()["count"] == 0

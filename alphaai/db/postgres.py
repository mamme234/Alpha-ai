"""PostgreSQL persistence for AlphaAI (Supabase-hosted or self-hosted).

Why a direct ``DATABASE_URL`` connection, and not the Supabase API:

* AlphaAI's FastAPI server is the only component that talks to the database. It
  uses the ordinary PostgreSQL wire protocol, so the schema, the migrations and
  the queries are portable to any PostgreSQL — and no Supabase key of any kind
  is needed server-side (``SUPABASE_SERVICE_ROLE_KEY`` is *not* used, and the
  dashboard never sees a Supabase URL or key).
* The connection is created per operation and closed again, with the transaction
  pooler in mind: the store sets no session state it depends on and uses
  ``prepare_threshold=None`` so nothing breaks when Supavisor (port 6543) hands
  the next statement to a different backend.
* Row level security is still enabled on every table (see the migrations).
  AlphaAI connects as the database owner, which owns the tables and therefore is
  not restricted by the policies; they exist so that the Data API (PostgREST)
  with an anon key can never read someone's conversations.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Iterator, Sequence

from ..config.schema import AlphaAIConfig
from ..core.errors import (
    DatabaseNotFoundError,
    DatabaseUnavailableError,
    InvalidRequestError,
)
from .base import (
    DEFAULT_CONVERSATION_TITLE,
    MIGRATIONS_TABLE,
    connection_summary,
    validate_client_id,
    validate_message_role,
)
from .migrate import discover_migrations, plan_migrations, run_migrations

logger = logging.getLogger("alphaai.db")

#: Columns every conversation read returns, in one place so responses match.
_CONVERSATION_COLUMNS = """
    c.id::text as id,
    c.client_id,
    c.user_id::text as user_id,
    c.title,
    c.session_id,
    c.engine_id,
    c.model,
    c.metadata,
    c.created_at,
    c.updated_at
"""

_MESSAGE_COLUMNS = """
    m.id,
    m.conversation_id::text as conversation_id,
    m.role,
    m.content,
    m.engine_id,
    m.model,
    m.finish_reason,
    m.latency_ms,
    m.usage,
    m.metadata,
    m.created_at
"""

# -- statements -------------------------------------------------------------
SELECT_CONVERSATIONS = f"""
select {_CONVERSATION_COLUMNS},
    (select count(*) from public.messages m where m.conversation_id = c.id) as message_count,
    (select m.content from public.messages m
      where m.conversation_id = c.id and m.role = 'user'
      order by m.created_at, m.id limit 1) as first_user_message,
    (select m.content from public.messages m
      where m.conversation_id = c.id
      order by m.created_at desc, m.id desc limit 1) as last_message
from public.conversations c
where c.client_id = %s
order by c.updated_at desc
limit %s offset %s
"""

SELECT_CONVERSATION = f"""
select {_CONVERSATION_COLUMNS}
from public.conversations c
where c.id = %s::uuid and (%s::text is null or c.client_id = %s::text)
"""

SELECT_MESSAGES = f"""
select {_MESSAGE_COLUMNS}
from public.messages m
where m.conversation_id = %s::uuid
order by m.created_at, m.id
"""

INSERT_CONVERSATION = f"""
insert into public.conversations (id, client_id, title, session_id, engine_id, model, metadata)
values (coalesce(%s::uuid, gen_random_uuid()), %s, %s, %s, %s, %s, %s::jsonb)
returning
    id::text as id, client_id, user_id::text as user_id, title, session_id,
    engine_id, model, metadata, created_at, updated_at
"""

TOUCH_CONVERSATION = """
update public.conversations
set updated_at = now(),
    title = case
        when title = %s and %s = 'user' then left(%s, 120)
        else title
    end
where id = %s::uuid
"""

INSERT_MESSAGE = f"""
insert into public.messages
    (conversation_id, role, content, engine_id, model, finish_reason, latency_ms, usage, metadata)
values (%s::uuid, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
returning {_MESSAGE_COLUMNS.replace("m.", "")}
"""

DELETE_CONVERSATION = """
delete from public.conversations
where id = %s::uuid and (%s::text is null or client_id = %s::text)
returning id::text as id
"""

SELECT_PREFERENCES = """
select client_id, preferences, updated_at from public.user_preferences where client_id = %s
"""

UPSERT_PREFERENCES = """
insert into public.user_preferences (client_id, preferences, updated_at)
values (%s, %s::jsonb, now())
on conflict (client_id) do update
    set preferences = excluded.preferences, updated_at = now()
returning client_id, preferences, updated_at
"""

INSERT_USAGE = """
insert into public.usage_records
    (client_id, conversation_id, engine_id, model, prompt_tokens, completion_tokens,
     total_tokens, latency_ms, metadata)
values (%s, %s::uuid, %s, %s, %s, %s, %s, %s, %s::jsonb)
returning id, client_id, conversation_id::text as conversation_id, engine_id, model,
          prompt_tokens, completion_tokens, total_tokens, latency_ms, metadata, created_at
"""

USAGE_TOTALS = """
select count(*) as generations,
       coalesce(sum(prompt_tokens), 0) as prompt_tokens,
       coalesce(sum(completion_tokens), 0) as completion_tokens,
       coalesce(sum(total_tokens), 0) as total_tokens,
       max(created_at) as last_at
from public.usage_records
where (%s::text is null or client_id = %s::text)
"""

USAGE_RECENT = """
select id, client_id, conversation_id::text as conversation_id, engine_id, model,
       prompt_tokens, completion_tokens, total_tokens, latency_ms, metadata, created_at
from public.usage_records
where (%s::text is null or client_id = %s::text)
order by created_at desc
limit %s
"""


def _driver() -> tuple[Any, Any]:
    """Import psycopg, or explain exactly how to install it."""

    try:  # pragma: no cover - exercised by the uninstalled-driver test
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:  # noqa: BLE001 - ImportError subclasses vary
        raise DatabaseUnavailableError(
            f"PostgreSQL driver is not available: {exc}",
            remediation=(
                "Install AlphaAI's database extra (`pip install 'alphaai[postgres]'` or "
                "`pip install 'psycopg[binary]>=3.1'`) on the server that owns DATABASE_URL."
            ),
            details={"driver": "psycopg"},
        ) from exc
    return psycopg, dict_row


def _jsonable(value: Any) -> Any:
    """Convert one database value into something ``json.dumps`` handles."""

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: _jsonable(value) for key, value in dict(row).items()}


def _rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [item for item in (_row(dict(row)) for row in rows) if item is not None]


def as_uuid(value: str, *, field: str = "conversation_id") -> str:
    """Validate a client-supplied id before it reaches SQL."""

    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidRequestError(
            f"{field} must be a UUID.",
            details={field: str(value)[:64]},
        ) from exc


def _dump(value: Any, default: Any) -> str:
    return json.dumps(default if value is None else value, ensure_ascii=False, default=str)


class PostgresDatabase:
    """A :class:`~alphaai.db.base.ConversationStore` backed by PostgreSQL."""

    configured = True

    def __init__(self, config: AlphaAIConfig) -> None:
        self.config = config
        self.settings = config.database
        self.url = self.settings.url

    # -- connection ------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[Any]:
        """Open a connection, or raise a structured, actionable error.

        Nothing is cached across requests on purpose: a serverless deployment
        runs one request per process and the Supavisor transaction pooler is
        designed for short-lived connections.
        """

        psycopg, dict_row = _driver()
        kwargs: dict[str, Any] = {
            "row_factory": dict_row,
            "connect_timeout": int(self.settings.connect_timeout_s),
            "application_name": self.settings.application_name,
            # Supavisor's transaction mode may hand the next statement to a
            # different backend; a named prepared statement would then be
            # missing, so AlphaAI never relies on one.
            "prepare_threshold": None,
        }
        if "sslmode=" not in self.url:
            kwargs["sslmode"] = self.settings.ssl_mode
        try:
            connection = psycopg.connect(self.url, **kwargs)
        except Exception as exc:  # noqa: BLE001 - reported as a database failure
            raise self._connection_error(exc) from exc
        try:
            with connection.cursor() as cursor:
                cursor.execute("select set_config('statement_timeout', %s, false)", (str(int(self.settings.statement_timeout_ms)),))
            yield connection
        finally:
            connection.close()

    def _connection_error(self, exc: Exception) -> DatabaseUnavailableError:
        summary = connection_summary(self.url, default_sslmode=self.settings.ssl_mode)
        sqlstate = str(getattr(exc, "sqlstate", "") or "")
        hints = {
            "28P01": "The password in DATABASE_URL is wrong.",
            "3D000": "That database name does not exist. Supabase calls it `postgres`.",
            "08006": "The host refused the connection.",
            "08001": "The host could not be reached (check the project ref and network).",
        }
        hint = hints.get(sqlstate, "Check the Supabase connection string (host, port, password) and that the project is running.")
        return DatabaseUnavailableError(
            f"Could not connect to PostgreSQL at {summary.get('host') or '<unset>'}: "
            f"{type(exc).__name__}: {exc}",
            remediation=hint,
            details={"host": summary.get("host"), "mode": summary.get("mode"), "sqlstate": sqlstate},
        )

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """One statement batch inside one transaction, committed on success."""

        with self._connect() as connection:
            with connection.cursor() as cursor:
                yield cursor
            connection.commit()

    def run(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute one statement and return its rows.

        Used for single-statement work (health checks, the migration ledger);
        each call is its own transaction.
        """

        return self._execute(sql, params)

    @contextmanager
    def migration_executor(self) -> Iterator["_ConnectionExecutor"]:
        """An executor bound to **one** connection for a whole migration run.

        This is what makes a migration file atomic: every statement in the file
        runs on the same connection, ``commit()`` is the durable point, and any
        failure rolls the file back before the error is reported.
        """

        with self._connect() as connection:
            executor = _ConnectionExecutor(connection)
            try:
                yield executor
            except Exception:
                connection.rollback()
                raise
            finally:
                pass

    def _execute(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        try:
            with self._cursor() as cursor:
                cursor.execute(sql, tuple(params) if params else None)
                if cursor.description is None:
                    return []
                return _rows(cursor.fetchall())
        except (InvalidRequestError, DatabaseNotFoundError):
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a database failure
            raise self._query_error(sql, exc) from exc

    def _query_error(self, sql: str, exc: Exception) -> DatabaseUnavailableError:
        sqlstate = str(getattr(exc, "sqlstate", "") or "")
        statement = " ".join(sql.split())[:120]
        if sqlstate in {"42P01", "42703"}:
            return DatabaseUnavailableError(
                f"The AlphaAI schema is not applied ({type(exc).__name__}: {exc}).",
                remediation=(
                    "Apply the migrations: `alphaai db migrate` (or `supabase db push`) "
                    "against this DATABASE_URL."
                ),
                details={"statement": statement, "sqlstate": sqlstate},
            )
        if sqlstate == "23505":
            return DatabaseUnavailableError(
                f"A row with that identifier already exists ({exc}).",
                remediation="Retry without an explicit conversation_id.",
                details={"statement": statement, "sqlstate": sqlstate},
            )
        if sqlstate == "23514":
            return DatabaseUnavailableError(
                f"A stored value violates an AlphaAI schema check constraint ({exc}).",
                remediation="Shorten the value (titles are <= 200 characters).",
                details={"statement": statement, "sqlstate": sqlstate},
            )
        return DatabaseUnavailableError(
            f"PostgreSQL rejected the AlphaAI query ({type(exc).__name__}: {exc}).",
            remediation=(
                "This is a real database failure; AlphaAI reports it instead of returning "
                "an empty result. Check the database logs and the schema version "
                "(`alphaai db status`)."
            ),
            details={"statement": statement, "sqlstate": sqlstate},
        )

    # -- health ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Connection and schema facts. Never raises — unreachable is a status."""

        info = connection_summary(self.url, default_sslmode=self.settings.ssl_mode)
        base: dict[str, Any] = {
            "configured": True,
            **info,
            "application_name": self.settings.application_name,
            "migrations_dir": getattr(self.config.paths, "migrations_dir", ""),
        }
        try:
            row = _row(
                (self._execute("select version() as server_version, current_database() as database, now() at time zone 'utc' as server_time") or [None])[0]
            )
        except Exception as exc:  # noqa: BLE001 - health must answer
            error = exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}
            return {**base, "reachable": False, "error": error}
        return {**base, "reachable": True, **(row or {}), "migrations": self.migration_state()}

    def migration_state(self) -> dict[str, Any]:
        """Which migrations the database has, which are still pending on disk."""

        state: dict[str, Any] = {"ledger": False, "applied": 0, "latest": None, "pending": [], "drift": []}
        try:
            row = _row(
                (
                    self._execute(
                        f"select count(*) as applied, max(name) as latest from {MIGRATIONS_TABLE}"
                    )
                    or [None]
                )[0]
            ) or {}
            rows = self._execute(f"select name, checksum from {MIGRATIONS_TABLE}")
        except Exception:  # noqa: BLE001 - the ledger may not exist yet
            return state
        state.update({"ledger": True, "applied": int(row.get("applied") or 0), "latest": row.get("latest")})
        try:
            on_disk = discover_migrations(getattr(self.config.paths, "migrations_dir", ""))
        except OSError:  # pragma: no cover - unreadable migrations directory
            return state
        if on_disk:
            applied = {str(item["name"]): str(item["checksum"]) for item in rows}
            pending, drift = plan_migrations(on_disk, applied)
            state["pending"] = [item.name for item in pending]
            state["drift"] = drift
            state["available_on_disk"] = len(on_disk)
        return state

    # -- migrations ------------------------------------------------------
    def migrate(self, *, directory: str | None = None) -> dict[str, Any]:
        # Only ``None`` falls back to configuration: an explicit empty directory
        # is a caller error and is reported as one.
        if directory is None:
            directory = getattr(self.config.paths, "migrations_dir", "")
        migrations = discover_migrations(directory)
        if not migrations:
            raise DatabaseUnavailableError(
                f"No migration files found in {directory or '<unset>'}.",
                remediation=(
                    "Run the migration from a checkout of the AlphaAI repository (the files "
                    "live in supabase/migrations), or point ALPHAI_MIGRATIONS_DIR at them."
                ),
                details={"directory": str(directory)},
            )
        with self.migration_executor() as executor:
            report = run_migrations(executor, migrations)
        report["directory"] = str(directory)
        return report

    # -- conversations ---------------------------------------------------
    def list_conversations(
        self, *, client_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        client_id = validate_client_id(client_id)
        limit = max(1, min(int(limit), self.settings.max_list_limit))
        offset = max(0, int(offset))
        rows = self._execute(SELECT_CONVERSATIONS, (client_id, limit, offset))
        for item in rows:
            preview = item.get("first_user_message") or item.get("last_message") or ""
            item["preview"] = " ".join(str(preview).split())[:160]
            item["message_count"] = int(item.get("message_count") or 0)
            item.pop("first_user_message", None)
            item.pop("last_message", None)
        return rows

    def create_conversation(
        self,
        *,
        client_id: str,
        title: str | None = None,
        session_id: str | None = None,
        engine_id: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        client_id = validate_client_id(client_id)
        explicit = as_uuid(conversation_id) if conversation_id else None
        clean_title = (title or "").strip() or DEFAULT_CONVERSATION_TITLE
        row = self._execute(
            INSERT_CONVERSATION,
            (
                explicit,
                client_id,
                clean_title[:200],
                session_id,
                engine_id,
                model,
                _dump(metadata, {}),
            ),
        )
        return _row(row[0]) if row else {}

    def get_conversation(
        self,
        conversation_id: str,
        *,
        client_id: str | None = None,
        include_messages: bool = True,
    ) -> dict[str, Any] | None:
        identifier = as_uuid(conversation_id)
        owner = validate_client_id(client_id) if client_id else None
        rows = self._execute(SELECT_CONVERSATION, (identifier, owner, owner))
        if not rows:
            return None
        conversation = _row(rows[0]) or {}
        if include_messages:
            conversation["messages"] = self.list_messages(identifier, client_id=client_id)
        return conversation

    def list_messages(self, conversation_id: str, *, client_id: str | None = None) -> list[dict[str, Any]]:
        identifier = as_uuid(conversation_id)
        return self._execute(SELECT_MESSAGES, (identifier,))

    def delete_conversation(self, conversation_id: str, *, client_id: str | None = None) -> bool:
        identifier = as_uuid(conversation_id)
        owner = validate_client_id(client_id) if client_id else None
        rows = self._execute(DELETE_CONVERSATION, (identifier, owner, owner))
        # ``messages`` rows go with it through ON DELETE CASCADE.
        return bool(rows)

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        client_id: str | None = None,
        engine_id: str | None = None,
        model: str | None = None,
        finish_reason: str | None = None,
        latency_ms: float | None = None,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identifier = as_uuid(conversation_id)
        role = validate_message_role(role)
        text = content if isinstance(content, str) else str(content)
        try:
            with self._cursor() as cursor:
                owner = validate_client_id(client_id) if client_id else None
                cursor.execute(SELECT_CONVERSATION, (identifier, owner, owner))
                row = cursor.fetchone()
                if row is None:
                    raise DatabaseNotFoundError(
                        "No conversation with that id exists for this client.",
                        remediation=(
                            "Create the conversation first (POST /api/conversations) or send a "
                            "chat turn without conversation_id to start a new one."
                        ),
                        details={"conversation_id": identifier},
                    )
                cursor.execute(
                    INSERT_MESSAGE,
                    (
                        identifier,
                        role,
                        text,
                        engine_id,
                        model,
                        finish_reason,
                        float(latency_ms) if latency_ms is not None else None,
                        _dump(usage, {}) if usage else None,
                        _dump(metadata, {}),
                    ),
                )
                inserted = cursor.fetchone()
                cursor.execute(
                    TOUCH_CONVERSATION,
                    (DEFAULT_CONVERSATION_TITLE, role, " ".join(text.split())[:120], identifier),
                )
        except (InvalidRequestError, DatabaseNotFoundError):
            raise
        except Exception as exc:  # noqa: BLE001 - mapped like every other query
            raise self._query_error(INSERT_MESSAGE, exc) from exc
        return _row(inserted) or {}

    # -- preferences -----------------------------------------------------
    def get_preferences(self, *, client_id: str) -> dict[str, Any]:
        client_id = validate_client_id(client_id)
        rows = self._execute(SELECT_PREFERENCES, (client_id,))
        if not rows:
            # No row yet is a normal state, not an error: the dashboard falls
            # back to its defaults.
            return {"client_id": client_id, "preferences": {}, "stored": False}
        return {**(_row(rows[0]) or {}), "stored": True}

    def set_preferences(self, *, client_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        client_id = validate_client_id(client_id)
        if not isinstance(preferences, dict):
            raise InvalidRequestError("preferences must be a JSON object.")
        rows = self._execute(UPSERT_PREFERENCES, (client_id, _dump(preferences, {})))
        return _row(rows[0]) if rows else {}

    # -- usage -----------------------------------------------------------
    def record_usage(
        self,
        *,
        client_id: str | None = None,
        conversation_id: str | None = None,
        engine_id: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner = validate_client_id(client_id) if client_id else None
        identifier = as_uuid(conversation_id) if conversation_id else None
        payload = usage or {}

        def tokens(key: str) -> int | None:
            value = payload.get(key)
            return int(value) if isinstance(value, (int, float)) else None

        rows = self._execute(
            INSERT_USAGE,
            (
                owner,
                identifier,
                engine_id,
                model,
                tokens("prompt_tokens"),
                tokens("completion_tokens"),
                tokens("total_tokens"),
                float(latency_ms) if latency_ms is not None else None,
                _dump(metadata, {}),
            ),
        )
        return _row(rows[0]) if rows else {}

    def usage_summary(self, *, client_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        owner = validate_client_id(client_id) if client_id else None
        limit = max(1, min(int(limit), self.settings.max_list_limit))
        totals = (_execute_first(self, USAGE_TOTALS, (owner, owner)) or {})
        recent = self._execute(USAGE_RECENT, (owner, owner, limit))
        return {"totals": totals, "recent": recent}


def _execute_first(store: PostgresDatabase, sql: str, params: Sequence[Any]) -> dict[str, Any] | None:
    rows = store._execute(sql, params)
    return _row(rows[0]) if rows else None


class _ConnectionExecutor:
    """Single-connection executor used by the migration runner.

    Statements run on one cursor in one transaction; :meth:`commit` is the point
    at which an applied migration becomes durable.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._cursor = connection.cursor()

    def run(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        self._cursor.execute(sql, tuple(params) if params else None)
        if self._cursor.description is None:
            return []
        return _rows(self._cursor.fetchall())

    def commit(self) -> None:
        self._connection.commit()


__all__ = [
    "PostgresDatabase",
    "as_uuid",
    "USAGE_RECENT",
    "USAGE_TOTALS",
]

"""AlphaAI persistence contract.

AlphaAI keeps conversation history in a PostgreSQL database (Supabase is the
managed flavour AlphaAI documents). Three rules shape this module:

* **The database is optional.** A deployment without ``DATABASE_URL`` still
  answers every endpoint; only *persistence* is unavailable, and it says so with
  ``database_not_configured`` instead of inventing history.
* **Only the server talks to it.** The dashboard never receives a database URL
  or a Supabase key of any kind — see :func:`connection_summary`.
* **Errors are real errors.** A database that is down is reported as down, not
  masked by an empty list that would look like "no conversations yet".

:class:`ConversationStore` is the interface both the configured PostgreSQL store
and the unconfigured placeholder implement, so the API never branches on which
one it got.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, unquote, urlsplit

from ..core.errors import DatabaseNotConfiguredError, InvalidRequestError

#: Roles AlphaAI stores. Matches the ``messages_role_valid`` check constraint in
#: ``supabase/migrations`` — kept here too so a bad request fails before SQL.
MESSAGE_ROLES = ("system", "user", "assistant", "tool")

#: AlphaAI's own migration ledger (see :mod:`alphaai.db.migrate`). It lives
#: outside the Supabase CLI's ``supabase_migrations`` table on purpose: the CLI
#: would otherwise try to replay files it never recorded.
MIGRATIONS_TABLE = "public.alphaai_migrations"

#: Default first title for a thread, matching the schema default.
DEFAULT_CONVERSATION_TITLE = "New conversation"

#: Characters allowed in a client id. It arrives from the browser, so it is
#: validated before it is ever used as a parameter.
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")


def validate_client_id(client_id: str) -> str:
    """Return a usable client id or raise a clear error.

    ``client_id`` is the anonymous identifier the dashboard generates and
    stores locally. It is *not* authentication: it only scopes rows so two
    browsers cannot read each other's history through AlphaAI's own API.
    """

    value = (client_id or "").strip()
    if not _CLIENT_ID_RE.match(value):
        raise InvalidRequestError(
            "client_id must be 1-200 characters of [A-Za-z0-9._:-].",
            details={"client_id_length": len(value)},
        )
    return value


def validate_message_role(role: str) -> str:
    value = (role or "").strip().lower()
    if value not in MESSAGE_ROLES:
        raise InvalidRequestError(
            f"role must be one of {', '.join(MESSAGE_ROLES)}.",
            details={"role": role, "allowed": list(MESSAGE_ROLES)},
        )
    return value


@dataclass(frozen=True, slots=True)
class ConnectionInfo:
    """Facts about a PostgreSQL connection string that are safe to publish.

    The password is never part of this object. AlphaAI needs the host and mode
    (direct connection vs Supavisor pooler) to report *how* it connects without
    handing anything secret to a browser.
    """

    host: str = ""
    port: int | None = None
    database: str = ""
    user: str = ""
    sslmode: str = ""
    mode: str = "unknown"  # "direct" | "session-pooler" | "transaction-pooler"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "sslmode": self.sslmode,
            "mode": self.mode,
        }


def connection_summary(url: str, *, default_sslmode: str = "require") -> dict[str, Any]:
    """Describe a DSN without leaking its password.

    Recognises Supabase's three shapes so an operator can tell which connection
    method a deployment is actually using:

    ``direct``                ``db.<ref>.supabase.co:5432``
    ``transaction-pooler``    ``*.pooler.supabase.com:6543`` (serverless)
    ``session-pooler``        ``*.pooler.supabase.com:5432`` (long-lived)
    """

    if not url:
        return {"configured": False, **ConnectionInfo(sslmode=default_sslmode).to_dict()}
    try:
        parts = urlsplit(url)
    except ValueError:  # pragma: no cover - urlsplit is very permissive
        return {"configured": True, **ConnectionInfo().to_dict(), "parse_error": True}

    query = parse_qs(parts.query or "")
    sslmode = (query.get("sslmode", [""])[0] or default_sslmode).strip()
    host = parts.hostname or ""
    port = parts.port
    if ".pooler.supabase.com" in host:
        mode = "transaction-pooler" if port == 6543 else "session-pooler"
    elif host:
        mode = "direct"
    else:
        mode = "unknown"

    info = ConnectionInfo(
        host=host,
        port=port,
        database=unquote(parts.path.lstrip("/")),
        user=unquote(parts.username or ""),
        sslmode=sslmode,
        mode=mode,
    )
    return {"configured": True, **info.to_dict()}


def database_public_view(settings: Any) -> dict[str, Any]:
    """A :class:`~alphaai.config.schema.DatabaseConfig`, minus the password.

    ``GET /api/config`` and the dashboard use this: an operator can see *that* a
    database is configured and which connection method it uses, and a client
    never receives a connection string. ``settings`` is duck-typed to keep
    ``alphaai.config.schema`` free of imports into this package.
    """

    url = str(getattr(settings, "url", "") or "")
    ssl_mode = str(getattr(settings, "ssl_mode", "require") or "require")
    return {
        **connection_summary(url, default_sslmode=ssl_mode),
        "configured": bool(url),
        "connect_timeout_s": float(getattr(settings, "connect_timeout_s", 10.0)),
        "statement_timeout_ms": int(getattr(settings, "statement_timeout_ms", 15000)),
        "application_name": str(getattr(settings, "application_name", "alphaai")),
        "max_list_limit": int(getattr(settings, "max_list_limit", 200)),
        "migrate_on_start": bool(getattr(settings, "migrate_on_start", False)),
    }


@runtime_checkable
class ConversationStore(Protocol):
    """Everything AlphaAI persists, behind one interface.

    Implementations must never raise :class:`~alphaai.core.errors.AlphaAIError`
    subclasses other than the database ones defined in ``alphaai.core.errors``,
    so the API can map their failures to HTTP without leaking driver details
    into a 500.
    """

    configured: bool

    def status(self) -> dict[str, Any]:
        """Connection/health facts. Never raises: unreachable is a status."""

    def list_conversations(
        self, *, client_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]: ...

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
    ) -> dict[str, Any]: ...

    def get_conversation(
        self,
        conversation_id: str,
        *,
        client_id: str | None = None,
        include_messages: bool = True,
    ) -> dict[str, Any] | None: ...

    def delete_conversation(self, conversation_id: str, *, client_id: str | None = None) -> bool: ...

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
    ) -> dict[str, Any]: ...

    def get_preferences(self, *, client_id: str) -> dict[str, Any]: ...

    def set_preferences(self, *, client_id: str, preferences: dict[str, Any]) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...

    def usage_summary(self, *, client_id: str | None = None, limit: int = 50) -> dict[str, Any]: ...

    def migrate(self, *, directory: str | None = None) -> dict[str, Any]: ...


class UnconfiguredStore:
    """The store AlphaAI uses when no database is configured.

    It satisfies :class:`ConversationStore` so the API registers exactly the
    same routes, and every persistence call fails with a structured
    ``database_not_configured`` error. :meth:`status` is the one method that
    answers instead of raising: the health endpoint must keep working.
    """

    configured = False

    def __init__(self, *, detail: str = "No database is configured for this process.") -> None:
        self.detail = detail

    # -- health ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "configured": False,
            "reachable": False,
            "detail": self.detail,
            "remediation": (
                "Set DATABASE_URL to a PostgreSQL connection string (Supabase: Project "
                "Settings -> Database -> Connection string -> URI) on the AlphaAI server "
                "that owns the data, then restart it. The dashboard and inference keep "
                "working without it: only conversation history is unavailable."
            ),
        }

    def _unconfigured(self) -> None:
        raise DatabaseNotConfiguredError(
            self.detail,
            remediation=(
                "Configure DATABASE_URL on the AlphaAI API server, or stop requesting "
                "persistence. Nothing was saved and no history was invented."
            ),
        )

    # -- every persistence call -----------------------------------------
    def list_conversations(self, **_kwargs: Any) -> list[dict[str, Any]]:
        self._unconfigured()

    def create_conversation(self, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()

    def get_conversation(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        self._unconfigured()

    def delete_conversation(self, *_args: Any, **_kwargs: Any) -> bool:
        self._unconfigured()

    def append_message(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()

    def get_preferences(self, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()

    def set_preferences(self, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()

    def record_usage(self, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()

    def usage_summary(self, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()

    def migrate(self, **_kwargs: Any) -> dict[str, Any]:
        self._unconfigured()


__all__ = [
    "ConnectionInfo",
    "ConversationStore",
    "DEFAULT_CONVERSATION_TITLE",
    "MESSAGE_ROLES",
    "MIGRATIONS_TABLE",
    "UnconfiguredStore",
    "connection_summary",
    "database_public_view",
    "validate_client_id",
    "validate_message_role",
]

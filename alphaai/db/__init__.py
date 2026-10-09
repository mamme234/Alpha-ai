"""AlphaAI persistence: PostgreSQL (Supabase) conversations, preferences and usage.

The package exposes one factory, :func:`open_database`, which returns either a
configured :class:`~alphaai.db.postgres.PostgresDatabase` or an
:class:`~alphaai.db.base.UnconfiguredStore` that answers every persistence call
with a structured ``database_not_configured`` error. Callers therefore never
branch on "is there a database?" — the API registers the same routes either way.

Schema and migrations live in ``supabase/migrations`` (see
:mod:`alphaai.db.migrate`). No model weights and no secrets are ever stored here.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import AlphaAIConfig
from .base import (
    DEFAULT_CONVERSATION_TITLE,
    MESSAGE_ROLES,
    MIGRATIONS_TABLE,
    ConversationStore,
    UnconfiguredStore,
    connection_summary,
    validate_client_id,
    validate_message_role,
)
from .migrate import (
    DEFAULT_MIGRATIONS_DIR,
    Migration,
    checksum_sql,
    discover_migrations,
    plan_migrations,
    run_migrations,
    split_sql_statements,
)

__all__ = [
    "ConversationStore",
    "DEFAULT_CONVERSATION_TITLE",
    "DEFAULT_MIGRATIONS_DIR",
    "MESSAGE_ROLES",
    "MIGRATIONS_TABLE",
    "Migration",
    "open_database",
    "checksum_sql",
    "connection_summary",
    "discover_migrations",
    "plan_migrations",
    "run_migrations",
    "split_sql_statements",
    "UnconfiguredStore",
    "validate_client_id",
    "validate_message_role",
]


def open_database(config: AlphaAIConfig) -> ConversationStore:
    """Return the store AlphaAI should use for this configuration.

    No connection is opened here: the store connects per operation, so a
    deployment whose database is temporarily unreachable still starts, serves
    inference, and reports the database as unreachable instead of refusing to
    boot.
    """

    if not getattr(config, "database", None) or not config.database.url:
        return UnconfiguredStore()
    from .postgres import PostgresDatabase

    return PostgresDatabase(config)

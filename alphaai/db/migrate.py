"""Version-controlled SQL migrations.

AlphaAI's schema lives in plain ``.sql`` files under ``supabase/migrations`` (the
directory the Supabase CLI uses), so there is exactly one source of truth
whether the schema is applied with ``supabase db push`` or with AlphaAI's own
runner (``alphaai db migrate``).

The runner is deliberately small and explicit:

* Files are applied in filename order (``YYYYMMDDHHMMSS_name.sql``).
* Every applied file is recorded by name **and SHA-256 checksum** in
  ``public.alphaai_migrations``. A file that changed after it was applied is
  reported as drift instead of being silently re-run.
* Each file runs as one transaction together with its ledger row, so a failure
  leaves the database at the last fully applied migration.
* Nothing here ever creates or drops AlphaAI's own tables implicitly: the SQL
  files are the only place the schema is defined.

This module has no PostgreSQL dependency: it works against any executor that can
run a statement and return rows, which is what makes it testable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ..core.errors import DatabaseUnavailableError
from .base import MIGRATIONS_TABLE

#: ``20261009120000_alphaai_conversations.sql`` — a sortable version prefix and a
#: snake_case name. Files that do not match are ignored (never run blind).
MIGRATION_FILENAME_RE = re.compile(r"^(?P<version>\d{8,14})_(?P<name>[A-Za-z0-9_]+)\.sql$")

DEFAULT_MIGRATIONS_DIR = "supabase/migrations"


class MigrationExecutor(Protocol):
    """The two operations the runner needs from a database connection.

    ``run`` executes exactly one statement (the runner splits the files itself,
    so a failure names the statement that failed) and ``commit`` is the durable
    point the runner calls after each fully applied file.
    """

    def run(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute one statement and return its rows."""

    def commit(self) -> None:
        """Make everything since the last commit durable."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One migration file, read from disk."""

    name: str
    path: Path
    sql: str
    checksum: str

    @property
    def version(self) -> str:
        return self.name.split("_", 1)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "bytes": len(self.sql.encode("utf-8")),
        }


_DOLLAR_QUOTE_RE = re.compile(r"\$(?P<tag>[A-Za-z_][A-Za-z0-9_]*)?\$")


def _only_comments(statement: str) -> bool:
    """True when a statement is nothing but comments and blank lines."""

    without_block = re.sub(r"/\*.*?\*/", " ", statement, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    return not without_line.strip()


def split_sql_statements(sql: str) -> list[str]:
    """Split a ``.sql`` file into statements, respecting PostgreSQL quoting.

    A naive ``sql.split(';')`` breaks on a semicolon inside a string literal, an
    identifier or a dollar-quoted function body. The migrations AlphaAI ships
    stay simple, but the runner must not corrupt a future one that is not, so
    this walks the text and only treats ``;`` as a separator when it is outside
    single quotes, double quotes, line comments, block comments and dollar-quoted
    blocks. Comments are kept with the statement that follows them, which makes a
    failing statement easy to find in the log.
    """

    statements: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        pair = sql[index : index + 2]
        if pair == "--":
            end = sql.find("\n", index)
            end = length if end == -1 else end
            buffer.append(sql[index:end])
            index = end
            continue
        if pair == "/*":
            end = sql.find("*/", index + 2)
            end = length if end == -1 else end + 2
            buffer.append(sql[index:end])
            index = end
            continue
        if char == "$":
            match = _DOLLAR_QUOTE_RE.match(sql, index)
            if match:
                tag = match.group(0)
                end = sql.find(tag, match.end())
                end = length if end == -1 else end + len(tag)
                buffer.append(sql[index:end])
                index = end
                continue
        if char in {"'", '"'}:
            # Doubled quotes escape themselves inside a quoted run.
            end = index + 1
            while end < length:
                if sql[end] == char:
                    if end + 1 < length and sql[end + 1] == char:
                        end += 2
                        continue
                    end += 1
                    break
                end += 1
            buffer.append(sql[index:end])
            index = end
            continue
        if char == ";":
            statement = "".join(buffer).strip()
            if statement and not _only_comments(statement):
                statements.append(statement)
            buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1
    tail = "".join(buffer).strip()
    if tail and not _only_comments(tail):
        statements.append(tail)
    return statements


def checksum_sql(sql: str) -> str:
    """Stable content hash. Normalises line endings so a CRLF checkout matches."""

    normalized = sql.replace("\r\n", "\n").strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def discover_migrations(directory: str | Path) -> list[Migration]:
    """Return every migration in ``directory``, ordered as they must be applied."""

    root = Path(directory)
    if not root.is_dir():
        return []
    migrations: list[Migration] = []
    for path in sorted(root.glob("*.sql")):
        if not MIGRATION_FILENAME_RE.match(path.name):
            continue
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(name=path.name, path=path, sql=sql, checksum=checksum_sql(sql))
        )
    migrations.sort(key=lambda item: item.name)
    return migrations


def plan_migrations(
    migrations: Iterable[Migration], applied: Mapping[str, str]
) -> tuple[list[Migration], list[dict[str, str]]]:
    """Split discovered files into ``pending`` and checksum ``drift``.

    ``applied`` maps a migration name to the checksum recorded when it ran.
    A pending file is one that has never been applied; drift is a file that was
    applied with different content, which means the repository and the database
    disagree and a human has to decide.
    """

    pending: list[Migration] = []
    drift: list[dict[str, str]] = []
    for migration in migrations:
        recorded = applied.get(migration.name)
        if recorded is None:
            pending.append(migration)
        elif recorded != migration.checksum:
            drift.append(
                {
                    "name": migration.name,
                    "applied_checksum": recorded,
                    "file_checksum": migration.checksum,
                }
            )
    return pending, drift


LEDGER_SQL = f"""
create table if not exists {MIGRATIONS_TABLE} (
    name text primary key,
    checksum text not null,
    applied_at timestamptz not null default now()
)
"""


def ensure_ledger(executor: MigrationExecutor) -> None:
    executor.run(LEDGER_SQL)


def applied_migrations(executor: MigrationExecutor) -> dict[str, str]:
    rows = executor.run(f"select name, checksum from {MIGRATIONS_TABLE}")
    return {str(row["name"]): str(row["checksum"]) for row in rows}


def run_migrations(
    executor: MigrationExecutor,
    migrations: Sequence[Migration],
    *,
    applied: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply ``migrations`` that are not applied yet. Returns a report.

    The report is the same shape whether or not anything was applied, so a caller
    (CLI, API, health check) can print it verbatim.
    """

    ensure_ledger(executor)
    # Its own durable point: the ledger exists even when nothing needs applying.
    executor.commit()
    recorded = dict(applied) if applied is not None else applied_migrations(executor)
    pending, drift = plan_migrations(migrations, recorded)

    applied_now: list[str] = []
    skipped: list[str] = []
    for migration in pending:
        statements = split_sql_statements(migration.sql)
        for position, statement in enumerate(statements, start=1):
            try:
                executor.run(statement)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise DatabaseUnavailableError(
                    f"Migration {migration.name} failed at statement {position} of "
                    f"{len(statements)}: {exc}",
                    remediation=(
                        "Fix the migration (or restore the database to the previous state) "
                        "and run `alphaai db migrate` again. Files are applied in filename "
                        "order; each file is one transaction and its statements are "
                        "idempotent, so a retry is safe."
                    ),
                    details={
                        "migration": migration.name,
                        "statement": position,
                        "sql": " ".join(statement.split())[:200],
                    },
                ) from exc
        # The file and its ledger row become durable together, so a crash can
        # never leave a file applied but unrecorded (or recorded but unapplied).
        executor.run(
            f"insert into {MIGRATIONS_TABLE} (name, checksum) values (%s, %s) "
            f"on conflict (name) do nothing",
            (migration.name, migration.checksum),
        )
        executor.commit()
        applied_now.append(migration.name)

    if applied is not None:
        # A caller that supplied the ledger already knows what is applied; only
        # files inside this set were considered, so nothing else is reported as
        # skipped.
        skipped = [item.name for item in migrations if item.name not in {p.name for p in pending}]
    else:
        skipped = sorted(recorded)

    return {
        "ok": not drift,
        "applied": applied_now,
        "already_applied": skipped,
        "drift": drift,
        "total": len(migrations),
    }


__all__ = [
    "DEFAULT_MIGRATIONS_DIR",
    "LEDGER_SQL",
    "MIGRATION_FILENAME_RE",
    "Migration",
    "MigrationExecutor",
    "applied_migrations",
    "checksum_sql",
    "discover_migrations",
    "ensure_ledger",
    "plan_migrations",
    "run_migrations",
    "split_sql_statements",
]

# AlphaAI + PostgreSQL (Supabase)

AlphaAI keeps conversation history in **PostgreSQL**. Supabase is the managed
PostgreSQL AlphaAI documents, and any other PostgreSQL server works the same
way — the schema, the migrations and the queries are ordinary SQL.

Two things this page is explicit about, because getting them wrong causes most
of the confusion:

* **Supabase is the database, not an inference server.** It never runs the model.
  The model still runs on a host with a persistent disk (see
  [`DEPLOY.md`](./DEPLOY.md)); inference is llama.cpp + a GGUF model, unchanged.
* **AlphaAI does not use Supabase's API or its keys.** The FastAPI server
  connects with a PostgreSQL connection string (`DATABASE_URL`) over the normal
  wire protocol. There is no anon key and no service-role key anywhere in the
  deployment, and the browser never talks to Supabase.

---

## 1. What is stored

Four tables, all in `public`, created by the migrations in
[`supabase/migrations`](../supabase/migrations):

| Table | What it holds | Notes |
| ----- | ------------- | ----- |
| `conversations` | one row per chat thread: `client_id`, `title`, `session_id`, `engine_id`, `model`, `metadata` (jsonb), timestamps | titles itself from the first question (`left(content, 120)`) |
| `messages` | every turn: `role`, `content`, `engine_id`, `model`, `finish_reason`, `latency_ms`, `usage` (jsonb), `metadata` (jsonb), `created_at` | `on delete cascade` from its conversation |
| `user_preferences` | one row per `client_id`: free-form `preferences` (jsonb) | optional; the dashboard works without it |
| `usage_records` | one row per completed generation with the engine's own token counts and latency | kept separate so retention can differ |

Design rules:

* **No model weights, ever.** Weights live on the inference host's disk (or in a
  private model registry). The database stores text, ids, small jsonb objects
  and token counts.
* `metadata`/`usage` are `jsonb` so new AlphaAI fields never need a migration.
* `messages.role` is constrained to `system | user | assistant | tool`.
* `user_id` exists on every table but is unused today. It is the column a future
  **Supabase Auth** integration would own (see §5) — AlphaAI's own
  authentication architecture is not replaced by it.

---

## 2. Configuration

One variable is required:

| Variable | Where | Meaning |
| -------- | ----- | ------- |
| `DATABASE_URL` | server-side only | PostgreSQL connection string (Supabase: *Project Settings → Database → Connection string → URI*) |

Optional, with defaults that suit a small deployment:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `ALPHAI_DATABASE_URL` | – | wins over `DATABASE_URL` when both are present |
| `ALPHAI_DB_SSL_MODE` | `require` | applied when the connection string does not state `sslmode` |
| `ALPHAI_DB_CONNECT_TIMEOUT_S` | `10` | per-connection timeout |
| `ALPHAI_DB_STATEMENT_TIMEOUT_MS` | `15000` | server-side statement timeout |
| `ALPHAI_DB_APPLICATION_NAME` | `alphaai` | shows up in `pg_stat_activity` |
| `ALPHAI_DB_MAX_LIST_LIMIT` | `200` | upper bound for `limit` on list endpoints |
| `ALPHAI_DB_MIGRATE_ON_START` | `false` | apply pending migrations during API startup (opt-in) |
| `ALPHAI_MIGRATIONS_DIR` | `supabase/migrations` | where the `.sql` files live |

**Not used:** `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.
AlphaAI needs none of them. If your platform already defines them, that is fine —
nothing reads them, and nothing should ship them to a browser.

### Which component owns the database

A database is not an inference concern, so either tier can hold it — and both
can, as long as they point at the **same** database:

| Deployment | Connection method | Why |
| ---------- | ----------------- | --- |
| The AlphaAI **inference host** (long-lived container, `alphaai serve`) | direct connection (port 5432) or Supavisor **session** mode | it is a normal long-lived server, it already has a persistent disk, and it is what produces the chat turns (so it is what writes them) |
| The **Vercel gateway** (serverless FastAPI) | Supavisor **transaction** mode (port 6543) | functions are short-lived and recycled; session state cannot be assumed |

The connection methods are *not* interchangeable: a serverless function on the
direct connection exhausts Supabase's connection slots, and a long-lived server
on the transaction pooler pays a new backend per statement for nothing.

Recommended arrangement: put `DATABASE_URL` on the **inference host** — that is
where chat turns are produced, so that is where they can actually be written. Add
the same database (same project, transaction-pooler string) to the **gateway** as
well if you want history readable while the inference host is asleep. Both
pointing at one database is consistent; pointing the two tiers at *different*
databases is what splits a user's threads in half, so do not do that.

Whichever deployment holds `DATABASE_URL` serves `/api/conversations*` itself and
reports its store under `database` in `/api/health`. Without a local
`DATABASE_URL` those routes are proxied to the inference host.

---

## 3. Migrations

The `.sql` files are the only definition of the schema. Apply them with either
tool — both are safe on the same database, and every statement is idempotent:

```shell
# AlphaAI's own runner (works anywhere Python does, no CLI to install)
alphaai db status      # configured? reachable? which connection method? applied/pending/drift
alphaai db plan        # what would run (files + checksums), nothing applied
alphaai db migrate     # apply pending files, one transaction per file

# Or the Supabase CLI (same directory, same files)
supabase db push
```

What the runner does:

* applies files in filename order (`YYYYMMDDHHMMSS_name.sql`);
* records each applied file by name **and SHA-256 checksum** in
  `public.alphaai_migrations`, in the same transaction as the file;
* reports a file that changed after it was applied as **drift** instead of
  re-running it;
* refuses to run anything when no migration files are found.

The ledger is `public.alphaai_migrations`, deliberately separate from the
Supabase CLI's own `supabase_migrations` table: neither tool replays the other's
files.

On a host that already has the code, migrations can also be applied over HTTP —
`POST /api/database/migrate` (on an inference host that requires
`ALPHAI_INFERENCE_TOKEN`, that route is behind the same token check) — or
automatically at boot with `ALPHAI_DB_MIGRATE_ON_START=1`.

---

## 4. API

History endpoints (all take the anonymous `client_id` the dashboard generates and
stores locally; it scopes rows, it is not authentication):

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/api/conversations?client_id=…&limit=&offset=` | list threads, newest first, with `message_count` and `preview` |
| `POST` | `/api/conversations` | create a thread explicitly |
| `GET` | `/api/conversations/{id}?client_id=…` | one thread **with its messages** |
| `DELETE` | `/api/conversations/{id}?client_id=…` | delete a thread (messages cascade) |
| `POST` | `/api/conversations/{id}/messages` | append a message (role, content, usage, …) |
| `GET`/`PUT` | `/api/preferences?client_id=…` | per-client preferences |
| `GET` | `/api/usage?client_id=…` | usage totals + recent records |
| `GET` | `/api/database` | connection, schema and migration state (never fails) |
| `POST` | `/api/database/migrate` | apply pending migrations |

Chat turns are stored when the client asks for it:

```jsonc
POST /api/chat
{"message": "hello", "client_id": "client-…"}                     // new thread
{"message": "again", "client_id": "client-…", "conversation_id": "…"}  // continue it
{"message": "hello", "client_id": "client-…", "persist": false}   // answer, do not store
```

Every chat response carries what actually happened:

```jsonc
"persistence": {"requested": true, "persisted": true, "conversation_id": "…", "message_id": 42}
"persistence": {"requested": true, "persisted": false, "error": {"code": "database_not_configured", …}}
"persistence": {"requested": false, "persisted": false, "detail": "no client_id supplied: …"}
```

`POST /api/chat/stream` emits the same outcome as a final `{"type": "persisted", …}`
event, after the last token. **Nothing is ever reported as saved unless it was
saved**, and no history is ever invented to fill a gap.

### Errors

| Code | HTTP | When |
| ---- | ---- | ---- |
| `database_not_configured` | 503 | persistence was requested and no `DATABASE_URL` is set on this deployment |
| `database_unavailable` | 503 | driver missing, connection/auth failure, schema not applied, SQL rejected |
| `conversation_not_found` | 404 | unknown id, or a thread that belongs to another client |
| `invalid_request` | 400 | malformed `client_id`/`conversation_id`/`role` |

Each error carries a `message` and a `remediation` naming the next action. The
health endpoint keeps working in every one of those cases
(`database.configured`, `database.reachable`, `database.migrations.pending`).

---

## 5. Security

* **Credentials stay on the server.** `DATABASE_URL` is read from the process
  environment. It is never logged, never returned by `/api/config`
  (`database.url` is replaced with the redacted connection facts, so you can see
  *how* a deployment connects without seeing the password), and never reaches a
  browser.
* **No Supabase API keys**, so nothing to leak: no anon key, no service-role key.
  The frontend has no Supabase URL at all.
* **Least privilege.** You do not need more than one database role: the server
  connects as the schema owner, which is what applies migrations. If you prefer a
  narrower runtime role, create it after migrating and grant only
  `select, insert, update, delete` on the four tables (plus `usage` on the
  sequences and `usage` on the schema); no DDL grant is then needed.
* **Row Level Security is enabled on all four tables** by the migrations, and no
  policy grants `anon` anything — with RLS enabled and no matching policy,
  PostgreSQL returns no rows, so the Data API (PostgREST) with a publishable key
  can never read someone's conversations. The `authenticated` owner policies
  (`user_id = auth.uid()`) are installed automatically where the Supabase auth
  objects exist (`authenticated` role + `auth.uid()`), are dormant today (every
  `user_id` is `null`), and are exactly what a future Supabase Auth integration
  needs. On a plain PostgreSQL without those objects the migration installs RLS
  and skips the policies with a notice, instead of failing.
* **Supabase Auth is not introduced.** AlphaAI's own session model is unchanged.
  `client_id` scoping is what separates browsers today, and it is enforced
  server-side in every query.
* **No secrets in git.** `.env` files are gitignored; [`env.example`](../env.example)
  lists variable *names* only, with empty values.

---

## 6. Verifying it for real

```shell
# 1. configuration and connection, without touching anything
alphaai db status

# 2. schema
alphaai db migrate
alphaai db status          # migrations.applied should equal the number of files

# 3. end to end, over HTTP, on the host that owns the database
curl -s localhost:8090/api/health | python -m json.tool | grep -A5 database
curl -s "localhost:8090/api/chat" -H 'content-type: application/json' \
  -d '{"message":"hello","client_id":"client-manual-check"}'
curl -s "localhost:8090/api/conversations?client_id=client-manual-check"
```

The test suite covers migrations, schema assumptions, persistence, retrieval,
client isolation, RLS default-deny, the unconfigured-database behaviour and the
full chat → PostgreSQL → history path. PostgreSQL tests run against a real server
and are **skipped with an exact reason** when there is none:

```shell
ALPHAI_TEST_DATABASE_URL=postgresql://user:password@host:5432/postgres python -m pytest
# or, with no server of your own:
pip install pgserver && python -m pytest      # starts a private temporary PostgreSQL
```

---

## 7. Supabase checklist

1. **Create the project** (supabase.com) — any region; note the database
   password you set.
2. **Get the connection string**: *Project Settings → Database → Connection
   string → URI*.
   * For the **inference host**: the direct connection (`db.<ref>.supabase.co:5432`)
     or Supavisor **session** mode.
   * For the **Vercel gateway**: Supavisor **transaction** mode (`…pooler.supabase.com:6543`).
   * Replace the `[YOUR-PASSWORD]` placeholder with the database password; keep
     `?sslmode=require`.
3. **Provide it to AlphaAI** as `DATABASE_URL` in that deployment's environment
   (never commit it, never paste it into frontend code or `NEXT_PUBLIC_*`-style
   variables).
4. **Apply the migrations** (either one):
   * `alphaai db migrate` on the host, or
   * `supabase db push` with the Supabase CLI linked to the project, or
   * `POST /api/database/migrate` against the running server.
5. **Verify**: `alphaai db status` (or `GET /api/database`) must report
   `reachable: true`, `migrations.pending: []`, `drift: []`; then send one chat
   message and confirm the thread appears in `GET /api/conversations`.
6. **Optional hardening**: create a runtime role limited to the four tables (see
   §5) and use it for `DATABASE_URL` after migrating; enable Supabase's PITR
   backups for the project.

Nothing in this list involves the Supabase *API* keys, and nothing changes how
inference runs.

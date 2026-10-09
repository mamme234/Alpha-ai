-- AlphaAI conversation persistence — conversations and messages.
--
-- This is the first AlphaAI migration. It is deliberately small: AlphaAI stores
-- only what a chat turn produces, so the schema stays readable and can grow with
-- new columns rather than new tables.
--
-- Rules that shape this schema:
--   * Model weights never live in PostgreSQL (or in Supabase Storage). The
--     database stores text, ids and small metadata objects only.
--   * Every row belongs to a `client_id` — the anonymous, browser-generated id
--     the dashboard sends. `user_id` is present but unused today: it is the
--     column a future Supabase Auth integration (auth.uid()) would own, which is
--     why the row-level-security policies in 20261009120200 already reference it.
--   * `metadata`/`usage` are jsonb so newer AlphaAI fields do not need a
--     migration, and unknown keys are preserved verbatim.
--
-- Apply with the Supabase CLI (`supabase db push`) or with AlphaAI's own runner
-- (`alphaai db migrate`, which tracks applied files in public.alphaai_migrations).
-- Both are safe to run against the same database: every statement is idempotent.

create table if not exists public.conversations (
    id uuid primary key default gen_random_uuid(),
    client_id text not null,
    user_id uuid,
    title text not null default 'New conversation',
    session_id text,
    engine_id text,
    model text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint conversations_title_len check (char_length(title) <= 200),
    constraint conversations_client_id_len check (char_length(client_id) between 1 and 200)
);

comment on table public.conversations is
    'One AlphaAI chat thread. `client_id` is the anonymous dashboard id; `user_id` is reserved for a future Supabase Auth subject.';
comment on column public.conversations.session_id is
    'The AlphaAI runtime session id (sess_...) that produced this thread, when the server created one.';
comment on column public.conversations.metadata is
    'Extensible per-thread data (routing hints, UI state). Never model weights.';

-- Listing a client's threads is the hottest query: newest first, per client.
create index if not exists conversations_client_updated_idx
    on public.conversations (client_id, updated_at desc);

-- Only populated once auth exists; partial so it costs nothing today.
create index if not exists conversations_user_idx
    on public.conversations (user_id)
    where user_id is not null;

create table if not exists public.messages (
    id bigint generated always as identity primary key,
    conversation_id uuid not null references public.conversations (id) on delete cascade,
    role text not null,
    content text not null,
    engine_id text,
    model text,
    finish_reason text,
    latency_ms double precision,
    usage jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint messages_role_valid check (role in ('system', 'user', 'assistant', 'tool'))
);

comment on table public.messages is
    'Every turn of a conversation: role, content, the model that produced it and its measured usage.';
comment on column public.messages.usage is
    'Token usage as reported by the engine (prompt/completion/total, plus how it was measured).';

-- History is always read as "this conversation, oldest first".
create index if not exists messages_conversation_created_idx
    on public.messages (conversation_id, created_at, id);

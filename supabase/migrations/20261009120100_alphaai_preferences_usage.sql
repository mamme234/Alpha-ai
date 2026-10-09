-- AlphaAI preferences and usage records.
--
-- Both tables are optional to AlphaAI's runtime: the dashboard works without
-- them, and the API reports honestly when they are empty. They exist because
-- they are the two things worth keeping next to conversations:
--
--   * user_preferences — per client, free-form UI/runtime choices (engine,
--     sampling, tool toggles). One row per client_id, so it grows with users,
--     not with messages.
--   * usage_records    — one row per completed generation, so usage can be
--     reported from stored facts instead of recomputed or invented.

create table if not exists public.user_preferences (
    client_id text primary key,
    user_id uuid,
    preferences jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    constraint user_preferences_client_id_len check (char_length(client_id) between 1 and 200)
);

comment on table public.user_preferences is
    'Per-client AlphaAI preferences (engine choice, sampling, tool toggles). One row per client_id.';

create index if not exists user_preferences_user_idx
    on public.user_preferences (user_id)
    where user_id is not null;

create table if not exists public.usage_records (
    id bigint generated always as identity primary key,
    client_id text,
    user_id uuid,
    conversation_id uuid references public.conversations (id) on delete set null,
    engine_id text,
    model text,
    prompt_tokens integer,
    completion_tokens integer,
    total_tokens integer,
    latency_ms double precision,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

comment on table public.usage_records is
    'One row per completed AlphaAI generation, with the engine''s own measurements. Kept separate from messages so retention can differ.';

-- Usage is reported over a time window, and optionally per client.
create index if not exists usage_records_created_idx
    on public.usage_records (created_at desc);

create index if not exists usage_records_client_created_idx
    on public.usage_records (client_id, created_at desc);

create index if not exists usage_records_conversation_idx
    on public.usage_records (conversation_id)
    where conversation_id is not null;

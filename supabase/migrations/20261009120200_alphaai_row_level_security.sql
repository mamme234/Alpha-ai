-- AlphaAI row level security.
--
-- AlphaAI has no user accounts today: the dashboard sends an anonymous
-- `client_id`, and the *server* (FastAPI) is the only component that talks to
-- PostgreSQL — with its own DATABASE_URL connection, which is a database
-- connection, not a Supabase API key.
--
-- The rule this migration enforces: private conversation data must never be
-- reachable with a publishable/anon key. So:
--
--   1. Row level security is enabled on every AlphaAI table. With RLS enabled
--      and no policy matching the caller, PostgreSQL returns no rows — a
--      default-deny, including for the Data API (PostgREST) with an anon key.
--   2. No policy grants `anon` anything, and none will: the browser never talks
--      to Supabase. AlphaAI's frontend has no Supabase URL and no key of any
--      kind in it.
--   3. `authenticated` gets owner-scoped policies keyed on `user_id =
--      auth.uid()`. They are dormant today (every `user_id` is null, and
--      `auth.uid()` is null without a session), and they are exactly what a
--      future Supabase Auth integration needs — without changing this schema.
--   4. The policies are created inside one guarded block, because they are
--      Supabase-specific: they need the `authenticated` role **and** the
--      `auth.uid()` function. On a plain/self-hosted PostgreSQL neither exists,
--      and creating them would fail the whole migration. There, RLS stays
--      enabled with no policy — the same default-deny — and the block says so
--      with a NOTICE. Apply this file again (it is idempotent) in an environment
--      that has the Supabase auth objects to install the owner policies.
--
-- The service_role key (if it is ever used) bypasses RLS by design. It is
-- server-side only, is not a database password, and must never reach a browser.
-- AlphaAI itself does not use it: the FastAPI server connects as the database
-- owner, which owns these tables and is therefore not restricted by the
-- policies it created.

alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.user_preferences enable row level security;
alter table public.usage_records enable row level security;

do $$
begin
    if to_regprocedure('auth.uid()') is null
        or not exists (select 1 from pg_roles where rolname = 'authenticated') then
        raise notice
            'AlphaAI: the Supabase auth objects (role "authenticated" and auth.uid()) are '
            'absent, so no owner policy is installed. Row level security stays enabled with '
            'no policy, which is default deny for anon and authenticated callers alike.';
        return;
    end if;

    -- ----------------------------------------------------------------------
    -- conversations: the authenticated owner of the row may do anything with it
    -- ----------------------------------------------------------------------
    drop policy if exists conversations_owner_select on public.conversations;
    create policy conversations_owner_select on public.conversations
        for select to authenticated
        using ((select auth.uid()) = user_id);

    drop policy if exists conversations_owner_insert on public.conversations;
    create policy conversations_owner_insert on public.conversations
        for insert to authenticated
        with check ((select auth.uid()) = user_id);

    drop policy if exists conversations_owner_update on public.conversations;
    create policy conversations_owner_update on public.conversations
        for update to authenticated
        using ((select auth.uid()) = user_id)
        with check ((select auth.uid()) = user_id);

    drop policy if exists conversations_owner_delete on public.conversations;
    create policy conversations_owner_delete on public.conversations
        for delete to authenticated
        using ((select auth.uid()) = user_id);

    -- ----------------------------------------------------------------------
    -- messages: ownership is inherited from the parent conversation
    -- ----------------------------------------------------------------------
    drop policy if exists messages_owner_select on public.messages;
    create policy messages_owner_select on public.messages
        for select to authenticated
        using (
            exists (
                select 1 from public.conversations c
                where c.id = messages.conversation_id
                  and c.user_id = (select auth.uid())
            )
        );

    drop policy if exists messages_owner_insert on public.messages;
    create policy messages_owner_insert on public.messages
        for insert to authenticated
        with check (
            exists (
                select 1 from public.conversations c
                where c.id = messages.conversation_id
                  and c.user_id = (select auth.uid())
            )
        );

    -- ----------------------------------------------------------------------
    -- user_preferences / usage_records: one owner per row, same pattern
    -- ----------------------------------------------------------------------
    drop policy if exists user_preferences_owner_all on public.user_preferences;
    create policy user_preferences_owner_all on public.user_preferences
        for all to authenticated
        using ((select auth.uid()) = user_id)
        with check ((select auth.uid()) = user_id);

    drop policy if exists usage_records_owner_select on public.usage_records;
    create policy usage_records_owner_select on public.usage_records
        for select to authenticated
        using ((select auth.uid()) = user_id);

    drop policy if exists usage_records_owner_insert on public.usage_records;
    create policy usage_records_owner_insert on public.usage_records
        for insert to authenticated
        with check ((select auth.uid()) = user_id);
end
$$;

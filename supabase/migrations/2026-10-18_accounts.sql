-- 2026-10-18: per-user accounts, roles, query auditing, rate limiting.
--
-- Replaces the single shared basic-auth login with one row per technician in
-- app_users, ties every question and thumbs-up/down to the account that made
-- it, and adds audit_events: the log of logins, lockouts, rate-limit hits, and
-- account changes. The rate limiter reads its counts back out of queries and
-- audit_events, so these tables are the limiter's source of truth as well as
-- the audit trail -- see implementation/rate_limit.py.
--
-- Additive and idempotent: safe to re-run, and the pre-accounts app keeps
-- working against it (every new column is nullable or defaulted).
--
-- Apply once in the Supabase SQL editor (or psql), then create the first
-- admin with:  uv run python scripts/manage_users.py bootstrap <username>

begin;

-- ---------------------------------------------------------------------------
-- Accounts
-- ---------------------------------------------------------------------------

create table if not exists app_users (
    id bigserial primary key,
    -- The part before @newpaltz.edu, always stored lowercase. The login form
    -- also accepts the full address; implementation/accounts.py normalizes it.
    username text not null unique
        check (username ~ '^[a-z0-9][a-z0-9._-]{1,31}$'),
    email text generated always as (username || '@newpaltz.edu') stored,
    role text not null default 'worker'
        check (role in ('admin', 'worker')),
    -- argon2id hash, one per account even while accounts share a role's
    -- default password, so per-user passwords later need no schema change.
    password_hash text not null,
    is_active boolean not null default true,
    -- Reserved: nothing enforces this yet. Set on every account created with a
    -- shared default password, ready for a forced-change flow.
    must_change_password boolean not null default true,
    created_at timestamptz not null default now(),
    created_by bigint references app_users(id) on delete set null,
    last_login_at timestamptz
);

-- ---------------------------------------------------------------------------
-- Queries: who asked, in which conversation, and how it ended
-- ---------------------------------------------------------------------------

alter table queries
    add column if not exists user_id bigint references app_users(id) on delete set null;

-- The app now inserts the row when a question starts ('in_progress') and
-- updates it when the answer finishes, so a question whose technician closed
-- the tab mid-answer is still recorded ('abandoned') -- before this, those
-- were never logged at all. The 'completed' default only applies to rows from
-- before this migration and to an old app version still running mid-deploy.
alter table queries
    add column if not exists status text not null default 'completed'
        check (status in ('in_progress', 'completed', 'error', 'abandoned'));

-- Gradio's per-tab session hash: groups the turns of one conversation.
alter table queries
    add column if not exists session_id text;

-- Pre-migration rows that failed were logged with an error message; label
-- them as such instead of leaving them under the 'completed' default.
update queries set status = 'error'
where error is not null and status = 'completed';

-- Per-user quota checks (last minute / hour / day) and the Activity tab.
create index if not exists queries_user_created_idx
    on queries (user_id, created_at desc);
-- Global daily budget check and time-range filters.
create index if not exists queries_created_at_idx
    on queries (created_at);

-- ---------------------------------------------------------------------------
-- Feedback: tied to the account and to the exact question it rates
-- ---------------------------------------------------------------------------

alter table feedback
    add column if not exists user_id bigint references app_users(id) on delete set null;
alter table feedback
    add column if not exists query_id bigint references queries(id) on delete set null;

-- ---------------------------------------------------------------------------
-- Audit log
-- ---------------------------------------------------------------------------

-- Append-only. event_type is deliberately not constrained to a fixed list so a
-- new kind of event never needs a migration; the ones written today are
-- login_success, login_failure, login_locked, rate_limited, access_denied,
-- account_created, account_disabled, account_enabled, role_changed,
-- password_reset. username_attempted is what was typed at the login form, kept
-- even when it matches no account, so failed logins against made-up names are
-- still visible; for admin actions it's the account acted on, and user_id is
-- the admin who acted.
create table if not exists audit_events (
    id bigserial primary key,
    occurred_at timestamptz not null default now(),
    event_type text not null,
    user_id bigint references app_users(id) on delete set null,
    username_attempted text,
    ip inet,
    user_agent text,
    details jsonb
);

-- Per-username login lockout: failures for one name in the last N minutes.
create index if not exists audit_events_type_username_idx
    on audit_events (event_type, username_attempted, occurred_at desc);
-- Activity tab and v_security_events, newest first.
create index if not exists audit_events_occurred_idx
    on audit_events (occurred_at desc);

-- ---------------------------------------------------------------------------
-- Monitoring views (open these in the Supabase table editor)
-- ---------------------------------------------------------------------------

-- security_invoker: the views run with the querying role's permissions, so
-- they can never expose more than the underlying tables would.

create or replace view v_query_activity with (security_invoker = true) as
select
    q.id as query_id,
    q.created_at,
    q.created_at at time zone 'America/New_York' as created_at_local,
    u.username,
    u.email,
    u.role,
    q.question,
    q.status,
    q.latency_seconds,
    q.error,
    case when jsonb_typeof(q.sources) = 'array'
         then jsonb_array_length(q.sources) end as source_count,
    q.history_length,
    q.session_id,
    q.answer
from queries q
left join app_users u on u.id = q.user_id;

create or replace view v_user_usage with (security_invoker = true) as
select
    u.id as user_id,
    u.username,
    u.email,
    u.role,
    u.is_active,
    u.created_at,
    u.last_login_at,
    count(q.id) as total_queries,
    count(q.id) filter (where q.created_at > now() - interval '24 hours') as queries_24h,
    count(q.id) filter (where q.created_at > now() - interval '7 days') as queries_7d,
    count(q.id) filter (where q.status = 'error') as error_count,
    max(q.created_at) as last_query_at
from app_users u
left join queries q on q.user_id = u.id
group by u.id;

-- username is the account the event is about (the one signing in, refused,
-- or changed); actor is the signed-in account that caused it -- the admin, for
-- account changes. For a technician's own events the two are the same.
create or replace view v_security_events with (security_invoker = true) as
select
    e.id,
    e.occurred_at,
    e.occurred_at at time zone 'America/New_York' as occurred_at_local,
    e.event_type,
    coalesce(e.username_attempted, u.username) as username,
    e.ip,
    e.user_agent,
    e.details,
    u.username as actor
from audit_events e
left join app_users u on u.id = e.user_id;

-- ---------------------------------------------------------------------------
-- Lock everything away from Supabase's public REST API
-- ---------------------------------------------------------------------------

-- Supabase serves the public schema over PostgREST to its anon/authenticated
-- roles. app_users holds password hashes, so it and the audit log get RLS
-- with no policies (deny all) -- as do article_state and refresh_runs, which
-- were the two tables left without it. The app connects as postgres, which has
-- BYPASSRLS, so none of this changes what the app or refresh_kb.py can do.
alter table app_users enable row level security;
alter table audit_events enable row level security;
alter table article_state enable row level security;
alter table refresh_runs enable row level security;

-- Belt and braces on top of RLS: no table privileges for the API roles on the
-- new tables and views at all. Guarded so this file still runs on a plain
-- Postgres that has no Supabase roles.
do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke all on app_users, audit_events,
                      v_query_activity, v_user_usage, v_security_events
            from anon, authenticated;
    end if;
end
$$;

commit;

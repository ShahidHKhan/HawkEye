-- HawkEye production schema (Supabase / Postgres + pgvector).
--
-- This file is a reconciliation snapshot of the LIVE schema, taken 2026-08-04
-- via `pg_dump --schema-only` against Supabase. Earlier versions of this file
-- had drifted from reality in both directions: article_state, refresh_runs,
-- and chunks.article_id all existed live without ever being reflected here
-- (created directly against Supabase by a refresh_kb.py that, until this
-- session, was never committed to this repo), while chunks.created_at and
-- two indexes below were declared here but never actually existed live.
-- Treat this file as documentation of what exists, kept idempotent (safe to
-- run against a fresh database) but not guaranteed to match reality unless
-- someone re-reconciles it the same way in the future.
--
-- 2026-10-18: updated by hand for supabase/migrations/2026-10-18_accounts.sql
-- (app_users, audit_events, the user/status columns on queries and feedback,
-- the monitoring views, and RLS on every table). That migration file is what
-- actually changes a live database; this file only mirrors the result.

create extension if not exists vector;

-- gemini-embedding-001 defaults to 3072 dimensions. If you request a smaller
-- output dimensionality from the embeddings model, update the vector(...) size
-- below to match, or ingestion/retrieval will fail with a dimension mismatch.
create table if not exists chunks (
    id bigserial primary key,
    source text not null,
    type text,
    page_content text not null,
    embedding vector(3072) not null,
    -- Links a chunk back to the KB article it was generated from (see
    -- article_state.article_id and refresh_kb.py). Nullable because chunks
    -- from implementation/ingest.py's original bulk load predate this column
    -- and were never backfilled.
    article_id text
);

-- Indexed for refresh_kb.py's per-article lookups: find/replace one article's
-- chunks by id instead of matching on its source file path, which can differ
-- between machines (see refresh_kb.py's existing_chunk_source()).
create index if not exists chunks_article_id_idx
    on chunks (article_id);

-- DRIFT NOTE (2026-08-04): this file previously declared the two indexes
-- below, and also a chunks.created_at column, none of which exist on the live
-- database as of this reconciliation (confirmed via pg_indexes /
-- information_schema.columns). Left here as a record of past intent, not live
-- DDL -- re-adding them silently would just reintroduce the same
-- schema.sql-vs-reality drift this file is meant to fix. Decide deliberately
-- before uncommenting:
--
-- -- Would speed up ingest's resume-by-content lookup (get_existing_chunk_keys)
-- -- and enforce the assumption that logic already makes: one row per unique
-- -- (source, page_content).
-- create unique index if not exists chunks_source_content_idx
--     on chunks (source, page_content);
--
-- -- Approximate nearest-neighbor index for the <-> (L2) ORDER BY in
-- -- fetch_chunks(). ivfflat needs the table populated with representative
-- -- data before it's built well; if bootstrapping an empty table, ingest
-- -- first, then run this. Without it, fetch_chunks() falls back to an exact
-- -- sequential scan -- fine at ~4k rows, worth revisiting if it grows a lot.
-- create index chunks_embedding_idx
--     on chunks using ivfflat (embedding vector_l2_ops) with (lists = 100);

-- One row per technician. See implementation/accounts.py and the migration
-- for the reasoning behind each column.
create table if not exists app_users (
    id bigserial primary key,
    username text not null unique
        check (username ~ '^[a-z0-9][a-z0-9._-]{1,31}$'),
    email text generated always as (username || '@newpaltz.edu') stored,
    role text not null default 'worker'
        check (role in ('admin', 'worker')),
    password_hash text not null,
    is_active boolean not null default true,
    must_change_password boolean not null default true,
    created_at timestamptz not null default now(),
    created_by bigint references app_users(id) on delete set null,
    last_login_at timestamptz
);

-- Inserted when a question starts (status 'in_progress') and updated when it
-- ends; the per-user rate limiter counts these rows. user_id is null for rows
-- from before accounts existed.
create table if not exists queries (
    id bigserial primary key,
    question text not null,
    history_length integer,
    answer text,
    sources jsonb,
    latency_seconds numeric,
    error text,
    created_at timestamptz default now(),
    user_id bigint references app_users(id) on delete set null,
    status text not null default 'completed'
        check (status in ('in_progress', 'completed', 'error', 'abandoned')),
    session_id text
);

create index if not exists queries_user_created_idx
    on queries (user_id, created_at desc);
create index if not exists queries_created_at_idx
    on queries (created_at);

create table if not exists feedback (
    id bigserial primary key,
    answer text,
    liked boolean,
    created_at timestamptz default now(),
    user_id bigint references app_users(id) on delete set null,
    query_id bigint references queries(id) on delete set null
);

-- Append-only log of sign-ins, lockouts, rate-limit hits, refused admin
-- access, and account changes. username_attempted is the name typed at the
-- login form, or the account acted on for admin actions (user_id is then the
-- acting admin).
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

create index if not exists audit_events_type_username_idx
    on audit_events (event_type, username_attempted, occurred_at desc);
create index if not exists audit_events_occurred_idx
    on audit_events (occurred_at desc);

-- One row per known KB article, keyed by its TeamDynamix article id. Written
-- and read by refresh_kb.py's diff-based refresh: implementation/diff_articles.py
-- compares a fresh crawl against this table to classify new/changed/unchanged/removed.
create table if not exists article_state (
    article_id text primary key,
    url text not null,
    modified_date timestamptz,
    last_crawled timestamptz not null default now(),
    -- 'public' (the 7 public TeamDynamix categories) or 'internal'
    -- (Internal-Documentation, gated). A refresh run diffs one scope at a time
    -- via get_known_article_state(scope).
    scope text not null default 'public'
);

-- Append-only audit log, one row per refresh_kb.py run (success or failure).
-- Deliberately has no foreign keys to chunks or article_state -- a run's
-- outcome should stay readable even if the articles/chunks it touched are
-- later deleted or renumbered. changes is a free-form jsonb summary, e.g.
-- {"new": [{"article_id", "title", "chunk_count"}, ...], "changed": [...], "removed": [...],
--  "healed": [{"article_id", "title"}, ...],
--  "failed": [{"article_id", "title", "error"}, ...]} -- "healed" entries were classified as
-- new by the diff but already had chunks (article_state lost track of them); "failed" entries
-- hit an error and were skipped without aborting the rest of the run, and are also summarized
-- into the error column so the run still reads as failed. See refresh_kb.py.
create table if not exists refresh_runs (
    id bigserial primary key,
    scope text not null,
    started_at timestamptz not null,
    -- NULL means the run never reached its logging step. refresh_kb.py logs
    -- exactly one row per run (success or failure) from a single finally
    -- block, so a NULL finished_at today would mean something crashed hard
    -- enough to skip even that -- see the 8/3 run that was cleaned up for
    -- exactly this reason before refresh_kb.py's finally-block logging existed.
    finished_at timestamptz,
    new_count integer,
    changed_count integer,
    unchanged_count integer,
    removed_count integer,
    error text,
    changes jsonb
);

-- Monitoring views: v_query_activity (who asked what, when),
-- v_user_usage (per-account totals), v_security_events (readable audit log).
-- Defined in supabase/migrations/2026-10-18_accounts.sql, with
-- security_invoker = true and no privileges for anon/authenticated.

-- Row Level Security is enabled with zero policies on every table, which denies
-- all access through Supabase's REST API (anon/authenticated roles). That
-- matters most for app_users, which holds password hashes. The app and
-- refresh_kb.py connect as postgres, which has BYPASSRLS, so they're unaffected.
-- (Before 2026-10-18, article_state and refresh_runs were the two tables
-- without RLS.)
alter table chunks enable row level security;
alter table queries enable row level security;
alter table feedback enable row level security;
alter table article_state enable row level security;
alter table refresh_runs enable row level security;
alter table app_users enable row level security;
alter table audit_events enable row level security;

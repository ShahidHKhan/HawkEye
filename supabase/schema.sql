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

create table if not exists queries (
    id bigserial primary key,
    question text not null,
    history_length integer not null,
    answer text,
    sources jsonb,
    latency_seconds numeric,
    error text,
    created_at timestamptz default now()
);

create table if not exists feedback (
    id bigserial primary key,
    answer text,
    liked boolean,
    created_at timestamptz default now()
);

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
-- {"new": [...], "changed": [{"article_id", "title", "chunk_count"}, ...], "removed": [...]}.
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

-- Row Level Security is enabled (Supabase's default for dashboard-created
-- tables) on chunks, queries, and feedback, but not on article_state or
-- refresh_runs -- inconsistent, and there are currently zero policies defined
-- on any of the five tables. The app always connects as the postgres role
-- (table owner), which bypasses RLS by default, so this has no practical
-- effect today; it would matter if a lower-privileged role (e.g. Supabase's
-- anon/authenticated roles) were ever used against these tables directly.
alter table chunks enable row level security;
alter table queries enable row level security;
alter table feedback enable row level security;

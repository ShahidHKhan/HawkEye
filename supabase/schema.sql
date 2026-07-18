-- HawkEye production schema (Supabase / Postgres + pgvector).
-- Run once against a fresh Supabase project before running implementation/ingest.py.

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
    created_at timestamptz not null default now()
);

-- Speeds up ingest's resume-by-content lookup (get_existing_chunk_keys).
create unique index if not exists chunks_source_content_idx
    on chunks (source, page_content);

-- Approximate nearest-neighbor index for the <-> (L2) ORDER BY in fetch_chunks.
-- ivfflat needs the table populated with representative data before it's built
-- well; if you're bootstrapping an empty table, ingest first, then run:
--   create index chunks_embedding_idx on chunks using ivfflat (embedding vector_l2_ops) with (lists = 100);
create index if not exists chunks_embedding_idx
    on chunks using ivfflat (embedding vector_l2_ops) with (lists = 100);

create table if not exists queries (
    id bigserial primary key,
    question text not null,
    history_length integer not null,
    answer text,
    sources jsonb,
    latency_seconds numeric,
    error text,
    created_at timestamptz not null default now()
);

create table if not exists feedback (
    id bigserial primary key,
    answer text not null,
    liked boolean not null,
    created_at timestamptz not null default now()
);

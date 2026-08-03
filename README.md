# HawkEye

Internal IT help desk knowledge assistant for SUNY New Paltz technicians.
Not for public or customer-facing use — for authorized help desk staff only.

Technicians describe a customer's issue in plain language; HawkEye retrieves
relevant knowledge-base context (Supabase/pgvector) and returns a direct answer
via Gemini, showing the retrieved sources alongside the response.

## Local development

```bash
uv sync
uv run python app.py
```

Requires a `.env` file — see `.env.example` for the required variables.

## Ingesting the knowledge base

Apply `supabase/schema.sql` to your Supabase project once, then run the
chunking + embedding pipeline in `implementation/ingest.py` (see its
`fetch_documents` → `create_chunks` → `create_embeddings` flow).

## Deployment

Built as a container: see `Dockerfile`. Deploys to any container-based host
(Railway, Render, Fly.io, etc.) that supplies a `PORT` env var and the
variables listed in `.env.example`.

## Repo layout

- `app.py` — Gradio chat UI (production entry point)
- `implementation/` — retrieval + answer pipeline (`answer.py`) and the
  knowledge-base ingestion pipeline (`ingest.py`)
- `evaluation/`, `evaluator.py` — retrieval/answer quality evaluation tooling
- `scraper/` — regenerates `knowledge-base/` from source content
- `day1.ipynb`–`day3.ipynb`, `rag_build_guide.md` — R&D notebooks tracing the
  build from a naive RAG pipeline to the current one; local Chroma
  (`vector_db/`, `preprocessed_db/`) is used only in this R&D path, not in
  production

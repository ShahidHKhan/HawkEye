# HawkEye

Internal IT help desk knowledge assistant for SUNY New Paltz technicians.
Not for public or customer-facing use — for authorized help desk staff only.

Technicians describe a customer's issue in plain language; HawkEye retrieves
relevant knowledge-base context from Supabase/pgvector and returns a direct,
coworker-style answer via Gemini, streaming the response alongside the
retrieved sources.

## How it works

**1. Knowledge base ingestion.** `scraper/scrape_kb.py` crawls SUNY New
Paltz's TeamDynamix IT knowledge base (8 top-level categories, one gated —
`Internal-Documentation`) and saves each article as JSON.
`scraper/convert_to_markdown.py` turns that JSON into Markdown files with
YAML frontmatter (title, tags, category path, dates, source URL) under
`knowledge-base/{Category}/`, which is what the ingestion and retrieval
pipelines actually read.

**2. Chunking + embedding.** `implementation/ingest.py` loads every Markdown
file, asks Gemini (`gemini-2.5-flash-lite`) to split each document into
overlapping chunks (each with a generated headline + summary prepended to
the original text, to help retrieval), embeds them with
`gemini-embedding-001`, and writes them into the `chunks` table in Supabase
Postgres (`pgvector`). Chunking output is cached to `chunks_cache.jsonl` so
a partial run can resume without re-calling the LLM.

**3. Retrieval + answering.** `implementation/answer.py` runs the live
question through a multi-step RAG pipeline:
   - **Decompose** — split a compound question into self-contained
     sub-questions.
   - **Rewrite** — compress each sub-question + conversation history into a
     focused search query.
   - **Dual retrieve** — fetch nearest-neighbor chunks (pgvector `<->`
     distance) for both the original sub-question and its rewritten form,
     merged and deduped.
   - **Rerank** — an LLM call re-sorts the merged chunks by relevance to the
     original question.
   - **Answer** — the top chunks are passed as context to Gemini
     (`gemini-2.5-flash-lite`), which streams back a direct answer written
     for a technician, not the end customer.

**4. Chat UI.** `app.py` is a Gradio app (basic-auth gated) with three tabs:
an **Assistant** tab (chat + a live "Retrieved sources" panel showing the
chunks used for the current answer), a **Refresh History** tab (read-only
view of the weekly KB refresh runs, below), and a **Knowledge Map** tab (a
rotatable 3D point cloud of every chunk's embedding, colored by category —
see `visualize_embeddings.py`; computed once and cached server-side so only
the first technician to open it pays the ~1-2 minute PCA/t-SNE cost). Every
query and every thumbs-up/down is logged to Supabase (`queries`, `feedback`
tables) for later analysis.

**5. Weekly KB refresh.** `refresh_kb.py`, run by
`.github/workflows/refresh_kb.yml` every Sunday night, re-crawls the public
TeamDynamix categories, diffs the result against `article_state`
(`implementation/diff_articles.py`) into new / changed / unchanged /
removed, and only touches what changed: new and changed articles are
re-chunked and re-embedded, removed articles have their chunks and local
Markdown deleted. Mass removals (more than 10% of known articles in one run)
trip a circuit breaker and abort instead of deleting — treated as a probable
crawl failure rather than genuine content removal. Every run, success or
failure, is logged to `refresh_runs` and visible in the app's Refresh
History tab.

**6. Evaluation.** `evaluator.py` (a separate Gradio dashboard) and
`evaluation/eval.py` measure retrieval quality (MRR, nDCG, keyword coverage)
and answer quality (LLM-judged accuracy/completeness/relevance) against the
hand-built test set in `evaluation/tests.jsonl`.

## Local development

```bash
uv sync
uv run python app.py
```

Requires a `.env` file — see `.env.example` for the required variables
(`GOOGLE_API_KEY`, `SUPABASE_DB_URL`, `APP_USERNAME`/`APP_PASSWORD`).

## Ingesting the knowledge base

1. Apply `supabase/schema.sql` to your Supabase project once (creates
   `chunks`, `queries`, `feedback`, `article_state`, `refresh_runs`).
2. Populate `knowledge-base/` — either scrape it fresh
   (`uv run scraper/scrape_kb.py` then `uv run scraper/convert_to_markdown.py`)
   or use an existing copy.
3. Run `uv run implementation/ingest.py` to chunk and embed everything into
   Supabase. Use `--regenerate` to re-chunk via the LLM instead of loading
   `chunks_cache.jsonl`, `--reset` to truncate `chunks` first, and
   `--smoke-test` to sanity-check retrieval without ingesting.

After the initial ingest, `refresh_kb.py` keeps the KB in sync going
forward — it doesn't need to be run manually except via
`workflow_dispatch` on the GitHub Actions workflow.

## Deployment

Built as a container: see `Dockerfile`. Deploys to Fly.io via
`.github/workflows/fly-deploy.yml` on every push to `master`
(`fly.toml` configures the app). The same image works on any
container-based host that supplies a `PORT` env var and the variables
listed in `.env.example`.

## Repo layout

- `app.py` — Gradio chat UI (production entry point)
- `implementation/` — retrieval + answer pipeline (`answer.py`), knowledge-base
  ingestion (`ingest.py`), and the diffing logic behind the weekly refresh
  (`diff_articles.py`)
- `refresh_kb.py` — weekly KB refresh pipeline (crawl → diff → re-ingest
  changes), run by `.github/workflows/refresh_kb.yml`
- `scraper/` — crawls TeamDynamix and regenerates `knowledge-base/` from
  source content (`scrape_kb.py`, `convert_to_markdown.py`)
- `knowledge-base/` — the Markdown + frontmatter source documents, one
  folder per top-level category
- `supabase/schema.sql` — production schema (Postgres + pgvector), kept as
  a reconciled snapshot of the live Supabase schema
- `evaluation/`, `evaluator.py` — retrieval/answer quality evaluation
  tooling and dashboard
- `migrate_to_supabase.py` — one-off migration of chunks from the old local
  Chroma store into Supabase; not part of the normal workflow
- `visualize_embeddings.py` — chunking-space 3D visualization (PCA + t-SNE
  + Plotly). Powers the app's Knowledge Map tab, and doubles as a standalone
  Gradio app (`uv run visualize_embeddings.py`) for grabbing screenshots
- `day1.ipynb`–`day3.ipynb`, `rag_build_guide.md` — R&D notebooks tracing the
  build from a naive RAG pipeline to the current one; local Chroma
  (`vector_db/`, `preprocessed_db/`) is used only in this R&D path, not in
  production

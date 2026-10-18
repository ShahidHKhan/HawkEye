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

**4. Chat UI.** `app.py` is a Gradio app behind per-person accounts (see
**Accounts, roles, and rate limits** below). Every technician gets the
**Assistant** tab: chat plus a live "Retrieved sources" panel showing the
chunks used for the current answer. Admins also get **Refresh History**
(read-only view of the weekly KB refresh runs, below), **Knowledge Map** (a
rotatable 3D point cloud of every chunk's embedding, colored by category, see
`visualize_embeddings.py`; a single PCA projection, computed once and cached
server-side), **Accounts** (create, disable, promote, and reset accounts), and
**Activity** (who asked what and when, per-account usage, and security
events). Every question is logged to Supabase the moment it's asked, tied to
the account and conversation, and every thumbs-up/down is tied to the
question it rates (`queries`, `feedback` tables).

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

**6. Evaluation.** `evaluation/evaluator.py` (a separate Gradio dashboard,
`uv run python -m evaluation.evaluator`) and `evaluation/eval.py` measure retrieval quality (MRR, nDCG, keyword coverage)
and answer quality (LLM-judged accuracy/completeness/relevance) against the
hand-built test set in `evaluation/tests.jsonl`.

## Accounts, roles, and rate limits

**Accounts.** Only accounts an admin has created can sign in. A username is the
part of a SUNY New Paltz address before `@newpaltz.edu` (`khans45` for
`khans45@newpaltz.edu`); the login form accepts either form in any case. During
the account-testing phase every account starts with its role's shared default
password, `WORKER_DEFAULT_PASSWORD` or `ADMIN_DEFAULT_PASSWORD`. Each account
still stores its own argon2id hash, so moving to per-person passwords later
needs no schema change.

**Roles.** `worker` sees the Assistant tab only. `admin` sees every tab.
Hiding tabs is cosmetic; every admin handler checks the role on the server,
because any Gradio event can be called directly. The two default passwords
must differ, and changing someone's role resets their password to the new
role's default, so knowing the worker password never gets anyone into an
admin account. An admin can't disable or demote themselves, and the last
active admin can't be removed.

**Auditing.** `audit_events` records sign-ins (including failed ones against
made-up names), lockouts, rate-limit hits, refused admin access, and every
account change with the admin who made it, along with IP and user agent. In
Supabase, three views make this readable: `v_query_activity` (who asked what,
when, and how it ended), `v_user_usage` (per-account totals), and
`v_security_events`. The admin Activity tab shows the same data.

**Rate limits.** Defaults below; each can be overridden with the environment
variable named in `.env.example`, and 0 disables a window.

| Surface | Limit |
|---|---|
| Sign-in attempts, per IP | 10 per 5 minutes (HTTP 429) |
| Sign-in failures, per username | locked for 15 minutes after 5 |
| Questions, worker | 5 per minute, 60 per hour, 300 per day |
| Questions, admin | 10 per minute |
| Questions, everyone combined | 3,000 per day (Gemini spend ceiling) |
| Questions in flight | 1 per account at a time |
| Question size | 1,500 characters; last 10 messages of history sent along |
| Thumbs up/down | 30 per minute per account |
| Admin tab actions | 30 per minute; 50 new accounts per hour |
| Gradio queue | 4 events processed at once, 50 waiting |
| Fly.io edge | soft 50 / hard 100 concurrent requests |

Question quotas are counted from the `queries` table itself, so they survive
restarts and would hold across several machines. The other in-memory limits
assume the single Fly machine `fly.toml` runs today.

**First admin, and break-glass.** Day to day, accounts are managed in the
Accounts tab. To create the first admin, or to get back in if every admin is
locked out:

```bash
uv run python scripts/manage_users.py bootstrap khans45
uv run python scripts/manage_users.py list
```

**Known limits.** Shared default passwords mean the audit trail shows which
account was used, not who was typing. The intended next step is signing in
with the campus Microsoft account, which proves the person owns the address
and removes shared passwords. Gradio sessions also don't expire until sign-out
or a restart; a disabled account stops working within 30 seconds regardless.

## License

Proprietary — see [LICENSE](LICENSE). The code is mine; the knowledge-base
articles it indexes belong to SUNY New Paltz and are neither licensed nor
distributed here (`knowledge-base/` is gitignored, and the gated
`Internal-Documentation` category is excluded from the automated refresh).

## Local development

```bash
uv sync
uv run python app.py
```

Requires a `.env` file — see `.env.example` for the required variables
(`GOOGLE_API_KEY`, `SUPABASE_DB_URL`, `WORKER_DEFAULT_PASSWORD`,
`ADMIN_DEFAULT_PASSWORD`). Sign in with an account created as described in
**Accounts, roles, and rate limits**.

## Ingesting the knowledge base

1. Apply `supabase/schema.sql` to your Supabase project once (creates
   `chunks`, `queries`, `feedback`, `article_state`, `refresh_runs`,
   `app_users`, `audit_events`), then
   `supabase/migrations/2026-10-18_accounts.sql` for the monitoring views.
   An existing database only needs the migration.
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

The app refuses to start without both default-password secrets, so set them
before the first deploy that includes accounts:

```bash
fly secrets set WORKER_DEFAULT_PASSWORD=... ADMIN_DEFAULT_PASSWORD=...
```

Remove the old basic-auth secrets only **after** that deploy is live. The
pre-accounts code refuses to start without them, and unsetting a secret
restarts the running machine:

```bash
fly secrets unset APP_USERNAME APP_PASSWORD
```

## Repo layout

- `app.py` — Gradio chat UI (production entry point)
- `implementation/` — retrieval + answer pipeline (`answer.py`), accounts and
  roles (`accounts.py`), rate limiting (`rate_limit.py`), the audit log
  (`audit.py`), the shared database pool (`db.py`), knowledge-base
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
- `evaluation/` — retrieval/answer quality evaluation tooling (`eval.py`,
  `tests.jsonl`) and the metrics dashboard (`evaluator.py`). Both are run as
  modules from the repo root, e.g. `uv run python -m evaluation.evaluator`
- `visualize_embeddings.py` — chunk-space 3D visualization (PCA + Plotly).
  Powers the app's Knowledge Map tab, and doubles as a standalone
  Gradio app (`uv run visualize_embeddings.py`) for grabbing screenshots
- `tests/` — offline tests plus a read-only live-database audit; see
  `tests/README.md`
- `notebooks/` — R&D notebooks (`day1`–`day3`) tracing the build from a naive
  RAG pipeline to the current one, plus the `rag_build_guide.md` scaffold they
  were worked through against. Local Chroma (`vector_db/`, `preprocessed_db/`)
  is used only in this R&D path, never in production
- `scripts/manage_users.py` — break-glass account management (first admin,
  lockout recovery)
- `supabase/migrations/` — schema changes to apply to the live database, in date order
- `scripts/archive/` — completed one-off scripts, kept as a record: the
  original Chroma → Supabase chunk migration
- `docs/archive/` — a detailed snapshot of the superseded Chroma-era
  architecture, retained for its design reasoning

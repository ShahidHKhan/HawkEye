# Tests

```
python tests/test_refresh_kb.py     # 15 offline tests
python tests/test_app.py            # 11 offline tests
python tests/check_live_state.py    # read-only audit of the live database
```

## No test dependency, on purpose

`pytest` isn't in this project's dependency groups, and adding it means relocking
`uv.lock` — which `uv sync --frozen` in both workflows rejects the moment it
drifts from `pyproject.toml`. So the tests are plain `test_*()` functions using
bare `assert`, run by the ~30-line runner in `_runner.py`. If pytest is ever
added, it collects these files unchanged; nothing here needs rewriting.

Each file puts the repo root on `sys.path` itself, so they run from anywhere.
They also set placeholder `SUPABASE_DB_URL` / `GOOGLE_API_KEY` values before
importing, because `implementation/diff_articles.py` and `implementation/ingest.py`
build a connection pool and a Gemini client at import time and refuse to import
without them. Nothing in the offline tests connects to either.

## What the offline tests cover

`test_refresh_kb.py` guards the failure that broke four consecutive weekly runs:
a changed article whose markdown file isn't on the machine doing the refresh.
`knowledge-base/` is gitignored, so a CI checkout has no mirror at all, and the
old code raised `FileNotFoundError` and aborted the whole run. **These tests
build every path in a temp directory and never read the real `knowledge-base/`** —
assuming that directory exists is the exact bug being guarded against.

The rest of that file covers the pieces that were easy to get wrong:

- **Cross-listed articles** (one article filed under two categories, so two files
  and two `chunks.source` values) resolve deterministically instead of raising.
  `check_live_state.py` lists the 18 that exist today.
- **Per-article failure isolation** — one bad article is recorded in
  `changes.failed` and the run continues, instead of aborting every article
  behind it plus the whole removal phase.
- **The behaviors that had to survive that change**: a circuit-breaker abort
  still exits 0, a genuine crawl/DB crash still re-raises, and an article that
  fails while healing is never recorded as healed.

`test_app.py` covers the Refresh History tab's detail panel, including the
distinction between a run that processed nothing because nothing had changed and
one that processed nothing because it crashed.

## check_live_state.py

Read-only, asserts nothing, changes nothing — it prints the state that's easy to
get wrong and expensive to notice late: `chunks.article_id` coverage (a chunk
without one can never be replaced, since a refresh deletes by article_id),
cross-listed articles, which machine wrote which chunks, and how the last few
refresh runs went. Skips itself if `SUPABASE_DB_URL` isn't set.

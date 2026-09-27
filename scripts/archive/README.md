# Archived one-off scripts

Completed migrations, kept as a record of how the data got where it is. Nothing
here is part of any current workflow, and none of it is expected to run again.

- **`migrate_to_supabase.py`** — the one-time move of ~4k already-embedded chunks
  out of the local Chroma store (`preprocessed_db/`) into the Supabase `chunks`
  table, done as part of commit `92f01d5`. It re-used the existing
  `gemini-embedding-001` vectors rather than paying to re-embed everything.
  Requires `chromadb` (a `dev` dependency) and a local `preprocessed_db/`.

Ongoing knowledge-base maintenance is [`refresh_kb.py`](../../refresh_kb.py); a
first-time bulk load is [`implementation/ingest.py`](../../implementation/ingest.py).

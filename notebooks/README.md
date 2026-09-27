# R&D notebooks

The build log for HawkEye's retrieval pipeline, kept for the reasoning rather
than for reuse. Read top to bottom, `day1` → `day3`:

- **`day1.ipynb`** — loading `knowledge-base/`, first look at the corpus.
- **`day2.ipynb`** — mechanical chunking (`RecursiveCharacterTextSplitter`,
  1000/200) into 7,676 chunks, embedded into a local Chroma store (`vector_db/`),
  then t-SNE'd to see whether the categories separated at all.
- **`day3.ipynb`** — naive RAG: retrieve → format → generate, wired to a
  `gr.ChatInterface`. No query rewriting, reranking, or decomposition yet.

`rag_build_guide.md` is the generic tutorial scaffold these were worked through
against. It is not HawkEye documentation and describes dependencies this project
no longer uses (OpenAI, Chroma, litellm).

## These do not reflect production

Everything here predates the migration to Supabase + pgvector. The mechanical
chunking in `day2` was replaced by LLM-driven chunking in
[`implementation/ingest.py`](../implementation/ingest.py), and the naive
retrieval in `day3` by the decompose → rewrite → dual-retrieve → rerank pipeline
in [`implementation/answer.py`](../implementation/answer.py). See the
[README](../README.md) for what actually runs, and
[`docs/archive/`](../docs/archive/) for a detailed snapshot of the intermediate
Chroma-era design.

## Running them

Mostly of historical interest — they aren't reproducible from a fresh clone,
since they read `knowledge-base/`, `vector_db/` and `preprocessed_db/`, all of
which are gitignored (the first carries gated internal docs; the other two are
large local stores). If you do have those directories, note that the notebooks
reference them by **bare relative path** (`"vector_db"`, `"knowledge-base"`), so
they need the **repo root** as the working directory, not `notebooks/`:

```bash
uv run jupyter lab --notebook-dir=.
```

`chromadb` and `ipykernel` are in the `dev` dependency group, so `uv sync` gets
them and `uv sync --no-dev` (what CI and the Dockerfile use) does not.

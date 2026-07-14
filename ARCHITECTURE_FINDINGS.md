# HawkEye Repo Investigation — Findings for Architecture Documentation

Investigated by reading every file directly (not reconstructed from memory of similar projects). All function names, model names, file paths, and line numbers below reflect the exact state of the repo as of commit `200754d` ("DAY 6", 2026-07-14) on branch `master`.

---

## 1. Current production pipeline — `implementation/answer.py`

### Module-level setup (lines 1–24)

- `MODEL = "gemini-2.5-flash-lite"` — the chat/generation model, used for every LLM call in this file (decomposition, rewriting, reranking, final answer generation).
- `EMBEDDING_MODEL = "gemini-embedding-001"` — used only for embedding queries.
- `DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")` — points at the repo-root `preprocessed_db/` Chroma store (**not** `vector_db/`).
- `COLLECTION_NAME = "docs"`
- `RETRIEVAL_K = 10` (chunks pulled per embed call), `FINAL_K = 5` (chunks kept after rerank).
- Global singletons instantiated at import time: `embeddings_model` (`GoogleGenerativeAIEmbeddings`), `chroma`/`collection` (Chroma `PersistentClient` + `get_or_create_collection("docs")`), `llm` (`ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)`).
- `wait = wait_exponential(multiplier=1, min=10, max=240)` — shared retry backoff used by every `@retry`-decorated function.
- `SYSTEM_PROMPT_TEMPLATE` — the final-answer system prompt, framed as talking to an IT help-desk **technician**, not the end customer, with a `{context}` placeholder.

### Pydantic models

- `Result(page_content: str, metadata: dict)` — the universal chunk container.
- `RankOrder(order: list[int])` — reranker output schema.
- `SubQuestions(sub_questions: list[str])` — decomposition output schema.

### Function-by-function

**`decompose_question(question: str) -> list[str]`** (line 66, `@retry`)
Chat model call via `llm.with_structured_output(SubQuestions)`. Determines if the technician's question bundles multiple unrelated asks; returns either `[question]` unchanged or a list of split, self-contained sub-questions. **Model: chat (gemini-2.5-flash-lite), structured output.**

**`rewrite_query(question: str, history: list[dict] = []) -> str`** (line 86, `@retry`)
Chat model call (`llm.invoke(prompt)`, plain, not structured). Compresses conversation history + the current (sub-)question into one short, specific search-query string. **Model: chat.**

**`fetch_chunks(query: str, k: int = RETRIEVAL_K) -> list[Result]`** (line 111, no retry decorator)
Embeds `query` via `embeddings_model.embed_query(query)` (**embedding model**), then does `collection.query(query_embeddings=[...], n_results=k)` against Chroma. Returns `k` `Result` objects built from `results["documents"][0]` / `["metadatas"][0]`.

**`merge_chunks(chunks_a, chunks_b) -> list[Result]`** (line 120, pure function, no model call)
Dedupes by `page_content` string equality, preserving order with `chunks_a` first.

**`rerank(question: str, chunks: list[Result]) -> list[Result]`** (line 132, `@retry`)
Chat model call via `llm.with_structured_output(RankOrder)`. Given the *original* question and all merged candidate chunks (numbered `CHUNK ID: n`), asks the LLM to return every chunk id reordered by relevance, then re-indexes `chunks` by that order. **Model: chat, structured output.**

**`fetch_context(question: str, history: list[dict] = []) -> list[Result]`** (line 153)
Orchestrator — see full call trace below.

**`answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Result]]`** (line 171)
Calls `fetch_context`, builds a `"Source: {meta['source']}\n{page_content}"` block per doc joined with `"\n\n"`, formats it into `SYSTEM_PROMPT_TEMPLATE`, builds `[SystemMessage] + convert_to_messages(history) + [HumanMessage(question)]`, calls `llm.invoke(messages)` (**chat model — note `answer_question` itself is NOT `@retry`-wrapped**, only `decompose_question`/`rewrite_query`/`rerank` are). Returns `(response.content, docs)`.

### Full call trace for one user question, in order

1. `answer_question(question, history)` is entered.
2. → `fetch_context(question, history)`
   1. → `decompose_question(question)` — **runs once**. Returns `sub_questions` (a list of 1+ strings).
   2. **Loop over `sub_questions`** (once per sub-question: 1 iteration for a single-ask question, N iterations for a compound one). Each iteration:
      - a. → `rewrite_query(sub_q, history)` — chat call, produces `rewritten`.
      - b. → `fetch_chunks(sub_q)` — embed + Chroma query on the *original* (sub-)question text → `chunks_original`.
      - c. → `fetch_chunks(rewritten)` — embed + Chroma query on the *rewritten* query → `chunks_rewritten`.
      - d. → `merge_chunks(chunks_original, chunks_rewritten)`, then → `merge_chunks(all_chunks, <that result>)` — accumulates into a single `all_chunks` list across loop iterations, deduped by `page_content`.
   3. After the loop → `rerank(question, all_chunks)` — **runs once**, reranks against the *original, undecomposed* question, not the sub-questions.
   4. Returns `reranked[:FINAL_K]` (top 5).
3. Back in `answer_question`: build context string from the 5 returned `Result`s, format system prompt, call `llm.invoke(messages)` — **runs once**, this is the actual answer generation.
4. Return `(answer_text, docs)`.

**Cost per question:** single-ask → 1 decompose + 1 rewrite + 2 embed/query + 1 rerank + 1 generate = **4 chat calls + 2 embedding calls**. Two-part compound question → 1 decompose + 2×(1 rewrite + 2 embed/query) + 1 rerank + 1 generate = **5 chat calls + 4 embedding calls**.

---

## 2. Ingestion pipeline — `implementation/ingest.py`

**Config:** same `MODEL`/`EMBEDDING_MODEL`/`DB_NAME`/`COLLECTION_NAME` as `answer.py`. `KNOWLEDGE_BASE_PATH = .../knowledge-base`. `AVERAGE_CHUNK_SIZE = 500` chars (used only to *suggest* a chunk count to the LLM, not to mechanically split). `WORKERS = 3` (multiprocessing pool size for chunking), `EMBED_BATCH_SIZE = 50`.

**`fetch_documents() -> list[dict]`** (line 62) — "Homemade DirectoryLoader." Walks `knowledge-base/*/` (each top-level folder = `doc_type`), globs `**/*.md`, reads raw text, strips embedded base64 images via `strip_embedded_images()` (regex `BASE64_IMAGE_PATTERN`), returns `{"type", "source", "text"}` dicts. **No frontmatter stripping here** (unlike day1/day2 notebooks) — raw file text including YAML frontmatter is kept in `"text"`. No model calls.

**`make_prompt(document: dict) -> str`** (line 82) — builds the LLM chunking prompt; computes a suggested chunk count `len(text)//500 + 1` and instructs ~25% / ~50-word overlap.

**`process_document(document: dict) -> list[Result]`** (line 111, `@retry`)
**This is the chunking step — LLM-driven, not mechanical.** Calls `llm.with_structured_output(Chunks)` (chat model) with the prompt from `make_prompt`, gets back a `Chunks` object (`list[Chunk]`), and converts each `Chunk` to a `Result` via `Chunk.as_result(document)`, which concatenates `headline + "\n\n" + summary + "\n\n" + original_text` as the text that will be embedded, with `metadata = {"source": document["source"], "type": document["type"]}`.

**`create_chunks(documents: list[dict]) -> list[Result]`** (line 118)
Runs `process_document` over all documents in a `multiprocessing.Pool(processes=3)` via `pool.imap_unordered`, aggregating all `Result`s. **No embedding here — chunking only.**

**`load_chunks_cache(path="chunks_cache.jsonl") -> list[Result]`** (line 130)
Reads pre-computed `Result` objects back from `chunks_cache.jsonl` (one JSON object per line, `Result.model_validate_json`). **This function is defined but never called anywhere else in the codebase** (confirmed via repo-wide grep). `chunks_cache.jsonl` exists in the repo root with 3,974 lines/chunks, so it was populated by some ad-hoc/interactive run (e.g. a REPL session serializing `create_chunks()` output), not by any function currently in the file. There is also no function in the repo that *writes* to `chunks_cache.jsonl` — it's a manually-maintained cache, kept in git via a `.gitignore` carve-out (`# Chunk cache (regeneratable via ingest.py, kept in Drive backup)`), even though the round-trip logic to regenerate it isn't actually wired up as a runnable script.

**`embed_batch(texts: list[str]) -> list[list[float]]`** (line 142, `@retry`) — thin wrapper on `embeddings_model.embed_documents(texts)`. **Embedding model call.**

**`create_embeddings(chunks: list[Result], reset: bool = False) -> None`** (line 146)
Connects to `preprocessed_db` Chroma store. If `reset=True`, deletes the existing `"docs"` collection first. Otherwise: `existing_ids = set(collection.get(include=[])["ids"])` — **this is the resume/caching logic**: chunk ids are just `str(index)` (positional, based on batch start offset), so for any batch of `EMBED_BATCH_SIZE=50` chunks it filters out any id already present in Chroma (`todo = [(i,c) for i,c in zip(ids,batch) if i not in existing_ids]`) before calling `embed_batch` and `collection.add(...)`. This means re-running ingestion after an interruption skips chunks whose *positional id* was already embedded, without re-calling the embedding API for them — but it depends on `chunks` being passed in the exact same order every time (since ids are positional, not content-hashed).

**`__main__` block** (line 172): **does not run the ingestion pipeline at all.** It only reconnects to `preprocessed_db`, embeds the literal string `"How do I reset my password?"`, queries `n_results=3`, and prints the top 3 chunks. So `python -m implementation.ingest` as currently written is a smoke-test/sanity-check, not a way to (re)build the store — `fetch_documents()` → `create_chunks()` → `create_embeddings()` must be invoked manually (interactively or from another script) to actually run ingestion.

**What's stored in `preprocessed_db`:** Chroma collection `"docs"`, each entry has `id` (positional string index), `embeddings` (from `gemini-embedding-001`, matching `page_content = headline + summary + original_text`), `documents` = that same concatenated text, `metadatas` = `{"source": <file path>, "type": <doc_type folder name>}`. Notably, unlike `vector_db` (Day 2/3), `preprocessed_db`'s metadata does **not** carry `title`, `url`, `tags`, etc. — only `source` and `type`.

---

## 3. Day-by-day notebook history

Only `day1.ipynb`, `day2.ipynb`, and `day3.ipynb` exist as notebooks in this repo — **there is no `day4.ipynb` or `day5.ipynb`** (confirmed via filesystem search and full git history — `git log --all -- day4.ipynb day5.ipynb` returns nothing). The commit log shows a single `"Day 4-5"` commit (`edac8d6`) that introduced `implementation/ingest.py`, `implementation/answer.py`, `evaluation/`, and `evaluator.py` directly as `.py` modules, skipping the notebook format entirely for those two days. So Day 4/5 concepts exist only in production-module form, not notebook form, in this repo.

### `day1.ipynb` — Naive baseline

Deliberately deviates from a generic "guide" (see §7 below): uses a real ~67-article subset of `knowledge-base/Hardware`, strips YAML frontmatter via `python-frontmatter`, builds a `word -> list[(title, body)]` inverted index (not `word -> single doc`) since article titles are multi-word, and calls Gemini (`gemini-2.5-flash-lite`) via the raw `google.genai` SDK instead of OpenAI, translating Gradio's OpenAI-shaped history into `types.Content(role="model"/"user")` objects. Ends with `gr.ChatInterface(chat).launch()`, which threw a `pydantic.ValidationError` in the last executed cell (Gradio passed history content as a list of `{'text':..., 'type':'text'}` dicts instead of a string — an unresolved bug left in the notebook's output).

**Not reflected in production:** the entire keyword/inverted-index retrieval approach, the raw `google.genai` client usage, and the Hardware-only subset — production uses the full 8-category KB with vector retrieval via LangChain wrappers.

### `day2.ipynb` — Real chunking + persistent vector store

Loads all 854 raw `.md` documents across all 8 `knowledge-base/` category folders (frontmatter stripped via `python-frontmatter`, metadata kept separately), builds LangChain `Document` objects, splits mechanically with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` → 7,676 chunks, embeds with `gemini-embedding-001` via `GoogleGenerativeAIEmbeddings` + `langchain_chroma.Chroma`, storing into `vector_db/` (batch size 100, resumable by checking `vectorstore._collection.count()`). Ends with 2D/3D t-SNE + Plotly visualization colored by `doc_type`.

**Not reflected in production:** mechanical/character-based chunking (production uses LLM-driven chunking with headline/summary/original_text), `RecursiveCharacterTextSplitter`, and the `vector_db` store itself — production reads/writes `preprocessed_db`, a completely separate store built by `ingest.py`'s LLM-chunking pipeline.

### `day3.ipynb` — Minimal end-to-end RAG

Reconnects to the Day 2 `vector_db` Chroma store (same embedding model), builds `retriever = vectorstore.as_retriever()` and `llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)`, defines `answer_question(question, history)` as pure retrieve → format → generate (no rewriting, no reranking, no decomposition), and wires it to `gr.ChatInterface(answer_question).launch()`. Uses the *exact same* `SYSTEM_PROMPT_TEMPLATE` string that's still in `implementation/answer.py` today (verbatim match).

**Not reflected in production:** the LangChain `retriever.as_retriever()` abstraction and single-shot retrieval with no query rewriting/dual retrieval/reranking/decomposition.

### Is Day 3's `vector_db` + LangChain retriever approach still used anywhere?

**No — fully superseded.** `implementation/answer.py` and `implementation/ingest.py` both point at `preprocessed_db`, not `vector_db`, and use `chromadb.PersistentClient` + manual `collection.query()`/`collection.add()` directly rather than LangChain's `Chroma`/`retriever.as_retriever()` wrapper (LangChain is only still used for `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings` model wrappers and message types, not for vector-store or retrieval abstractions).

The one place the LangChain `Chroma` wrapper *is* still used is `evaluation/generate_spanning.py`, which imports `langchain_chroma.Chroma` and reads against `vector_db` (`DB_NAME = "vector_db"`) to mine "spanning" (multi-hop) test questions from that older store — this is a test-set-generation utility, not part of the runtime answer pipeline. Both `vector_db/` (177MB) and `preprocessed_db/` (84MB) still physically exist on disk.

---

## 4. Evaluation system

### `evaluation/test.py`

Defines `TestQuestion(BaseModel)`: `question: str`, `keywords: list[str]` (must appear in retrieved context), `reference_answer: str`, `category: str`. `load_tests(path=TEST_FILE) -> list[TestQuestion]` reads `evaluation/tests.jsonl` line-by-line (`TEST_FILE = str(Path(__file__).parent / "tests.jsonl")`).

### `evaluation/tests.jsonl`

Contains **27 test cases** (`wc -l` = 27), each a JSON object with `question` (phrased as a technician relaying a customer's issue, e.g. `"customer is asking how to set up text message verification..."`), `keywords` (2–4 specific phrases pulled from source articles), `reference_answer`, and `category` (values seen in the file: `procedural`, `troubleshooting`; per `generate_tests.py`'s intended taxonomy the full set is also `direct_fact`, `policy`, `numerical`, `relationship`; `generate_spanning.py` additionally tags multi-hop questions `category: "spanning"`).

These are hand-curated/reviewed outputs originally produced by two generator scripts:
- **`evaluation/generate_tests.py`** — stratified sampling (`DOCS_PER_CATEGORY = 3` per knowledge-base folder), an LLM prompt against `ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)` with `structured_llm = llm.with_structured_output(DraftTest)`, writes to `tests_draft.jsonl` for manual review (with `_source` field for reviewer reference, meant to be stripped before merging into the final `tests.jsonl`).
- **`evaluation/generate_spanning.py`** — finds nearest-neighbor chunk pairs from *different* sources in `vector_db` (via `vectorstore.similarity_search`), asks an LLM (`SpanningCandidate` structured output) whether a genuine two-hop question is possible that requires both excerpts, writes candidates to `spanning_draft.jsonl` (target count 5, up to 30 attempts).

Both draft generators are separate from the final `tests.jsonl`, which is the reviewed/merged output actually used at eval time.

### `evaluation/eval.py`

- **`evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval`** — calls `fetch_context(test.question)` (imported from `implementation.answer`, so this exercises the **full production retrieval pipeline including decompose/rewrite/rerank**), then for each keyword computes `calculate_mrr` (reciprocal rank of first chunk containing the keyword, case-insensitive substring match) and `calculate_ndcg` (binary relevance, discounted by `log2(rank+1)`, normalized against ideal ordering, `k=10`), plus `keyword_coverage` (% of keywords found anywhere in results). Returns a `RetrievalEval` with `mrr`, `ndcg`, `keywords_found`, `total_keywords`, `keyword_coverage`.
- **`evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]`** — calls `answer_question(test.question)` (again, full production pipeline, generating a real answer), then sends `{question, generated_answer, reference_answer}` to a **separate judge LLM** (`judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)`, module-level, distinct instance from `answer.py`'s `llm` but same model) via `judge_llm.with_structured_output(AnswerEval)`. `AnswerEval` has `feedback: str`, `accuracy`, `completeness`, `relevance` (each 1–5 floats, with explicit judge-prompt rules: any wrong answer must score accuracy=1, completeness=5 only if all reference info present, relevance=5 only if no extraneous info).
- **`evaluate_all_retrieval()` / `evaluate_all_answers()`** — generators that `yield (test, result, progress)` per test over `load_tests()`, for streaming progress into a UI.
- **`run_cli_evaluation(test_number)` / `main()`** — CLI entry (`uv run python -m evaluation.eval <row>`) that prints both retrieval and answer evaluation for one test row.

### `evaluator.py` (repo root)

The only Gradio UI in the repo (there is no `app.py` — see §5). `main()` builds a `gr.Blocks(title="RAG Evaluation Dashboard")` with two independent sections:

- **"Retrieval Evaluation"**: a button wired via `retrieval_button.click(fn=run_retrieval_evaluation, outputs=[retrieval_metrics, retrieval_chart])`. `run_retrieval_evaluation` drains `evaluate_all_retrieval()`, accumulates totals and a `category -> [mrr scores]` map, and returns (a) an HTML block of color-coded metric tiles (green ≥0.9 MRR/nDCG or ≥90% coverage, amber ≥0.75/75%, else red) via `format_metric_html`, and (b) a `pandas.DataFrame` of average MRR per category rendered as a `gr.BarPlot`.
- **"Answer Evaluation"**: same pattern via `answer_button.click(fn=run_answer_evaluation, ...)`, thresholds green ≥4.5/5, amber ≥4.0/5, bar-charting average accuracy per category.

So: clicking either button in the dashboard triggers a full loop over all 27 `tests.jsonl` rows through the *actual* production `fetch_context`/`answer_question` functions (real API calls, no mocking), aggregates metrics, and renders both a scorecard and a per-category bar chart.

---

## 5. App layer

**There is no `app.py` anywhere in this repository.** Full repo search (excluding `.venv`) confirms the only Python files are `implementation/{answer,ingest}.py`, `evaluation/{test,eval,generate_tests,generate_spanning}.py`, and `evaluator.py`.

`rag_build_guide.md` (a generic scaffold/checklist document, see §7) references an `app.py` with a `format_context` function and a `gr.Chatbot` + side panel, but that file does not exist in this repo's working tree or git history under that name.

The closest thing to a "chat UI" that actually exists is the Day 1 and Day 3 notebooks' inline `gr.ChatInterface(...)` cells (one-off, launched from within the notebook, not a standalone module), and `evaluator.py`'s Gradio dashboard (which is an eval/metrics UI, not a chat UI — it never calls `answer_question` interactively for a live user question, only in bulk over the test set).

**There is currently no production chat interface wired to `answer_question()`.** This should be flagged explicitly in any architecture diagram/documentation rather than assumed to exist.

---

## 6. Agentic feature added this session — `decompose_question`

Confirmed present at `implementation/answer.py:66-82`, and confirmed via `git show 200754d -- implementation/answer.py` that it was introduced in the most recent commit, `200754d "DAY 6"` (2026-07-14, i.e. today), which also deleted two now-gone ad-hoc test scripts (`implementation/test_retrieval.py`, `implementation/test_rewrite.py` — visible only as stale `.pyc` cache entries under `implementation/__pycache__/`, no longer in the working tree).

**Before this commit**, `fetch_context` was:

```python
def fetch_context(question, history=[]):
    rewritten = rewrite_query(question, history)
    chunks_original = fetch_chunks(question)
    chunks_rewritten = fetch_chunks(rewritten)
    merged = merge_chunks(chunks_original, chunks_rewritten)
    reranked = rerank(question, merged)
    return reranked[:FINAL_K]
```

i.e., exactly one rewrite call, one dual-retrieval (2 embed/query calls), one merge, one rerank — regardless of whether the question bundled multiple asks.

**After this commit** (current state), `fetch_context` first calls `decompose_question(question)`, then loops `rewrite_query` + dual `fetch_chunks` + `merge_chunks` once per sub-question, accumulating into a single `all_chunks` list, and only reranks once at the end against the *original* (non-decomposed) question (full trace in §1).

### Behavioral difference, single-ask vs. compound question

- **Single-ask question** (e.g. "customer can't connect to eduroam"): `decompose_question` returns `[question]` unchanged (per its own system prompt's explicit instruction), so the loop runs exactly once — behavior is functionally identical to the pre-Day-6 code, just with one extra LLM call (the decomposition check itself) that always returns a 1-element list. Retrieval volume and cost are unchanged except for that added classification call.
- **Compound question** (e.g. "customer wants to reset their password and also asks about the print quota"): `decompose_question` splits it into N self-contained sub-questions. The loop then independently rewrites and dual-retrieves for *each* sub-question, so retrieval breadth scales linearly with N (2N embed/Chroma-query calls instead of 2, N rewrite calls instead of 1) before everything is merged into one deduped pool and reranked once against the full original question. This directly targets a failure mode the single-shot pipeline had: a single embedding of a two-topic question tends to retrieve chunks relevant to neither topic well, whereas decomposing first lets each topic get its own targeted retrieval pass before the results are pooled and re-ranked together.

---

## 7. Other structurally relevant files

- **`scraper/scrape_kb.py`** — Public scraper for SUNY New Paltz's TeamDynamix KB (`BASE_URL = "https://newpaltz.teamdynamix.com"`). Recursively crawls 8 hardcoded `TOP_LEVEL_CATEGORIES` URLs (`Internal-Documentation`, `Getting-Started-Guides`, `Accounts-Access-Security`, `Software-and-Apps`, `Hardware`, `Networking-WiFi`, `Digital-Accessibility`, `Policies`), parses each article's title/breadcrumb/tags/dates/body-as-Markdown (via `markdownify`), and writes one JSON file per article into `scraper/output/{category}/`. No model calls, no dependency from production code.
- **`scraper/scrape_kb_authenticated.py`** — Same logic, but for the one gated category (`Internal-Documentation`) that requires an authenticated session; loads cookies from `scraper/cookies.txt` (Netscape format, gitignored, must be manually exported from a logged-in browser), writes to `scraper/output-internal/` separately so it can be reviewed before merging.
- **`scraper/convert_to_markdown.py`** — Converts the scraped JSON (`scraper/output/**/*.json`) into the final `knowledge-base/{Category}/{slug}.md` files with YAML frontmatter (`title`, `tags`, `category_path`, `published_date`, `modified_date`, `url`, `article_id`) via `python-frontmatter`. This is the actual producer of everything under `knowledge-base/`.

  **Dependency status:** nothing in `implementation/` or `evaluation/` calls into `scraper/` at runtime — `knowledge-base/` is a static, already-generated artifact (gitignored, "regeneratable via scraper scripts" per `.gitignore`) that `ingest.py`'s `fetch_documents()` reads directly. The scraper is a one-time/occasional data-refresh tool, not part of the live pipeline.

- **`.env`** — two keys actually present: `HF_TOKEN` and `GOOGLE_API_KEY` (values redacted, not read here). Only `GOOGLE_API_KEY` is actually consumed by the current code path (`langchain_google_genai` reads it implicitly); `HF_TOKEN` isn't referenced by any `.py` file found in the repo (`langchain-huggingface` is a listed dependency in `pyproject.toml` but unused by `implementation/`/`evaluation/`/`evaluator.py`).
- **`chunks_cache.jsonl`** (repo root, 3,974 lines) — cache of pre-chunked `Result` objects for `preprocessed_db`; see §2, it's read-only-if-called and currently has no caller in the codebase.
- **`preprocessed_db/`** (84MB, Chroma `PersistentClient` store, collection `"docs"`) — the store actually used by production `answer.py`/`ingest.py`.
- **`vector_db/`** (177MB, LangChain `Chroma` store) — the Day 2/3 store, still on disk, still read by `evaluation/generate_spanning.py`, otherwise unused by production.
- **`pyproject.toml`** — notable dependencies not otherwise mentioned above: `langchain-openai`, `openai`, `google-genai`, `langchain-huggingface`, `tiktoken`, `scikit-learn`, `plotly` — these support the notebooks (Day 1's raw `google.genai` client, Day 2's t-SNE/plotly viz, token counting) but are not imported by `implementation/` or `evaluation/`.
- **`rag_build_guide.md`** (repo root) — **this is a generic, aspirational scaffold document** ("A step-by-step scaffold based on the 'Insurellm' RAG week"), not a description of this specific repo's actual state. It explicitly describes things that do not exist here: an `app.py` with `format_context`/`combined_question`, a `gpt-4.1-nano` chunking model, a Groq-hosted `gpt-oss-120b` answer model, `litellm.completion` for provider-swapping, and OpenAI embeddings — none of which are true of this repo (which uses `gemini-2.5-flash-lite` + `gemini-embedding-001` exclusively via `langchain_google_genai`, no litellm, no Groq, no app.py).

  **Flag this explicitly to whoever builds the architecture diagram:** treat `rag_build_guide.md` as a template/reference the developer was following loosely, not as ground truth for this codebase — always prefer what's actually in `implementation/`, `evaluation/`, and `evaluator.py`.

- **`README.md`** — effectively empty (1 blank line).
- **`evaluation/__pycache__/`, `implementation/__pycache__/`** — compiled bytecode only; the `implementation/__pycache__` entries for `test_decompose_e2e.py`/`test_rewrite.py` are stale remnants of deleted ad-hoc test scripts (see §6), not evidence of currently-existing files.

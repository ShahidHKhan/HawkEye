# Building a RAG System: From Naive to Production-Grade

A step-by-step scaffold based on the "Insurellm" RAG week, organized by day. Each day builds on the last — start at Day 1 even if your end goal is the Day 5 "pro" pipeline, since the pro version is a rewrite of the same ideas, not a separate thing.

---

## Day 0: Environment Setup

Before any of the notebooks/modules run, get a reproducible environment with `uv`.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the project
uv init my-rag-project
cd my-rag-project

# Create/activate a virtual environment
uv venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

Add the core dependencies you'll need across all five days:

```bash
uv add openai python-dotenv gradio pandas numpy
uv add langchain-openai langchain-chroma langchain-huggingface langchain-community langchain-text-splitters
uv add chromadb litellm tenacity tqdm pydantic
uv add scikit-learn plotly tiktoken
```

Create a `.env` file in your project root:

```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...      # only needed if you use Groq models in Day 5
HF_TOKEN=...              # only needed if you use HuggingFace embeddings
```

Recommended folder layout (matches what the later modules assume):

```
my-rag-project/
├── .env
├── knowledge-base/        # your source .md documents, one subfolder per doc_type
│   ├── employees/
│   ├── products/
│   ├── contracts/
│   └── company/
├── vector_db/              # created by Day 2 ingestion (LangChain path)
├── preprocessed_db/         # created by Day 5 ingestion (pro path)
├── implementation/
│   ├── ingest.py
│   └── answer.py
├── evaluation/
│   ├── test.py
│   ├── eval.py
│   └── tests.jsonl
├── app.py
└── evaluator.py
```

Every module below calls `load_dotenv(override=True)` at import time — keep that pattern so `.env` values always win over any stale shell exports.

---

## Day 1: The Naive Baseline (no vector DB at all)

**Goal:** prove the concept — a chatbot that gets smarter by stuffing relevant text into the system prompt — before introducing any infrastructure.

1. **Load every document into memory as a flat dictionary**, keyed by a recognizable name (employee surname, product slug, etc.):
   ```python
   knowledge = {}
   for filename in glob.glob("knowledge-base/employees/*"):
       name = Path(filename).stem.split(" ")[-1].lower()
       knowledge[name] = open(filename, encoding="utf-8").read()
   ```
2. **Write a keyword-matching retriever.** Strip punctuation, lowercase, split into words, and pull any dictionary entry whose key appears as a whole word in the question:
   ```python
   def get_relevant_context(message):
       text = "".join(ch for ch in message if ch.isalpha() or ch.isspace())
       words = text.lower().split()
       return [knowledge[w] for w in words if w in knowledge]
   ```
3. **Build a system prompt template** with a placeholder for whatever context was found, and a fallback string ("no additional context relevant") when nothing matched.
4. **Wire it to `openai.chat.completions.create`** — no framework, just the raw SDK — passing `[system] + history + [user]`.
5. **Prototype the UI with `gr.ChatInterface(chat, type="messages")`** — this is the fastest way to get a working chat window, and Gradio manages the `history` list for you.

**What this teaches:** RAG is fundamentally "find relevant text, put it in the prompt." Everything from here on is about doing "find relevant text" better than exact keyword matching.

**Limitation to notice:** this only works if the question contains a word that's an exact dictionary key. It can't handle synonyms, typos, or questions about concepts spread across multiple documents — which motivates Day 2.

---

## Day 2: Real Chunking + a Vector Store

**Goal:** replace the toy dictionary with actual chunking, embeddings, and a persistent vector database (Chroma), then *look* at what you built.

### Part A — Load and chunk
1. Use LangChain's `DirectoryLoader` to load every `.md` file per subfolder, tagging each with `doc_type` metadata from the folder name:
   ```python
   loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
   ```
2. Split with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`. Overlap matters — it keeps a sentence from being severed exactly at a chunk boundary and losing its context.
3. Sanity-check size: count characters and tokens (`tiktoken.encoding_for_model(MODEL)`) across the whole knowledge base so you know roughly what you're embedding.

### Part B — Embed and store
4. Pick an embedding model. Two options, both used across the notebooks:
   - `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` — free, runs locally, lower dimensionality.
   - `OpenAIEmbeddings(model="text-embedding-3-large")` — paid, higher quality, higher dimensionality.
5. Build the store: `Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)`. Delete any existing collection first if you're re-ingesting, or you'll get duplicates.
6. Inspect what you made: `collection.count()` and the dimensionality of a sample embedding (`collection.get(limit=1, include=["embeddings"])`).

### Part C — Visualize
7. Pull every embedding + its `doc_type` back out of Chroma, reduce to 2D/3D with `sklearn.manifold.TSNE`, and plot with `plotly.graph_objects` (`Scatter` for 2D, `Scatter3d` for 3D), coloring points by `doc_type`. This is a genuinely useful debugging step — clusters that don't separate by document type suggest your chunking or embedding choice isn't capturing meaningful structure.

**What this teaches:** chunking strategy and embedding model choice are the two biggest levers on retrieval quality, and both are cheap to inspect visually before you ever ask an LLM a question.

---

## Day 3: Assembling the Actual RAG Pipeline

**Goal:** connect the Day 2 vector store to an LLM and get a real chat loop, using LangChain's abstractions end-to-end.

1. Reconnect to the persisted store: `Chroma(persist_directory=DB_NAME, embedding_function=embeddings)` — note you must use the *same* embedding model you ingested with.
2. Create the two key objects:
   ```python
   retriever = vectorstore.as_retriever()
   llm = ChatOpenAI(temperature=0, model_name=MODEL)
   ```
   `temperature=0` means "always pick the most likely token" — appropriate for a factual assistant that shouldn't improvise.
3. Write the RAG system prompt template with a `{context}` placeholder.
4. Write `answer_question`:
   ```python
   def answer_question(question, history):
       docs = retriever.invoke(question)
       context = "\n\n".join(doc.page_content for doc in docs)
       system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
       response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
       return response.content
   ```
5. Plug straight into `gr.ChatInterface(answer_question).launch()`.

**What this teaches:** the minimal shape of RAG is retrieve → format into prompt → generate. Everything from Day 2 (good chunks, good embeddings) determines whether step 1 returns anything useful; everything from Day 5 is about making step 1 and step 2 smarter.

At this point you also have a standalone `implementation/answer.py` + `implementation/ingest.py` module pair (see doc 1 & 2 from your first upload) that generalizes this into a small library: `fetch_context()`, `combined_question()` (concatenates prior user turns with the new question before embedding — a cheap way to keep multi-turn context in retrieval), and `answer_question()` returning both the answer *and* the source documents so a UI can display them (see `app.py`'s `format_context`).

---

## Day 4: Evaluation

**Goal:** stop eyeballing answers — build a test set and quantitative metrics for both retrieval and generation quality.

### Build the test set
1. Define a schema (`evaluation/test.py`):
   ```python
   class TestQuestion(BaseModel):
       question: str
       keywords: list[str]        # must appear in retrieved context if retrieval "worked"
       reference_answer: str
       category: str              # direct_fact, temporal, spanning, comparative, numerical, relationship, holistic
   ```
2. Store as JSONL (`tests.jsonl`), one question per line. Categorize deliberately — a good test set covers simple lookups *and* harder multi-hop/comparative questions, not just easy ones.

### Retrieval metrics (`evaluate_retrieval`)
3. **MRR (Mean Reciprocal Rank)** — for each keyword, find the rank of the first retrieved chunk containing it (case-insensitive substring match), score `1/rank`, average across keywords. Rewards keywords showing up *early*.
4. **nDCG (Normalized Discounted Cumulative Gain)** — binary relevance per chunk (keyword present or not), discounted by `log2(rank+1)`, normalized against the ideal ordering. More forgiving than MRR about multiple relevant chunks.
5. **Keyword coverage** — simple percentage of keywords found anywhere in the top-k at all.

### Answer metrics (`evaluate_answer`) — LLM-as-judge
6. Generate the answer with your RAG pipeline, then send `{question, generated_answer, reference_answer}` to a judge LLM with a **structured output schema**:
   ```python
   class AnswerEval(BaseModel):
       feedback: str
       accuracy: float      # 1-5, must be 1 if the answer is factually wrong
       completeness: float  # 1-5, only 5 if every fact from the reference is present
       relevance: float     # 1-5, only 5 if it answers the question and nothing extraneous
   ```
7. Key judge-prompt discipline: tell the model explicitly what a 1 and a 5 mean for each dimension, and force it via `response_format` — free-text judge output is much noisier to parse and to keep consistent.

### Aggregate + visualize
8. Loop `evaluate_all_retrieval()` / `evaluate_all_answers()` as generators that `yield (test, result, progress)` per test — this lets a Gradio progress bar update live instead of blocking.
9. Build a small dashboard (`evaluator.py`): average each metric, color-code against green/amber/red thresholds, and bar-chart the average score per question `category`. Category-level breakdown is the single most useful view — it tells you *which kind* of question your system is bad at, not just an overall score.
10. Keep a CLI path too (`eval.py`'s `run_cli_evaluation`) for debugging one specific test row without launching the full UI.

**What this teaches:** you need two separate scores — did retrieval find the right text, and did the LLM use it well — because a bad final answer could be caused by either one, and you fix them differently (chunking/embeddings vs. prompt/model).

---

## Day 5: Going Pro — Advanced Retrieval Techniques

**Goal:** once Day 4 tells you *where* the system is weak, here's the toolbox for fixing retrieval specifically, applied without LangChain for full control.

### Pro ingestion (`implementation/ingest.py`)
1. **Drop LangChain's loaders** — write your own `fetch_documents()` walking the knowledge base with `Path.iterdir()` / `rglob("*.md")`. Simple, and removes a dependency you don't need once you're doing custom chunking anyway.
2. **Replace mechanical chunking with LLM-driven chunking.** For each document, prompt an LLM to split it into `Chunk` objects, each with:
   - `headline` — short, query-shaped heading
   - `summary` — a few sentences distilling the chunk
   - `original_text` — the verbatim source text, unmodified
   
   Suggest a target chunk count (`len(text) // AVERAGE_CHUNK_SIZE + 1`) but let the model deviate. Ask for ~25% overlap between chunks, same rationale as Day 2's `chunk_overlap`, just LLM-judged instead of character-counted.
3. **Store `headline + summary + original_text` concatenated as the embedded text**, not just the raw chunk. Embedding a distilled, query-shaped headline/summary alongside the source text tends to retrieve better than raw prose alone, because it's closer in style to how users phrase questions.
4. **Enforce the shape with structured outputs**: `completion(model=MODEL, messages=messages, response_format=Chunks)` where `Chunks` wraps `list[Chunk]` — validate with `Chunks.model_validate_json(reply)`.
5. **Parallelize** document processing with `multiprocessing.Pool(processes=WORKERS)` — LLM calls are I/O-bound and independent per document, so this is a straightforward throughput win. Drop `WORKERS` to 1 if you hit rate limits.
6. **Wrap every LLM call in retry logic**: `@retry(wait=wait_exponential(multiplier=1, min=10, max=240))` from `tenacity`. LLM APIs fail transiently far more often than a plain function call would; naive chunking never needed this, LLM-driven chunking does.
7. Embed with the raw `openai.embeddings.create()` call (batched — pass the whole `texts` list in one request) and store directly via `chromadb.PersistentClient` + `collection.add(ids=..., embeddings=..., documents=..., metadatas=...)`. No LangChain wrapper in between.
8. Re-run the same t-SNE/plotly visualization from Day 2 against this new store as a gut-check that the LLM-chunked embeddings still cluster sensibly by `doc_type`.

### Pro retrieval (`implementation/answer.py`)
9. **Query rewriting.** Before embedding the user's question, ask an LLM to compress the conversation history + current question into one short, specific search query. This helps most on multi-turn conversations where the literal last message is underspecified ("what about her manager?" only makes sense with history attached).
10. **Dual retrieval + merge.** Embed *both* the original question and the rewritten query, retrieve `RETRIEVAL_K` chunks for each, and merge/dedupe by `page_content`. This hedges against the rewrite occasionally being worse than the original phrasing.
11. **LLM reranking.** Send all merged candidate chunks back to an LLM with numbered `CHUNK ID`s and ask it to return `RankOrder.order` — a re-sorted list of IDs — via structured output. Take only the top `FINAL_K`. This is the single highest-leverage step in the pro pipeline: initial vector search is fast but approximate; a reranker gets to read the actual question against actual chunk text and make a more informed relevance judgment.
12. Keep the same `make_rag_messages` shape as Day 3, but cite `chunk.metadata["source"]` inline in the context block — this is what lets `app.py`'s `format_context` show users *where* an answer came from.
13. Same `tenacity` retry decorator on every network call (`rewrite_query`, `rerank`, `answer_question`).
14. Model choice as a deliberate cost/latency tradeoff: cheap model (`gpt-4.1-nano`) for the bulk ingestion-time chunking calls (many calls, less individually critical), a fast model (e.g. `groq/openai/gpt-oss-120b` via Groq) for the final user-facing answer where latency matters most. `litellm.completion` makes swapping providers a one-line change (`"provider/model-name"` string).

### Tie it together
15. `app.py` imports `answer_question` from whichever implementation you're running, keeps a `gr.Chatbot` + a side panel rendering retrieved context via `format_context`, and chains `message.submit(...).then(chat, ...)` so the UI shows the user's message immediately, then streams in the assistant's reply.
16. Re-run your Day 4 evaluation suite against the pro pipeline. Compare MRR/nDCG/coverage and accuracy/completeness/relevance against the Day 3 baseline, broken down by category — this is how you prove (or disprove) that each added technique was worth its extra cost and latency. Don't assume "pro" is strictly better for your data until the numbers say so.

---

## Suggested order for your project

1. Get Day 1 running first, even if trivially — it validates your `.env`, API keys, and Gradio setup with zero infrastructure risk.
2. Stand up Day 2's vector store and actually look at the t-SNE plot before trusting it.
3. Get Day 3's LangChain pipeline answering real questions end-to-end.
4. Build your Day 4 test set **early** — even 20-30 questions — so every change you make afterward is measured, not vibes-based.
5. Layer in Day 5 techniques one at a time (semantic chunking → query rewriting → dual retrieval → reranking), re-running evaluation after each, so you know which one is actually earning its keep for your specific knowledge base.

# HawkEye

An internal RAG (retrieval-augmented generation) knowledge assistant for SUNY New Paltz's IT help desk. Technicians describe a customer's issue in plain language and get a direct answer sourced from the department's internal knowledge base, with retrieved sources shown alongside the answer for verification.

**Pipeline:** scraped/authenticated ingestion of the TeamDynamix knowledge base → Markdown conversion with structured frontmatter → LLM-driven semantic chunking → embeddings in Postgres/pgvector (Supabase) → query decomposition + rewriting + dual retrieval + LLM reranking → answer generation → Gradio chat UI with query/feedback logging → automated retrieval + answer-quality evaluation harness.

**Stack:** Python, LangChain, Google Gemini (`gemini-2.5-flash-lite`, `gemini-embedding-001`), Postgres/pgvector (Supabase), ChromaDB (legacy store), Gradio, psycopg2.

---


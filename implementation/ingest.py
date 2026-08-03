import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 500  # chars — tune later if chunks come out too big/small

WORKERS = 3  # keep low for Gemini rate limits
EMBED_BATCH_SIZE = 50  # keep small to avoid rate limits

BASE64_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)')

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file")

db_pool = SimpleConnectionPool(1, 5, SUPABASE_DB_URL)


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def strip_embedded_images(text: str) -> str:
    """Replace embedded base64 image data with a short placeholder."""
    return BASE64_IMAGE_PATTERN.sub('[embedded image removed]', text)


def fetch_documents() -> list[dict]:
    """Homemade version of LangChain's DirectoryLoader - no LangChain needed."""
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            text = strip_embedded_images(text)
            documents.append({"type": doc_type, "source": file.as_posix(), "text": text})
    print(f"Loaded {len(documents)} documents")
    return documents


llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
wait = wait_exponential(multiplier=1, min=10, max=240)


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and split it into overlapping chunks for a KnowledgeBase.

The document is from the IT knowledge base of SUNY New Paltz.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

An IT help-desk assistant will use these chunks to answer technician questions.
You should divide up the document as you see fit, being sure that the entire document
is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have
more or less as appropriate, ensuring individual chunks can answer specific questions.
There should be overlap between chunks as appropriate; typically about 25% overlap or
about 50 words, so the same text appears in multiple chunks for best retrieval results.

For each chunk, provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


@retry(wait=wait, stop=stop_after_attempt(5))
def process_document(document: dict) -> list[Result]:
    structured_llm = llm.with_structured_output(Chunks)
    prompt = make_prompt(document)
    reply = structured_llm.invoke(prompt)
    return [chunk.as_result(document) for chunk in reply.chunks]


def create_chunks(documents: list[dict]) -> list[Result]:
    """
    Create chunks using a number of workers in parallel.
    If you get repeated rate-limit errors, drop WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def load_chunks_cache(path: str = "chunks_cache.jsonl") -> list[Result]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Result.model_validate_json(line))
    return chunks


def save_chunks_cache(chunks: list[Result], path: str = "chunks_cache.jsonl") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@retry(wait=wait, stop=stop_after_attempt(5))
def embed_batch(texts: list[str]) -> list[list[float]]:
    return embeddings_model.embed_documents(texts)


def embedding_to_vector_literal(embedding: list[float]) -> str:
    """Format a python float list as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def get_existing_chunk_keys() -> set[tuple[str, str]]:
    """
    (source, page_content) pairs already in Supabase — lets a partial ingest resume
    without re-embedding chunks that already made it in. Keyed on content rather than
    a positional index, since a bigserial primary key doesn't map to a batch offset
    the way Chroma's manually-assigned string ids used to.
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT source, page_content FROM chunks")
            return set(cur.fetchall())
    finally:
        db_pool.putconn(conn)


def create_embeddings(chunks: list[Result], reset: bool = False) -> None:
    if reset:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks")
            conn.commit()
        finally:
            db_pool.putconn(conn)

    existing_keys = get_existing_chunk_keys()
    print(f"{len(existing_keys)} chunks already embedded, resuming...")

    todo = [c for c in chunks if (c.metadata["source"], c.page_content) not in existing_keys]

    for start in tqdm(range(0, len(todo), EMBED_BATCH_SIZE)):
        batch = todo[start:start + EMBED_BATCH_SIZE]
        texts = [c.page_content for c in batch]
        vectors = embed_batch(texts)
        rows = [
            (c.metadata["source"], c.metadata.get("type"), c.page_content, embedding_to_vector_literal(v))
            for c, v in zip(batch, vectors)
        ]
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO chunks (source, type, page_content, embedding) VALUES %s",
                    rows,
                    template="(%s, %s, %s, %s::vector)",
                )
            conn.commit()
        finally:
            db_pool.putconn(conn)

    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            total = cur.fetchone()[0]
    finally:
        db_pool.putconn(conn)
    print(f"chunks table now has {total:,} rows")


def run_ingest(reset: bool = False, regenerate: bool = False, cache_path: str = "chunks_cache.jsonl") -> None:
    if regenerate or not Path(cache_path).exists():
        documents = fetch_documents()
        chunks = create_chunks(documents)
        save_chunks_cache(chunks, cache_path)
    else:
        print(f"Loading chunks from cache: {cache_path}")
        chunks = load_chunks_cache(cache_path)
    print(f"{len(chunks)} chunks ready to embed")
    create_embeddings(chunks, reset=reset)


def smoke_test() -> None:
    """Sanity-check retrieval against whatever is already in the chunks table."""
    query_embedding = embeddings_model.embed_query("How do I reset my password?")
    embedding_literal = embedding_to_vector_literal(query_embedding)

    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, page_content
                FROM chunks
                ORDER BY embedding <-> %s::vector
                LIMIT 3
                """,
                (embedding_literal,),
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)

    for source, page_content in rows:
        print("---")
        print(page_content[:300])
        print("source:", source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge-base/ into the Supabase chunks table")
    parser.add_argument("--reset", action="store_true", help="Truncate the chunks table before ingesting")
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Re-chunk knowledge-base/ via the LLM instead of loading the cache",
    )
    parser.add_argument("--cache-path", default="chunks_cache.jsonl", help="Path to the chunk cache file")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Skip ingestion; just run a sample similarity query against the existing table",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        run_ingest(reset=args.reset, regenerate=args.regenerate, cache_path=args.cache_path)

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.genai.errors import ServerError
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 500  # chars — tune later if chunks come out too big/small

WORKERS = 3  # keep low for Gemini rate limits
EMBED_BATCH_SIZE = 50  # keep small to avoid rate limits
# Longest document process_document will store whole when Gemini can't chunk it. About
# twice the largest LLM-made chunk -- still one coherent thing for retrieval to match.
FALLBACK_MAX_CHARS = 4000

BASE64_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)')

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file")

db_pool = ConnectionPool(
    SUPABASE_DB_URL,
    min_size=1,
    max_size=5,
    check=ConnectionPool.check_connection,
)


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


# timeout is per HTTP attempt: a normal chunking call takes ~5s, while a prompt Gemini
# gets stuck on hangs ~2 minutes before it 500s. max_retries=1 leaves retrying to the
# tenacity @retry on process_document -- the client's own default of 6 attempts stacked
# underneath it, which turned one stuck article into an hour of a refresh run.
LLM_TIMEOUT_SECONDS = 90
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
wait = wait_exponential(multiplier=1, min=10, max=240)


def prompt_source(source: str) -> str:
    """
    The source path as the chunking prompt shows it, relative to knowledge-base/. The
    stored source is an absolute path from whichever machine last ran ingest, which tells
    the model nothing -- and article 42989's (D:/mrsha/Projects/...) reliably made Gemini
    hang and 500 on every attempt, while the same prompt with a relative path chunked in ~5s.
    """
    normalized = source.replace("\\", "/")
    marker = "knowledge-base/"
    idx = normalized.lower().find(marker)
    if idx != -1:
        return normalized[idx + len(marker):]
    return normalized.split("/")[-1]


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and split it into overlapping chunks for a KnowledgeBase.

The document is from the IT knowledge base of SUNY New Paltz.
The document is of type: {document["type"]}
The document has been retrieved from: {prompt_source(document["source"])}

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
def chunk_with_llm(document: dict) -> list[Result]:
    structured_llm = llm.with_structured_output(Chunks)
    prompt = make_prompt(document)
    reply = structured_llm.invoke(prompt)
    check_chunks(reply.chunks, document)
    return [chunk.as_result(document) for chunk in reply.chunks]


def process_document(document: dict) -> list[Result]:
    """
    Chunk a document with the LLM, falling back to storing it whole as a single chunk
    when Gemini won't chunk it and it's short enough to be one.

    Some prompts make Gemini hang and 500 (or time out) on every attempt, and some make
    it loop -- deterministically, at temperature 0, so retrying never helps. Articles
    32121, 67137 and 169560 did this and couldn't be re-chunked at all. The fallback is
    deliberately narrow: only those failure modes and only short documents, so a real
    outage or a bad API key still fails loudly instead of quietly storing whole articles.
    """
    try:
        return chunk_with_llm(document)
    except RetryError as e:
        cause = e.last_attempt.exception()
        if not isinstance(cause, UNCHUNKABLE_ERRORS) or len(document["text"]) > FALLBACK_MAX_CHARS:
            raise
        print(
            f"WARNING: Gemini could not chunk {prompt_source(document['source'])} "
            f"({type(cause).__name__}); storing it whole as a single chunk."
        )
        return [whole_document_chunk(document)]


def whole_document_chunk(document: dict) -> Result:
    """The whole document as one chunk, headed by its title, the way an LLM chunk is."""
    filename = prompt_source(document["source"]).split("/")[-1]
    title = document.get("title") or re.sub(r"-\d+\.md$", "", filename).replace("-", " ")
    return Result(
        page_content=title + "\n\n" + document["text"].strip(),
        metadata={"source": document["source"], "type": document["type"]},
    )


class LoopedChunkError(ValueError):
    """The model returned a chunk longer than the document it was splitting."""


# What a prompt Gemini can't chunk looks like once every retry is spent: a hang that
# ends in a 500 or in our own timeout, or a reply that looped.
UNCHUNKABLE_ERRORS = (ServerError, httpx.TimeoutException, LoopedChunkError)


def check_chunks(chunks: list[Chunk], document: dict) -> None:
    """
    Reject a reply where some chunk's original_text is longer than the whole document,
    which no honest split can produce. It's the signature of the model looping on one
    token: article 156931 came back with a 1,024,995-char chunk that was almost entirely
    dashes from a table underline, and was stored and embedded as-is. Raising makes
    the @retry on chunk_with_llm try again, and if every attempt loops, process_document
    stores a short document whole and fails a long one -- never the garbage.
    """
    limit = len(document["text"]) + 500  # slack for whitespace/markdown the model normalizes
    for chunk in chunks:
        if len(chunk.original_text) > limit:
            raise LoopedChunkError(
                f"chunk original_text is {len(chunk.original_text):,} chars, longer than the "
                f"{len(document['text']):,}-char document -- the model looped; refusing to store it"
            )


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
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source, page_content FROM chunks")
            return set(cur.fetchall())


def create_embeddings(chunks: list[Result], reset: bool = False) -> None:
    if reset:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks")
            conn.commit()

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
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (source, type, page_content, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    rows,
                )
            conn.commit()

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            total = cur.fetchone()[0]
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

    with db_pool.connection() as conn:
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

import json
import re
from multiprocessing import Pool

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from pathlib import Path

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "gemini-embedding-001"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 500  # chars — tune later if chunks come out too big/small

WORKERS = 3  # keep low for Gemini rate limits
EMBED_BATCH_SIZE = 50  # keep small to avoid rate limits

BASE64_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)')


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


embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@retry(wait=wait, stop=stop_after_attempt(5))
def embed_batch(texts: list[str]) -> list[list[float]]:
    return embeddings_model.embed_documents(texts)


def create_embeddings(chunks: list[Result], reset: bool = False) -> None:
    chroma = PersistentClient(path=DB_NAME)
    if reset and COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    existing_ids = set(collection.get(include=[])["ids"])
    print(f"{len(existing_ids)} chunks already embedded, resuming...")

    for start in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE)):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        ids = [str(start + i) for i in range(len(batch))]
        # skip ids already done (lets us resume after a failure)
        todo = [(i, c) for i, c in zip(ids, batch) if i not in existing_ids]
        if not todo:
            continue
        todo_ids = [i for i, c in todo]
        todo_chunks = [c for i, c in todo]
        texts = [c.page_content for c in todo_chunks]
        vectors = embed_batch(texts)
        metas = [c.metadata for c in todo_chunks]
        collection.add(ids=todo_ids, embeddings=vectors, documents=texts, metadatas=metas)

    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    query_embedding = embeddings_model.embed_query("How do I reset my password?")
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print("---")
        print(doc[:300])
        print("source:", meta["source"])
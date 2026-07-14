from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt

from test_rewrite import rewrite_query

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "gemini-embedding-001"
RETRIEVAL_K = 10

RERANK_MODEL = "gemini-2.5-flash-lite"
FINAL_K = 5

embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

rerank_llm = ChatGoogleGenerativeAI(model=RERANK_MODEL, temperature=0)
wait = wait_exponential(multiplier=1, min=10, max=240)


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def fetch_chunks(query: str, k: int = RETRIEVAL_K):
    """Embed a query and retrieve top-k chunks as (page_content, metadata) pairs."""
    query_embedding = embeddings_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def merge_chunks(chunks_a, chunks_b):
    """Dedupe by page_content, preserving order (chunks_a first)."""
    merged = list(chunks_a)
    seen = {content for content, _ in chunks_a}
    for content, meta in chunks_b:
        if content not in seen:
            merged.append((content, meta))
            seen.add(content)
    return merged


def fetch_context_dual(question: str, history: list[dict] = []):
    rewritten = rewrite_query(question, history)
    print(f"Rewritten query: {rewritten}")

    chunks_original = fetch_chunks(question)
    chunks_rewritten = fetch_chunks(rewritten)
    merged = merge_chunks(chunks_original, chunks_rewritten)

    print(f"Original alone: {len(chunks_original)} | Rewritten alone: {len(chunks_rewritten)} | Merged: {len(merged)}")
    return merged


@retry(wait=wait, stop=stop_after_attempt(5))
def rerank(question: str, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Re-sort merged chunks by relevance to the original question via LLM judgment."""
    system_prompt = """
You are a document re-ranker for an IT help desk knowledge base.
You are given a question and a list of chunks retrieved from the knowledge base.
The chunks are roughly ordered by relevance already, but you may be able to improve on that.
Rank all the chunks by relevance to the question, most relevant first.
Include every chunk id you are given, reranked — don't drop any.
"""
    user_prompt = f"Question:\n{question}\n\nChunks:\n\n"
    for i, (content, meta) in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}\n{content}\n\n"
    user_prompt += "Reply with the ranked list of chunk ids."

    structured_llm = rerank_llm.with_structured_output(RankOrder)
    reply = structured_llm.invoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    return [chunks[i - 1] for i in reply.order]


if __name__ == "__main__":
    question = "customer can't get their printer to connect to wifi"
    merged = fetch_context_dual(question)

    print(f"\n--- Before rerank (top 5 of {len(merged)}) ---")
    for content, meta in merged[:5]:
        print(content[:80].replace("\n", " "))

    reranked = rerank(question, merged)[:FINAL_K]

    print(f"\n--- After rerank (top {FINAL_K}) ---")
    for content, meta in reranked:
        print(content[:80].replace("\n", " "))
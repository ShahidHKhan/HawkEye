import json
import random
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
DB_NAME = "vector_db"
OUTPUT_PATH = Path(__file__).parent / "spanning_draft.jsonl"

TARGET_COUNT = 5       # how many spanning questions we want to end up with
MAX_ATTEMPTS = 30      # how many candidate pairs to try before giving up
NEIGHBORS_PER_SEED = 5

wait = wait_exponential(multiplier=1, min=10, max=120)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.3)


class SpanningCandidate(BaseModel):
    possible: bool = Field(
        description=(
            "True only if a genuine question can be written that REQUIRES combining "
            "facts from BOTH excerpts — i.e. it is NOT answerable using either excerpt alone."
        )
    )
    question: str = Field(
        default="",
        description="The spanning question, phrased as a technician relaying a customer's issue. Only fill in if possible=True.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="2-4 specific keywords, drawn from BOTH excerpts, that must appear across the retrieved context. Only fill in if possible=True.",
    )
    reference_answer: str = Field(
        default="",
        description="The correct answer, combining facts from both excerpts. Only fill in if possible=True.",
    )


PROMPT = """
You are helping build a test set for a RAG system over a college IT help desk knowledge base.

Below are two excerpts from DIFFERENT articles in the knowledge base.

Excerpt A (source: {source_a}):
{text_a}

Excerpt B (source: {source_b}):
{text_b}

Determine if it's possible to write ONE realistic test question that a help desk
TECHNICIAN would ask an internal assistant, where answering it CORRECTLY REQUIRES
combining specific facts from BOTH excerpts — not answerable using only one of them.

If genuinely possible, phrase the question the way a technician relays a customer's
issue (e.g. "customer is asking..."), give specific keywords drawn from both excerpts,
and a reference answer that combines facts from both.

If the two excerpts don't share any real connection that supports a genuine two-hop
question, set possible=False and leave the other fields empty. Do not force a question
that's really answerable from just one excerpt.
"""


@retry(wait=wait)
def try_pair(text_a, source_a, text_b, source_b):
    structured_llm = llm.with_structured_output(SpanningCandidate)
    prompt = PROMPT.format(source_a=source_a, text_a=text_a, source_b=source_b, text_b=text_b)
    return structured_llm.invoke(prompt)


def load_all_chunks():
    result = vectorstore._collection.get(include=["documents", "metadatas"])
    return list(zip(result["documents"], result["metadatas"]))


def main():
    random.seed(7)
    all_chunks = load_all_chunks()
    print(f"Loaded {len(all_chunks):,} chunks from the vector store")

    seed_sample = random.sample(all_chunks, min(MAX_ATTEMPTS, len(all_chunks)))
    found = []

    for attempt, (doc, meta) in enumerate(seed_sample, start=1):
        if len(found) >= TARGET_COUNT:
            break

        neighbors = vectorstore.similarity_search(doc, k=NEIGHBORS_PER_SEED + 1)
        candidates = [n for n in neighbors if n.metadata.get("source") != meta.get("source")]
        if not candidates:
            continue
        neighbor = candidates[0]

        print(f"[{attempt}] {meta.get('source')}  <->  {neighbor.metadata.get('source')}")
        result = try_pair(doc, meta.get("source"), neighbor.page_content, neighbor.metadata.get("source"))

        if result.possible:
            found.append({
                "question": result.question,
                "keywords": result.keywords,
                "reference_answer": result.reference_answer,
                "category": "spanning",
                "_sources": [meta.get("source"), neighbor.metadata.get("source")],
            })
            print("  -> spanning question found")
        else:
            print("  -> no genuine overlap, skipped")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for d in found:
            f.write(json.dumps(d) + "\n")

    print(f"\nFound {len(found)} spanning questions, wrote to {OUTPUT_PATH}")
    print("Review these, then merge the good ones (with '_sources' stripped) into tests.jsonl")


if __name__ == "__main__":
    main()

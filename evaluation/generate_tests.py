import json
import random
from pathlib import Path

import frontmatter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
OUTPUT_PATH = Path(__file__).parent / "tests_draft.jsonl"

# ~3 docs per category folder -> roughly 24 draft questions across your 8 categories.
# Bump this up if you want a bigger draft batch to pick and choose from.
DOCS_PER_CATEGORY = 3

QUESTION_CATEGORIES = [
    "direct_fact",     # simple lookup, answer is stated plainly in one place
    "procedural",      # multi-step "how do I do X" instructions
    "policy",          # rules, permissions, who's allowed to do what
    "troubleshooting", # diagnosing/fixing a problem
    "numerical",       # a specific number, limit, date, or threshold
    "relationship",    # which team/tool/system handles or connects to what
]

wait = wait_exponential(multiplier=1, min=10, max=120)

llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.3)


class DraftTest(BaseModel):
    question: str = Field(
        description=(
            "A realistic question an IT help desk TECHNICIAN would ask this internal "
            "assistant while helping a customer, phrased as the technician relaying the "
            "customer's issue (e.g. 'customer is asking how to reset their password' or "
            "'customer can't connect to eduroam on their laptop') — NOT a question the "
            "customer asks the assistant directly."
        )
    )
    keywords: list[str] = Field(
        description=(
            "2-4 specific keywords or short phrases, taken from the article, that MUST "
            "appear in the correct source chunk for this question. Used to check whether "
            "retrieval found the right content — pick specific terms, not generic words."
        )
    )
    reference_answer: str = Field(
        description="A concise, correct reference answer based solely on the provided article."
    )
    category: str = Field(description=f"One of: {', '.join(QUESTION_CATEGORIES)}")


GENERATION_PROMPT = """
You are helping build a test set to evaluate a RAG system for a college IT help desk.
Below is one article from the knowledge base (category: {doc_type}, source: {source}).

Write ONE realistic test question that an IT help desk TECHNICIAN would ask an internal
assistant tool while helping a customer, based ONLY on the content of this article.
Phrase it the way a technician would relay a customer's issue, e.g.
"customer is asking how to reset their password" or "customer can't connect to eduroam
on their laptop" — NOT a customer directly asking the assistant.

Pick the keywords carefully: they should be specific terms/phrases from the article that
would only appear in the chunk(s) actually relevant to this question, not generic words
that show up everywhere.

Here is the article:

{text}

Respond with the test question data.
"""


def sample_documents():
    """Stratified sample: a few docs per knowledge-base category folder."""
    samples = []
    for folder in sorted(KNOWLEDGE_BASE_PATH.iterdir()):
        if not folder.is_dir():
            continue
        doc_type = folder.name
        files = list(folder.rglob("*.md"))
        if not files:
            continue
        chosen = random.sample(files, min(DOCS_PER_CATEGORY, len(files)))
        for file in chosen:
            post = frontmatter.load(file)
            samples.append({"type": doc_type, "source": file.as_posix(), "text": post.content})
    return samples


@retry(wait=wait)
def generate_question(document):
    prompt = GENERATION_PROMPT.format(
        doc_type=document["type"], source=document["source"], text=document["text"]
    )
    structured_llm = llm.with_structured_output(DraftTest)
    result = structured_llm.invoke(prompt)
    return {
        "question": result.question,
        "keywords": result.keywords,
        "reference_answer": result.reference_answer,
        "category": result.category,
        "_source": document["source"],  # for your review only — strip before final tests.jsonl
    }


def main():
    random.seed(42)
    documents = sample_documents()
    print(f"Sampled {len(documents)} documents across categories")

    drafts = []
    for doc in documents:
        try:
            drafts.append(generate_question(doc))
            print(f"  ok: {doc['source']}")
        except Exception as e:
            print(f"  FAILED: {doc['source']} -> {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d) + "\n")

    print(f"\nWrote {len(drafts)} draft questions to {OUTPUT_PATH}")
    print("Review/edit these, then save the final (with '_source' stripped) as tests.jsonl")


if __name__ == "__main__":
    main()

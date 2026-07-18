import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

RETRIEVAL_K = 10
FINAL_K = 5

wait = wait_exponential(multiplier=1, min=10, max=240)

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file")

db_pool = SimpleConnectionPool(1, 5, SUPABASE_DB_URL)

embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

SYSTEM_PROMPT_TEMPLATE = """
You are an internal knowledge assistant for IT Help Desk technicians at SUNY New Paltz.
You do not talk to students, faculty, or staff directly — you are only ever talking to
the technician, who is the middleman handling the actual customer interaction.

The technician will describe what the customer is asking or the issue they're facing
(e.g. "customer is asking how to reset their password"). Give the technician a direct,
straightforward answer: the relevant steps, procedure, or policy, in plain language —
like a coworker quickly telling another coworker what to do. Don't script out exact
phrases to relay to the customer unless the technician specifically asks for wording.
Keep it tight — no unnecessary preamble, headers, or restating the question.

If you don't know the answer or it isn't in the context, say so rather than guessing —
don't leave the technician to relay a wrong answer.

Context:
{context}
"""


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description="One or more self-contained sub-questions. If the original question "
                    "only has a single distinct ask, return a list containing just that "
                    "question, unchanged."
    )


@retry(wait=wait, stop=stop_after_attempt(5))
def decompose_question(question: str) -> list[str]:
    """Split a compound question into self-contained sub-questions, if needed."""
    system_prompt = """
You help an IT help desk search system prepare a technician's question for retrieval.
Determine whether the question contains more than one distinct, unrelated ask
(e.g. two separate topics or tasks bundled into one sentence).

If there is only one ask, return a list containing the original question, unchanged.
If there are multiple distinct asks, split it into separate, self-contained sub-questions
— each one should make sense on its own, without needing the rest of the sentence.
Do not split a single ask into smaller pieces just because it's detailed or has multiple steps.
"""
    structured_llm = llm.with_structured_output(SubQuestions)
    reply = structured_llm.invoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    )
    return reply.sub_questions


@retry(wait=wait, stop=stop_after_attempt(5))
def rewrite_query(question: str, history: list[dict] = []) -> str:
    """
    Compress conversation history + current question into one short,
    specific search query for the knowledge base.
    """
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = f"""
You are helping an IT help desk technician search a knowledge base.
You are about to look up information to answer their question.

This is the history of the conversation so far:
{history_text}

And this is the technician's current question:
{question}

Respond only with a single, short, specific search query that will surface
the most relevant content in the knowledge base. Don't mention SUNY New Paltz
or "IT" unless it's actually part of what's being searched for.
IMPORTANT: Respond ONLY with the search query, nothing else.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def fetch_chunks(query: str, k: int = RETRIEVAL_K) -> list[Result]:
    """Embed a query and retrieve the k nearest chunks from Supabase (pgvector)."""
    query_embedding = embeddings_model.embed_query(query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, type, page_content
                FROM chunks
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (embedding_literal, k),
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)

    return [
        Result(page_content=page_content, metadata={"source": source, "type": doc_type})
        for source, doc_type, page_content in rows
    ]


def merge_chunks(chunks_a: list[Result], chunks_b: list[Result]) -> list[Result]:
    """Dedupe by page_content, preserving order (chunks_a first)."""
    merged = list(chunks_a)
    seen = {chunk.page_content for chunk in chunks_a}
    for chunk in chunks_b:
        if chunk.page_content not in seen:
            merged.append(chunk)
            seen.add(chunk.page_content)
    return merged


@retry(wait=wait, stop=stop_after_attempt(5))
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """Re-sort merged chunks by relevance to the original question via LLM judgment."""
    system_prompt = """
You are a document re-ranker for an IT help desk knowledge base.
You are given a question and a list of chunks retrieved from the knowledge base.
The chunks are roughly ordered by relevance already, but you may be able to improve on that.
Rank all the chunks by relevance to the question, most relevant first.
Include every chunk id you are given, reranked — don't drop any.
"""
    user_prompt = f"Question:\n{question}\n\nChunks:\n\n"
    for i, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}\n{chunk.page_content}\n\n"
    user_prompt += "Reply with the ranked list of chunk ids."

    structured_llm = llm.with_structured_output(RankOrder)
    reply = structured_llm.invoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    return [chunks[i - 1] for i in reply.order]


def fetch_context(question: str, history: list[dict] = []) -> list[Result]:
    """
    Full pro retrieval: decompose, rewrite+dual-retrieve per sub-question,
    N-way merge, rerank against the original question, truncate to FINAL_K.
    """
    sub_questions = decompose_question(question)

    all_chunks: list[Result] = []
    for sub_q in sub_questions:
        rewritten = rewrite_query(sub_q, history)
        chunks_original = fetch_chunks(sub_q)
        chunks_rewritten = fetch_chunks(rewritten)
        all_chunks = merge_chunks(all_chunks, merge_chunks(chunks_original, chunks_rewritten))

    reranked = rerank(question, all_chunks)
    return reranked[:FINAL_K]


@retry(wait=wait, stop=stop_after_attempt(5))
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """
    Answer the given question with pro RAG; return the answer and the context chunks.
    """
    docs = fetch_context(question, history)
    context = "\n\n".join(
        f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs

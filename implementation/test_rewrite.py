from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

wait = wait_exponential(multiplier=1, min=10, max=240)


@retry(wait=wait, stop=stop_after_attempt(5))
def rewrite_query(question: str, history: list[dict] = []) -> str:
    """
    Compress conversation history + current question into one short,
    specific search query for the knowledge base.
    """
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history
    )
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


if __name__ == "__main__":
    # Simple case, no history
    print(rewrite_query("customer can't get their printer to connect to wifi"))

    # Multi-turn case — this is the one that actually tests the point of rewriting
    history = [
        {"role": "user", "content": "customer is asking about their student email"},
        {"role": "assistant", "content": "Are they asking about accessing it, or a forwarding/alias issue?"},
    ]
    print(rewrite_query("they can't log in, says password is wrong but they just reset it", history))

    # Vague follow-up that only makes sense with history
    history2 = [
        {"role": "user", "content": "how do I reset a password for a locked account"},
    ]
    print(rewrite_query("what about for a shared department account instead?", history2))
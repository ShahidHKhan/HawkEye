from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
RETRIEVAL_K = 10

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

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the technician's prior turns into a single string, for retrieval only.
    Cheap way to keep multi-turn context in the retrieval step.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs

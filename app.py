import json
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
QUERY_LOG = LOG_DIR / "queries.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"


def log_query(question: str, history: list[dict], answer: str | None,
              sources: list[str] | None, latency: float, error: str | None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "history_length": len(history),
        "answer": answer,
        "sources": sources,
        "latency_seconds": round(latency, 2),
        "error": error,
    }
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_feedback(answer: str, liked: bool) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answer": answer,
        "liked": liked,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def format_context(chunks) -> str:
    if not chunks:
        return "*No sources retrieved.*"
    result = "### Retrieved sources\n\n"
    for chunk in chunks:
        result += f"**Source:** {chunk.metadata.get('source', 'unknown')}\n\n"
        result += chunk.page_content + "\n\n---\n\n"
    return result


def chat(message: str, history: list[dict]) -> tuple[str, str]:
    """
    Call the real production pipeline; log every attempt (success or failure);
    never let a raw exception reach the technician's screen.
    """
    start = time.time()
    try:
        answer, chunks = answer_question(message, history)
        sources = [chunk.metadata.get("source") for chunk in chunks]
        latency = time.time() - start
        log_query(message, history, answer, sources, latency, error=None)
        return answer, format_context(chunks)
    except Exception as e:
        latency = time.time() - start
        log_query(message, history, None, None, latency, error=str(e))
        friendly = (
            "Something went wrong reaching the knowledge base or the model just now. "
            "Try again in a moment — this has been logged."
        )
        return friendly, "*Error retrieving context — see logs/queries.jsonl.*"


def main():
    with gr.Blocks(title="HawkEye IT Assistant") as ui:
        gr.Markdown(
            "# HawkEye\nInternal IT help desk knowledge assistant — technicians only. "
            "Describe the customer's issue as you would to a coworker."
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Conversation", height=600)
                message = gr.Textbox(
                    label="Question",
                    placeholder="e.g. customer can't connect to eduroam on their laptop",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        def put_message_in_chatbot(msg, hist):
            return "", hist + [{"role": "user", "content": msg}]

        def respond(hist):
            last_message = hist[-1]["content"]
            prior = hist[:-1]
            answer, context = chat(last_message, prior)
            hist.append({"role": "assistant", "content": answer})
            return hist, context

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(respond, inputs=chatbot, outputs=[chatbot, context_markdown])

        def on_like(evt: gr.LikeData):
            # Gradio's LikeData gives us the message content and whether it was
            # liked/disliked, but not the question that produced it — good enough
            # for a first pass at "is this answer any good" signal.
            log_feedback(answer=str(evt.value), liked=bool(evt.liked))

        chatbot.like(on_like)

    ui.launch(inbrowser=True, theme=gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"]))


if __name__ == "__main__":
    main()

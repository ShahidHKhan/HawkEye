import os
import time

import gradio as gr
from dotenv import load_dotenv
from psycopg2.extras import Json

from implementation.answer import answer_question, db_pool

load_dotenv(override=True)


def extract_text(content) -> str:
    """
    Gradio 6's Chatbot returns message content as either a plain string or a list
    of content-part dicts (e.g. [{'type': 'text', 'text': '...'}]), even for
    plain-text messages. Normalize to plain text so nothing downstream — the RAG
    pipeline, query rewriting, or logging — has to special-case this.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def normalize_history(hist: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": extract_text(m["content"])} for m in hist]


def log_query(question: str, history: list[dict], answer: str | None,
              sources: list[str] | None, latency: float, error: str | None) -> None:
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (question, history_length, answer, sources, latency_seconds, error)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    question,
                    len(history),
                    answer,
                    Json(sources) if sources is not None else None,
                    round(latency, 2),
                    error,
                ),
            )
        conn.commit()
    finally:
        db_pool.putconn(conn)


def log_feedback(answer: str, liked: bool) -> None:
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (answer, liked) VALUES (%s, %s)",
                (answer, liked),
            )
        conn.commit()
    finally:
        db_pool.putconn(conn)


def clean_source(source: str) -> str:
    """Show a clean relative path instead of a local absolute file path."""
    normalized = source.replace("\\", "/")
    marker = "knowledge-base/"
    idx = normalized.lower().find(marker)
    if idx != -1:
        return normalized[idx + len(marker):]
    return normalized.split("/")[-1]


def format_context(chunks) -> str:
    if not chunks:
        return "*No sources retrieved.*"
    result = "### Retrieved sources\n\n"
    for chunk in chunks:
        result += f"**Source:** {clean_source(chunk.metadata.get('source', 'unknown'))}\n\n"
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
        return friendly, "*Error retrieving context — this attempt has been logged.*"


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
                reset_button = gr.Button("New customer / reset chat", variant="secondary")

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        def put_message_in_chatbot(msg, hist):
            return "", hist + [{"role": "user", "content": msg}]

        def respond(hist):
            hist = normalize_history(hist)
            last_message = hist[-1]["content"]
            prior = hist[:-1]
            answer, context = chat(last_message, prior)
            hist.append({"role": "assistant", "content": answer})
            return hist, context

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(respond, inputs=chatbot, outputs=[chatbot, context_markdown])

        def reset_chat():
            return [], "*Retrieved context will appear here*", ""

        reset_button.click(
            reset_chat, inputs=None, outputs=[chatbot, context_markdown, message]
        )

        def on_like(evt: gr.LikeData):
            # Gradio's LikeData gives us the message content and whether it was
            # liked/disliked, but not the question that produced it — good enough
            # for a first pass at "is this answer any good" signal.
            log_feedback(answer=str(evt.value), liked=bool(evt.liked))

        chatbot.like(on_like)

    ui.launch(
        theme=gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"]),
        auth=[(os.getenv("APP_USERNAME"), os.getenv("APP_PASSWORD"))],
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )


if __name__ == "__main__":
    main()

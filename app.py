import os
import time

import gradio as gr
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from implementation.answer import answer_question_stream, db_pool
from visualize_embeddings import build_figure, fetch_chunks, reduce_to_3d

ASSISTANT_AVATAR = "static/hawkeye-icon.svg"
FAVICON = "static/hawkeye-icon.svg"

load_dotenv(override=True)

# Shared across every technician's session -- the embedding map is the same
# knowledge base for everyone, so one person pays the ~1-2 minute PCA/t-SNE
# cost and everyone else who opens the tab afterward gets it instantly.
_kb_map_cache = {"coords": None, "doc_types": None, "hover_texts": None}


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
    with db_pool.connection() as conn:
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
                    Jsonb(sources) if sources is not None else None,
                    round(latency, 2),
                    error,
                ),
            )
        conn.commit()


def log_feedback(answer: str, liked: bool) -> None:
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (answer, liked) VALUES (%s, %s)",
                (answer, liked),
            )
        conn.commit()


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


REFRESH_RUNS_HEADERS = [
    "Started", "Finished", "Scope", "New", "Changed", "Unchanged", "Removed", "Status", "Error",
]
REFRESH_RUNS_DATATYPES = ["str", "str", "str", "number", "number", "number", "number", "str", "str"]

REFRESH_RUNS_DETAIL_PLACEHOLDER = "*Select a run above to see which articles it touched.*"

# refresh_kb.py's changes jsonb buckets, in the order they're worth reading.
REFRESH_RUNS_BUCKETS = [
    ("new", "Added"),
    ("changed", "Updated"),
    ("removed", "Removed"),
    ("healed", "Healed — had chunks but article_state had lost track of them"),
    ("failed", "Failed — skipped, the rest of the run continued"),
]


def fetch_refresh_runs(limit: int = 20) -> list[tuple]:
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT started_at, finished_at, scope, new_count, changed_count,
                       unchanged_count, removed_count, error, changes
                FROM refresh_runs
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def format_run_details(changes: dict | None, error: str | None = None) -> str:
    """
    Render one run's changes jsonb as markdown: which articles it actually
    touched, not just how many. Worth having as its own panel because the table's
    counts come from the diff, so they say what a run set out to do -- this says
    what it did. `error` is only used to tell an empty successful run ("nothing
    had changed upstream") apart from an empty failed one ("it never got that
    far"), which otherwise look identical here.
    """
    sections = []
    for key, label in REFRESH_RUNS_BUCKETS:
        items = (changes or {}).get(key) or []
        if not items:
            continue
        lines = [f"**{label}** ({len(items)})", ""]
        for item in items:
            line = f"- `{item.get('article_id')}` {item.get('title') or '*untitled*'}"
            if item.get("chunk_count") is not None:
                line += f" — {item['chunk_count']} chunks"
            if item.get("error"):
                line += f" — {item['error']}"
            lines.append(line)
        sections.append("\n".join(lines))

    if sections:
        return "\n\n".join(sections)
    if error:
        return "*No articles were processed — this run stopped on an error before reaching any. See the Error column.*"
    if changes is None:
        return "*This run recorded no per-article detail.*"
    return "*This run touched no articles — everything was already up to date.*"


def format_refresh_runs(rows: list[tuple]) -> list[list]:
    formatted = []
    for started_at, finished_at, scope, new, changed, unchanged, removed, error, _changes in rows:
        if error:
            status = "Failed"
        elif finished_at is None:
            status = "Running"
        else:
            status = "Success"
        formatted.append([
            started_at.strftime("%Y-%m-%d %H:%M UTC"),
            finished_at.strftime("%Y-%m-%d %H:%M UTC") if finished_at else "—",
            scope,
            new,
            changed,
            unchanged,
            removed,
            status,
            error or "",
        ])
    return formatted


def load_refresh_runs():
    """Read-only fetch of the most recent KB refresh runs; no writes to refresh_runs."""
    rows = fetch_refresh_runs(20)
    if not rows:
        return gr.update(value=[]), "*No refresh runs recorded yet.*", [], REFRESH_RUNS_DETAIL_PLACEHOLDER
    # Each run's error + changes jsonb is parked in a State alongside the table so
    # a row click can render its detail by index without going back to the database.
    return (
        gr.update(value=format_refresh_runs(rows)),
        "",
        [{"error": row[7], "changes": row[8]} for row in rows],
        REFRESH_RUNS_DETAIL_PLACEHOLDER,
    )


def show_run_details(run_details: list, evt: gr.SelectData) -> str:
    """Row-click handler for the runs table: show the selected run's article list."""
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not run_details or row is None or row >= len(run_details):
        return REFRESH_RUNS_DETAIL_PLACEHOLDER
    return format_run_details(run_details[row]["changes"], run_details[row]["error"])


KB_MAP_THEME = "light"
KB_MAP_MARKER_SIZE = 3


def load_kb_map(progress=gr.Progress()):
    """
    Cached load, used when the tab is first opened -- instant after the first
    technician warms it. No controls on this tab by design: fixed theme/size,
    just click and look. Both steps are fast (a streamed DB fetch, then a
    single PCA projection), so no heartbeat/threading is needed here -- see
    reduce_to_3d's docstring for why t-SNE was dropped in favor of plain PCA.
    """
    if _kb_map_cache["coords"] is None:
        progress(0, desc="Fetching chunks from Supabase...")
        doc_types, hover_texts, vectors = fetch_chunks()
        progress(0.7, desc="Projecting embeddings to 3D...")
        coords = reduce_to_3d(vectors)
        _kb_map_cache.update(coords=coords, doc_types=doc_types, hover_texts=hover_texts)

    categories = sorted(set(_kb_map_cache["doc_types"]))
    return build_figure(
        _kb_map_cache["coords"], _kb_map_cache["doc_types"], _kb_map_cache["hover_texts"],
        KB_MAP_THEME, KB_MAP_MARKER_SIZE, categories,
    )


def chat_stream(message: str, history: list[dict]):
    """
    Streaming version of the production pipeline: yields (partial_answer, context)
    as the answer streams in. Logs exactly once, after the stream finishes
    (success or failure); never lets a raw exception reach the technician's screen.
    """
    start = time.time()
    docs = []
    answer = ""
    try:
        for answer, docs in answer_question_stream(message, history):
            yield answer, format_context(docs)
        latency = time.time() - start
        sources = [chunk.metadata.get("source") for chunk in docs]
        log_query(message, history, answer, sources, latency, error=None)
    except Exception as e:
        latency = time.time() - start
        log_query(message, history, None, None, latency, error=str(e))
        friendly = (
            "Something went wrong reaching the knowledge base or the model just now. "
            "Try again in a moment — this has been logged."
        )
        yield friendly, "*Error retrieving context — this attempt has been logged.*"


def require_auth_credentials() -> tuple[str, str]:
    """
    APP_USERNAME / APP_PASSWORD, or a hard failure.

    Checked here rather than at import time because tests/test_app.py imports this
    module, and reading the Refresh History helpers shouldn't require the server's
    credentials -- but launching the server must. Without this, a missing variable
    made os.getenv return None and Gradio received auth=[(None, None)], standing
    the app up with a login form in an unclear state instead of refusing to start.
    For an internal tool reachable on the public internet, failing loudly at boot
    is the only safe reading of "the password variable isn't set."
    """
    username = os.getenv("APP_USERNAME")
    password = os.getenv("APP_PASSWORD")
    missing = [
        name for name, value in (("APP_USERNAME", username), ("APP_PASSWORD", password))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not set — refusing to start without basic auth. "
            f"Add them to your .env file (see .env.example), or to the host's secrets "
            f"when deploying."
        )
    return username, password


def main():
    username, password = require_auth_credentials()
    theme = gr.themes.Soft(
        primary_hue="blue", secondary_hue="slate", font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(title="HawkEye IT Assistant") as ui:
        gr.Markdown(
            "# HawkEye\nInternal IT help desk knowledge assistant — technicians only. "
            "Describe the customer's issue as you would to a coworker."
        )

        with gr.Tabs():
            with gr.Tab("Assistant"):
                with gr.Row():
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=420,
                            avatar_images=(None, ASSISTANT_AVATAR),
                            buttons=["copy", "copy_all"],
                        )
                        with gr.Row():
                            message = gr.Textbox(
                                label="Question",
                                placeholder="e.g. customer can't connect to eduroam on their laptop",
                                show_label=False,
                                scale=5,
                            )
                            send_button = gr.Button("Send", variant="primary", scale=1)

                    with gr.Column(scale=1):
                        context_markdown = gr.Markdown(
                            value="*Retrieved context will appear here*",
                            container=True,
                            height=420,
                        )
                        reset_button = gr.Button("New customer / reset chat", variant="secondary")

            with gr.Tab("Refresh History") as refresh_tab:
                gr.Markdown(
                    "Most recent knowledge-base refresh runs (weekly pipeline). Read-only."
                )
                refresh_runs_table = gr.Dataframe(
                    headers=REFRESH_RUNS_HEADERS,
                    datatype=REFRESH_RUNS_DATATYPES,
                    interactive=False,
                    wrap=True,
                )
                refresh_runs_empty = gr.Markdown(value="")
                refresh_runs_button = gr.Button("Refresh", size="sm")
                refresh_runs_state = gr.State([])
                refresh_runs_details = gr.Markdown(value=REFRESH_RUNS_DETAIL_PLACEHOLDER)

            with gr.Tab("Knowledge Map") as kb_map_tab:
                gr.Markdown(
                    "3D map of every chunk in the knowledge base, colored by category."
                )
                kb_map_plot = gr.Plot()

            refresh_runs_outputs = [
                refresh_runs_table, refresh_runs_empty, refresh_runs_state, refresh_runs_details,
            ]
            refresh_tab.select(load_refresh_runs, inputs=None, outputs=refresh_runs_outputs)
            refresh_runs_button.click(load_refresh_runs, inputs=None, outputs=refresh_runs_outputs)
            refresh_runs_table.select(
                show_run_details, inputs=refresh_runs_state, outputs=refresh_runs_details
            )

            kb_map_tab.select(load_kb_map, inputs=None, outputs=kb_map_plot)

        def put_message_in_chatbot(msg, hist):
            return "", hist + [{"role": "user", "content": msg}]

        def respond(hist):
            hist = normalize_history(hist)
            last_message = hist[-1]["content"]
            prior = hist[:-1]
            hist.append({"role": "assistant", "content": "_Searching the knowledge base…_"})
            yield hist, "*Retrieving context…*"
            for partial_answer, context in chat_stream(last_message, prior):
                hist[-1]["content"] = partial_answer
                yield hist, context

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(respond, inputs=chatbot, outputs=[chatbot, context_markdown])

        send_button.click(
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
        theme=theme,
        auth=[(username, password)],
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        favicon_path=FAVICON,
    )


if __name__ == "__main__":
    main()

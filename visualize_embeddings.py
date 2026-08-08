"""
3D visualization of the HawkEye knowledge-base embedding space: pulls every
chunk's embedding out of Supabase (pgvector), reduces it to 3 dimensions, and
renders a rotatable Plotly scatter in a small Gradio app -- built for grabbing
screenshots (e.g. Plotly's built-in camera-icon PNG export), not day-to-day use.

Run from the project root:
    uv run visualize_embeddings.py
"""

import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import psycopg
from dotenv import load_dotenv
from sklearn.decomposition import PCA

load_dotenv(override=True)

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file")

HOVER_TEXT_LEN = 160

# Validated categorical order (see the dataviz skill's references/palette.md),
# assigned alphabetically to HawkEye's knowledge-base categories in fixed order
# -- categorical color must never be cycled or reassigned per-render.
CATEGORY_COLORS_LIGHT = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
CATEGORY_COLORS_DARK = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}


FETCH_BATCH_SIZE = 200


def fetch_chunks() -> tuple[list[str], list[str], np.ndarray]:
    """
    Pull every chunk's category, a hover snippet, and its embedding out of Supabase.

    Streamed via a server-side cursor and converted to float32 one row at a time
    -- each embedding round-trips as ~39KB of text (a 3,072-dim vector as ASCII
    numbers), and pulling all of them with fetchall() plus a naive per-element
    float() list comprehension briefly holds tens of millions of individual
    Python float/str objects in memory at once. On the 1GB Fly machine this app
    runs on, that was enough to OOM-kill the whole container mid-request --
    which looks to the browser like the server connection dying outright.
    page_content is truncated in SQL for the same reason: only a hover snippet
    is ever needed here, so there's no reason to pull the full chunk text over
    the wire.
    """
    doc_types: list[str] = []
    hover_texts: list[str] = []
    vectors: list[np.ndarray] = []

    with psycopg.connect(SUPABASE_DB_URL) as conn:
        with conn.cursor(name="kb_map_chunks") as cur:
            cur.execute(
                "SELECT type, left(page_content, %s), embedding::text FROM chunks ORDER BY id",
                (HOVER_TEXT_LEN,),
            )
            while True:
                batch = cur.fetchmany(FETCH_BATCH_SIZE)
                if not batch:
                    break
                for doc_type, snippet, embedding_text in batch:
                    doc_types.append(doc_type or "unknown")
                    hover_texts.append((snippet or "").replace("\n", " "))
                    vectors.append(
                        np.array(embedding_text.strip("[]").split(","), dtype=np.float32)
                    )

    return doc_types, hover_texts, np.stack(vectors)


def reduce_to_3d(vectors: np.ndarray) -> np.ndarray:
    """
    Straight PCA down to 3 dimensions -- a single SVD, near-instant regardless
    of environment. Tried t-SNE first for its tighter-looking clusters, but it
    ran roughly 10x slower on the production Fly box (shared vCPU) than in
    local testing, turning the Knowledge Map tab's first load into a
    multi-minute wait with no reliable way to predict or bound it.
    """
    return PCA(n_components=3, random_state=42).fit_transform(vectors)


def build_figure(
    coords: np.ndarray,
    doc_types: list[str],
    hover_texts: list[str],
    theme: str,
    marker_size: int,
    visible_categories: list[str],
) -> go.Figure:
    colors = CATEGORY_COLORS_DARK if theme == "dark" else CATEGORY_COLORS_LIGHT
    unique_types = sorted(set(doc_types))
    color_map = dict(zip(unique_types, colors))

    fig = go.Figure()
    for doc_type in unique_types:
        if doc_type not in visible_categories:
            continue
        idx = [i for i, t in enumerate(doc_types) if t == doc_type]
        fig.add_trace(go.Scatter3d(
            x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
            mode="markers",
            name=doc_type,
            marker=dict(size=marker_size, color=color_map[doc_type], opacity=0.8, line=dict(width=0)),
            text=[hover_texts[i] for i in idx],
            hoverinfo="text",
        ))

    surface = SURFACE[theme]
    ink = INK[theme]
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(
        title=dict(text="HawkEye knowledge base — embedding space", font=dict(color=ink, size=18)),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, bgcolor=surface),
        paper_bgcolor=surface,
        legend=dict(font=dict(color=ink)),
        margin=dict(l=0, r=0, t=50, b=0),
        height=780,
    )
    return fig


def load_and_reduce(progress=gr.Progress()):
    progress(0, desc="Fetching chunks from Supabase...")
    doc_types, hover_texts, vectors = fetch_chunks()
    progress(0.3, desc=f"Projecting {len(doc_types):,} embeddings to 3D (PCA)...")
    coords = reduce_to_3d(vectors)
    progress(0.9, desc="Rendering...")

    categories = sorted(set(doc_types))
    state = {"coords": coords, "doc_types": doc_types, "hover_texts": hover_texts}
    fig = build_figure(coords, doc_types, hover_texts, "dark", 4, categories)
    return state, gr.update(choices=categories, value=categories), fig


def rebuild(state, theme, marker_size, categories):
    if state is None:
        return None
    return build_figure(state["coords"], state["doc_types"], state["hover_texts"], theme, marker_size, categories)


def main():
    with gr.Blocks(title="HawkEye — embedding space") as ui:
        gr.Markdown(
            "# HawkEye knowledge-base embedding space\n"
            "Each point is one chunk from the `chunks` table, colored by knowledge-base "
            "category. Drag to rotate; use the camera icon in the plot toolbar to export a PNG."
        )

        state = gr.State(None)
        load_button = gr.Button("Load embeddings & compute layout", variant="primary")

        with gr.Row():
            theme = gr.Radio(["dark", "light"], value="dark", label="Background")
            marker_size = gr.Slider(1, 10, value=4, step=1, label="Marker size")
        categories = gr.CheckboxGroup(choices=[], value=[], label="Categories shown")

        plot = gr.Plot()

        load_button.click(load_and_reduce, outputs=[state, categories, plot])
        theme.change(rebuild, inputs=[state, theme, marker_size, categories], outputs=plot)
        marker_size.change(rebuild, inputs=[state, theme, marker_size, categories], outputs=plot)
        categories.change(rebuild, inputs=[state, theme, marker_size, categories], outputs=plot)

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

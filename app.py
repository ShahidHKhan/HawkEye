import json
import logging
import os
import time
from contextlib import closing

import anyio
import gradio as gr
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from psycopg.types.json import Jsonb

from implementation import accounts, audit, rate_limit
from implementation.accounts import AccountError, User
from implementation.answer import answer_question_stream, db_pool
from implementation.rate_limit import RateLimited
from visualize_embeddings import build_figure, fetch_chunks, reduce_to_3d

ASSISTANT_AVATAR = "static/hawkeye-icon.svg"
FAVICON = "static/hawkeye-icon.svg"

load_dotenv(override=True)

log = logging.getLogger("hawkeye")

# Longest question accepted, and how much of the conversation is sent along with
# it. Both bound what one request can cost in Gemini tokens; the full history
# length is still logged.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", 1500))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 10))

CONTEXT_PLACEHOLDER = "*Retrieved context will appear here*"
FRIENDLY_ERROR = (
    "Something went wrong reaching the knowledge base or the model just now. "
    "Try again in a moment — this has been logged."
)
SESSION_ENDED = (
    "Your session has ended or your account was disabled. Sign out and sign in again, "
    "or ask a HawkEye admin."
)

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


# ---------------------------------------------------------------------------
# Who is asking: identity and role checks for every handler
# ---------------------------------------------------------------------------
#
# Hiding a tab from workers is only cosmetic: every Gradio event can be called
# directly over its HTTP API by anyone signed in, whatever the page shows. So
# each handler resolves the account behind the request itself and refuses
# anything its role doesn't allow.

def audit_refusal(event_type: str, request, user: User | None, details: dict) -> None:
    """Record an access_denied / rate_limited event, capped per user so refusals can't flood the log."""
    ip, agent = audit.request_context(request)
    key = (event_type, user.id if user else ip or "unknown")
    if rate_limit.audit_noise_limiter.hit(key) is not None:
        return
    typed = getattr(request, "username", None) if request is not None else None
    audit.record_event(
        event_type,
        user_id=user.id if user else None,
        username=user.username if user else (accounts.normalize_username(typed) or typed),
        details=details, ip=ip, agent=agent,
    )


def require_user(request) -> User:
    user = accounts.current_user(request)
    if user is None:
        raise gr.Error(SESSION_ENDED)
    return user


def require_admin(request, action: str, *, count_action: bool = True) -> User:
    """
    The signed-in admin, or a gr.Error. `count_action` charges the admin action
    limiter; it's off for handlers that only re-render data already on the page.
    """
    user = accounts.current_user(request)
    if user is None or not user.is_admin:
        audit_refusal("access_denied", request, user, {"action": action})
        raise gr.Error(SESSION_ENDED if user is None else "That part of HawkEye is for admins only.")
    if count_action:
        try:
            rate_limit.check(
                rate_limit.admin_action_limiter, user.id, "admin_actions",
                "That's a lot of admin actions in a short time.",
            )
        except RateLimited as refusal:
            audit_refusal("rate_limited", request, user, {**refusal.audit_details(), "action": action})
            raise gr.Error(refusal.user_message())
    return user


# ---------------------------------------------------------------------------
# Query and feedback logging
# ---------------------------------------------------------------------------

def start_query(user: User, question: str, history_length: int, session_id: str | None) -> int:
    """
    Log the question the moment it's asked, before any model call. Inserting up
    front is what makes the per-user quota count in-flight questions too, and it
    means a question is on record even if the technician closes the tab before
    the answer finishes (finish_query then marks it 'abandoned').
    """
    with db_pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO queries (question, history_length, user_id, session_id, status)
            VALUES (%s, %s, %s, %s, 'in_progress')
            RETURNING id
            """,
            (question, history_length, user.id, session_id),
        ).fetchone()
    return row[0]


def finish_query(query_id: int, status: str, answer: str | None, sources: list[str] | None,
                 latency: float, error: str | None) -> None:
    """Close out a start_query row. Never raises: it also runs while a stream is being torn down."""
    try:
        with db_pool.connection() as conn:
            conn.execute(
                """
                UPDATE queries
                SET status = %s, answer = %s, sources = %s, latency_seconds = %s, error = %s
                WHERE id = %s
                """,
                (
                    status,
                    answer,
                    Jsonb(sources) if sources is not None else None,
                    round(latency, 2),
                    error,
                    query_id,
                ),
            )
    except Exception:
        log.exception("Failed to finish query %s as %s", query_id, status)


def log_feedback(answer: str, liked: bool, user_id: int | None = None, query_id: int | None = None) -> None:
    with db_pool.connection() as conn:
        conn.execute(
            "INSERT INTO feedback (answer, liked, user_id, query_id) VALUES (%s, %s, %s, %s)",
            (answer, liked, user_id, query_id),
        )


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


def on_refresh_history(request: gr.Request):
    require_admin(request, "refresh_history")
    return load_refresh_runs()


def on_run_selected(run_details: list, evt: gr.SelectData, request: gr.Request) -> str:
    # Re-renders data already in the page's state, so no database hit to rate limit.
    require_admin(request, "refresh_history", count_action=False)
    return show_run_details(run_details, evt)


def on_kb_map(request: gr.Request, progress=gr.Progress()):
    require_admin(request, "knowledge_map")
    return load_kb_map(progress)


# ---------------------------------------------------------------------------
# Assistant: the question flow
# ---------------------------------------------------------------------------

def _refuse(user: User, refusal: RateLimited, request=None):
    audit_refusal("rate_limited", request, user, {**refusal.audit_details(), "surface": "question"})
    yield refusal.user_message(), "*No sources: this question wasn't run.*", None


def chat_stream(message: str, history: list[dict], user: User, request=None):
    """
    Streaming version of the production pipeline: yields (partial_answer, context,
    query_id) as the answer streams in. In order: validate the question, take the
    user's one-at-a-time slot, check their quota, log the question, stream the
    answer, then record how it ended -- completed, error, or abandoned if the
    technician left mid-answer. Never lets a raw exception reach the screen.
    """
    message = (message or "").strip()
    if not message:
        yield "Type a question first.", CONTEXT_PLACEHOLDER, None
        return
    if len(message) > MAX_QUESTION_CHARS:
        yield (
            f"That question is {len(message):,} characters; the limit is {MAX_QUESTION_CHARS:,}. "
            "Trim it to the customer's actual issue and try again.",
            CONTEXT_PLACEHOLDER,
            None,
        )
        return

    try:
        rate_limit.question_guard.acquire(user.id)
    except RateLimited as refusal:
        yield from _refuse(user, refusal, request)
        return
    try:
        yield from _answer(message, history, user, request)
    finally:
        rate_limit.question_guard.release(user.id)


def _answer(message: str, history: list[dict], user: User, request=None):
    try:
        rate_limit.check_question_quota(user)
    except RateLimited as refusal:
        yield from _refuse(user, refusal, request)
        return
    except Exception:
        log.exception("Quota check failed for %s", user.username)
        yield FRIENDLY_ERROR, "*Error retrieving context — this attempt has been logged.*", None
        return

    start = time.time()
    try:
        query_id = start_query(user, message, len(history), getattr(request, "session_hash", None))
    except Exception:
        log.exception("Couldn't log the start of a question for %s", user.username)
        yield FRIENDLY_ERROR, "*Error retrieving context — this attempt has been logged.*", None
        return

    answer, docs = "", []
    try:
        for answer, docs in answer_question_stream(message, history[-MAX_HISTORY_MESSAGES:]):
            yield answer, format_context(docs), query_id
    except GeneratorExit:
        # The technician closed the tab or reset mid-answer. Record what they got.
        finish_query(query_id, "abandoned", answer or None,
                     [c.metadata.get("source") for c in docs] if docs else None,
                     time.time() - start, None)
        raise
    except Exception as e:
        finish_query(query_id, "error", None, None, time.time() - start, str(e))
        yield FRIENDLY_ERROR, "*Error retrieving context — this attempt has been logged.*", query_id
        return
    finish_query(query_id, "completed", answer, [c.metadata.get("source") for c in docs],
                 time.time() - start, None)


def put_message_in_chatbot(msg, hist):
    if not (msg or "").strip():
        return msg, hist
    return "", hist + [{"role": "user", "content": msg}]


def respond(hist, query_ids, request: gr.Request):
    user = require_user(request)
    hist = normalize_history(hist)
    query_ids = dict(query_ids or {})
    if not hist or hist[-1]["role"] != "user":
        yield hist, gr.update(), query_ids
        return
    last_message = hist[-1]["content"]
    prior = hist[:-1]
    hist.append({"role": "assistant", "content": "_Searching the knowledge base…_"})
    assistant_index = len(hist) - 1
    yield hist, "*Retrieving context…*", query_ids
    # closing(): if this generator is torn down mid-stream, chat_stream is closed
    # right away (and logs the question as abandoned) instead of whenever it's
    # garbage collected.
    with closing(chat_stream(last_message, prior, user, request)) as stream:
        for partial_answer, context, query_id in stream:
            hist[-1]["content"] = partial_answer
            if query_id is not None:
                query_ids[assistant_index] = query_id
            yield hist, context, query_ids


def on_like(query_ids, evt: gr.LikeData, request: gr.Request):
    """Thumbs up/down, tied to the account and to the exact question it rates."""
    user = require_user(request)
    try:
        rate_limit.check(rate_limit.feedback_limiter, user.id, "feedback", "That's a lot of feedback clicks.")
    except RateLimited as refusal:
        audit_refusal("rate_limited", request, user, {**refusal.audit_details(), "surface": "feedback"})
        gr.Warning(refusal.user_message())
        return
    index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    log_feedback(
        answer=extract_text(evt.value), liked=bool(evt.liked),
        user_id=user.id, query_id=(query_ids or {}).get(index),
    )


# ---------------------------------------------------------------------------
# Accounts tab (admins)
# ---------------------------------------------------------------------------

USERS_HEADERS = [
    "Username", "Email", "Role", "Active", "Created", "Last sign-in (ET)",
    "Questions (7d)", "Last question (ET)",
]
USERS_DATATYPES = ["str", "str", "str", "str", "str", "str", "number", "str"]


def format_users(rows: list[tuple]) -> list[list]:
    return [
        [username, email, role, "yes" if active else "no", created or "", last_login or "—",
         queries_7d, last_query or "—"]
        for username, email, role, active, created, last_login, queries_7d, last_query in rows
    ]


def users_outputs():
    rows = format_users(accounts.list_users())
    return rows, gr.update(choices=[row[0] for row in rows])


def on_accounts_tab(request: gr.Request):
    require_admin(request, "accounts")
    return users_outputs()


def password_note(role: str) -> str:
    return "the shared admin password" if role == "admin" else "the shared worker password"


def on_create_account(username: str, role: str, request: gr.Request):
    admin = require_admin(request, "create_account")
    try:
        rate_limit.check(rate_limit.account_creation_limiter, admin.id, "account_creation",
                         "That's the hourly limit on new accounts.")
        user = accounts.create_user(admin, username, role)
    except RateLimited as refusal:
        audit_refusal("rate_limited", request, admin, refusal.audit_details())
        return f"**Not created.** {refusal.user_message()}", gr.update(), gr.update(), username
    except AccountError as e:
        return f"**Not created.** {e}", gr.update(), gr.update(), username
    rows, choices = users_outputs()
    return (
        f"Created **{user.username}** ({user.email}) as {user.role}. They sign in with "
        f"`{user.username}` and {password_note(user.role)}.",
        rows, choices, "",
    )


def format_bulk_results(results: list[tuple[str, bool, str]]) -> str:
    if not results:
        return "*Nothing to add: enter one username per line.*"
    created = sum(1 for _, ok, _ in results if ok)
    lines = [f"**{created} of {len(results)} created.** New accounts sign in with {password_note('worker')}.", ""]
    lines += [f"- {'Created' if ok else 'Skipped'} `{entry}`: {message}" for entry, ok, message in results]
    return "\n".join(lines)


def on_bulk_create(text: str, request: gr.Request):
    admin = require_admin(request, "bulk_create_accounts")

    def creation_budget():
        retry = rate_limit.account_creation_limiter.hit(admin.id)
        if retry is not None:
            raise AccountError(
                f"hourly limit on new accounts reached; try again in {rate_limit.format_wait(retry)}."
            )

    results = accounts.bulk_create(admin, text, "worker", budget_check=creation_budget)
    rows, choices = users_outputs()
    return format_bulk_results(results), rows, choices


def manage_account(action: str, username: str, request) -> tuple:
    admin = require_admin(request, f"account_{action}")
    if not username:
        return "Pick an account first (or click its row in the table).", gr.update(), gr.update()
    try:
        if action == "disable":
            user = accounts.set_active(admin, username, False)
            message = f"Disabled **{user.username}**. Their open sessions stop working within a minute."
        elif action == "enable":
            user = accounts.set_active(admin, username, True)
            message = f"Enabled **{user.username}**."
        elif action in ("admin", "worker"):
            user = accounts.set_role(admin, username, action)
            message = (f"**{user.username}** is now {'an admin' if action == 'admin' else 'a worker'}. "
                       f"Their password was reset to {password_note(action)}.")
        elif action == "reset":
            user = accounts.reset_password(admin, username)
            message = (f"Reset **{user.username}**'s password to {password_note(user.role)}. "
                       "This also clears a sign-in lockout.")
        else:
            raise AccountError(f"Unknown action {action!r}.")
    except AccountError as e:
        return f"**Not changed.** {e}", gr.update(), gr.update()
    rows, choices = users_outputs()
    return message, rows, choices


def manage_handler(action: str):
    # A real function per button (not functools.partial) so Gradio sees the
    # gr.Request annotation and injects the request.
    def handler(username: str, request: gr.Request):
        return manage_account(action, username, request)
    handler.__name__ = f"manage_{action}"
    return handler


def on_user_row_selected(evt: gr.SelectData, request: gr.Request):
    require_admin(request, "accounts", count_action=False)
    row = evt.row_value or []
    return gr.update(value=row[0]) if row else gr.update()


# ---------------------------------------------------------------------------
# Activity tab (admins)
# ---------------------------------------------------------------------------

ACTIVITY_RANGES = {"24 hours": "24 hours", "7 days": "7 days", "30 days": "30 days"}
ACTIVITY_QUERY_HEADERS = ["Time (ET)", "User", "Question", "Status", "Latency (s)", "Sources", "Error"]
ACTIVITY_USAGE_HEADERS = ["Username", "Role", "Active", "Questions", "Errors", "Last question (ET)", "Last sign-in (ET)"]
ACTIVITY_SECURITY_HEADERS = ["Time (ET)", "Event", "Account", "By", "IP", "Details"]
ACTIVITY_ROW_LIMIT = 200


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fetch_activity(range_label: str, username: str | None) -> dict:
    """Everything the Activity tab shows, for one time range and optionally one user."""
    interval = ACTIVITY_RANGES.get(range_label, "24 hours")
    params = {"interval": interval, "username": username}
    with db_pool.connection() as conn:
        summary = conn.execute(
            """
            SELECT
                (SELECT count(*) FROM v_query_activity
                 WHERE created_at > now() - %(interval)s::interval
                   AND (%(username)s::text IS NULL OR username = %(username)s)),
                (SELECT count(DISTINCT username) FROM v_query_activity
                 WHERE created_at > now() - %(interval)s::interval AND username IS NOT NULL
                   AND (%(username)s::text IS NULL OR username = %(username)s)),
                (SELECT count(*) FROM v_query_activity
                 WHERE created_at > now() - %(interval)s::interval AND status = 'error'
                   AND (%(username)s::text IS NULL OR username = %(username)s)),
                (SELECT count(*) FROM v_query_activity
                 WHERE created_at > now() - %(interval)s::interval AND status = 'abandoned'
                   AND (%(username)s::text IS NULL OR username = %(username)s)),
                (SELECT count(*) FROM v_security_events
                 WHERE occurred_at > now() - %(interval)s::interval AND event_type = 'rate_limited'
                   AND (%(username)s::text IS NULL OR username = %(username)s)),
                (SELECT count(*) FROM v_security_events
                 WHERE occurred_at > now() - %(interval)s::interval
                   AND event_type IN ('login_failure', 'login_locked')
                   AND (%(username)s::text IS NULL OR username = %(username)s))
            """,
            params,
        ).fetchone()
        questions = conn.execute(
            """
            SELECT to_char(created_at_local, 'YYYY-MM-DD HH24:MI:SS'), coalesce(username, '(pre-accounts)'),
                   question, status, latency_seconds, source_count, error
            FROM v_query_activity
            WHERE created_at > now() - %(interval)s::interval
              AND (%(username)s::text IS NULL OR username = %(username)s)
            ORDER BY created_at DESC
            LIMIT """ + str(ACTIVITY_ROW_LIMIT),
            params,
        ).fetchall()
        usage = conn.execute(
            """
            SELECT u.username, u.role, u.is_active,
                   count(q.id) FILTER (WHERE q.created_at > now() - %(interval)s::interval),
                   count(q.id) FILTER (WHERE q.created_at > now() - %(interval)s::interval AND q.status = 'error'),
                   to_char(max(q.created_at) AT TIME ZONE 'America/New_York', 'YYYY-MM-DD HH24:MI'),
                   to_char(u.last_login_at AT TIME ZONE 'America/New_York', 'YYYY-MM-DD HH24:MI')
            FROM app_users u
            LEFT JOIN queries q ON q.user_id = u.id
            WHERE %(username)s::text IS NULL OR u.username = %(username)s
            GROUP BY u.id
            ORDER BY 4 DESC, u.username
            """,
            params,
        ).fetchall()
        security = conn.execute(
            """
            SELECT to_char(occurred_at_local, 'YYYY-MM-DD HH24:MI:SS'), event_type,
                   coalesce(username, ''),
                   CASE WHEN actor IS DISTINCT FROM username THEN coalesce(actor, '') ELSE '' END,
                   coalesce(host(ip), ''), details
            FROM v_security_events
            WHERE occurred_at > now() - %(interval)s::interval
              AND (%(username)s::text IS NULL OR username = %(username)s OR actor = %(username)s)
            ORDER BY occurred_at DESC
            LIMIT """ + str(ACTIVITY_ROW_LIMIT),
            params,
        ).fetchall()
    return {"summary": summary, "questions": questions, "usage": usage, "security": security}


def format_activity_summary(summary: tuple, range_label: str, username: str | None) -> str:
    questions, users, errors, abandoned, limited, failed_logins = summary
    scope = f"**{username}**" if username else "everyone"
    return (
        f"Last {range_label}, {scope}:\n\n"
        "| Questions | Active users | Errors | Abandoned | Rate-limit hits | Failed sign-ins |\n"
        "|---|---|---|---|---|---|\n"
        f"| {questions} | {users} | {errors} | {abandoned} | {limited} | {failed_logins} |"
    )


def format_activity(data: dict) -> tuple[list, list, list]:
    questions = [
        [when, who, _clip(question, 200), status,
         float(latency) if latency is not None else None, sources, _clip(error, 120)]
        for when, who, question, status, latency, sources, error in data["questions"]
    ]
    usage = [
        [username, role, "yes" if active else "no", count, errors, last_query or "—", last_login or "—"]
        for username, role, active, count, errors, last_query, last_login in data["usage"]
    ]
    security = [
        [when, event, who, by, ip, _clip(json.dumps(details, separators=(", ", ": ")) if details else "", 160)]
        for when, event, who, by, ip, details in data["security"]
    ]
    return questions, usage, security


def on_activity(range_label: str, username_filter: str, request: gr.Request):
    require_admin(request, "activity")
    username = None
    if (username_filter or "").strip():
        username = accounts.normalize_username(username_filter)
        if username is None:
            message = f"*'{username_filter.strip()}' isn't a valid username.*"
            return message, [], [], []
    data = fetch_activity(range_label, username)
    questions, usage, security = format_activity(data)
    return format_activity_summary(data["summary"], range_label, username), questions, usage, security


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def on_page_load(request: gr.Request):
    """Banner for the signed-in account, and admin tabs shown only to admins."""
    user = accounts.current_user(request)
    is_admin = bool(user and user.is_admin)
    if user is None:
        banner = f"**{SESSION_ENDED}** [Sign out](/logout)"
    else:
        banner = (
            f"Signed in as **{user.username}** ({user.email}) · {user.role} · [Sign out](/logout)"
        )
    return (banner,) + tuple(gr.update(visible=is_admin) for _ in range(4))


def build_ui() -> gr.Blocks:
    private = {"api_visibility": "private"}

    with gr.Blocks(title="HawkEye IT Assistant") as ui:
        gr.Markdown(
            "# HawkEye\nInternal IT help desk knowledge assistant — technicians only. "
            "Describe the customer's issue as you would to a coworker."
        )
        session_banner = gr.Markdown("")

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
                                max_length=MAX_QUESTION_CHARS,
                            )
                            send_button = gr.Button("Send", variant="primary", scale=1)

                    with gr.Column(scale=1):
                        context_markdown = gr.Markdown(
                            value=CONTEXT_PLACEHOLDER,
                            container=True,
                            height=420,
                        )
                        reset_button = gr.Button("New customer / reset chat", variant="secondary")
                # Maps each assistant message's index in the chat to its queries.id,
                # so a thumbs up/down can be filed against the question it rates.
                query_ids = gr.State({})

            # Admin tabs start hidden and are revealed by on_page_load for admins,
            # so a worker never sees them flash in. Hiding is cosmetic: the
            # handlers below each check the role themselves.
            with gr.Tab("Refresh History", visible=False) as refresh_tab:
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

            with gr.Tab("Knowledge Map", visible=False) as kb_map_tab:
                gr.Markdown(
                    "3D map of every chunk in the knowledge base, colored by category."
                )
                kb_map_plot = gr.Plot()

            with gr.Tab("Accounts", visible=False) as accounts_tab:
                gr.Markdown(
                    "Only accounts listed here can sign in. The username is the part of a New Paltz "
                    "address before @newpaltz.edu. New workers sign in with the shared worker "
                    "password and new admins with the admin password. Changing someone's role "
                    "resets their password to the new role's default."
                )
                with gr.Row():
                    new_username = gr.Textbox(
                        label="Username", placeholder="e.g. khans45 or khans45@newpaltz.edu", scale=3,
                    )
                    new_role = gr.Radio(["worker", "admin"], value="worker", label="Role", scale=2)
                    create_button = gr.Button("Create account", variant="primary", scale=1)
                create_result = gr.Markdown("")
                with gr.Accordion("Add several workers at once", open=False):
                    bulk_text = gr.Textbox(
                        label="One username or New Paltz address per line", lines=6,
                    )
                    bulk_button = gr.Button("Create worker accounts")
                    bulk_result = gr.Markdown("")

                gr.Markdown("### Accounts\nClick a row to pick that account below.")
                users_table = gr.Dataframe(
                    headers=USERS_HEADERS, datatype=USERS_DATATYPES, interactive=False, wrap=True,
                )
                with gr.Row():
                    manage_username = gr.Dropdown(label="Account", choices=[], scale=2)
                    disable_button = gr.Button("Disable", variant="stop", scale=1)
                    enable_button = gr.Button("Enable", scale=1)
                    make_admin_button = gr.Button("Make admin", scale=1)
                    make_worker_button = gr.Button("Make worker", scale=1)
                    reset_password_button = gr.Button("Reset password", scale=1)
                manage_result = gr.Markdown("")
                users_refresh_button = gr.Button("Refresh list", size="sm")

            with gr.Tab("Activity", visible=False) as activity_tab:
                with gr.Row():
                    activity_range = gr.Radio(
                        list(ACTIVITY_RANGES), value="24 hours", label="Time range", scale=2,
                    )
                    activity_user = gr.Textbox(
                        label="Username (blank for everyone)", placeholder="e.g. khans45", scale=2,
                    )
                    activity_button = gr.Button("Refresh", variant="primary", scale=1)
                activity_summary = gr.Markdown("")
                gr.Markdown("### Questions")
                activity_questions = gr.Dataframe(
                    headers=ACTIVITY_QUERY_HEADERS, interactive=False, wrap=True,
                )
                gr.Markdown("### Usage by account")
                activity_usage = gr.Dataframe(headers=ACTIVITY_USAGE_HEADERS, interactive=False)
                gr.Markdown("### Sign-ins, lockouts, rate limits, account changes")
                activity_security = gr.Dataframe(
                    headers=ACTIVITY_SECURITY_HEADERS, interactive=False, wrap=True,
                )

        # --- Wiring -----------------------------------------------------------

        admin_tabs = [refresh_tab, kb_map_tab, accounts_tab, activity_tab]
        ui.load(on_page_load, inputs=None, outputs=[session_banner, *admin_tabs], **private)

        refresh_runs_outputs = [
            refresh_runs_table, refresh_runs_empty, refresh_runs_state, refresh_runs_details,
        ]
        refresh_tab.select(on_refresh_history, inputs=None, outputs=refresh_runs_outputs, **private)
        refresh_runs_button.click(on_refresh_history, inputs=None, outputs=refresh_runs_outputs, **private)
        refresh_runs_table.select(
            on_run_selected, inputs=refresh_runs_state, outputs=refresh_runs_details, **private
        )

        kb_map_tab.select(on_kb_map, inputs=None, outputs=kb_map_plot, **private)

        users_outputs_list = [users_table, manage_username]
        accounts_tab.select(on_accounts_tab, inputs=None, outputs=users_outputs_list, **private)
        users_refresh_button.click(on_accounts_tab, inputs=None, outputs=users_outputs_list, **private)
        create_button.click(
            on_create_account, inputs=[new_username, new_role],
            outputs=[create_result, *users_outputs_list, new_username], **private,
        )
        bulk_button.click(
            on_bulk_create, inputs=bulk_text, outputs=[bulk_result, *users_outputs_list], **private,
        )
        for button, action in (
            (disable_button, "disable"), (enable_button, "enable"),
            (make_admin_button, "admin"), (make_worker_button, "worker"),
            (reset_password_button, "reset"),
        ):
            button.click(
                manage_handler(action), inputs=manage_username,
                outputs=[manage_result, *users_outputs_list], **private,
            )
        users_table.select(on_user_row_selected, inputs=None, outputs=manage_username, **private)

        activity_outputs = [activity_summary, activity_questions, activity_usage, activity_security]
        activity_inputs = [activity_range, activity_user]
        activity_tab.select(on_activity, inputs=activity_inputs, outputs=activity_outputs, **private)
        activity_button.click(on_activity, inputs=activity_inputs, outputs=activity_outputs, **private)
        activity_user.submit(on_activity, inputs=activity_inputs, outputs=activity_outputs, **private)

        respond_outputs = [chatbot, context_markdown, query_ids]
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot], **private
        ).then(respond, inputs=[chatbot, query_ids], outputs=respond_outputs, **private)
        send_button.click(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot], **private
        ).then(respond, inputs=[chatbot, query_ids], outputs=respond_outputs, **private)

        def reset_chat():
            return [], CONTEXT_PLACEHOLDER, "", {}

        reset_button.click(
            reset_chat, inputs=None, outputs=[chatbot, context_markdown, message, query_ids], **private
        )

        chatbot.like(on_like, inputs=query_ids, outputs=None, **private)

    return ui


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class ClientContextMiddleware:
    """
    Runs in front of Gradio on every HTTP request. Records the real client IP and
    user agent for the audit log (Gradio's login hook only gets a username and
    password), and turns away a single IP's sign-in flood with a 429 before any
    password is hashed or database row touched. Pure ASGI rather than
    BaseHTTPMiddleware so Gradio's streaming responses pass through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        peer = scope["client"][0] if scope.get("client") else None
        ip = audit.resolve_client_ip(headers, peer)
        ip_token = audit.client_ip.set(ip)
        agent_token = audit.user_agent.set(headers.get("user-agent"))
        try:
            if scope["method"] == "POST" and scope["path"].rstrip("/").endswith("/login"):
                retry = rate_limit.login_ip_limiter.hit(ip or "unknown")
                if retry is not None:
                    if rate_limit.audit_noise_limiter.hit(("login_ip", ip)) is None:
                        await anyio.to_thread.run_sync(
                            lambda: audit.record_event(
                                "rate_limited", ip=ip, agent=headers.get("user-agent"),
                                details={"scope": "login_ip", "retry_after_seconds": int(retry) + 1},
                            )
                        )
                    response = JSONResponse(
                        {"detail": "Too many sign-in attempts from your network. "
                                   f"Try again in {rate_limit.format_wait(retry)}."},
                        status_code=429,
                        headers={"Retry-After": str(int(retry) + 1)},
                    )
                    await response(scope, receive, send)
                    return
            await self.app(scope, receive, send)
        finally:
            audit.client_ip.reset(ip_token)
            audit.user_agent.reset(agent_token)


def check_startup() -> None:
    """Fail fast on missing secrets; warn loudly if nobody could manage accounts."""
    accounts.require_default_passwords()
    try:
        if not accounts.has_active_admin():
            log.warning(
                "No active admin account exists, so nobody can create accounts. Run "
                "`uv run python scripts/manage_users.py bootstrap <username>`."
            )
    except Exception:
        log.warning("Couldn't check for an admin account at startup.", exc_info=True)


def build_app() -> FastAPI:
    theme = gr.themes.Soft(
        primary_hue="blue", secondary_hue="slate", font=["Inter", "system-ui", "sans-serif"]
    )
    ui = build_ui()
    # Four answers stream at once at most (the database pool holds five
    # connections), and at most 50 events wait in line before new ones are refused.
    ui.queue(default_concurrency_limit=4, max_size=50)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ClientContextMiddleware)
    return gr.mount_gradio_app(
        app,
        ui,
        path="/",
        auth=accounts.authenticate,
        auth_message=(
            "Sign in with your New Paltz username (the part of your email before "
            "@newpaltz.edu). Accounts are created by HawkEye admins."
        ),
        theme=theme,
        favicon_path=FAVICON,
        allowed_paths=["static"],
        footer_links=["gradio", "settings"],
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    check_startup()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()

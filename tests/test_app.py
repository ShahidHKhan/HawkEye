"""
Offline tests for app.py: the Refresh History tab's rendering, role checks on
every admin handler, and the question flow (validation, one-at-a-time, quotas,
start/finish logging). No database or model: refresh runs are hand-written
changes payloads of the shape refresh_kb.py logs, and everything chat_stream
calls out to is stubbed.

    python tests/test_app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://placeholder/unused")
os.environ.setdefault("GOOGLE_API_KEY", "placeholder-unused")

import app  # noqa: E402
from _runner import run_tests  # noqa: E402


class FakeSelect:
    """Stands in for gr.SelectData, which carries [row, column]."""
    def __init__(self, row):
        self.index = [row, 0]


def test_lists_the_articles_a_run_added():
    rendered = app.format_run_details({
        "new": [
            {"article_id": "170662", "title": "My Courses Tabs", "chunk_count": 3},
            {"article_id": "170648", "title": "PT Greenroom", "chunk_count": 1},
        ],
    })
    assert "**Added** (2)" in rendered
    assert "`170662` My Courses Tabs — 3 chunks" in rendered
    assert "`170648` PT Greenroom — 1 chunks" in rendered


def test_buckets_render_in_reading_order():
    rendered = app.format_run_details({
        "removed": [{"article_id": "3", "title": "C"}],
        "new": [{"article_id": "1", "title": "A"}],
        "changed": [{"article_id": "2", "title": "B"}],
    })
    assert rendered.index("**Added**") < rendered.index("**Updated**") < rendered.index("**Removed**")


def test_empty_successful_run_reads_as_up_to_date():
    assert "already up to date" in app.format_run_details({"new": [], "changed": []})


def test_empty_failed_run_does_not_claim_it_was_up_to_date():
    """The 8/10 and 8/17 runs processed nothing because they crashed, not because
    there was nothing to do -- the panel must not conflate the two."""
    rendered = app.format_run_details({"new": []}, "No existing knowledge-base file found for 111141")
    assert "already up to date" not in rendered
    assert "stopped on an error" in rendered


def test_null_changes_is_handled():
    assert app.format_run_details(None).startswith("*")


def test_failed_articles_render_with_their_reason():
    rendered = app.format_run_details(
        {"failed": [{"article_id": "57276", "title": "Getting Started", "error": "embedding timeout"}]},
        "1 article(s) failed",
    )
    assert "`57276` Getting Started — embedding timeout" in rendered
    assert "already up to date" not in rendered


def test_untitled_article_still_renders():
    rendered = app.format_run_details({"removed": [{"article_id": "999", "title": None}]})
    assert "`999` *untitled*" in rendered


def test_row_click_shows_that_row():
    state = [
        {"error": None, "changes": {"new": [{"article_id": "1", "title": "First"}]}},
        {"error": None, "changes": {"changed": [{"article_id": "2", "title": "Second"}]}},
    ]
    assert "Second" in app.show_run_details(state, FakeSelect(1))
    assert "First" in app.show_run_details(state, FakeSelect(0))


def test_row_click_out_of_range_falls_back_to_the_placeholder():
    state = [{"error": None, "changes": {}}]
    assert app.show_run_details(state, FakeSelect(99)) == app.REFRESH_RUNS_DETAIL_PLACEHOLDER
    assert app.show_run_details([], FakeSelect(0)) == app.REFRESH_RUNS_DETAIL_PLACEHOLDER


def test_row_click_passes_the_error_through():
    state = [{"error": "boom", "changes": {"new": []}}]
    assert "stopped on an error" in app.show_run_details(state, FakeSelect(0))


def test_summary_table_still_has_nine_columns():
    """Guards the row unpacking, which now has to skip the changes column."""
    from datetime import datetime, timezone

    started = datetime(2026, 8, 31, 9, 34, tzinfo=timezone.utc)
    finished = datetime(2026, 8, 31, 9, 51, tzinfo=timezone.utc)
    rows = [
        (started, finished, "public", 7, 15, 631, 0, "some error", {"new": []}),
        (started, finished, "public", 0, 0, 644, 0, None, {"new": []}),
        (started, None, "public", 0, 0, 0, 0, None, None),
    ]
    formatted = app.format_refresh_runs(rows)
    assert all(len(row) == len(app.REFRESH_RUNS_HEADERS) for row in formatted)
    assert [row[7] for row in formatted] == ["Failed", "Success", "Running"]
    assert formatted[2][1] == "—", "an unfinished run shows a dash, not a crash"


# --- accounts, roles, and the question flow --------------------------------------

import gradio as gr  # noqa: E402

from implementation.accounts import User  # noqa: E402
from implementation.rate_limit import RateLimited  # noqa: E402

ADMIN = User(id=1, username="khans45", role="admin", is_active=True)
WORKER = User(id=3, username="jdoe12", role="worker", is_active=True)


class FakeRequest:
    def __init__(self, username="jdoe12"):
        self.username = username
        self.headers = {"user-agent": "test"}
        self.client = None
        self.session_hash = "session-1"


class Stubs:
    """Swap module attributes for the length of a test, then put them back."""

    def __init__(self):
        self._saved = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def signed_in_as(stubs, user):
    stubs.set(app.accounts, "current_user", lambda request: user)
    events = []
    stubs.set(app.audit, "record_event", lambda event_type, **kw: events.append((event_type, kw)))
    return events


def expect_gr_error(fn) -> str:
    try:
        fn()
    except gr.Error as e:
        return str(e)
    raise AssertionError("expected gr.Error")


def test_workers_are_refused_by_every_admin_handler():
    """Tabs are hidden from workers, but the handlers are callable directly -- they must check."""
    stubs = Stubs()
    events = signed_in_as(stubs, WORKER)
    app.rate_limit.audit_noise_limiter._hits.clear()
    try:
        request = FakeRequest()
        for call in (
            lambda: app.on_refresh_history(request),
            lambda: app.on_run_selected([], FakeSelect(0), request),
            lambda: app.on_kb_map(request),
            lambda: app.on_accounts_tab(request),
            lambda: app.on_create_account("newuser1", "admin", request),
            lambda: app.on_bulk_create("a1\nb2", request),
            lambda: app.manage_account("admin", "jdoe12", request),
            lambda: app.on_activity("24 hours", "", request),
        ):
            assert "admins only" in expect_gr_error(call)
    finally:
        stubs.restore()
    assert events and all(event == "access_denied" for event, _ in events)


def test_signed_out_requests_are_refused():
    stubs = Stubs()
    signed_in_as(stubs, None)
    try:
        assert "session has ended" in expect_gr_error(lambda: app.on_refresh_history(FakeRequest("ghost")))
        assert "session has ended" in expect_gr_error(lambda: app.require_user(FakeRequest("ghost")))
    finally:
        stubs.restore()


def test_admin_can_use_admin_handlers():
    stubs = Stubs()
    signed_in_as(stubs, ADMIN)
    try:
        state = [{"error": None, "changes": {"new": [{"article_id": "1", "title": "First"}]}}]
        assert "First" in app.on_run_selected(state, FakeSelect(0), FakeRequest("khans45"))
    finally:
        stubs.restore()


def test_page_load_shows_admin_tabs_only_to_admins():
    stubs = Stubs()
    try:
        signed_in_as(stubs, ADMIN)
        banner, *tabs = app.on_page_load(FakeRequest("khans45"))
        assert "khans45@newpaltz.edu" in banner and all(t["visible"] for t in tabs)
        signed_in_as(stubs, WORKER)
        banner, *tabs = app.on_page_load(FakeRequest())
        assert "worker" in banner and not any(t["visible"] for t in tabs)
    finally:
        stubs.restore()


def question_stubs(answer_stream, quota=lambda user: None):
    """Stub everything chat_stream touches outside itself; returns (stubs, record)."""
    stubs = Stubs()
    record = {"started": [], "finished": [], "events": []}

    def start(user, question, history_length, session):
        record["started"].append((user.id, question, session))
        return 42

    def finish(query_id, status, answer, sources, latency, error):
        record["finished"].append((query_id, status, answer, error))

    stubs.set(app.rate_limit, "check_question_quota", quota)
    stubs.set(app, "start_query", start)
    stubs.set(app, "finish_query", finish)
    stubs.set(app, "answer_question_stream", answer_stream)
    stubs.set(app.audit, "record_event", lambda event_type, **kw: record["events"].append(event_type))
    app.rate_limit.question_guard._held.clear()
    app.rate_limit.audit_noise_limiter._hits.clear()
    return stubs, record


class FakeChunk:
    page_content = "Restart the router."
    metadata = {"source": "knowledge-base/Network/wifi.md"}


def streaming_answer(question, history):
    yield "Restart", [FakeChunk()]
    yield "Restart the router.", [FakeChunk()]


def test_completed_question_is_logged_start_and_finish():
    stubs, record = question_stubs(streaming_answer)
    try:
        outputs = list(app.chat_stream("wifi down", [], WORKER, FakeRequest()))
    finally:
        stubs.restore()
    assert outputs[-1][0] == "Restart the router." and outputs[-1][2] == 42
    assert record["started"] == [(3, "wifi down", "session-1")]
    assert record["finished"] == [(42, "completed", "Restart the router.", None)]
    assert 3 not in app.rate_limit.question_guard._held


def test_abandoned_question_is_still_recorded():
    """Closing the tab mid-answer used to skip logging entirely."""
    stubs, record = question_stubs(streaming_answer)
    try:
        stream = app.chat_stream("wifi down", [], WORKER, FakeRequest())
        next(stream)
        stream.close()
    finally:
        stubs.restore()
    assert record["finished"] == [(42, "abandoned", "Restart", None)]
    assert 3 not in app.rate_limit.question_guard._held


def test_pipeline_error_is_logged_and_friendly():
    def broken(question, history):
        raise TimeoutError("gemini timed out")
        yield  # pragma: no cover

    stubs, record = question_stubs(broken)
    try:
        outputs = list(app.chat_stream("wifi down", [], WORKER, FakeRequest()))
    finally:
        stubs.restore()
    assert outputs[-1][0] == app.FRIENDLY_ERROR
    assert record["finished"] == [(42, "error", None, "gemini timed out")]


def test_rate_limited_question_never_reaches_the_model():
    def over_quota(user):
        raise RateLimited("user_minute", retry_after=40, limit=5, period="minute")

    stubs, record = question_stubs(streaming_answer, quota=over_quota)
    try:
        outputs = list(app.chat_stream("wifi down", [], WORKER, FakeRequest()))
    finally:
        stubs.restore()
    assert outputs == [("You've reached the limit of 5 questions per minute. Try again in 40 seconds.",
                        "*No sources: this question wasn't run.*", None)]
    assert record["started"] == [] and record["events"] == ["rate_limited"]
    assert 3 not in app.rate_limit.question_guard._held


def test_second_question_while_one_is_streaming_is_refused():
    stubs, record = question_stubs(streaming_answer)
    try:
        first = app.chat_stream("wifi down", [], WORKER, FakeRequest())
        next(first)
        second = list(app.chat_stream("printer jam", [], WORKER, FakeRequest()))
        assert "still answering your previous question" in second[0][0]
        list(first)
        assert len(list(app.chat_stream("printer jam", [], WORKER, FakeRequest()))) == 2
    finally:
        stubs.restore()


def test_oversized_and_empty_questions_are_refused_before_anything_runs():
    stubs, record = question_stubs(streaming_answer)
    try:
        long_out = list(app.chat_stream("x" * (app.MAX_QUESTION_CHARS + 1), [], WORKER, FakeRequest()))
        empty_out = list(app.chat_stream("   ", [], WORKER, FakeRequest()))
    finally:
        stubs.restore()
    assert "the limit is" in long_out[0][0]
    assert empty_out[0][0] == "Type a question first."
    assert record["started"] == []


def test_only_recent_history_is_sent_to_the_pipeline():
    seen = {}

    def capture(question, history):
        seen["history"] = history
        yield "ok", []

    stubs, record = question_stubs(capture)
    history = [{"role": "user", "content": str(i)} for i in range(25)]
    try:
        list(app.chat_stream("follow-up", history, WORKER, FakeRequest()))
    finally:
        stubs.restore()
    assert len(seen["history"]) == app.MAX_HISTORY_MESSAGES
    assert seen["history"][-1]["content"] == "24"


def test_bulk_results_summarize_each_line():
    rendered = app.format_bulk_results([("jdoe12", True, "created jdoe12"), ("bad name", False, "invalid")])
    assert rendered.startswith("**1 of 2 created.**")
    assert "- Skipped `bad name`: invalid" in rendered


def test_activity_summary_names_its_scope():
    assert "everyone" in app.format_activity_summary((3, 2, 1, 0, 4, 5), "24 hours", None)
    assert "**jdoe12**" in app.format_activity_summary((3, 1, 0, 0, 0, 0), "7 days", "jdoe12")


# --- /healthz ----------------------------------------------------------------------

from contextlib import contextmanager  # noqa: E402


class FakePool:
    """Stands in for db_pool: counts probes, and can fail the way an unreachable database does."""

    def __init__(self, fail=False):
        self.fail = fail
        self.probes = 0

    @contextmanager
    def connection(self, timeout=None):
        self.probes += 1
        if self.fail:
            raise TimeoutError("couldn't get a connection after 5.00 sec")
        yield type("FakeConn", (), {"execute": lambda self, sql: None})()


def health_stubs(pool):
    stubs = Stubs()
    stubs.set(app, "db_pool", pool)
    app._health.update(checked_at=None, ok=False)
    return stubs


def test_healthz_is_ok_when_the_database_answers():
    stubs = health_stubs(FakePool())
    try:
        response = app.healthz()
    finally:
        stubs.restore()
    assert response.status_code == 200
    assert response.body == b'{"status":"ok"}'


def test_healthz_is_503_when_the_database_is_unreachable():
    """The 2026-10-26 outage: every pool checkout timed out while the page still loaded."""
    stubs = health_stubs(FakePool(fail=True))
    try:
        response = app.healthz()
    finally:
        stubs.restore()
    assert response.status_code == 503
    assert b"5.00 sec" not in response.body, "no error details to an unauthenticated caller"


def test_healthz_probes_at_most_once_per_cache_window():
    pool = FakePool()
    stubs = health_stubs(pool)
    try:
        for _ in range(20):
            app.healthz()
        assert pool.probes == 1
        app._health["checked_at"] -= app.HEALTH_CACHE_SECONDS
        app.healthz()
        assert pool.probes == 2
    finally:
        stubs.restore()


def test_healthz_is_served_without_signing_in():
    """Fly's checker and the uptime workflow have no account; the route must sit outside Gradio's auth."""
    from fastapi.testclient import TestClient

    stubs = health_stubs(FakePool())
    try:
        with TestClient(app.build_app()) as client:
            response = client.get("/healthz")
    finally:
        stubs.restore()
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))

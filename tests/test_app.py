"""
Offline tests for the Refresh History tab's rendering. No database: every run
here is a hand-written changes payload of the shape refresh_kb.py logs.

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


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))

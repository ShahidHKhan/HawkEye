"""
Offline tests for refresh_kb.py's article resolution and per-article failure
isolation. Touches no database, no network, and -- deliberately -- no local
knowledge-base/ mirror: assuming that mirror is present is the exact bug these
tests exist to guard against, so every path here is built in a temp directory.

    python tests/test_refresh_kb.py
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing refresh_kb pulls in modules that build a connection pool and a Gemini
# client at import time and refuse to import without these set. Nothing here
# actually connects, so placeholders keep the suite runnable without secrets.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://placeholder/unused")
os.environ.setdefault("GOOGLE_API_KEY", "placeholder-unused")

import refresh_kb as rk  # noqa: E402
from _runner import run_tests  # noqa: E402
from implementation.diff_articles import ArticleDiff  # noqa: E402


@contextmanager
def empty_mirror():
    """Point refresh_kb at an empty knowledge-base/, i.e. a bare CI checkout."""
    original = rk.KNOWLEDGE_BASE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        rk.KNOWLEDGE_BASE_PATH = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            rk.KNOWLEDGE_BASE_PATH = original


@contextmanager
def mirror_containing(*relative_paths):
    """A knowledge-base/ holding exactly the given 'Category/file.md' entries."""
    original = rk.KNOWLEDGE_BASE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for relative in relative_paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("---\ntitle: Existing\n---\nbody\n", encoding="utf-8")
        rk.KNOWLEDGE_BASE_PATH = root
        try:
            yield root
        finally:
            rk.KNOWLEDGE_BASE_PATH = original


def article(article_id="57276", title="Getting Started: Faculty and Staff",
            category="Getting-Started-Guides"):
    return {"article_id": article_id, "title": title, "top_level_category": category, "url": "u"}


# --- path resolution -------------------------------------------------------

def test_missing_mirror_resolves_instead_of_raising():
    """The regression: a changed article whose file isn't on this machine."""
    with empty_mirror() as root:
        path = rk.resolve_markdown_path(article())
        assert path.parent == root / "Getting-Started-Guides", path
        assert path.name.endswith("-57276.md"), path.name


def test_missing_mirror_is_not_an_error():
    with empty_mirror():
        assert rk.find_existing_markdown_paths("57276") == []


def test_existing_file_is_rewritten_in_place():
    """An upstream retitle must not orphan the old file under a new slug."""
    existing = "Getting-Started-Guides/Old-Slug-57276.md"
    with mirror_containing(existing) as root:
        path = rk.resolve_markdown_path(article(title="A Completely New Title"))
        assert path == root / existing, path


def test_cross_listed_article_follows_the_crawl_category():
    """Two files for one article is normal data, not corruption -- pick, don't raise."""
    with mirror_containing(
        "Digital-Accessibility/Accessible-Syllabus-Template-102445.md",
        "Software-and-Apps/Accessible-Syllabus-Template-102445.md",
    ):
        for category in ("Software-and-Apps", "Digital-Accessibility"):
            path = rk.resolve_markdown_path(article("102445", "Accessible Syllabus Template", category))
            assert path.parent.name == category, (category, path)


def test_cross_listed_fallback_is_deterministic():
    """With no category to steer by, the same file must win every run."""
    files = (
        "Software-and-Apps/Accessible-Syllabus-Template-102445.md",
        "Digital-Accessibility/Accessible-Syllabus-Template-102445.md",
    )
    with mirror_containing(*files) as root:
        chosen = {rk.resolve_markdown_path(article("102445", "x", None)) for _ in range(5)}
        assert len(chosen) == 1, chosen
        assert chosen.pop() == sorted(root.glob("*/*-102445.md"))[0]


def test_article_with_no_category_lands_in_uncategorized():
    with empty_mirror() as root:
        path = rk.resolve_markdown_path(article(category=None))
        assert path.parent == root / "Uncategorized", path


def test_prefer_category_falls_back_to_first_candidate():
    candidates = [("src-a", "Hardware"), ("src-b", "Policies")]
    assert rk._prefer_category(candidates, "Policies", lambda row: row[1]) == ("src-b", "Policies")
    assert rk._prefer_category(candidates, "Nonexistent", lambda row: row[1]) == ("src-a", "Hardware")
    assert rk._prefer_category(candidates, None, lambda row: row[1]) == ("src-a", "Hardware")


# --- chunking prompt -------------------------------------------------------

def test_chunking_prompt_never_carries_a_machine_path():
    """Article 42989's stored D:/ source made Gemini hang and 500 on every attempt."""
    from implementation.ingest import make_prompt, prompt_source

    relative = "Software-and-Apps/OneDrive-Synchronize-your-OneDrive-to-your-computer-Windows-42989.md"
    for stored in (
        "D:/mrsha/Projects/HawkEye/knowledge-base/" + relative,
        "/home/runner/work/HawkEye/HawkEye/knowledge-base/" + relative,
        "C:\\Users\\someone\\HawkEye\\knowledge-base\\" + relative.replace("/", "\\"),
    ):
        assert prompt_source(stored) == relative, stored
        prompt = make_prompt({"type": "Software-and-Apps", "source": stored, "text": "body"})
        assert "knowledge-base" not in prompt and relative in prompt, stored
    assert prompt_source("somewhere/else/file-1.md") == "file-1.md"


def test_looped_chunk_is_rejected_not_stored():
    """Article 156931 was stored with a 1,024,995-char chunk of dashes."""
    from implementation.ingest import Chunk, check_chunks

    document = {"type": "Internal-Documentation", "source": "x.md", "text": "a" * 3567}
    fine = Chunk(headline="h", summary="s", original_text="a" * 3567)
    check_chunks([fine], document)  # a chunk may be the whole document

    looped = Chunk(headline="h", summary="s", original_text="-" * 1_024_995)
    try:
        check_chunks([fine, looped], document)
    except ValueError as e:
        assert "looped" in str(e), e
    else:
        raise AssertionError("a chunk longer than its document must be rejected")


@contextmanager
def chunker_that_always_fails(exc):
    """Stand in for chunk_with_llm once all its retries are spent, failing with exc."""
    from tenacity import Future, RetryError
    import implementation.ingest as ingest

    def failing(document):
        attempt = Future(5)
        attempt.set_exception(exc)
        raise RetryError(attempt)

    original = ingest.chunk_with_llm
    ingest.chunk_with_llm = failing
    try:
        yield ingest
    finally:
        ingest.chunk_with_llm = original


def unchunkable_500():
    from google.genai.errors import ServerError
    return ServerError(500, {"error": {"code": 500, "message": "internal", "status": "INTERNAL"}})


def test_short_unchunkable_document_is_stored_whole():
    """32121, 67137 and 169560: Gemini 500s on every attempt, at temperature 0."""
    with chunker_that_always_fails(unchunkable_500()) as ingest:
        document = {
            "type": "Networking-WiFi",
            "source": "D:/x/knowledge-base/Networking-WiFi/Google-Chrome-Pop-up-Blocker-Settings-32121.md",
            "title": "Google Chrome: Pop-up Blocker Settings",
            "text": "To allow pop-ups, open Settings.\n",
        }
        [chunk] = ingest.process_document(document)
        assert chunk.page_content == "Google Chrome: Pop-up Blocker Settings\n\nTo allow pop-ups, open Settings."
        assert chunk.metadata == {"source": document["source"], "type": "Networking-WiFi"}

        del document["title"]  # a full ingest has no title; fall back to the filename
        [chunk] = ingest.process_document(document)
        assert chunk.page_content.startswith("Google Chrome Pop up Blocker Settings\n\n"), chunk.page_content


def test_long_or_otherwise_failing_documents_still_raise():
    """An outage or a bad key must fail loudly, not quietly store whole articles."""
    from tenacity import RetryError

    short = {"type": "t", "source": "s-1.md", "text": "body"}
    long = {"type": "t", "source": "s-1.md", "text": "x" * 4001}
    for exc, document in (
        (unchunkable_500(), long),
        (PermissionError("API key not valid"), short),
    ):
        with chunker_that_always_fails(exc) as ingest:
            try:
                ingest.process_document(document)
            except RetryError:
                pass
            else:
                raise AssertionError(f"{type(exc).__name__} on a {len(document['text'])}-char doc must raise")


# --- circuit breaker -------------------------------------------------------

def test_removal_safety_threshold():
    assert rk.removal_exceeds_safety_threshold(50, 100) is True
    assert rk.removal_exceeds_safety_threshold(10, 100) is False   # exactly at the threshold
    assert rk.removal_exceeds_safety_threshold(11, 100) is True
    assert rk.removal_exceeds_safety_threshold(5, 0) is False      # nothing to compare against


# --- per-article failure isolation ----------------------------------------

def test_record_article_failure():
    failed = []
    rk.record_article_failure(failed, "12345", "Some Article", RuntimeError("boom"))
    assert failed == [{"article_id": "12345", "title": "Some Article", "error": "boom"}]


@contextmanager
def stubbed_run(diff, known, breaks=frozenset(), crawl=None):
    """
    Run run_refresh() against fabricated data with every DB call and per-article
    processor replaced, so nothing is written anywhere. Yields the list that
    log_refresh_run rows land in.
    """
    logged = []
    originals = {name: getattr(rk, name) for name in (
        "crawl_public_articles", "get_known_article_state", "diff_articles", "has_existing_chunks",
        "process_new_article", "process_changed_article", "process_removed_article", "log_refresh_run",
    )}

    def guard(article_id):
        if article_id in breaks:
            raise RuntimeError(f"synthetic failure for {article_id}")

    def fake_process(article_or_id, scope=None, chunk_count=3):
        article_id = article_or_id if isinstance(article_or_id, str) else article_or_id["article_id"]
        guard(article_id)
        title = None if isinstance(article_or_id, str) else article_or_id.get("title")
        return {"article_id": article_id, "title": title, "chunk_count": chunk_count}

    rk.crawl_public_articles = crawl or (lambda: ([], []))
    rk.get_known_article_state = lambda scope: known
    rk.diff_articles = lambda crawled, known_, failed_ids: diff
    rk.has_existing_chunks = lambda article_id: False
    rk.process_new_article = fake_process
    rk.process_changed_article = fake_process
    rk.process_removed_article = lambda article_id: fake_process(article_id)
    rk.log_refresh_run = lambda **kw: logged.append(kw)
    try:
        yield logged
    finally:
        for name, value in originals.items():
            setattr(rk, name, value)


def run_and_capture(diff, known, breaks=frozenset(), crawl=None):
    """Returns (the single logged row, the exception raised or None)."""
    with stubbed_run(diff, known, breaks, crawl) as logged:
        raised = None
        try:
            rk.run_refresh("public")
        except Exception as e:  # noqa: BLE001 -- the point is to inspect whatever came out
            raised = e
        assert len(logged) == 1, f"expected exactly one refresh_runs row, got {len(logged)}"
        return logged[0], raised


def art(article_id):
    return {"article_id": article_id, "title": f"Title {article_id}", "top_level_category": "Hardware"}


KNOWN_100 = {f"k{i}": None for i in range(100)}


def test_one_failure_per_phase_does_not_abort_the_run():
    diff = ArticleDiff(
        new=[art("n1"), art("n2-BAD"), art("n3")],
        changed=[art("c1"), art("c2-BAD"), art("c3")],
        unchanged=[art("u1")],
        removed_article_ids=["r1", "r2-BAD"],
        failed=[],
    )
    row, raised = run_and_capture(diff, KNOWN_100, {"n2-BAD", "c2-BAD", "r2-BAD"})
    changes = row["changes"]

    assert isinstance(raised, rk.PartialRefreshFailure), raised
    assert [c["article_id"] for c in changes["new"]] == ["n1", "n3"]
    assert [c["article_id"] for c in changes["changed"]] == ["c1", "c3"]
    assert [c["article_id"] for c in changes["removed"]] == ["r1"], "removal phase must still run"
    assert [f["article_id"] for f in changes["failed"]] == ["n2-BAD", "c2-BAD", "r2-BAD"]
    assert "synthetic failure" in changes["failed"][0]["error"]
    assert "3 article(s) failed" in row["error"], row["error"]
    # Counts describe the diff, not what got through -- that's what changes is for.
    assert (row["new_count"], row["changed_count"], row["removed_count"]) == (3, 3, 2)


def test_clean_run_records_no_error():
    diff = ArticleDiff(new=[art("n1")], changed=[art("c1")], unchanged=[],
                       removed_article_ids=[], failed=[])
    row, raised = run_and_capture(diff, KNOWN_100)
    assert raised is None, raised
    assert row["error"] is None
    assert row["changes"]["failed"] == []


def test_healed_is_only_recorded_when_healing_succeeds():
    diff = ArticleDiff(new=[art("h1-BAD")], changed=[], unchanged=[],
                       removed_article_ids=[], failed=[])
    with stubbed_run(diff, KNOWN_100, {"h1-BAD"}) as logged:
        rk.has_existing_chunks = lambda article_id: True  # forces the healing path
        try:
            rk.run_refresh("public")
        except rk.PartialRefreshFailure:
            pass
    changes = logged[0]["changes"]
    assert changes["healed"] == [], "an article that failed to heal must not be listed as healed"
    assert [f["article_id"] for f in changes["failed"]] == ["h1-BAD"]


def test_breaker_and_article_failures_are_both_reported():
    diff = ArticleDiff(new=[], changed=[art("c1-BAD")], unchanged=[],
                       removed_article_ids=[f"r{i}" for i in range(50)], failed=[])
    row, raised = run_and_capture(diff, KNOWN_100, {"c1-BAD"})
    assert "safety threshold" in row["error"], row["error"]
    assert "1 article(s) failed" in row["error"], row["error"]
    assert row["changes"]["removed"] == [], "the breaker must still block every removal"
    assert isinstance(raised, rk.PartialRefreshFailure)


def test_breaker_alone_still_exits_zero():
    """Unchanged behavior: a breaker abort is not a crash, so it must not raise."""
    diff = ArticleDiff(new=[], changed=[], unchanged=[],
                       removed_article_ids=[f"r{i}" for i in range(50)], failed=[])
    row, raised = run_and_capture(diff, KNOWN_100)
    assert raised is None, raised
    assert "safety threshold" in row["error"]


def test_genuine_crash_still_aborts_and_reraises():
    """A crawl or DB failure is not per-article, so it must not be swallowed."""
    def boom():
        raise ConnectionError("TeamDynamix unreachable")

    diff = ArticleDiff(new=[], changed=[], unchanged=[], removed_article_ids=[], failed=[])
    row, raised = run_and_capture(diff, KNOWN_100, crawl=boom)
    assert isinstance(raised, ConnectionError), raised
    assert "unreachable" in row["error"]
    assert row["new_count"] == 0, "counts stay zero when the diff never happened"


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))

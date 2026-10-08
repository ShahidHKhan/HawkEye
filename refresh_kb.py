"""
Weekly KB refresh pipeline. Diffs the live TeamDynamix public categories against
article_state, and acts on diff.new, diff.changed, and diff.removed_article_ids
(removal is gated by a circuit breaker -- see REMOVAL_SAFETY_THRESHOLD).

Run from the project root:
    uv run refresh_kb.py
"""

from datetime import datetime, timezone
from pathlib import Path

import frontmatter
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from implementation.diff_articles import crawl_public_articles, db_pool, diff_articles, get_known_article_state
from implementation.ingest import embed_batch, embedding_to_vector_literal, process_document, strip_embedded_images
from scraper.convert_to_markdown import safe_filename, write_markdown_file

load_dotenv(override=True)

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge-base"

# Circuit breaker: if a run would remove more than this fraction of all known
# articles in one go, something is more likely wrong with the crawl (a TD outage,
# a category getting restructured, a bug) than that many articles genuinely
# vanished at once. Tune here -- nowhere else references this threshold directly.
REMOVAL_SAFETY_THRESHOLD = 0.10


class PartialRefreshFailure(RuntimeError):
    """
    Raised at the very end of a run that finished its work but had individual
    articles fail along the way. Exists so those failures still exit the process
    non-zero (keeping the Actions failure notification honest) without any one
    article being able to abort the articles queued behind it.
    """


def _prefer_category(candidates: list, category: str | None, key):
    """
    Pick one candidate deterministically, preferring whichever one sits under the
    fresh crawl's top_level_category. An article cross-listed under more than one
    top-level category legitimately has one markdown file and one chunks.source
    per category (scraper/convert_to_markdown.py writes a file per category
    folder), so "more than one" is normal data, not corruption -- picking rather
    than raising lets a refresh collapse such an article onto a single category
    instead of dying on it. `candidates` must already be sorted, so the fallback
    is stable run to run.
    """
    if category is not None:
        for candidate in candidates:
            if key(candidate) == category:
                return candidate
    return candidates[0]


def find_existing_markdown_paths(article_id: str) -> list[Path]:
    """
    Every local markdown file for this article, sorted for stable ordering.
    Usually 0 or 1: 0 whenever the mirror simply isn't on this machine, which is
    the normal case in CI (knowledge-base/ is gitignored because it carries gated
    internal docs, so a workflow run starts from a checkout with no mirror at
    all), 1 on a machine that has it. 2+ for a cross-listed article.
    """
    return sorted(KNOWLEDGE_BASE_PATH.glob(f"*/*-{article_id}.md"))


def resolve_markdown_path(article: dict) -> Path:
    """
    Where this article's markdown mirror lives, or should live if it isn't here.

    Prefers a file that already exists so an edit rewrites it in place, and falls
    back to computing the path exactly the way process_new_article does. That
    fallback is the whole point: the markdown tree is a best-effort local artifact
    -- the chunks and article_state rows in Postgres are a run's real output -- so
    an absent mirror must never fail an article. Before this, a changed article
    whose file wasn't on the machine raised FileNotFoundError, which meant every
    scheduled CI run died on the first upstream edit it found.
    """
    article_id = article["article_id"]
    category = article.get("top_level_category")
    existing = find_existing_markdown_paths(article_id)
    if existing:
        return _prefer_category(existing, category, lambda path: path.parent.name)
    return KNOWLEDGE_BASE_PATH / (category or "Uncategorized") / safe_filename(article.get("title"), article_id)


def existing_chunk_identity(article_id: str, category: str | None = None) -> tuple[str | None, str | None]:
    """
    The (source, type) pair already stored in chunks for this article, if any,
    looked up by the chunks.article_id key (indexed, and populated for every
    chunk) rather than by matching on source -- source is whatever path the
    machine that last ran ingest.py happened to have, which may not match this
    machine's local knowledge-base/ path at all.

    Reusing the stored pair is what lets a refresh run correctly on a machine
    with no markdown mirror: source stays stable instead of churning to whatever
    path this runner had, and type keeps matching what the Knowledge Map colors
    by, neither of which can be read off the filesystem when the file isn't there.
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT source, type FROM chunks WHERE article_id = %s ORDER BY source",
                (article_id,),
            )
            rows = cur.fetchall()
    if not rows:
        return None, None
    return _prefer_category(rows, category, lambda row: row[1])


def has_existing_chunks(article_id: str) -> bool:
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chunks WHERE article_id = %s LIMIT 1", (article_id,))
            return cur.fetchone() is not None


def process_changed_article(article: dict, scope: str) -> dict:
    """
    Rewrite the local markdown file, re-chunk + re-embed it, and in one transaction:
    replace its chunks and upsert its article_state row. Returns the summary entry
    recorded in refresh_runs.changes.

    Upsert (not a plain UPDATE) so this same function also serves the "healing"
    path in run_refresh(): an article diff_articles() classified as new, but that
    already has chunks, means article_state lost track of it rather than it being
    genuinely new -- there is no article_state row yet to UPDATE in that case.

    Side effect worth knowing about: the DELETE is by article_id, so it clears
    every copy of a cross-listed article's chunks, and the re-insert writes one
    source/type. Such an article therefore collapses onto a single category the
    first time it's edited, which also stops its text being embedded twice and
    taking two slots in one top-k retrieval.
    """
    article_id = article["article_id"]
    category = article.get("top_level_category")
    local_path = resolve_markdown_path(article)
    write_markdown_file(article, local_path)

    # Both fall back to the crawl/computed path rather than the filesystem, so this
    # works identically on a machine holding the full mirror and on a bare CI checkout.
    known_source, known_type = existing_chunk_identity(article_id, category)
    source = known_source or local_path.as_posix()
    doc_type = known_type or category or "Uncategorized"
    document = {
        "type": doc_type,
        "source": source,
        "title": article.get("title"),
        "text": strip_embedded_images(article.get("body", "")),
    }

    chunks = process_document(document)
    vectors = embed_batch([c.page_content for c in chunks])
    modified_date = datetime.fromisoformat(article["modified_date"]) if article.get("modified_date") else None

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE article_id = %s", (article_id,))
            rows = [
                (c.metadata["source"], c.metadata.get("type"), c.page_content, embedding_to_vector_literal(v), article_id)
                for c, v in zip(chunks, vectors)
            ]
            cur.executemany(
                "INSERT INTO chunks (source, type, page_content, embedding, article_id) "
                "VALUES (%s, %s, %s, %s::vector, %s)",
                rows,
            )
            cur.execute(
                """
                INSERT INTO article_state (article_id, url, modified_date, last_crawled, scope)
                VALUES (%s, %s, %s, now(), %s)
                ON CONFLICT (article_id) DO UPDATE
                    SET modified_date = EXCLUDED.modified_date, last_crawled = EXCLUDED.last_crawled
                """,
                (article_id, article["url"], modified_date, scope),
            )
        conn.commit()

    return {"article_id": article_id, "title": article.get("title"), "chunk_count": len(chunks)}


def process_new_article(article: dict, scope: str) -> dict:
    """
    Write the new local markdown file, chunk + embed it, then INSERT (never UPDATE)
    both its chunks and its article_state row -- this article has never been seen
    before, so there is nothing to replace.
    """
    article_id = article["article_id"]
    category = article.get("top_level_category") or "Uncategorized"
    local_path = KNOWLEDGE_BASE_PATH / category / safe_filename(article.get("title"), article_id)
    write_markdown_file(article, local_path)

    document = {
        "type": category,
        "source": local_path.as_posix(),
        "title": article.get("title"),
        "text": strip_embedded_images(article.get("body", "")),
    }
    chunks = process_document(document)
    vectors = embed_batch([c.page_content for c in chunks])
    modified_date = datetime.fromisoformat(article["modified_date"]) if article.get("modified_date") else None

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            rows = [
                (c.metadata["source"], c.metadata.get("type"), c.page_content, embedding_to_vector_literal(v), article_id)
                for c, v in zip(chunks, vectors)
            ]
            cur.executemany(
                "INSERT INTO chunks (source, type, page_content, embedding, article_id) "
                "VALUES (%s, %s, %s, %s::vector, %s)",
                rows,
            )
            cur.execute(
                "INSERT INTO article_state (article_id, url, modified_date, last_crawled, scope) "
                "VALUES (%s, %s, %s, now(), %s)",
                (article_id, article["url"], modified_date, scope),
            )
        conn.commit()

    return {"article_id": article_id, "title": article.get("title"), "chunk_count": len(chunks)}


def removal_exceeds_safety_threshold(removed_count: int, known_count: int) -> bool:
    """
    True if removed_count/known_count exceeds REMOVAL_SAFETY_THRESHOLD. Pure and
    DB-free by design, so the circuit breaker itself can be tested with fabricated
    numbers without needing a live crawl. known_count == 0 never trips it -- there
    is nothing to compare a removal count against.
    """
    if known_count == 0:
        return False
    return (removed_count / known_count) > REMOVAL_SAFETY_THRESHOLD


def process_removed_article(article_id: str) -> dict:
    """
    Delete an article's chunks, its article_state row, and its local markdown files
    if still present. article_state has no title column, so the title (best-effort,
    for logging only) is read from the first local file's frontmatter before it's
    deleted; if no file is here, title is just None. Every step no-ops gracefully if
    its target doesn't exist -- a partially-cleaned-up article must never crash a run,
    and on CI there is no markdown mirror to clean up in the first place. All copies
    are removed, so a cross-listed article doesn't leave a file behind in its other
    category folder.
    """
    title = None
    for local_path in find_existing_markdown_paths(article_id):
        if title is None:
            try:
                title = frontmatter.load(local_path).get("title")
            except Exception:
                title = None
        local_path.unlink(missing_ok=True)

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE article_id = %s", (article_id,))
            cur.execute("DELETE FROM article_state WHERE article_id = %s", (article_id,))
        conn.commit()

    return {"article_id": article_id, "title": title}


def log_refresh_run(
    scope: str, started_at: datetime, finished_at: datetime,
    new_count: int, changed_count: int, unchanged_count: int, removed_count: int,
    changes: dict, error: str | None,
) -> None:
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO refresh_runs
                    (scope, started_at, finished_at, new_count, changed_count,
                     unchanged_count, removed_count, error, changes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (scope, started_at, finished_at, new_count, changed_count,
                 unchanged_count, removed_count, error, Jsonb(changes)),
            )
        conn.commit()


def record_article_failure(failed: list, article_id: str, title: str | None, exc: Exception) -> None:
    """
    Note one article's failure and let the run continue. Kept to id/title/error --
    the traceback goes to the workflow log, while refresh_runs.changes only needs
    to answer "which articles didn't make it, and why" when read back in the UI.
    """
    print(f"ERROR: article {article_id} ({title}) failed, skipping it: {exc}")
    failed.append({"article_id": article_id, "title": title, "error": str(exc)})


def run_refresh(scope: str = "public") -> None:
    """
    Handles diff.new, diff.changed, and diff.removed_article_ids. Removal is gated
    by removal_exceeds_safety_threshold() -- new/changed are per-article safe and
    always proceed regardless, since only mass-removal is the actual danger. Always
    logs exactly one refresh_runs row, success or failure, so a crash never leaves
    a silent gap.

    A failure on one article is contained to that article: it's recorded in
    changes.failed and the run carries on, then reports the failures at the end.
    Without this, one bad article aborted every article queued behind it *and* the
    whole removal phase, and since its own article_state row was never updated it
    came back unfixed the next week -- which is exactly how a single article blocked
    four consecutive weekly runs.
    """
    started_at = datetime.now(timezone.utc)
    diff = None
    new_processed = []
    changed_processed = []
    removed_processed = []
    healed = []
    failed = []
    error = None

    try:
        crawled, failed_ids = crawl_public_articles()
        known = get_known_article_state(scope)
        diff = diff_articles(crawled, known, failed_ids)

        print(
            f"new={len(diff.new)} changed={len(diff.changed)} unchanged={len(diff.unchanged)} "
            f"removed={len(diff.removed_article_ids)} failed={len(diff.failed)}"
        )

        for article in diff.new:
            article_id = article["article_id"]
            try:
                if has_existing_chunks(article_id):
                    print(
                        f"WARNING: article {article_id} ({article['title']}) was classified as new, "
                        f"but chunks already exist for it -- article_state lost track of it rather than "
                        f"it being genuinely new. Healing via the changed-article path instead of "
                        f"duplicating its chunks."
                    )
                    changed_processed.append(process_changed_article(article, scope))
                    healed.append({"article_id": article_id, "title": article.get("title")})
                else:
                    print(f"Processing new article {article_id}: {article['title']}")
                    new_processed.append(process_new_article(article, scope))
            except Exception as e:
                record_article_failure(failed, article_id, article.get("title"), e)

        for article in diff.changed:
            article_id = article["article_id"]
            print(f"Processing changed article {article_id}: {article['title']}")
            try:
                changed_processed.append(process_changed_article(article, scope))
            except Exception as e:
                record_article_failure(failed, article_id, article.get("title"), e)

        if removal_exceeds_safety_threshold(len(diff.removed_article_ids), len(known)):
            ratio = len(diff.removed_article_ids) / len(known) if known else 0.0
            error = (
                f"removal count exceeds safety threshold — aborted, 0 removals processed "
                f"({len(diff.removed_article_ids)}/{len(known)} = {ratio:.1%} "
                f"> {REMOVAL_SAFETY_THRESHOLD:.0%})"
            )
            print(f"WARNING: {error}")
        else:
            for article_id in diff.removed_article_ids:
                print(f"Processing removed article {article_id}")
                try:
                    removed_processed.append(process_removed_article(article_id))
                except Exception as e:
                    record_article_failure(failed, article_id, None, e)

        if failed:
            # Raised only now that every article that could be processed has been.
            # Folds in a circuit-breaker message if there is one, since `error` is
            # about to be overwritten by the handler below with this exception's text.
            summary = (
                f"{len(failed)} article(s) failed and were skipped, the rest of the run completed "
                f"(see changes.failed): {', '.join(f['article_id'] for f in failed[:5])}"
                f"{' ...' if len(failed) > 5 else ''}"
            )
            raise PartialRefreshFailure(f"{error} | {summary}" if error else summary)

    except Exception as e:
        error = str(e)
        print(f"Refresh run failed: {error}")
        raise  # re-raise after `finally` logs the row, so the process exits non-zero on a
               # genuine crash, and on a run that finished with per-article failures
               # (PartialRefreshFailure) -- a circuit-breaker abort never reaches this
               # branch, so it still exits 0. This is what makes GitHub Actions'
               # failure notification meaningful.

    finally:
        finished_at = datetime.now(timezone.utc)
        changes = {
            "new": new_processed,
            "changed": changed_processed,
            "removed": removed_processed,
            "healed": healed,
            "failed": failed,
        }
        log_refresh_run(
            scope=scope,
            started_at=started_at,
            finished_at=finished_at,
            new_count=len(diff.new) if diff else 0,
            changed_count=len(diff.changed) if diff else 0,
            unchanged_count=len(diff.unchanged) if diff else 0,
            removed_count=len(diff.removed_article_ids) if diff else 0,
            changes=changes,
            error=error,
        )
        print(f"Logged refresh_runs row (error={error!r})")


if __name__ == "__main__":
    run_refresh()

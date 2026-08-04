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


def find_existing_markdown_path(article_id: str) -> Path:
    matches = list(KNOWLEDGE_BASE_PATH.glob(f"*/*-{article_id}.md"))
    if not matches:
        raise FileNotFoundError(f"No existing knowledge-base file found for article_id={article_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple knowledge-base files found for article_id={article_id}: {matches}")
    return matches[0]


def existing_chunk_source(article_id: str) -> str | None:
    """
    The exact `source` string already stored in chunks for this article, if any,
    looked up by the chunks.article_id key (indexed, and the same key ingest's
    original run must have populated) rather than by matching on source -- source
    is whatever path the machine that last ran ingest.py happened to have, which
    may not match this machine's local knowledge-base/ path at all.
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT source FROM chunks WHERE article_id = %s", (article_id,))
            rows = cur.fetchall()
    if len(rows) > 1:
        raise RuntimeError(f"Multiple chunk sources found for article_id={article_id}: {rows}")
    return rows[0][0] if rows else None


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
    """
    article_id = article["article_id"]
    local_path = find_existing_markdown_path(article_id)
    write_markdown_file(article, local_path)

    source = existing_chunk_source(article_id) or local_path.as_posix()
    doc_type = local_path.parent.name
    document = {"type": doc_type, "source": source, "text": strip_embedded_images(article.get("body", ""))}

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
    Delete an article's chunks, its article_state row, and its local markdown file
    if still present. article_state has no title column, so the title (best-effort,
    for logging only) is read from the local file's frontmatter before it's deleted;
    if the file is already gone, title is just None. Every step no-ops gracefully if
    its target doesn't exist -- a partially-cleaned-up article must never crash a run.
    """
    title = None
    try:
        local_path = find_existing_markdown_path(article_id)
    except FileNotFoundError:
        local_path = None

    if local_path is not None:
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


def run_refresh(scope: str = "public") -> None:
    """
    Handles diff.new, diff.changed, and diff.removed_article_ids. Removal is gated
    by removal_exceeds_safety_threshold() -- new/changed are per-article safe and
    always proceed regardless, since only mass-removal is the actual danger. Always
    logs exactly one refresh_runs row, success or failure, so a crash never leaves
    a silent gap.
    """
    started_at = datetime.now(timezone.utc)
    diff = None
    new_processed = []
    changed_processed = []
    removed_processed = []
    healed = []
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
            if has_existing_chunks(article_id):
                print(
                    f"WARNING: article {article_id} ({article['title']}) was classified as new, "
                    f"but chunks already exist for it -- article_state lost track of it rather than "
                    f"it being genuinely new. Healing via the changed-article path instead of "
                    f"duplicating its chunks."
                )
                healed.append({"article_id": article_id, "title": article.get("title")})
                changed_processed.append(process_changed_article(article, scope))
            else:
                print(f"Processing new article {article_id}: {article['title']}")
                new_processed.append(process_new_article(article, scope))

        for article in diff.changed:
            print(f"Processing changed article {article['article_id']}: {article['title']}")
            changed_processed.append(process_changed_article(article, scope))

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
                removed_processed.append(process_removed_article(article_id))

    except Exception as e:
        error = str(e)
        print(f"Refresh run failed: {error}")
        raise  # re-raise after `finally` logs the row, so the process exits non-zero on a
               # genuine crash -- a circuit-breaker abort never reaches this branch, so it
               # still exits 0. This is what makes GitHub Actions' failure notification meaningful.

    finally:
        finished_at = datetime.now(timezone.utc)
        changes = {
            "new": new_processed,
            "changed": changed_processed,
            "removed": removed_processed,
            "healed": healed,
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

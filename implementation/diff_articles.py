import os
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv(override=True)

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file")

db_pool = ConnectionPool(
    SUPABASE_DB_URL,
    min_size=1,
    max_size=5,
    check=ConnectionPool.check_connection,
)


def get_known_article_state(scope: str = "public") -> dict[str, datetime | None]:
    """{article_id: modified_date} already recorded in article_state for the given scope."""
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT article_id, modified_date FROM article_state WHERE scope = %s",
                (scope,),
            )
            return dict(cur.fetchall())


@dataclass
class ArticleDiff:
    new: list[dict]
    changed: list[dict]
    unchanged: list[dict]
    removed_article_ids: list[str]
    failed: list[str]


def _parsed_modified_date(article: dict) -> datetime | None:
    """
    article['modified_date'] is the raw ISO string from parse_article() (kept
    as-is so downstream markdown writing preserves the scraper's original
    format) -- parse it here, just for comparison, without mutating the article.
    """
    value = article.get("modified_date")
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def diff_articles(
    crawled: list[dict], known: dict[str, datetime | None], failed_ids: list[str] = []
) -> ArticleDiff:
    """
    Classify a fresh crawl against the known {article_id: modified_date} state:
    not in known -> new, modified_date differs -> changed, else -> unchanged.
    Anything in `known` but absent from both `crawled` and `failed_ids` is removed.
    Ids in `failed_ids` are excluded from every other bucket — a fetch failure means
    "unknown this run," never "unchanged" and never "removed."
    """
    new, changed, unchanged = [], [], []
    seen_ids = set()
    failed_id_set = set(failed_ids)

    for article in crawled:
        article_id = article["article_id"]
        seen_ids.add(article_id)
        if article_id not in known:
            new.append(article)
        elif _parsed_modified_date(article) != known[article_id]:
            changed.append(article)
        else:
            unchanged.append(article)

    removed_article_ids = [
        article_id for article_id in known
        if article_id not in seen_ids and article_id not in failed_id_set
    ]

    return ArticleDiff(
        new=new,
        changed=changed,
        unchanged=unchanged,
        removed_article_ids=removed_article_ids,
        failed=list(failed_ids),
    )


def crawl_public_articles() -> tuple[list[dict], list[str]]:
    """
    Crawl the 7 public categories (Internal-Documentation untouched). parse_article
    already retries transient failures internally (tenacity), so anything that still
    raises here is tracked as a real failure in failed_ids rather than silently
    dropped — a fetch failure must never look like a removal to the caller.
    """
    from scraper.scrape_kb import ARTICLE_ID_RE, TOP_LEVEL_CATEGORIES, crawl_category_tree, parse_article

    public_categories = {
        name: url for name, url in TOP_LEVEL_CATEGORIES.items() if name != "Internal-Documentation"
    }

    crawled_by_id: dict[str, dict] = {}
    failed_ids: list[str] = []

    for name, url in public_categories.items():
        print(f"Crawling {name}...")
        article_urls = crawl_category_tree(url)
        print(f"  {len(article_urls)} article URLs found")
        for article_url in article_urls:
            try:
                article = parse_article(article_url)
            except Exception as e:
                match = ARTICLE_ID_RE.search(article_url)
                failed_ids.append(match.group(1) if match else article_url)
                print(f"  FAILED (after retries): {article_url} ({e})")
                continue
            # an article can be cross-listed under more than one top-level category;
            # last one crawled wins, same as which folder scrape_kb.py's own main() would leave it in
            article["top_level_category"] = name
            crawled_by_id[article["article_id"]] = article

    return list(crawled_by_id.values()), failed_ids


def _dry_run() -> None:
    """
    Read-only sanity check: crawl the 7 public categories fresh, diff against the
    real article_state("public") rows, print a summary. Makes no writes to
    article_state, chunks, or refresh_runs.
    """
    crawled, failed_ids = crawl_public_articles()
    print(f"\n{len(crawled)} unique articles crawled across public categories")
    if failed_ids:
        print(f"{len(failed_ids)} article(s) failed to fetch even after retries: {failed_ids}")

    known = get_known_article_state("public")
    print(f"{len(known)} known public articles in article_state")

    diff = diff_articles(crawled, known, failed_ids)

    print("\n=== Diff summary (scope=public) ===")
    print(f"new:       {len(diff.new)}")
    print(f"changed:   {len(diff.changed)}")
    print(f"unchanged: {len(diff.unchanged)}")
    print(f"removed:   {len(diff.removed_article_ids)}")
    print(f"failed:    {len(diff.failed)}")

    if diff.new:
        print("\n--- new ---")
        for a in diff.new:
            print(f"  {a['article_id']}: {a['title']}")

    if diff.changed:
        print("\n--- changed ---")
        for a in diff.changed:
            print(f"  {a['article_id']}: {a['title']}")

    if diff.removed_article_ids:
        print("\n--- removed (id only — not in fresh crawl, no title available) ---")
        for article_id in diff.removed_article_ids:
            print(f"  {article_id}")

    if diff.failed:
        print("\n--- failed (fetch failed after retries; excluded from new/changed/unchanged/removed) ---")
        for article_id in diff.failed:
            print(f"  {article_id}")


if __name__ == "__main__":
    _dry_run()

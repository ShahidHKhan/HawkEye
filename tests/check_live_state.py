"""
Read-only audit of the live Supabase state behind the refresh pipeline. Not a
unit test -- it asserts nothing and changes nothing, it just prints the handful
of facts that are easy to get wrong and expensive to notice late. Needs a real
SUPABASE_DB_URL in .env; exits quietly if there isn't one.

    python tests/check_live_state.py

Reports:
  * chunks.article_id coverage -- rows without it can't be replaced by a refresh,
    since process_changed_article deletes by article_id.
  * cross-listed articles -- one article with several chunks.source values, i.e.
    its text embedded more than once. These collapse to one source the next time
    the article is edited upstream.
  * source path prefixes -- which machine wrote each batch of chunks.
  * article_state vs chunks -- the two should agree on which articles exist.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

if not os.getenv("SUPABASE_DB_URL"):
    print("SUPABASE_DB_URL not set — skipping the live audit.")
    raise SystemExit(0)

from implementation.diff_articles import db_pool  # noqa: E402


def query(sql, params=()):
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


print("\n=== chunks.article_id coverage ===")
total, missing, distinct = query("""
    SELECT count(*), count(*) FILTER (WHERE article_id IS NULL), count(DISTINCT article_id)
    FROM chunks
""")[0]
print(f"  {total} chunks across {distinct} articles")
if missing:
    print(f"  WARNING: {missing} chunk(s) have no article_id. A refresh deletes by article_id,")
    print("           so those rows can never be replaced and will duplicate on the next edit.")
else:
    print("  every chunk carries an article_id — refreshes can replace all of them")

print("\n=== article_state vs chunks ===")
for scope, count in query("SELECT scope, count(*) FROM article_state GROUP BY scope ORDER BY scope"):
    print(f"  article_state: {count} {scope}")
orphans = query("""
    SELECT count(DISTINCT c.article_id) FROM chunks c
    LEFT JOIN article_state a ON a.article_id = c.article_id
    WHERE a.article_id IS NULL AND c.article_id IS NOT NULL
""")[0][0]
print(f"  {orphans} article(s) have chunks but no article_state row"
      f"{' — these would be re-classified as new and healed' if orphans else ''}")

print("\n=== cross-listed articles (text embedded more than once) ===")
cross_listed = query("""
    SELECT c.article_id, a.scope, count(DISTINCT c.source), array_agg(DISTINCT c.type)
    FROM chunks c
    LEFT JOIN article_state a ON a.article_id = c.article_id
    GROUP BY c.article_id, a.scope
    HAVING count(DISTINCT c.source) > 1
    ORDER BY a.scope, c.article_id
""")
if not cross_listed:
    print("  none — every article has exactly one source")
else:
    print(f"  {len(cross_listed)} article(s), each collapsing to one source when next edited:")
    for article_id, scope, source_count, types in cross_listed:
        print(f"    {article_id} ({scope}): {source_count} sources — {', '.join(sorted(t or '?' for t in types))}")

print("\n=== source path prefixes (which machine wrote what) ===")
for prefix, count in query("""
    SELECT split_part(source, 'knowledge-base/', 1), count(*)
    FROM chunks GROUP BY 1 ORDER BY 2 DESC
"""):
    print(f"  {count:>5}  {prefix or '(relative path)'}")

print("\n=== most recent refresh runs ===")
for started, error, changes in query("""
    SELECT started_at, error, changes FROM refresh_runs ORDER BY started_at DESC LIMIT 5
"""):
    buckets = ", ".join(
        f"{key}={len(changes.get(key) or [])}"
        for key in ("new", "changed", "removed", "healed", "failed")
        if (changes or {}).get(key)
    ) or "nothing processed"
    print(f"  {started:%Y-%m-%d %H:%M UTC}  {'FAILED ' if error else 'ok     '} {buckets}")
    if error:
        print(f"      {error[:120]}")

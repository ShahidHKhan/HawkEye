"""
Converts scraped JSON articles (scraper/output/**/*.json) into Markdown files
with YAML frontmatter, saved into knowledge-base/{TopLevelCategory}/{slug}.md

This matches the folder-per-doc_type convention the RAG build guide expects,
while keeping title/tags/dates/url/category_path as parseable frontmatter
(via python-frontmatter) rather than plain text that would pollute embeddings.

Run from the project root:
    uv run scraper/convert_to_markdown.py
"""

import json
import re
import sys
from pathlib import Path

import frontmatter

DEFAULT_INPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR = Path(__file__).parent.parent / "knowledge-base"


def safe_filename(title: str, article_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\- ]", "", title or "untitled").strip().replace(" ", "-")
    return f"{slug[:80]}-{article_id}.md"


def convert_file(json_path: Path, category_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        article = json.load(f)

    post = frontmatter.Post(article.get("body", ""))
    post["title"] = article.get("title")
    post["tags"] = article.get("tags", [])
    post["category_path"] = article.get("category_path", [])
    post["published_date"] = article.get("published_date")
    post["modified_date"] = article.get("modified_date")
    post["url"] = article.get("url")
    post["article_id"] = article.get("article_id")

    filename = safe_filename(article.get("title"), article.get("article_id"))
    output_path = category_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        frontmatter.dump(post, f)


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_DIR

    if not input_dir.exists():
        print(f"No scraped output found at {input_dir}. Run the scraper first.")
        return

    total = 0
    for category_dir in sorted(input_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        out_category_dir = OUTPUT_DIR / category_dir.name
        out_category_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(category_dir.glob("*.json"))
        print(f"{category_dir.name}: converting {len(json_files)} articles")

        for json_path in json_files:
            convert_file(json_path, out_category_dir)
            total += 1

    print(f"\nDone. Converted {total} articles into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

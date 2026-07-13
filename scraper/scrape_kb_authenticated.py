"""
Authenticated scraper for gated TeamDynamix content (e.g. Internal-Documentation).

Uses cookies exported from a logged-in admin browser session (scraper/cookies.txt,
Netscape format) to access category/article pages that anonymous requests can't see.

Saves output separately from the public scrape, into scraper/output-internal/,
so it can be reviewed before merging into the main knowledge-base/.

Run from the project root:
    uv run scraper/scrape_kb_authenticated.py
"""

import http.cookiejar
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE_URL = "https://newpaltz.teamdynamix.com"
COOKIE_FILE = Path(__file__).parent / "cookies.txt"
OUTPUT_DIR = Path(__file__).parent / "output-internal"
REQUEST_DELAY = 0.6

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Just the gated category/categories we couldn't see anonymously.
# Add more URLs here if other categories turn out to be gated too.
GATED_CATEGORIES = {
    "Internal-Documentation": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/4039/Internal-Documentation",
}

ARTICLE_ID_RE = re.compile(r"/KB/Article/(\d+)/")


def load_session() -> requests.Session:
    if not COOKIE_FILE.exists():
        raise FileNotFoundError(
            f"Cookie file not found at {COOKIE_FILE}. Export cookies from your logged-in "
            "browser session first (see project notes)."
        )

    jar = http.cookiejar.MozillaCookieJar(str(COOKIE_FILE))
    jar.load(ignore_discard=True, ignore_expires=True)

    session = requests.Session()
    session.cookies = jar
    session.headers.update(HEADERS)
    return session


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(response.text, "html.parser")


def parse_category(session: requests.Session, url: str):
    soup = fetch_soup(session, url)
    container = soup.find("div", id="divCats")
    if container is None:
        return [], []

    subcategory_urls = []
    for box in container.select(".category-box"):
        link = box.select_one("h3.category-title a")
        if link and link.get("href"):
            subcategory_urls.append(BASE_URL + link["href"])

    article_urls = []
    for item in container.select(".gutter-bottom-lg"):
        link = item.select_one("h3.gutter-bottom-xs a")
        if link and link.get("href"):
            article_urls.append(BASE_URL + link["href"])

    return subcategory_urls, article_urls


def crawl_category_tree(session: requests.Session, start_url: str):
    visited_categories = set()
    all_article_urls = set()
    queue = [start_url]

    while queue:
        url = queue.pop()
        if url in visited_categories:
            continue
        visited_categories.add(url)

        print(f"  Crawling category: {url}")
        subcats, articles = parse_category(session, url)
        queue.extend(subcats)
        all_article_urls.update(articles)

    return all_article_urls


def parse_article(session: requests.Session, url: str) -> dict:
    soup = fetch_soup(session, url)

    main = soup.find("div", id="divMainContent")
    title_tag = main.find("h1") if main else None
    title = title_tag.get_text(strip=True) if title_tag else None

    breadcrumb = soup.select("ol.breadcrumb.pull-left li")
    category_path = [li.get_text(strip=True) for li in breadcrumb[1:-1]] if len(breadcrumb) > 2 else []

    tags_container = soup.find("div", id=lambda x: x and x.endswith("divTags"))
    tags = [a.get_text(strip=True) for a in tags_container.find_all("a")] if tags_container else []

    published = soup.find("meta", attrs={"property": "article:published_time"})
    modified = soup.find("meta", attrs={"property": "article:modified_time"})
    published_date = published["content"] if published else None
    modified_date = modified["content"] if modified else None

    body_container = soup.find("div", id=lambda x: x and x.endswith("divBody"))
    body_markdown = md(str(body_container), heading_style="ATX").strip() if body_container else ""

    match = ARTICLE_ID_RE.search(url)
    article_id = match.group(1) if match else None

    return {
        "article_id": article_id,
        "url": url,
        "title": title,
        "category_path": category_path,
        "tags": tags,
        "published_date": published_date,
        "modified_date": modified_date,
        "body": body_markdown,
        "gated": True,
    }


def safe_filename(title: str, article_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\- ]", "", title or "untitled").strip().replace(" ", "-")
    return f"{slug[:80]}-{article_id}.json"


def main():
    session = load_session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_article_urls = set()

    for category_name, category_url in GATED_CATEGORIES.items():
        print(f"\n=== Gated category: {category_name} ===")
        article_urls = crawl_category_tree(session, category_url)
        print(f"  Found {len(article_urls)} articles")

        category_dir = OUTPUT_DIR / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        for article_url in article_urls:
            if article_url in seen_article_urls:
                continue
            seen_article_urls.add(article_url)

            try:
                article = parse_article(session, article_url)
            except Exception as e:
                print(f"  FAILED: {article_url} ({e})")
                continue

            filename = safe_filename(article["title"], article["article_id"])
            filepath = category_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {filepath.name}")

    print(f"\nDone. Total gated articles scraped: {len(seen_article_urls)}")


if __name__ == "__main__":
    main()

"""
Scraper for the SUNY New Paltz TeamDynamix IT Knowledge Base.

Crawls each top-level category recursively, finds all articles, and saves
each article as a JSON file (with a Markdown body) organized into folders
by top-level category -- e.g. scraper/output/Hardware/IT-Loaner-Equipment.json

Run from the project root:
    uv run scraper/scrape_kb.py
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE_URL = "https://newpaltz.teamdynamix.com"
OUTPUT_DIR = Path(__file__).parent / "output"
REQUEST_DELAY = 0.6  # seconds between requests, be polite to their server

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

TOP_LEVEL_CATEGORIES = {
    "Internal-Documentation": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/4039/Internal-Documentation",
    "Getting-Started-Guides": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/9889/Getting-Started-Guides",
    "Accounts-Access-Security": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/3921/Accounts-Access-Security",
    "Software-and-Apps": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/22048/Software-and-Apps",
    "Hardware": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/22049/Hardware",
    "Networking-WiFi": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/3925/Networking-WiFi",
    "Digital-Accessibility": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/10857/Digital-Accessibility",
    "Policies": "https://newpaltz.teamdynamix.com/TDClient/1905/Portal/KB/Category/4271/Policies",
}

ARTICLE_ID_RE = re.compile(r"/KB/Article/(\d+)/")


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(response.text, "html.parser")


def parse_category(url: str):
    """Return (list of subcategory urls, list of article urls) found directly in this category."""
    soup = fetch_soup(url)
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


def crawl_category_tree(start_url: str):
    """Recursively crawl a category and all its subcategories, returning all unique article URLs found."""
    visited_categories = set()
    all_article_urls = set()
    queue = [start_url]

    while queue:
        url = queue.pop()
        if url in visited_categories:
            continue
        visited_categories.add(url)

        print(f"  Crawling category: {url}")
        subcats, articles = parse_category(url)
        queue.extend(subcats)
        all_article_urls.update(articles)

    return all_article_urls


def parse_article(url: str) -> dict:
    soup = fetch_soup(url)

    main = soup.find("div", id="divMainContent")
    title_tag = main.find("h1") if main else None
    title = title_tag.get_text(strip=True) if title_tag else None

    # Breadcrumb -> category path (excluding "Knowledge Base" root and the article itself)
    breadcrumb = soup.select("ol.breadcrumb.pull-left li")
    category_path = [li.get_text(strip=True) for li in breadcrumb[1:-1]] if len(breadcrumb) > 2 else []

    # Tags
    tags_container = soup.find("div", id=lambda x: x and x.endswith("divTags"))
    tags = []
    if tags_container:
        tags = [a.get_text(strip=True) for a in tags_container.find_all("a")]

    # Dates
    published = soup.find("meta", attrs={"property": "article:published_time"})
    modified = soup.find("meta", attrs={"property": "article:modified_time"})
    published_date = published["content"] if published else None
    modified_date = modified["content"] if modified else None

    # Body -> Markdown
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
    }


def safe_filename(title: str, article_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\- ]", "", title or "untitled").strip().replace(" ", "-")
    return f"{slug[:80]}-{article_id}.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_article_urls = set()

    for category_name, category_url in TOP_LEVEL_CATEGORIES.items():
        print(f"\n=== Top-level category: {category_name} ===")
        article_urls = crawl_category_tree(category_url)
        print(f"  Found {len(article_urls)} articles")

        category_dir = OUTPUT_DIR / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        for article_url in article_urls:
            if article_url in seen_article_urls:
                continue
            seen_article_urls.add(article_url)

            try:
                article = parse_article(article_url)
            except Exception as e:
                print(f"  FAILED: {article_url} ({e})")
                continue

            filename = safe_filename(article["title"], article["article_id"])
            filepath = category_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {filepath.name}")

    print(f"\nDone. Total unique articles scraped: {len(seen_article_urls)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scrape paginated job ads from rejobs.org and export to CSV/JSON.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_URL = "https://rejobs.org/en/renewable-energy-jobs"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Rejobs paginated listings.")
    parser.add_argument("--base-url", default=DEFAULT_URL, help="Rejobs listing URL.")
    parser.add_argument("--start-page", type=int, default=1, help="First page to scrape.")
    parser.add_argument("--max-pages", type=int, default=300, help="Hard page cap.")
    parser.add_argument("--delay-seconds", type=float, default=0.7, help="Delay per page.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per page.")
    parser.add_argument("--timeout", type=int, default=40, help="Request timeout in seconds.")
    parser.add_argument(
        "--out-csv",
        default="/Users/HP/Documents/scrapping/scrapped_jobs/rejobs_jobs.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--out-json",
        default="/Users/HP/Documents/scrapping/scrapped_jobs/rejobs_jobs.json",
        help="JSON output path.",
    )
    return parser.parse_args()


def fetch_page(session: requests.Session, url: str, page: int, timeout: int, retries: int) -> Optional[str]:
    params = {"page": page} if page > 1 else {}
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            # Rejobs uses 410 Gone for out-of-range pagination on some queries.
            if r.status_code in (404, 410):
                return None
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(5, attempt * 1.2))
                continue
            raise RuntimeError(f"Failed page {page}: {last_err}") from last_err
    return None


def extract_jobs(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("li.relative.overflow-hidden.white-box")
    rows: List[Dict[str, str]] = []

    for li in cards:
        title_el = li.select_one("a.rejobs-link[href*='/renewable-energy-jobs/']")
        employer_el = li.select_one("a.rejobs-employer-link")
        location_el = li.select_one("a.rejobs-regular-link[href*='/location/']")

        # Keep only job cards.
        if not (title_el and employer_el):
            continue

        tags = []
        for a in li.select("a.hover\\:underline[href*='/renewable-energy-jobs/']"):
            txt = a.get_text(" ", strip=True)
            href = a.get("href", "")
            if "/location/" in href or "/employer/" in href:
                continue
            if txt:
                tags.append(txt)

        job_href = title_el.get("href", "").strip()
        rows.append(
            {
                "title": title_el.get_text(" ", strip=True),
                "description": title_el.get("title", "").strip(),
                "employer": employer_el.get_text(" ", strip=True),
                "location": location_el.get_text(" ", strip=True) if location_el else "",
                "job_url": urljoin("https://rejobs.org", job_href),
                "tags": " | ".join(dict.fromkeys(tags)),
                "featured": "featured-job-ad" in (li.get("class") or []),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    all_rows: List[Dict[str, str]] = []
    seen_urls = set()
    consecutive_empty = 0

    for page in range(args.start_page, args.start_page + args.max_pages):
        html = fetch_page(session, args.base_url, page, args.timeout, args.retries)
        if html is None:
            print(f"Page {page}: 404/end reached.")
            break

        page_rows = extract_jobs(html)
        new_count = 0
        for row in page_rows:
            url = row["job_url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            all_rows.append(row)
            new_count += 1

        print(f"Page {page}: found={len(page_rows)} new={new_count} total={len(all_rows)}")

        if new_count == 0:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        # Stop if two pages in a row add nothing.
        if consecutive_empty >= 2:
            print("No new jobs across two consecutive pages. Stopping.")
            break

        time.sleep(args.delay_seconds)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["job_url"]).reset_index(drop=True)
    print(f"Final unique jobs: {len(df)}")

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", force_ascii=False, indent=2)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()

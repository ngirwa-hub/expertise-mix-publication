#!/usr/bin/env python3
"""
Paginated scraper for energyjobsearch.com jobs.

Approach:
- Request SSR page /jobs with query parameters and page index.
- Parse embedded JSON from <script id="__NEXT_DATA__">.
- Read jobs from pageProps.jobsModel.items.
- Continue until reported total pages or until empty page.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_BASE_URL = "https://energyjobsearch.com/jobs"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


class PageNotFoundError(Exception):
    """Raised when requested pagination page does not exist (HTTP 404)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Energy JobSearch paginated jobs.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Jobs search URL.")
    parser.add_argument(
        "--category",
        default="ENGINEERING,GENERAL_MANAGEMENT,MAINTENANCE_MANAGER,PLANT_MANAGER,TECHNICIANS_SERVICE,CONSULTING,RISK_MANAGEMENT,OTHER",
        help="Comma-separated category values.",
    )
    parser.add_argument(
        "--seniority",
        default="SENIOR,MID_LEVEL",
        help="Comma-separated seniority values.",
    )
    parser.add_argument("--title", default="", help="Title keyword.")
    parser.add_argument("--limit", type=int, default=100, help="Rows per page.")
    parser.add_argument("--start-page", type=int, default=0, help="First page index.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=250,
        help="Hard stop for pages to request.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.8,
        help="Delay between page requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per failed page request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="HTTP request timeout seconds.",
    )
    parser.add_argument(
        "--out-csv",
        default="/Users/HP/Documents/scrapping/scrapped_jobs/energyjobsearch_jobs.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--out-json",
        default="/Users/HP/Documents/scrapping/scrapped_jobs/energyjobsearch_jobs.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--debug-html-dir",
        default="",
        help="Optional directory for saving raw HTML per page.",
    )
    return parser.parse_args()


def fetch_html_with_retries(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int,
    retries: int,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 404:
                raise PageNotFoundError(f"Page not found for params: {params}")
            response.raise_for_status()
            return response.text
        except PageNotFoundError:
            raise
        except requests.RequestException as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(5, attempt * 1.5))
                continue
            break
    raise RuntimeError(f"Failed to fetch page after {retries} retries: {last_err}") from last_err


def parse_next_data(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.select_one("script#__NEXT_DATA__")
    if not script:
        raise ValueError("Could not find __NEXT_DATA__ script payload in HTML.")
    script_text = script.get_text()
    if not script_text.strip():
        raise ValueError("__NEXT_DATA__ script is empty.")
    return json.loads(script_text)


def safe_get(dct: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = dct
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def normalize_job(item: Dict[str, Any]) -> Dict[str, Any]:
    location = item.get("location") or {}
    company = item.get("company") or {}
    partner = item.get("partner") or {}
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "short_description": item.get("shortDescription"),
        "category": item.get("category"),
        "seniority": item.get("seniority"),
        "occupation_type": item.get("occupationType"),
        "remote": item.get("remote"),
        "published_at": item.get("publishedAt"),
        "created_at": item.get("createdAt"),
        "expires_at": item.get("expiresAt"),
        "city": location.get("city"),
        "region": location.get("region"),
        "country": location.get("country"),
        "company_name": company.get("name"),
        "partner_name": partner.get("name"),
        "external_apply_url": item.get("externalApplyUrl"),
        "url": f"https://energyjobsearch.com/jobs/{item.get('category', '').lower().replace('_', '-')}/{item.get('id')}" if item.get("id") else None,
    }


def extract_page_data(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    page_props = safe_get(payload, ["props", "pageProps"], default={})
    jobs_model = page_props.get("jobsModel") or page_props.get("jobs") or {}
    items = jobs_model.get("items") or []
    pages = jobs_model.get("pages")
    return items, pages


def run_scrape(args: argparse.Namespace) -> pd.DataFrame:
    out_html_dir: Optional[Path] = None
    if args.debug_html_dir:
        out_html_dir = Path(args.debug_html_dir)
        out_html_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "category": args.category,
        "seniority": args.seniority,
        "title": args.title,
        "limit": args.limit,
    }

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    all_rows: List[Dict[str, Any]] = []
    seen_ids: set = set()
    reported_total_pages: Optional[int] = None

    for page in range(args.start_page, args.start_page + args.max_pages):
        params["page"] = page
        try:
            html = fetch_html_with_retries(
                session=session,
                url=args.base_url,
                params=params,
                timeout=args.timeout,
                retries=args.retries,
            )
        except PageNotFoundError:
            print(
                f"Page {page}: HTTP 404 reached. "
                "Treating this as end of available pages for this route."
            )
            break

        if out_html_dir:
            (out_html_dir / f"page_{page}.html").write_text(html, encoding="utf-8")

        payload = parse_next_data(html)
        items, pages = extract_page_data(payload)

        if pages is not None:
            reported_total_pages = pages

        if not items:
            print(f"Page {page}: no jobs found, stopping.")
            break

        page_new = 0
        for item in items:
            job_id = item.get("id")
            dedupe_key = job_id if job_id is not None else item.get("externalApplyUrl")
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            all_rows.append(normalize_job(item))
            page_new += 1

        print(
            f"Page {page}: fetched={len(items)} new={page_new} total_unique={len(all_rows)} "
            f"(reported_pages={reported_total_pages})"
        )

        if page_new == 0:
            print(f"Page {page}: no new jobs after dedupe, stopping.")
            break

        if reported_total_pages is not None and page >= reported_total_pages - 1:
            print(f"Reached reported last page ({reported_total_pages - 1}).")
            break

        time.sleep(args.delay_seconds)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"], keep="first")
    return df


def save_outputs(df: pd.DataFrame, out_csv: str, out_json: str) -> None:
    csv_path = Path(out_csv)
    json_path = Path(out_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def main() -> None:
    args = parse_args()
    df = run_scrape(args)
    print(f"Final unique jobs: {len(df)}")
    if not df.empty:
        print(df[["id", "title", "company_name", "city", "country"]].head(10).to_string(index=False))
    save_outputs(df, args.out_csv, args.out_json)


if __name__ == "__main__":
    main()

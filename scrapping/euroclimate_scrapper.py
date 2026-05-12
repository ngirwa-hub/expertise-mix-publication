#!/usr/bin/env python3
"""
Robust scraper for EuroClimateJobs listings.

Features:
- Retries + timeout + user-agent headers
- Parses real job cards only (skips alert/subscribe blocks)
- Dedupe by job_id (fallback title+company+location)
- Pagination support:
  1) follow explicit next link if present
  2) fallback to ?page=N probing, stop when no new jobs
- Optional detail-page summary backfill with 429-safe behavior
- CSV + JSON output
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_URL = (
    "https://www.euroclimatejobs.com/job_search/"
    "category/bioenergy_and_biofuel/"
    "category/carbon_capture_and_storage/"
    "category/climate_and_energy_analyst/"
    "category/consultant/category/electricity/"
    "category/energy_engineer/"
    "category/geothermal_energy/"
    "category/government_and_associations/"
    "category/hydropower_energy/"
    "category/oil_and_gas/"
    "category/policy_and_regulation/"
    "category/production_and_operations/"
    "category/project_manager/"
    "category/renewable_energy/"
    "category/research_and_development/"
    "category/solar_energy/"
    "category/sustainable_transport/"
    "category/wave_and_tidal_energy/"
    "category/wind_energy/experience/"
    "3-4_years_experience/experience/5+_years_experience/"
    "experience/manager_and_executive"

)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape EuroClimateJobs listings.")
    p.add_argument("--url", default=DEFAULT_URL, help="Search URL.")
    p.add_argument("--max-pages", type=int, default=60, help="Hard pagination cap.")
    p.add_argument("--delay-seconds", type=float, default=0.8, help="Delay between listing requests.")
    p.add_argument("--timeout", type=int, default=40, help="HTTP timeout.")
    p.add_argument("--retries", type=int, default=3, help="Retries per request.")
    p.add_argument(
        "--out-csv",
        default="/Users/HP/Documents/scrapping/scrapped_jobs/euroclimatejobs_jobs.csv",
        help="CSV output path.",
    )
    p.add_argument(
        "--out-json",
        default="/Users/HP/Documents/scrapping/scrapped_jobs/euroclimatejobs_jobs.json",
        help="JSON output path.",
    )
    p.add_argument(
        "--debug-html-dir",
        default="",
        help="Optional folder to save fetched HTML pages.",
    )
    p.add_argument(
        "--no-backfill-summary",
        action="store_true",
        help="Disable detail-page backfill for missing summaries.",
    )
    p.add_argument(
        "--detail-delay-seconds",
        type=float,
        default=0.5,
        help="Delay between detail-page requests when backfilling summaries.",
    )
    return p.parse_args()


def request_with_retries(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
    fail_silently: bool = False,
) -> Optional[str]:
    last_err: Optional[Exception] = None
    rate_limited = 0
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (404, 410):
                return None
            if r.status_code == 429:
                rate_limited += 1
                retry_after = r.headers.get("Retry-After")
                try:
                    retry_after_s = float(retry_after) if retry_after is not None else 0
                except ValueError:
                    retry_after_s = 0
                wait_s = max(retry_after_s, min(20, 3 * attempt))
                print(f"429 Too Many Requests for {url}. Waiting {wait_s:.1f}s before retry.")
                if fail_silently and rate_limited >= 2:
                    return None
                if attempt < retries:
                    time.sleep(wait_s)
                    continue
                if fail_silently:
                    return None
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(8, attempt * 1.6))
                continue
            if fail_silently:
                return None
            raise RuntimeError(f"Failed to fetch {url}: {last_err}") from last_err
    return None


def make_page_url(base_url: str, page: int) -> str:
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["page"] = [str(page)]
    new_query = urlencode(query, doseq=True)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def extract_jobs(html: str, page_url: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

    for li in soup.select("ul.searchList > li"):
        title_el = li.select_one("h3 a")
        company_el = li.select_one(".companyName")
        location_el = li.select_one(".location")
        if not (title_el and company_el and location_el):
            continue

        posted_el = li.select_one(".postedDate")
        save_el = li.select_one("a.jobSave[data-job-id]")
        tags = [t.get_text(" ", strip=True) for t in li.select("span.badge.bg-job-tag")]

        summary = ""
        for p in li.select("p"):
            txt = p.get_text(" ", strip=True)
            if txt and "Email me jobs like this" not in txt and txt != "Subscribe":
                summary = txt
                break

        href = title_el.get("href", "").strip()
        rows.append(
            {
                "job_id": save_el.get("data-job-id", "") if save_el else "",
                "title": title_el.get_text(" ", strip=True),
                "company": company_el.get_text(" ", strip=True),
                "location": location_el.get_text(" ", strip=True),
                "posted": posted_el.get_text(" ", strip=True) if posted_el else "",
                "summary": summary,
                "job_url": urljoin(base, href) if href else "",
                "tags": " | ".join(tags),
                "source": "euroclimatejobs",
            }
        )

    next_href = None
    next_link = soup.select_one("a[rel='next']")
    if next_link and next_link.get("href"):
        next_href = urljoin(base, next_link["href"])
    else:
        for a in soup.select("a[href]"):
            txt = a.get_text(" ", strip=True).lower()
            if txt in {"next", "next page", "older"}:
                next_href = urljoin(base, a["href"])
                break

    return rows, next_href


def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        key = (
            r.get("job_id", "").strip(),
            r.get("title", "").strip().lower(),
            r.get("company", "").strip().lower(),
            r.get("location", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def extract_detail_summary(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    def normalize(txt: str) -> str:
        return " ".join(txt.split()).strip()

    containers = []
    selectors = [
        ".job-description",
        ".jobDescription",
        "#jobDescription",
        ".job-details",
        ".jobDetails",
        ".description",
        ".content",
        "main",
    ]
    for sel in selectors:
        containers.extend(soup.select(sel))

    best = ""
    for c in containers:
        parts = []
        for node in c.select("p, li"):
            txt = normalize(node.get_text(" ", strip=True))
            if not txt:
                continue
            lower = txt.lower()
            if "save this job" in lower or "email me jobs like this" in lower:
                continue
            parts.append(txt)
        candidate = normalize(" ".join(parts))
        if len(candidate) > len(best):
            best = candidate

    if not best:
        paras = []
        for p in soup.select("p"):
            txt = normalize(p.get_text(" ", strip=True))
            if len(txt) < 80:
                continue
            lower = txt.lower()
            if any(
                x in lower
                for x in [
                    "cookie",
                    "privacy policy",
                    "save this job",
                    "email me jobs like this",
                    "subscribe",
                ]
            ):
                continue
            paras.append(txt)
        if paras:
            best = max(paras, key=len)

    return best


def backfill_missing_summaries(
    rows: List[Dict[str, str]],
    session: requests.Session,
    timeout: int,
    retries: int,
    delay_seconds: float,
) -> int:
    updated = 0
    failed = 0
    consecutive_misses = 0
    targets = [r for r in rows if not str(r.get("summary", "")).strip() and str(r.get("job_url", "")).strip()]
    if not targets:
        return updated

    print(f"Backfill: trying to fill summaries for {len(targets)} jobs from detail pages...")
    for i, row in enumerate(targets, 1):
        detail_url = row["job_url"]
        html = request_with_retries(
            session=session,
            url=detail_url,
            timeout=timeout,
            retries=retries,
            fail_silently=True,
        )
        if not html:
            failed += 1
            consecutive_misses += 1
            if consecutive_misses >= 25:
                print(
                    "Backfill stopped: too many consecutive misses (likely rate-limited). "
                    "Run later or increase detail delay."
                )
                break
            continue

        consecutive_misses = 0
        detail_summary = extract_detail_summary(html)
        if detail_summary:
            row["summary"] = detail_summary
            updated += 1

        if i % 25 == 0 or i == len(targets):
            print(f"Backfill progress: {i}/{len(targets)} checked, {updated} updated, {failed} failed")
        time.sleep(delay_seconds)
    return updated


def run() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    debug_dir: Optional[Path] = None
    if args.debug_html_dir:
        debug_dir = Path(args.debug_html_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    visited_urls: Set[str] = set()
    url = args.url
    used_explicit_next = False

    for idx in range(1, args.max_pages + 1):
        if url in visited_urls:
            print(f"Page {idx}: detected loop in pagination URL, stopping.")
            break
        visited_urls.add(url)

        html = request_with_retries(session, url, args.timeout, args.retries)
        if html is None:
            print(f"Page {idx}: reached 404/410 at {url}, stopping.")
            break

        if debug_dir:
            (debug_dir / f"page_{idx}.html").write_text(html, encoding="utf-8")

        page_rows, next_url = extract_jobs(html, url)
        before = len(all_rows)
        all_rows.extend(page_rows)
        all_rows = dedupe_rows(all_rows)
        added = len(all_rows) - before
        print(f"Page {idx}: parsed={len(page_rows)} new={added} total={len(all_rows)}")

        if added == 0 and idx > 1:
            print("No new jobs found on current page, stopping.")
            break

        if next_url:
            url = next_url
            used_explicit_next = True
        elif used_explicit_next:
            print("No explicit next link found, stopping.")
            break
        else:
            url = make_page_url(args.url, idx + 1)

        time.sleep(args.delay_seconds)

    if not args.no_backfill_summary and all_rows:
        updated = backfill_missing_summaries(
            rows=all_rows,
            session=session,
            timeout=args.timeout,
            retries=args.retries,
            delay_seconds=args.detail_delay_seconds,
        )
        print(f"Backfill completed: {updated} summaries filled from detail pages.")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["job_id", "title", "company", "location"], keep="first")

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2, force_ascii=False)

    print(f"Final jobs: {len(df)}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    run()

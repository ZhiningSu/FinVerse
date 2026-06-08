"""Collect stable US corporate event data from SEC EDGAR.

This script builds the news/event layer for HMSC using official SEC filings.
It focuses on reproducible high-signal events:

  - IPO / offering related filings: S-1, F-1, 424B*, S-3
  - Major event reports: 8-K, 6-K
  - Earnings / fundamentals: 10-K, 10-Q, 20-F
  - Governance / ownership: DEF 14A, SC 13D/G

Outputs:
  - data/raw/sec/company_tickers.json
  - data/raw/sec/submissions/{TICKER}.json
  - data/processed/sec_events.jsonl
  - data/processed/sec_events.csv
  - data/processed/sec_event_daily_features.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = PROJECT_ROOT / "data" / "tickers" / "hmsc_us_90.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sec"
SUBMISSIONS_DIR = RAW_DIR / "submissions"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# SEC asks automated clients to identify themselves. Replace the email if you
# have a project-specific contact.
HEADERS = {
    "User-Agent": "FinWorldModel HMSC research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
TICKER_HEADERS = {
    "User-Agent": "FinWorldModel HMSC research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

IMPORTANT_FORMS = {
    "S-1", "S-1/A", "F-1", "F-1/A", "424B1", "424B2", "424B3", "424B4", "424B5",
    "S-3", "S-3/A", "F-3", "F-3/A",
    "8-K", "8-K/A", "6-K",
    "10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A",
    "DEF 14A", "DEFA14A", "PRE 14A",
    "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
}

EVENT_WEIGHTS = {
    "ipo_or_offering": 3.0,
    "major_event": 2.0,
    "earnings_or_fundamentals": 1.5,
    "governance_or_ownership": 1.0,
    "other_important": 0.5,
}


def load_universe(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_company_ticker_map(force_refresh: bool = False) -> dict[str, dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RAW_DIR / "company_tickers.json"
    if out_file.exists() and not force_refresh:
        payload = json.loads(out_file.read_text(encoding="utf-8"))
    else:
        LOG.info("Fetching SEC company ticker map")
        resp = requests.get(COMPANY_TICKERS_URL, headers=TICKER_HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ticker_map = {}
    for item in payload.values():
        ticker = str(item.get("ticker", "")).upper()
        if not ticker:
            continue
        cik = str(item.get("cik_str", "")).zfill(10)
        ticker_map[ticker] = {
            "cik": cik,
            "title": item.get("title", ""),
        }
    return ticker_map


def fetch_submission(ticker: str, cik: str, force_refresh: bool = False) -> dict | None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SUBMISSIONS_DIR / f"{ticker}.json"
    if out_file.exists() and not force_refresh:
        return json.loads(out_file.read_text(encoding="utf-8"))

    url = SUBMISSIONS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            LOG.warning("[%s] SEC submissions HTTP %d", ticker, resp.status_code)
            return None
        payload = resp.json()
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    except Exception as exc:
        LOG.warning("[%s] SEC submissions failed: %s", ticker, exc)
        return None


def event_category(form: str) -> str:
    if form in {"S-1", "S-1/A", "F-1", "F-1/A", "424B1", "424B2", "424B3", "424B4", "424B5"}:
        return "ipo_or_offering"
    if form in {"S-3", "S-3/A", "F-3", "F-3/A"}:
        return "ipo_or_offering"
    if form in {"8-K", "8-K/A", "6-K"}:
        return "major_event"
    if form in {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"}:
        return "earnings_or_fundamentals"
    if form in {"DEF 14A", "DEFA14A", "PRE 14A", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}:
        return "governance_or_ownership"
    return "other_important"


def parse_recent_filings(
    ticker: str,
    cik: str,
    company_name: str,
    sector: str,
    submission: dict,
    start: str,
    end: str,
) -> list[dict]:
    recent = submission.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    items = recent.get("items", [])

    events = []
    for idx, form in enumerate(forms):
        if form not in IMPORTANT_FORMS:
            continue
        filing_date = filing_dates[idx] if idx < len(filing_dates) else ""
        if not filing_date or filing_date < start or filing_date > end:
            continue

        category = event_category(form)
        accession = accession_numbers[idx] if idx < len(accession_numbers) else ""
        primary_doc = primary_documents[idx] if idx < len(primary_documents) else ""
        report_date = report_dates[idx] if idx < len(report_dates) else ""
        item_text = items[idx] if idx < len(items) else ""
        accession_compact = accession.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_compact}/{primary_doc}"
            if accession and primary_doc else ""
        )

        events.append({
            "date": filing_date,
            "ticker": ticker,
            "cik": cik,
            "company_name": company_name,
            "sector": sector,
            "source": "sec_edgar",
            "form": form,
            "event_category": category,
            "event_weight": EVENT_WEIGHTS[category],
            "report_date": report_date,
            "accession_number": accession,
            "primary_document": primary_doc,
            "items": item_text,
            "url": filing_url,
            "headline": f"{ticker} filed {form}",
        })
    return events


def build_daily_features(events: list[dict], start: str, end: str) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    rows = []
    by_key = defaultdict(list)
    for event in events:
        by_key[(event["ticker"], event["date"])].append(event)

    for (ticker, date), items in sorted(by_key.items()):
        dt = pd.to_datetime(date)
        if dt < start_dt or dt > end_dt:
            continue
        counts = defaultdict(int)
        weighted = 0.0
        forms = []
        headlines = []
        for item in items:
            counts[item["event_category"]] += 1
            weighted += float(item["event_weight"])
            forms.append(item["form"])
            headlines.append(item["headline"])

        rows.append({
            "date": date,
            "ticker": ticker,
            "sec_event_count": len(items),
            "sec_event_intensity": weighted,
            "ipo_or_offering_count": counts["ipo_or_offering"],
            "major_event_count": counts["major_event"],
            "earnings_or_fundamentals_count": counts["earnings_or_fundamentals"],
            "governance_or_ownership_count": counts["governance_or_ownership"],
            "forms": "|".join(sorted(set(forms))),
            "headlines": " || ".join(headlines[:5]),
        })
    return pd.DataFrame(rows)


def save_events(events: list[dict], start: str, end: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_file = PROCESSED_DIR / "sec_events.jsonl"
    csv_file = PROCESSED_DIR / "sec_events.csv"
    daily_file = PROCESSED_DIR / "sec_event_daily_features.csv"

    with jsonl_file.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(["date", "ticker", "form"])
    df.to_csv(csv_file, index=False)

    daily = build_daily_features(events, start, end)
    if not daily.empty:
        daily = daily.sort_values(["date", "ticker"])
    daily.to_csv(daily_file, index=False)

    LOG.info("Saved events jsonl: %s (%d rows)", jsonl_file, len(events))
    LOG.info("Saved events csv: %s", csv_file)
    LOG.info("Saved daily features: %s (%d rows)", daily_file, len(daily))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SEC EDGAR event/news layer for HMSC.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    universe = load_universe(args.universe)
    if args.limit:
        universe = universe[:args.limit]
    ticker_map = load_company_ticker_map(force_refresh=args.force_refresh)

    all_events = []
    report = {"success": [], "missing_cik": [], "failed": [], "event_counts": {}}
    for idx, row in enumerate(universe, start=1):
        ticker = row["ticker"].upper().strip()
        if row.get("type") == "etf":
            LOG.info("[%d/%d] Skipping ETF %s for SEC company events", idx, len(universe), ticker)
            continue
        meta = ticker_map.get(ticker)
        if not meta:
            LOG.warning("[%s] Missing CIK", ticker)
            report["missing_cik"].append(ticker)
            continue

        LOG.info("[%d/%d] Fetching SEC submissions %s", idx, len(universe), ticker)
        submission = fetch_submission(ticker, meta["cik"], force_refresh=args.force_refresh)
        if not submission:
            report["failed"].append(ticker)
            time.sleep(args.sleep)
            continue

        events = parse_recent_filings(
            ticker=ticker,
            cik=meta["cik"],
            company_name=row.get("name") or meta.get("title", ""),
            sector=row.get("sector", ""),
            submission=submission,
            start=args.start,
            end=args.end,
        )
        all_events.extend(events)
        report["success"].append(ticker)
        report["event_counts"][ticker] = len(events)
        LOG.info("[%s] events=%d", ticker, len(events))
        time.sleep(args.sleep)

    save_events(all_events, args.start, args.end)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report["total_events"] = len(all_events)
    (RAW_DIR / "sec_event_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("SEC event collection done. total_events=%d", len(all_events))


if __name__ == "__main__":
    main()

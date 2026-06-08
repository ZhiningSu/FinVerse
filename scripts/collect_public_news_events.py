"""Collect public financial news/events for HMSC.

Stable no-key sources used here:
  1. Nasdaq IPO calendar API for IPO and offering events.
  2. Yahoo Finance RSS for company-level news headlines.

The script normalizes both sources into:
  - data/raw/news/nasdaq_ipo_calendar/*.json
  - data/raw/news/yahoo_rss/*.xml
  - data/processed/public_financial_events.jsonl
  - data/processed/public_financial_events.csv
  - data/processed/public_event_daily_features.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = PROJECT_ROOT / "data" / "tickers" / "hmsc_us_90.csv"
RAW_NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NASDAQ_IPO_URL = "https://api.nasdaq.com/api/ipo/calendar"
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/xml,application/xml,*/*",
}


def load_universe(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def month_iter(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield f"{year}-{month:02d}"


def fetch_nasdaq_ipo_month(month: str) -> dict | None:
    try:
        resp = requests.get(
            NASDAQ_IPO_URL,
            params={"date": month},
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code != 200:
            LOG.warning("Nasdaq IPO %s HTTP %d", month, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:
        LOG.warning("Nasdaq IPO %s failed: %s", month, exc)
        return None


def _parse_money(value) -> float | None:
    if value is None:
        return None
    text = str(value).replace("$", "").replace(",", "").strip()
    if not text or text in {"N/A", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_date(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return text[:10] if len(text) >= 10 else text


def normalize_ipo_row(row: dict, section: str, month: str) -> dict:
    symbol = (
        row.get("proposedTickerSymbol")
        or row.get("ticker")
        or row.get("symbol")
        or row.get("proposedTicker")
        or ""
    )
    company = row.get("companyName") or row.get("name") or row.get("company") or ""
    if section == "withdrawn":
        raw_date = row.get("withdrawDate") or row.get("filedDate") or row.get("date") or ""
    elif section == "filed":
        raw_date = row.get("filedDate") or row.get("date") or ""
    elif section == "upcoming":
        raw_date = row.get("expectedPriceDate") or row.get("pricedDate") or row.get("date") or ""
    else:
        raw_date = row.get("pricedDate") or row.get("date") or row.get("offerDate") or ""
    price = row.get("price") or row.get("offerPrice") or row.get("proposedSharePrice") or ""
    shares = row.get("sharesOffered") or row.get("shares") or ""
    amount = row.get("dollarValueOfSharesOffered") or row.get("dealSize") or row.get("proceeds") or ""
    exchange = row.get("exchange") or row.get("proposedExchange") or ""

    date = _parse_date(raw_date)
    if not date or len(date) != 10:
        date = f"{month}-01"

    return {
        "date": date,
        "ticker": str(symbol).upper(),
        "company_name": company,
        "source": "nasdaq_ipo_calendar",
        "event_category": "ipo_or_offering",
        "event_type": section,
        "event_weight": 3.0,
        "headline": f"{company} IPO calendar event ({section})".strip(),
        "price": _parse_money(price),
        "shares_offered": _parse_money(shares),
        "deal_value": _parse_money(amount),
        "exchange": exchange,
        "raw": row,
    }


def collect_ipo_events(start_year: int, end_year: int, sleep: float, use_cache: bool = True) -> list[dict]:
    out_dir = RAW_NEWS_DIR / "nasdaq_ipo_calendar"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = []
    report = {"success": [], "failed": []}

    for month in month_iter(start_year, end_year):
        LOG.info("Fetching Nasdaq IPO calendar %s", month)
        cache_file = out_dir / f"{month}.json"
        if use_cache and cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            payload = fetch_nasdaq_ipo_month(month)
        if not payload:
            report["failed"].append(month)
            time.sleep(sleep)
            continue
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report["success"].append(month)

        data = payload.get("data") or {}
        for section in ["priced", "upcoming", "withdrawn", "filed"]:
            block = data.get(section) or {}
            rows = block.get("rows") or []
            if isinstance(rows, list):
                for row in rows:
                    events.append(normalize_ipo_row(row, section, month))
        time.sleep(sleep)

    (out_dir / "fetch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Nasdaq IPO events collected: %d", len(events))
    return events


def fetch_yahoo_rss(symbol: str) -> str | None:
    try:
        resp = requests.get(
            YAHOO_RSS_URL,
            params={"s": symbol, "region": "US", "lang": "en-US"},
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code != 200:
            LOG.warning("[%s] Yahoo RSS HTTP %d", symbol, resp.status_code)
            return None
        return resp.text
    except Exception as exc:
        LOG.warning("[%s] Yahoo RSS failed: %s", symbol, exc)
        return None


def parse_yahoo_rss(symbol: str, company_name: str, sector: str, xml_text: str) -> list[dict]:
    events = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        LOG.warning("[%s] Yahoo RSS parse failed: %s", symbol, exc)
        return events

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        try:
            dt = parsedate_to_datetime(pub_date).strftime("%Y-%m-%d")
        except Exception:
            dt = ""
        if not title:
            continue
        events.append({
            "date": dt,
            "ticker": symbol,
            "company_name": company_name,
            "sector": sector,
            "source": "yahoo_finance_rss",
            "event_category": "company_news",
            "event_type": "headline",
            "event_weight": 0.8,
            "headline": title,
            "summary": description,
            "url": link,
        })
    return events


def collect_yahoo_rss_events(universe: list[dict], sleep: float, limit: int | None = None) -> list[dict]:
    out_dir = RAW_NEWS_DIR / "yahoo_rss"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = []
    report = {"success": [], "failed": []}
    rows = universe[:limit] if limit else universe

    for idx, row in enumerate(rows, start=1):
        symbol = row["ticker"].upper().strip()
        LOG.info("[%d/%d] Fetching Yahoo RSS %s", idx, len(rows), symbol)
        xml_text = fetch_yahoo_rss(symbol)
        if not xml_text:
            report["failed"].append(symbol)
            time.sleep(sleep)
            continue
        (out_dir / f"{symbol}.xml").write_text(xml_text, encoding="utf-8")
        parsed = parse_yahoo_rss(symbol, row.get("name", ""), row.get("sector", ""), xml_text)
        events.extend(parsed)
        report["success"].append(symbol)
        LOG.info("[%s] Yahoo RSS headlines=%d", symbol, len(parsed))
        time.sleep(sleep)

    (out_dir / "fetch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Yahoo RSS events collected: %d", len(events))
    return events


def build_daily_features(events: list[dict]) -> pd.DataFrame:
    by_key = defaultdict(list)
    for event in events:
        date = event.get("date", "")
        ticker = event.get("ticker", "")
        if not date or not ticker:
            continue
        by_key[(ticker, date)].append(event)

    rows = []
    for (ticker, date), items in sorted(by_key.items()):
        counts = defaultdict(int)
        intensity = 0.0
        headlines = []
        for event in items:
            counts[event.get("event_category", "unknown")] += 1
            intensity += float(event.get("event_weight", 0.0))
            if event.get("headline"):
                headlines.append(event["headline"])
        rows.append({
            "date": date,
            "ticker": ticker,
            "public_event_count": len(items),
            "public_event_intensity": intensity,
            "ipo_or_offering_count": counts["ipo_or_offering"],
            "company_news_count": counts["company_news"],
            "headlines": " || ".join(headlines[:5]),
        })
    return pd.DataFrame(rows)


def save_events(events: list[dict], start: str, end: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_file = PROCESSED_DIR / "public_financial_events.jsonl"
    csv_file = PROCESSED_DIR / "public_financial_events.csv"
    daily_file = PROCESSED_DIR / "public_event_daily_features.csv"

    filtered = [
        event for event in events
        if event.get("date") and start <= str(event["date"]) <= end
    ]

    with jsonl_file.open("w", encoding="utf-8") as f:
        for event in filtered:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    df = pd.DataFrame(filtered)
    if not df.empty:
        df = df.sort_values(["date", "source", "ticker"], na_position="last")
    df.to_csv(csv_file, index=False)

    daily = build_daily_features(filtered)
    if not daily.empty:
        daily = daily.sort_values(["date", "ticker"])
    daily.to_csv(daily_file, index=False)

    LOG.info("Saved public events jsonl: %s (%d rows)", jsonl_file, len(filtered))
    LOG.info("Saved public events csv: %s", csv_file)
    LOG.info("Saved daily features: %s (%d rows)", daily_file, len(daily))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public financial news/events.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--tasks", nargs="+", default=["ipo", "yahoo_rss"], choices=["ipo", "yahoo_rss"])
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--rss-limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    universe = load_universe(args.universe)
    events = []
    if "ipo" in args.tasks:
        events.extend(collect_ipo_events(args.start_year, args.end_year, args.sleep, use_cache=not args.no_cache))
    if "yahoo_rss" in args.tasks:
        events.extend(collect_yahoo_rss_events(universe, args.sleep, limit=args.rss_limit))
    start = f"{args.start_year}-01-01"
    end = f"{args.end_year}-12-31"
    save_events(events, start, end)


if __name__ == "__main__":
    main()

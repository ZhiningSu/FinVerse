"""Collect US data for HMSC experiments.

Sources:
  - Yahoo Finance chart endpoint for OHLCV and market proxies.
  - FRED public CSV endpoint for macro variables.
  - GDELT Doc 2.0 timeline endpoint for public news/event signals.

The script is intentionally modular: use --tasks to collect only the data you
need, and --limit for lightweight smoke tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from io import StringIO
from urllib.parse import quote_plus

import pandas as pd
import requests
import urllib3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = PROJECT_ROOT / "data" / "tickers" / "hmsc_us_90.csv"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

YAHOO_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "T10Y2Y": "treasury_10y_2y_spread",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
    "PAYEMS": "nonfarm_payrolls",
    "INDPRO": "industrial_production",
    "BAA10Y": "baa_10y_spread",
    "BAMLH0A0HYM2": "high_yield_oas",
}

MARKET_PROXIES = {
    "^VIX": "vix",
    "DX-Y.NYB": "dollar_index",
    "CL=F": "wti_crude",
    "GC=F": "gold",
    "^TNX": "treasury_10y_yahoo",
}


def load_universe(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def unix_ts(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def _safe_get(items, idx):
    return items[idx] if isinstance(items, list) and idx < len(items) else None


def fetch_yahoo_symbol(symbol: str, start: str, end: str, retries: int = 3) -> dict | None:
    params = {
        "period1": unix_ts(start),
        "period2": unix_ts(end),
        "interval": "1d",
        "events": "div,split",
    }
    url = YAHOO_URL.format(symbol=quote_plus(symbol))
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                LOG.warning("[%s] Yahoo rate limited, sleeping %ds", symbol, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                LOG.warning("[%s] Yahoo HTTP %d", symbol, resp.status_code)
                time.sleep(1 + attempt)
                continue

            payload = resp.json()
            result = payload.get("chart", {}).get("result")
            if not result:
                LOG.warning("[%s] Yahoo empty result", symbol)
                return None

            item = result[0]
            timestamps = item.get("timestamp", [])
            quote = item.get("indicators", {}).get("quote", [{}])[0] or {}
            adjclose = item.get("indicators", {}).get("adjclose") or []
            adjclose = adjclose[0].get("adjclose", []) if adjclose else []

            rows = []
            for idx, ts in enumerate(timestamps):
                rows.append({
                    "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
                    "open": _safe_get(quote.get("open"), idx),
                    "high": _safe_get(quote.get("high"), idx),
                    "low": _safe_get(quote.get("low"), idx),
                    "close": _safe_get(quote.get("close"), idx),
                    "volume": _safe_get(quote.get("volume"), idx),
                    "adj_close": _safe_get(adjclose, idx),
                })
            rows = [row for row in rows if row["close"] is not None]
            return {"symbol": symbol, "source": "yahoo", "data": rows}
        except Exception as exc:
            LOG.warning("[%s] Yahoo attempt %d failed: %s", symbol, attempt + 1, exc)
            time.sleep(1 + attempt)
    return None


def collect_ohlcv(universe: list[dict], start: str, end: str, sleep: float) -> None:
    out_dir = RAW_ROOT / "yfinance_hmsc"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"success": [], "failed": []}

    for idx, row in enumerate(universe, start=1):
        symbol = row["ticker"].strip()
        LOG.info("[%d/%d] Fetching OHLCV %s", idx, len(universe), symbol)
        result = fetch_yahoo_symbol(symbol, start, end)
        if result and result["data"]:
            result["name"] = row.get("name", "")
            result["sector"] = row.get("sector", "")
            result["type"] = row.get("type", "")
            out_file = out_dir / f"{symbol.replace('^', '').replace('=', '_')}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            LOG.info("[%s] OK: %d rows", symbol, len(result["data"]))
            report["success"].append(symbol)
        else:
            LOG.warning("[%s] FAILED", symbol)
            report["failed"].append(symbol)
        time.sleep(sleep)

    with (out_dir / "fetch_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOG.info("OHLCV success=%d failed=%d", len(report["success"]), len(report["failed"]))


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    resp = requests.get(FRED_URL, params={"id": series_id}, headers=HEADERS, timeout=8, verify=False)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    date_col = "observation_date"
    if date_col not in df.columns:
        raise ValueError(f"Unexpected FRED response for {series_id}: {df.columns.tolist()}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
    df = df.rename(columns={date_col: "date", series_id: FRED_SERIES[series_id]})
    df[FRED_SERIES[series_id]] = pd.to_numeric(df[FRED_SERIES[series_id]], errors="coerce")
    return df[["date", FRED_SERIES[series_id]]]


def collect_fred_macro(start: str, end: str) -> None:
    out_dir = RAW_ROOT / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = None
    report = {"success": [], "failed": []}
    for series_id in FRED_SERIES:
        try:
            LOG.info("Fetching FRED %s", series_id)
            df = fetch_fred_series(series_id, start, end)
            merged = df if merged is None else merged.merge(df, on="date", how="outer")
            report["success"].append(series_id)
        except Exception as exc:
            LOG.warning("FRED %s failed: %s", series_id, exc)
            report["failed"].append(series_id)
        time.sleep(0.2)

    if merged is None:
        with (out_dir / "fred_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        LOG.warning("No FRED macro series were collected; continuing without FRED macro data")
        return

    merged = merged.sort_values("date").ffill()
    merged.to_csv(out_dir / "fred_macro.csv", index=False)
    with (out_dir / "fred_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOG.info("FRED macro saved: %s rows=%d", out_dir / "fred_macro.csv", len(merged))


def collect_market_proxies(start: str, end: str, sleep: float) -> None:
    out_dir = RAW_ROOT / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    report = {"success": [], "failed": []}
    for symbol, name in MARKET_PROXIES.items():
        LOG.info("Fetching market proxy %s (%s)", symbol, name)
        result = fetch_yahoo_symbol(symbol, start, end)
        if result and result["data"]:
            df = pd.DataFrame(result["data"])[["date", "close"]].rename(columns={"close": name})
            frames.append(df)
            report["success"].append(symbol)
        else:
            report["failed"].append(symbol)
        time.sleep(sleep)

    if frames:
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on="date", how="outer")
        merged = merged.sort_values("date").ffill()
        merged.to_csv(out_dir / "yahoo_market_proxies.csv", index=False)

    with (out_dir / "yahoo_proxy_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOG.info("Market proxy success=%d failed=%d", len(report["success"]), len(report["failed"]))


def gdelt_query(row: dict) -> str:
    name = row.get("name") or row["ticker"]
    ticker = row["ticker"]
    return f'"{name}" OR "{ticker}"'


def fetch_gdelt_timeline(query: str, start: str, end: str) -> dict | None:
    params = {
        "query": query,
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "235959",
        "timelinesmooth": "0",
        "timelineres": "month",
    }
    try:
        resp = requests.get(GDELT_URL, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            LOG.warning("GDELT HTTP %d for %s", resp.status_code, query)
            return None
        return resp.json()
    except Exception as exc:
        LOG.warning("GDELT failed for %s: %s", query, exc)
        return None


def collect_gdelt_news(universe: list[dict], start: str, end: str, sleep: float) -> None:
    out_dir = RAW_ROOT / "news" / "gdelt_timeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"success": [], "failed": []}

    for idx, row in enumerate(universe, start=1):
        symbol = row["ticker"].strip()
        query = gdelt_query(row)
        LOG.info("[%d/%d] Fetching GDELT timeline %s", idx, len(universe), symbol)
        payload = fetch_gdelt_timeline(query, start, end)
        if payload:
            out = {
                "symbol": symbol,
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "source": "gdelt_doc_2_timeline_tone",
                "query": query,
                "payload": payload,
            }
            with (out_dir / f"{symbol}.json").open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            report["success"].append(symbol)
        else:
            report["failed"].append(symbol)
        time.sleep(sleep)

    with (out_dir / "fetch_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOG.info("GDELT success=%d failed=%d", len(report["success"]), len(report["failed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect HMSC US market, macro, and news data.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--tasks", nargs="+", default=["ohlcv", "macro", "proxies"],
                        choices=["ohlcv", "macro", "proxies", "gdelt"])
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    universe = load_universe(args.universe, args.limit)
    LOG.info("Loaded universe: %d symbols from %s", len(universe), args.universe)
    LOG.info("Tasks: %s | range: %s -> %s", ", ".join(args.tasks), args.start, args.end)

    if "ohlcv" in args.tasks:
        collect_ohlcv(universe, args.start, args.end, args.sleep)
    if "macro" in args.tasks:
        collect_fred_macro(args.start, args.end)
    if "proxies" in args.tasks:
        collect_market_proxies(args.start, args.end, args.sleep)
    if "gdelt" in args.tasks:
        collect_gdelt_news(universe, args.start, args.end, args.sleep)


if __name__ == "__main__":
    main()

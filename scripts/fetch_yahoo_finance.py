"""Fetch daily OHLCV data for all tickers from Yahoo Finance (query2 endpoint, no auth)."""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}
BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
PERIOD1 = int(datetime(2015, 1, 1).timestamp())
PERIOD2 = int(datetime(2025, 12, 31).timestamp())
OUT_ROOT = Path(__file__).parent.parent / "data" / "raw" / "yfinance"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def fetch_ticker(symbol: str, retries: int = 3) -> dict | None:
    url = BASE_URL.format(symbol=symbol)
    params = {"period1": PERIOD1, "period2": PERIOD2, "interval": "1d"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                wait = 5 * (2 ** attempt)
                LOGGER.warning("[%s] Rate limited, sleeping %ds", symbol, wait)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                LOGGER.warning("[%s] HTTP %d", symbol, r.status_code)
                time.sleep(2)
                continue
            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result or result is None:
                LOGGER.warning("[%s] No result", symbol)
                return None
            ts_data = result[0]
            timestamps = ts_data.get("timestamp", [])
            if not timestamps:
                LOGGER.warning("[%s] No timestamps", symbol)
                return None

            quote = ts_data.get("indicators", {}).get("quote", [{}])[0] or {}
            adj_close_raw = ts_data.get("indicators", {}).get("adjclose")
            adj_close_list = (adj_close_raw[0].get("adjclose", []) if adj_close_raw else []) or []

            rows = []
            for i, ts in enumerate(timestamps):
                date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                rows.append({
                    "date": date,
                    "open": _safe_get(quote.get("open"), i),
                    "high": _safe_get(quote.get("high"), i),
                    "low": _safe_get(quote.get("low"), i),
                    "close": _safe_get(quote.get("close"), i),
                    "volume": _safe_get(quote.get("volume"), i),
                    "adj_close": _safe_get(adj_close_list, i),
                })
            return {"symbol": symbol, "data": rows}
        except Exception as e:
            LOGGER.warning("[%s] Attempt %d failed: %s", symbol, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def _safe_get(lst, i):
    return lst[i] if isinstance(lst, list) and i < len(lst) else None


def main():
    ticker_file = Path("/Users/samli/Downloads/tickers.csv")
    if not ticker_file.exists():
        LOGGER.error("tickers.csv not found: %s", ticker_file)
        return

    tickers = []
    with open(ticker_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append(row["ticker"].strip())

    LOGGER.info("Loaded %d tickers", len(tickers))
    LOGGER.info("Date range: %s to %s (timestamps %d - %d)",
               datetime.fromtimestamp(PERIOD1).date(),
               datetime.fromtimestamp(PERIOD2).date(),
               PERIOD1, PERIOD2)

    results = {"success": [], "failed": []}

    for i, symbol in enumerate(tickers):
        LOGGER.info("[%d/%d] Fetching %s ...", i + 1, len(tickers), symbol)
        result = fetch_ticker(symbol)
        if result and result["data"]:
            out_file = OUT_ROOT / f"{symbol}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            LOGGER.info("[%s] OK: %d days (%s → %s)",
                        symbol, len(result["data"]),
                        result["data"][0]["date"], result["data"][-1]["date"])
            results["success"].append(symbol)
        else:
            LOGGER.warning("[%s] FAILED", symbol)
            results["failed"].append(symbol)
        time.sleep(0.5)

    LOGGER.info("")
    LOGGER.info("=== Summary ===")
    LOGGER.info("Success: %d / %d", len(results["success"]), len(tickers))
    LOGGER.info("Failed: %d (%s)", len(results["failed"]), results["failed"])

    report_path = OUT_ROOT / "fetch_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
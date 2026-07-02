"""Fetch daily OHLCV data from Eastmoney push2his kline endpoint.

The output schema matches scripts/fetch_yahoo_finance.py so the same processing
pipeline can build FinWorldDataset episodes from either data source.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}
FIELDS1 = "f1,f2,f3,f4,f5,f6"
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
OUT_ROOT = Path(__file__).parent.parent / "data" / "raw" / "eastmoney"


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("-", ".")


def eastmoney_secid(symbol: str) -> str:
    """Convert common A-share symbols to Eastmoney secid.

    Examples:
      600519, 600519.SH, SH600519 -> 1.600519
      000001, 000001.SZ, SZ000001 -> 0.000001
    """
    s = normalize_symbol(symbol)
    if s.startswith("SH") and len(s) >= 8:
        return f"1.{s[-6:]}"
    if s.startswith("SZ") and len(s) >= 8:
        return f"0.{s[-6:]}"
    if s.endswith(".SH"):
        return f"1.{s.split('.')[0]}"
    if s.endswith(".SZ") or s.endswith(".BJ"):
        return f"0.{s.split('.')[0]}"
    code = "".join(ch for ch in s if ch.isdigit())
    if len(code) != 6:
        raise ValueError(f"Cannot infer Eastmoney secid from symbol: {symbol}")
    if code.startswith(("5", "6", "9")):
        return f"1.{code}"
    return f"0.{code}"


def _to_float(value: str):
    if value in {"", "-", "None", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str):
    val = _to_float(value)
    return int(val) if val is not None else None


def parse_kline(line: str) -> dict:
    parts = line.split(",")
    if len(parts) < 11:
        raise ValueError(f"Unexpected kline format: {line}")
    date, open_, close, high, low, volume, amount, amplitude, pct_chg, change, turnover = parts[:11]
    return {
        "date": date,
        "open": _to_float(open_),
        "high": _to_float(high),
        "low": _to_float(low),
        "close": _to_float(close),
        "volume": _to_int(volume),
        "adj_close": _to_float(close),
        "amount": _to_float(amount),
        "amplitude": _to_float(amplitude),
        "pct_chg": _to_float(pct_chg),
        "change": _to_float(change),
        "turnover": _to_float(turnover),
    }


def fetch_symbol(
    symbol: str,
    begin: str,
    end: str,
    fq: int = 1,
    retries: int = 3,
    sleep_base: float = 1.0,
    timeout: float = 30,
) -> dict | None:
    secid = eastmoney_secid(symbol)
    params = {
        "secid": secid,
        "fields1": FIELDS1,
        "fields2": FIELDS2,
        "klt": "101",
        "fqt": str(fq),
        "beg": begin,
        "end": end,
        "lmt": "1000000",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=timeout)
            if resp.status_code != 200:
                wait = sleep_base * (2 ** attempt)
                LOGGER.warning("[%s] HTTP %d, sleeping %.1fs", symbol, resp.status_code, wait)
                time.sleep(wait)
                continue

            payload = resp.json()
            data = payload.get("data") or {}
            klines = data.get("klines") or []
            if not klines:
                LOGGER.warning("[%s] No kline data returned for secid=%s", symbol, secid)
                return None

            rows = [parse_kline(line) for line in klines]
            return {
                "symbol": normalize_symbol(symbol),
                "source": "eastmoney",
                "secid": secid,
                "adjustment": fq,
                "data": rows,
            }
        except Exception as exc:
            wait = sleep_base * (2 ** attempt)
            LOGGER.warning("[%s] Attempt %d failed: %s", symbol, attempt + 1, exc)
            time.sleep(wait)
    return None


def load_symbols(ticker_file: Path) -> list[str]:
    with ticker_file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        symbol_col = "ticker" if "ticker" in fieldnames else "symbol"
        if symbol_col not in fieldnames:
            raise ValueError(f"{ticker_file} must contain a 'ticker' or 'symbol' column")
        return [row[symbol_col].strip() for row in reader if row.get(symbol_col, "").strip()]


def main():
    parser = argparse.ArgumentParser(description="Fetch A-share daily data from Eastmoney.")
    parser.add_argument("--ticker_file", type=Path, default=Path("/Users/samli/Downloads/tickers.csv"))
    parser.add_argument("--begin", type=str, default="20150101")
    parser.add_argument("--end", type=str, default="20251231")
    parser.add_argument("--fq", type=int, default=1, choices=[0, 1, 2],
                        help="Eastmoney adjustment: 0=none, 1=qfq, 2=hfq")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.ticker_file.exists():
        raise FileNotFoundError(f"Ticker file not found: {args.ticker_file}")

    symbols = load_symbols(args.ticker_file)
    if args.limit:
        symbols = symbols[:args.limit]

    LOGGER.info("Loaded %d symbols from %s", len(symbols), args.ticker_file)
    LOGGER.info("Date range: %s -> %s, adjustment fq=%d", args.begin, args.end, args.fq)

    results = {"success": [], "failed": []}
    for i, symbol in enumerate(symbols, start=1):
        LOGGER.info("[%d/%d] Fetching %s ...", i, len(symbols), symbol)
        try:
            result = fetch_symbol(symbol, args.begin, args.end, fq=args.fq)
        except ValueError as exc:
            LOGGER.warning("[%s] FAILED: %s", symbol, exc)
            result = None

        if result and result["data"]:
            safe_symbol = normalize_symbol(symbol).replace(".", "_")
            out_file = OUT_ROOT / f"{safe_symbol}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            first, last = result["data"][0]["date"], result["data"][-1]["date"]
            LOGGER.info("[%s] OK: %d days (%s -> %s)", symbol, len(result["data"]), first, last)
            results["success"].append(symbol)
        else:
            LOGGER.warning("[%s] FAILED", symbol)
            results["failed"].append(symbol)
        time.sleep(args.sleep)

    report_path = OUT_ROOT / "fetch_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    LOGGER.info("Success: %d / %d", len(results["success"]), len(symbols))
    LOGGER.info("Failed: %d (%s)", len(results["failed"]), results["failed"])
    LOGGER.info("Report: %s", report_path)


if __name__ == "__main__":
    main()

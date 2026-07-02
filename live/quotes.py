from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from scripts.fetch_eastmoney import eastmoney_secid, normalize_symbol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
US_TICKER_FILE = PROJECT_ROOT / "data" / "tickers" / "hmsc_us_90.csv"
CN_TICKER_FILE = PROJECT_ROOT / "data" / "tickers" / "hmsc_cn_50.csv"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 FinVerse-Dashboard/0.1",
    "Referer": "https://quote.eastmoney.com/",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_ticker_info(market: str) -> dict[str, dict[str, str]]:
    ticker_file = CN_TICKER_FILE if market == "cn" else US_TICKER_FILE
    if not ticker_file.exists():
        return {}
    with ticker_file.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {
            row["ticker"].upper(): {
                "ticker": row["ticker"].upper(),
                "name": row.get("name", row["ticker"]),
                "sector": row.get("sector", "Unknown"),
                "type": row.get("type", "stock"),
            }
            for row in reader
            if row.get("ticker")
        }


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _float(value: Any) -> float | None:
    if value in {None, "", "-", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _percent(value: Any) -> float | None:
    number = _float(value)
    return number / 100.0 if number is not None else None


def _quote_time(epoch_seconds: Any) -> str | None:
    ts = _int(epoch_seconds)
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _change(price: float | None, previous_close: float | None, provided: float | None = None) -> float | None:
    if provided is not None:
        return provided
    if price is None or previous_close in {None, 0}:
        return None
    return price - previous_close


def _change_percent(price: float | None, previous_close: float | None, provided: float | None = None) -> float | None:
    if provided is not None:
        return provided
    if price is None or previous_close in {None, 0}:
        return None
    return price / previous_close - 1.0


def _base_quote(
    market: str,
    ticker: str,
    info: dict[str, dict[str, str]],
    source: str,
    is_realtime: bool,
) -> dict[str, Any]:
    ticker = ticker.upper()
    meta = info.get(ticker, {})
    return {
        "market": market,
        "ticker": ticker,
        "name": meta.get("name", ticker),
        "source": source,
        "is_realtime": is_realtime,
    }


def fetch_us_quotes(symbols: list[str], info: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = []
    for chunk in _chunks([symbol.upper() for symbol in symbols], 40):
        try:
            response = requests.get(
                YAHOO_QUOTE_URL,
                headers=HTTP_HEADERS,
                params={"symbols": ",".join(chunk)},
                timeout=8,
            )
            if response.status_code != 200:
                continue
            results = response.json().get("quoteResponse", {}).get("result", [])
        except Exception:
            continue
        for item in results:
            ticker = str(item.get("symbol", "")).upper()
            price = _float(item.get("regularMarketPrice"))
            previous_close = _float(item.get("regularMarketPreviousClose"))
            change = _change(price, previous_close, _float(item.get("regularMarketChange")))
            change_percent = _change_percent(price, previous_close, _percent(item.get("regularMarketChangePercent")))
            quotes.append(
                {
                    **_base_quote("us", ticker, info, "yahoo_finance_quote", True),
                    "price": price,
                    "previous_close": previous_close,
                    "open": _float(item.get("regularMarketOpen")),
                    "high": _float(item.get("regularMarketDayHigh")),
                    "low": _float(item.get("regularMarketDayLow")),
                    "volume": _int(item.get("regularMarketVolume")),
                    "change": change,
                    "change_percent": change_percent,
                    "currency": item.get("currency", "USD"),
                    "market_state": item.get("marketState"),
                    "quote_time": _quote_time(item.get("regularMarketTime")),
                }
            )
    seen = {quote["ticker"].upper() for quote in quotes}
    missing = [symbol for symbol in symbols if symbol.upper() not in seen]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_us_chart_quote, symbol, info): symbol for symbol in missing}
        for future in as_completed(futures):
            quote = future.result()
            if quote:
                quotes.append(quote)
    return quotes


def _last_number(values: list[Any]) -> float | None:
    for value in reversed(values):
        number = _float(value)
        if number is not None:
            return number
    return None


def _clean_numbers(values: list[Any]) -> list[float]:
    return [number for number in (_float(value) for value in values) if number is not None]


def fetch_us_chart_quote(symbol: str, info: dict[str, dict[str, str]]) -> dict[str, Any] | None:
    symbol = symbol.upper()
    try:
        response = requests.get(
            f"{YAHOO_CHART_URL}/{symbol}",
            headers=HTTP_HEADERS,
            params={"range": "1d", "interval": "1m", "includePrePost": "true"},
            timeout=8,
        )
        if response.status_code != 200:
            return None
        result = (response.json().get("chart", {}).get("result") or [None])[0]
        if not result:
            return None
        meta = result.get("meta", {})
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    except Exception:
        return None

    closes = quote.get("close") or []
    highs = _clean_numbers(quote.get("high") or [])
    lows = _clean_numbers(quote.get("low") or [])
    opens = _clean_numbers(quote.get("open") or [])
    volumes = [_int(value) or 0 for value in quote.get("volume") or []]
    price = _float(meta.get("regularMarketPrice")) or _last_number(closes)
    previous_close = _float(meta.get("chartPreviousClose")) or _float(meta.get("previousClose"))
    change = _change(price, previous_close)
    change_percent = _change_percent(price, previous_close)
    return {
        **_base_quote("us", symbol, info, "yahoo_finance_chart_quote", True),
        "price": price,
        "previous_close": previous_close,
        "open": opens[0] if opens else None,
        "high": max(highs) if highs else None,
        "low": min(lows) if lows else None,
        "volume": _int(meta.get("regularMarketVolume")) or sum(volumes) or None,
        "change": change,
        "change_percent": change_percent,
        "currency": meta.get("currency", "USD"),
        "market_state": meta.get("marketState"),
        "quote_time": _quote_time(timestamps[-1] if timestamps else meta.get("regularMarketTime")),
    }


def fetch_cn_quotes(symbols: list[str], info: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = []
    normalized = [normalize_symbol(symbol) for symbol in symbols]
    code_to_symbol = {"".join(ch for ch in symbol if ch.isdigit())[-6:]: symbol for symbol in normalized}
    for chunk in _chunks(normalized, 80):
        try:
            secids = ",".join(eastmoney_secid(symbol) for symbol in chunk)
            response = requests.get(
                EASTMONEY_QUOTE_URL,
                headers=HTTP_HEADERS,
                params={
                    "fltt": "2",
                    "secids": secids,
                    "fields": "f12,f13,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18,f124",
                },
                timeout=8,
            )
            if response.status_code != 200:
                continue
            results = response.json().get("data", {}).get("diff", [])
        except Exception:
            continue
        for item in results:
            code = str(item.get("f12", "")).zfill(6)
            ticker = code_to_symbol.get(code)
            if not ticker:
                ticker = f"{code}.SH" if str(item.get("f13")) == "1" else f"{code}.SZ"
            price = _float(item.get("f2"))
            previous_close = _float(item.get("f18"))
            change = _change(price, previous_close, _float(item.get("f4")))
            change_percent = _change_percent(price, previous_close, _percent(item.get("f3")))
            base = _base_quote("cn", ticker, info, "eastmoney_push2_quote", True)
            quotes.append(
                {
                    **base,
                    "name": info.get(ticker.upper(), {}).get("name") or item.get("f14") or base["name"],
                    "price": price,
                    "previous_close": previous_close,
                    "open": _float(item.get("f17")),
                    "high": _float(item.get("f15")),
                    "low": _float(item.get("f16")),
                    "volume": _int(item.get("f5")),
                    "amount": _float(item.get("f6")),
                    "change": change,
                    "change_percent": change_percent,
                    "currency": "CNY",
                    "market_state": None,
                    "quote_time": _quote_time(item.get("f124")),
                }
            )
    return quotes


def snapshot_quotes_from_recommendation(
    market: str,
    payload: dict[str, Any],
    symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested = {symbol.upper() for symbol in symbols or []}
    assets = payload.get("all_assets") or payload.get("top_assets", [])
    quotes = []
    for asset in assets:
        ticker = str(asset.get("ticker", "")).upper()
        if requested and ticker not in requested:
            continue
        history = asset.get("history_close") or []
        previous_close = None
        if len(history) >= 2:
            previous_close = _float(history[-2].get("close"))
        price = _float(asset.get("close"))
        quotes.append(
            {
                "market": market,
                "ticker": ticker,
                "name": asset.get("name", ticker),
                "price": price,
                "previous_close": previous_close,
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "change": _change(price, previous_close),
                "change_percent": _change_percent(price, previous_close),
                "currency": "CNY" if market == "cn" else "USD",
                "market_state": "SNAPSHOT",
                "quote_time": payload.get("last_updated_at"),
                "source": "daily_snapshot_fallback",
                "is_realtime": False,
            }
        )
    return quotes


def fetch_live_quotes(market: str, symbols: list[str]) -> dict[str, Any]:
    market = market.lower()
    info = load_ticker_info(market)
    unique_symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols if symbol))
    if not unique_symbols:
        return {"market": market, "as_of": utc_now(), "source": "empty", "is_realtime": False, "quotes": []}
    quotes = fetch_cn_quotes(unique_symbols, info) if market == "cn" else fetch_us_quotes(unique_symbols, info)
    sources = sorted({quote["source"] for quote in quotes})
    return {
        "market": market,
        "as_of": utc_now(),
        "source": "+".join(sources) if sources else ("eastmoney_push2_quote" if market == "cn" else "yahoo_finance_quote"),
        "is_realtime": bool(quotes),
        "quotes": quotes,
    }

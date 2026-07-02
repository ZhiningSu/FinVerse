from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from scripts.fetch_eastmoney import fetch_symbol, normalize_symbol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKER_FILE = PROJECT_ROOT / "data" / "tickers" / "hmsc_us_90.csv"
DEFAULT_CN_TICKER_FILE = PROJECT_ROOT / "data" / "tickers" / "hmsc_cn_50.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "yfinance_hmsc"
DEFAULT_STOOQ_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stooq"
DEFAULT_CN_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eastmoney"
DEFAULT_LIVE_DIR = PROJECT_ROOT / "outputs" / "live"
DEFAULT_DATA_LIVE_DIR = PROJECT_ROOT / "data" / "live"
STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
YAHOO_CHART_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
STOCKTWITS_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
HTTP_HEADERS = {"User-Agent": "FinVerse-Dashboard/0.1 (+https://github.com/ZhiningSu/FinVerse)"}
POSITIVE_NEWS_TERMS = {
    "beat",
    "beats",
    "growth",
    "upgrade",
    "upgrades",
    "surge",
    "rally",
    "strong",
    "profit",
    "record",
    "outperform",
    "bullish",
    "raises",
}
NEGATIVE_NEWS_TERMS = {
    "miss",
    "misses",
    "downgrade",
    "downgrades",
    "fall",
    "falls",
    "drop",
    "weak",
    "loss",
    "lawsuit",
    "probe",
    "bearish",
    "cuts",
}

THEME_KEYWORDS = {
    "ai": {"ai", "artificial", "intelligence", "genai", "gpu", "accelerator", "datacenter", "data center"},
    "semiconductor": {"semiconductor", "chip", "chips", "foundry", "hbm", "dram", "nand", "memory", "wafer"},
    "cloud": {"cloud", "software", "saas", "database", "infrastructure"},
    "crypto": {"bitcoin", "crypto", "blockchain", "ethereum"},
    "obesity_drugs": {"obesity", "glp-1", "weight-loss", "weight loss", "zepbound", "mounjaro"},
    "energy": {"oil", "gas", "energy", "crude", "lng", "opec"},
    "rates_financials": {"rate", "rates", "yield", "treasury", "fed", "bank", "lending"},
    "defense": {"defense", "aerospace", "missile", "contract", "pentagon"},
}

THEME_TICKERS = {
    "ai": {"NVDA", "AMD", "AVGO", "MSFT", "GOOGL", "META", "ORCL", "CRM", "XLK", "QQQ"},
    "semiconductor": {"NVDA", "AMD", "AVGO", "INTC", "XLK", "QQQ"},
    "cloud": {"MSFT", "ORCL", "CRM", "AMZN", "GOOGL"},
    "crypto": {"MSTR", "COIN", "RIOT", "MARA"},
    "obesity_drugs": {"LLY", "MRK", "ABBV", "PFE", "XLV"},
    "energy": {"XOM", "CVX", "COP", "SLB", "EOG", "OXY", "XLE"},
    "rates_financials": {"JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "XLF"},
    "defense": {"LMT", "RTX", "BA", "HON", "GE"},
}

NEWS_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "after", "before", "about",
    "into", "over", "under", "stock", "stocks", "share", "shares", "market", "markets",
    "company", "companies", "says", "said", "will", "can", "are", "was", "were", "its",
}

MODAL_WEIGHTS = {
    "sector_macro": 0.05,
    "discussion": 0.36,
    "theme_heat": 0.20,
    "price_momentum": 0.16,
    "graph_corr": 0.09,
    "historical_retrieval": 0.06,
    "model_vq": 0.08,
}


@dataclass(frozen=True)
class LivePipelineConfig:
    market: str = "us"
    ticker_file: Path = DEFAULT_TICKER_FILE
    raw_dir: Path = DEFAULT_RAW_DIR
    data_live_dir: Path = DEFAULT_DATA_LIVE_DIR
    output_dir: Path = DEFAULT_LIVE_DIR
    top_k: int = 20
    mode: str = "heuristic_adapter"
    model_checkpoint: str = "outputs/paper_experiments/finverse/best_checkpoint.pt"
    model_name: str = "finverse"
    hidden_dim: int = 128
    latent_dim: int = 128
    device: str = "cpu"
    fetch_online: bool = False
    fetch_begin: str = "20230101"
    fetch_end: str = "20500101"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def market_language(market: str) -> str:
    return "zh" if market.lower() == "cn" else "en"


def effective_paths(config: LivePipelineConfig) -> tuple[Path, Path, Path, Path]:
    market = config.market.lower()
    ticker_file = config.ticker_file
    raw_dir = config.raw_dir
    if market == "cn":
        if ticker_file == DEFAULT_TICKER_FILE:
            ticker_file = DEFAULT_CN_TICKER_FILE
        if raw_dir == DEFAULT_RAW_DIR:
            raw_dir = DEFAULT_CN_RAW_DIR
    output_dir = config.output_dir / market
    data_live_dir = config.data_live_dir / market
    return ticker_file, raw_dir, data_live_dir, output_dir


def load_ticker_info(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return {
            row["ticker"]: {
                "ticker": row["ticker"],
                "name": row.get("name", row["ticker"]),
                "sector": row.get("sector", "Unknown"),
                "type": row.get("type", "stock"),
            }
            for row in reader
        }


def load_raw_market_data(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        if path.name == "fetch_report.json":
            continue
        payload = json.loads(path.read_text())
        symbol = payload.get("symbol", path.stem)
        rows = payload.get("data", [])
        if rows:
            data[symbol] = sorted(rows, key=lambda item: item["date"])
    return data


def fetch_eastmoney_universe(
    raw_dir: Path,
    ticker_info: dict[str, dict[str, str]],
    begin: str,
    end: str,
    force_fetch: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, Path]] = []
    for symbol in ticker_info:
        safe_symbol = normalize_symbol(symbol).replace(".", "_")
        out_file = raw_dir / f"{safe_symbol}.json"
        if out_file.exists() and not force_fetch:
            continue
        pending.append((symbol, out_file))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_symbol, symbol, begin=begin, end=end, fq=1, retries=1, sleep_base=0.2, timeout=8): (symbol, out_file)
            for symbol, out_file in pending
        }
        for future in as_completed(futures):
            _, out_file = futures[future]
            result = future.result()
            if result and result.get("data"):
                out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_raw_market_data(raw_dir)


def stooq_symbol(symbol: str) -> str:
    ticker = symbol.strip().lower().replace("-", ".")
    return f"{ticker}.us"


def _stooq_row(row: dict[str, str]) -> dict[str, Any] | None:
    try:
        close = float(row["Close"])
        return {
            "date": row["Date"],
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": close,
            "volume": int(float(row.get("Volume") or 0)),
            "adj_close": close,
        }
    except (KeyError, TypeError, ValueError):
        return None


def fetch_stooq_symbol(symbol: str, retries: int = 1, sleep_base: float = 0.4) -> dict[str, Any] | None:
    params = {"s": stooq_symbol(symbol), "i": "d"}
    for attempt in range(retries):
        try:
            resp = requests.get(STOOQ_BASE_URL, headers=HTTP_HEADERS, params=params, timeout=3)
            if resp.status_code != 200 or not resp.text.strip().startswith("Date,"):
                time.sleep(sleep_base * (2 ** attempt))
                continue
            rows = [
                parsed
                for parsed in (_stooq_row(row) for row in csv.DictReader(io.StringIO(resp.text)))
                if parsed is not None
            ]
            if rows:
                return {"symbol": symbol.upper(), "source": "stooq", "stooq_symbol": params["s"], "data": rows}
        except Exception:
            time.sleep(sleep_base * (2 ** attempt))
    return None


def fetch_yahoo_chart_symbol(symbol: str, retries: int = 1, sleep_base: float = 0.4) -> dict[str, Any] | None:
    url = f"{YAHOO_CHART_BASE_URL}/{symbol.upper()}"
    params = {"range": "5y", "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            if resp.status_code != 200:
                time.sleep(sleep_base * (2 ** attempt))
                continue
            result = (resp.json().get("chart", {}).get("result") or [None])[0]
            if not result:
                return None
            timestamps = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
            rows = []
            for idx, ts in enumerate(timestamps):
                try:
                    close = quote["close"][idx]
                    if close is None:
                        continue
                    rows.append(
                        {
                            "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat(),
                            "open": float(quote["open"][idx]) if quote["open"][idx] is not None else None,
                            "high": float(quote["high"][idx]) if quote["high"][idx] is not None else None,
                            "low": float(quote["low"][idx]) if quote["low"][idx] is not None else None,
                            "close": float(close),
                            "volume": int(quote["volume"][idx] or 0),
                            "adj_close": float(adj[idx]) if idx < len(adj) and adj[idx] is not None else float(close),
                        }
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            if rows:
                return {"symbol": symbol.upper(), "source": "yahoo_chart", "data": rows}
        except Exception:
            time.sleep(sleep_base * (2 ** attempt))
    return None


def fetch_us_online_symbol(symbol: str) -> dict[str, Any] | None:
    return fetch_stooq_symbol(symbol) or fetch_yahoo_chart_symbol(symbol)


def headline_sentiment(text: str) -> float:
    words = {word.strip(".,:;!?()[]{}\"'").lower() for word in text.split()}
    pos = len(words & POSITIVE_NEWS_TERMS)
    neg = len(words & NEGATIVE_NEWS_TERMS)
    if pos == 0 and neg == 0:
        return 0.0
    return _clip((pos - neg) / max(pos + neg, 1), -1.0, 1.0)


def parse_yahoo_rss_sentiment(xml_text: str) -> tuple[float, int, list[str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0.0, 0, []
    scores = []
    headlines = []
    for item in root.findall(".//item")[:8]:
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        text = f"{title} {desc}".strip()
        if not text:
            continue
        scores.append(headline_sentiment(text))
        if title:
            headlines.append(title)
    if not scores:
        return 0.0, 0, headlines
    return float(np.mean(scores)), len(scores), headlines[:3]


def fetch_yahoo_rss_sentiment(symbol: str) -> tuple[float, int, list[str]]:
    try:
        resp = requests.get(
            YAHOO_RSS_URL,
            headers=HTTP_HEADERS,
            params={"s": symbol.upper(), "region": "US", "lang": "en-US"},
            timeout=3,
        )
        if resp.status_code != 200:
            return 0.0, 0, []
        return parse_yahoo_rss_sentiment(resp.text)
    except Exception:
        return 0.0, 0, []


def _sentiment_from_titles(items: list[str]) -> tuple[float, int, list[str]]:
    titles = [item.strip() for item in items if item and item.strip()]
    if not titles:
        return 0.0, 0, []
    scores = [headline_sentiment(title) for title in titles]
    return float(np.mean(scores)), len(titles), titles[:3]


def parse_finnhub_company_news(payload: Any) -> tuple[float, int, list[str]]:
    if not isinstance(payload, list):
        return 0.0, 0, []
    titles = [str(item.get("headline") or "") for item in payload[:20] if isinstance(item, dict)]
    return _sentiment_from_titles(titles)


def fetch_finnhub_company_news(symbol: str) -> tuple[float, int, list[str]]:
    token = os.environ.get("FINNHUB_API_KEY")
    if not token:
        return 0.0, 0, []
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=14)
    try:
        resp = requests.get(
            FINNHUB_COMPANY_NEWS_URL,
            headers=HTTP_HEADERS,
            params={"symbol": symbol.upper(), "from": start.isoformat(), "to": end.isoformat(), "token": token},
            timeout=4,
        )
        if resp.status_code != 200:
            return 0.0, 0, []
        return parse_finnhub_company_news(resp.json())
    except Exception:
        return 0.0, 0, []


def parse_gdelt_doc(payload: Any) -> tuple[float, int, list[str]]:
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    titles = [str(item.get("title") or "") for item in articles[:20] if isinstance(item, dict)]
    return _sentiment_from_titles(titles)


def fetch_gdelt_doc_signal(symbol: str, name: str) -> tuple[float, int, list[str]]:
    query = f'("{symbol.upper()}" OR "{name}") sourcecountry:US'
    try:
        resp = requests.get(
            GDELT_DOC_URL,
            headers=HTTP_HEADERS,
            params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 20,
                "timespan": "7d",
            },
            timeout=4,
        )
        if resp.status_code != 200:
            return 0.0, 0, []
        return parse_gdelt_doc(resp.json())
    except Exception:
        return 0.0, 0, []


def parse_stocktwits_messages(payload: Any) -> tuple[float, int, list[str]]:
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    texts = [str(item.get("body") or "") for item in messages[:30] if isinstance(item, dict)]
    return _sentiment_from_titles(texts)


def fetch_stocktwits_signal(symbol: str) -> tuple[float, int, list[str]]:
    try:
        resp = requests.get(
            STOCKTWITS_STREAM_URL.format(symbol=symbol.upper()),
            headers=HTTP_HEADERS,
            timeout=4,
        )
        if resp.status_code != 200:
            return 0.0, 0, []
        return parse_stocktwits_messages(resp.json())
    except Exception:
        return 0.0, 0, []


def _news_keywords(texts: list[str], limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for text in texts:
        for raw in text.lower().replace("/", " ").replace("-", " ").split():
            token = raw.strip(".,:;!?()[]{}\"'#$")
            if len(token) < 3 or token in NEWS_STOPWORDS:
                continue
            if token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
    return [
        {"keyword": word, "count": count}
        for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _news_theme_profile(asset: dict[str, Any], texts: list[str]) -> tuple[float, list[dict[str, Any]]]:
    joined = " ".join(texts + [asset.get("ticker", ""), asset.get("name", ""), asset.get("sector", "")]).lower()
    ticker = asset.get("ticker", "").upper()
    theme_counts: dict[str, int] = {}
    for theme, terms in THEME_KEYWORDS.items():
        count = sum(joined.count(term) for term in terms)
        if ticker in THEME_TICKERS.get(theme, set()):
            count += 2
        if count > 0:
            theme_counts[theme] = count
    if not theme_counts:
        return 0.0, []
    total = sum(theme_counts.values())
    theme_heat = _clip(math.log1p(total) / math.log1p(12.0), 0.0, 1.0)
    themes = [
        {"theme": theme, "count": count, "score": float(count / max(total, 1))}
        for theme, count in sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    return theme_heat, themes


def _news_hotness_score(asset_signal: dict[str, Any], discussion_score: float) -> float:
    sentiment = float(asset_signal.get("sentiment", 0.0))
    sentiment_norm = (sentiment + 1.0) / 2.0
    source_diversity = float(asset_signal.get("source_diversity", 0.0))
    theme_heat = float(asset_signal.get("theme_heat", 0.0))
    heat = float(asset_signal.get("heat", 0.0))
    return _clip(
        0.44 * heat
        + 0.28 * discussion_score
        + 0.18 * theme_heat
        + 0.05 * sentiment_norm
        + 0.05 * source_diversity,
        0.0,
        1.0,
    )


def collect_news_signals(
    market: str,
    ranked_assets: list[dict[str, Any]],
    fetch_online: bool,
) -> dict[str, dict[str, Any]]:
    """Return sector-level news sentiment from local RSS cache or online RSS."""
    if market != "us":
        return {}
    per_sector: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    headlines: dict[str, list[str]] = {}
    local_dir = PROJECT_ROOT / "data" / "raw" / "news" / "yahoo_rss"
    local_symbols = set()

    for asset in ranked_assets[:60]:
        symbol = asset["ticker"]
        sector = asset["sector"]
        sentiment, count, items = 0.0, 0, []
        local_file = local_dir / f"{symbol}.xml"
        if local_file.exists():
            sentiment, count, items = parse_yahoo_rss_sentiment(local_file.read_text(encoding="utf-8", errors="ignore"))
            local_symbols.add(symbol)
        if count == 0:
            continue
        per_sector.setdefault(sector, []).append(sentiment)
        counts[sector] = counts.get(sector, 0) + count
        headlines.setdefault(sector, []).extend(items[:2])

    if fetch_online:
        pending = [asset for asset in ranked_assets[:40] if asset["ticker"] not in local_symbols]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_yahoo_rss_sentiment, asset["ticker"]): asset for asset in pending}
            for future in as_completed(futures):
                asset = futures[future]
                sentiment, count, items = future.result()
                if count == 0:
                    continue
                sector = asset["sector"]
                per_sector.setdefault(sector, []).append(sentiment)
                counts[sector] = counts.get(sector, 0) + count
                headlines.setdefault(sector, []).extend(items[:2])

    return {
        sector: {
            "score": float(np.mean(scores)),
            "count": counts.get(sector, 0),
            "headlines": headlines.get(sector, [])[:3],
            "source": "yahoo_finance_rss",
        }
        for sector, scores in per_sector.items()
    }


def collect_discussion_signals(
    market: str,
    asset_features: list[dict[str, Any]],
    fetch_online: bool,
) -> dict[str, Any]:
    """Approximate asset/sector discussion heat from news/forum-style text counts."""
    if market != "us":
        return {"assets": {}, "sectors": {}}
    yahoo_dir = PROJECT_ROOT / "data" / "raw" / "news" / "yahoo_rss"
    finnhub_dir = PROJECT_ROOT / "data" / "raw" / "news" / "finnhub"
    gdelt_dir = PROJECT_ROOT / "data" / "raw" / "news" / "gdelt"
    stocktwits_dir = PROJECT_ROOT / "data" / "raw" / "news" / "stocktwits"
    asset_counts: dict[str, int] = {}
    asset_sentiments: dict[str, list[float]] = {}
    asset_texts: dict[str, list[str]] = {}
    sector_counts: dict[str, int] = {}
    sector_sentiments: dict[str, list[float]] = {}
    asset_sources: dict[str, set[str]] = {}
    fetched_symbols: set[str] = set()

    def add_signal(asset: dict[str, Any], sentiment: float, count: int, source: str, items: list[str] | None = None) -> None:
        if count <= 0:
            return
        ticker = asset["ticker"]
        sector = asset["sector"]
        asset_counts[ticker] = asset_counts.get(ticker, 0) + count
        asset_sentiments.setdefault(ticker, []).append(sentiment)
        asset_texts.setdefault(ticker, []).extend((items or [])[:10])
        asset_sources.setdefault(ticker, set()).add(source)
        sector_counts[sector] = sector_counts.get(sector, 0) + count
        sector_sentiments.setdefault(sector, []).append(sentiment)

    def add_local_json(asset: dict[str, Any], path: Path, parser, source: str) -> None:
        if not path.exists():
            return
        try:
            sentiment, count, items = parser(json.loads(path.read_text(encoding="utf-8", errors="ignore")))
            add_signal(asset, sentiment, count, source, items)
            fetched_symbols.add(asset["ticker"])
        except Exception:
            return

    for asset in asset_features:
        symbol = asset["ticker"]
        local_file = yahoo_dir / f"{symbol}.xml"
        if local_file.exists():
            sentiment, count, items = parse_yahoo_rss_sentiment(local_file.read_text(encoding="utf-8", errors="ignore"))
            add_signal(asset, sentiment, count, "yahoo_rss", items)
            fetched_symbols.add(symbol)
        add_local_json(asset, finnhub_dir / f"{symbol}.json", parse_finnhub_company_news, "finnhub")
        add_local_json(asset, gdelt_dir / f"{symbol}.json", parse_gdelt_doc, "gdelt")
        add_local_json(asset, stocktwits_dir / f"{symbol}.json", parse_stocktwits_messages, "stocktwits")

    if fetch_online:
        pending = [asset for asset in asset_features[:60] if asset["ticker"] not in fetched_symbols]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for asset in pending:
                futures[executor.submit(fetch_yahoo_rss_sentiment, asset["ticker"])] = (asset, "yahoo_rss")
                futures[executor.submit(fetch_finnhub_company_news, asset["ticker"])] = (asset, "finnhub")
                futures[executor.submit(fetch_gdelt_doc_signal, asset["ticker"], asset["name"])] = (asset, "gdelt")
                futures[executor.submit(fetch_stocktwits_signal, asset["ticker"])] = (asset, "stocktwits")
            for future in as_completed(futures):
                asset, source = futures[future]
                sentiment, count, items = future.result()
                add_signal(asset, sentiment, count, source, items)

    max_asset = max(asset_counts.values(), default=1)
    max_sector = max(sector_counts.values(), default=1)
    asset_by_ticker = {asset["ticker"]: asset for asset in asset_features}
    sector_theme_heats: dict[str, list[float]] = {}
    asset_payload: dict[str, dict[str, Any]] = {}
    for ticker, count in asset_counts.items():
        asset = asset_by_ticker.get(ticker, {"ticker": ticker, "name": ticker, "sector": "Unknown"})
        texts = asset_texts.get(ticker, [])
        theme_heat, themes = _news_theme_profile(asset, texts)
        sector_theme_heats.setdefault(asset.get("sector", "Unknown"), []).append(theme_heat)
        sources = sorted(asset_sources.get(ticker, set()))
        asset_payload[ticker] = {
            "count": count,
            "heat": float(math.log1p(count) / math.log1p(max_asset)),
            "sentiment": float(np.mean(asset_sentiments.get(ticker, [0.0]))),
            "sources": sources,
            "source_diversity": _clip(len(sources) / 4.0, 0.0, 1.0),
            "keywords": _news_keywords(texts),
            "themes": themes,
            "theme_heat": theme_heat,
            "sample_headlines": texts[:5],
        }
    return {
        "assets": asset_payload,
        "sectors": {
            sector: {
                "count": count,
                "heat": float(math.log1p(count) / math.log1p(max_sector)),
                "sentiment": float(np.mean(sector_sentiments.get(sector, [0.0]))),
                "theme_heat": float(np.mean(sector_theme_heats.get(sector, [0.0]))),
            }
            for sector, count in sector_counts.items()
        },
    }


def fetch_stooq_universe(
    raw_dir: Path,
    ticker_info: dict[str, dict[str, str]],
    force_fetch: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, Path]] = []
    for symbol in ticker_info:
        out_file = raw_dir / f"{symbol.upper().replace('.', '_').replace('-', '_')}.json"
        if out_file.exists() and not force_fetch:
            continue
        pending.append((symbol, out_file))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_us_online_symbol, symbol): (symbol, out_file) for symbol, out_file in pending}
        for future in as_completed(futures):
            _, out_file = futures[future]
            result = future.result()
            if result and result.get("data"):
                out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_raw_market_data(raw_dir)


def choose_trade_date(raw: dict[str, list[dict[str, Any]]], requested_date: str | None = None) -> str:
    if requested_date:
        return requested_date
    latest_dates = [rows[-1]["date"] for rows in raw.values() if rows]
    if not latest_dates:
        raise ValueError("No raw market data found.")
    return max(latest_dates)


def rows_until_date(rows: list[dict[str, Any]], trade_date: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["date"] <= trade_date]


def fetch_latest_data(
    config: LivePipelineConfig,
    raw_dir: Path,
    data_live_dir: Path,
    ticker_info: dict[str, dict[str, str]],
    trade_date: str | None = None,
    force_fetch: bool = False,
) -> dict[str, Any]:
    raw = load_raw_market_data(raw_dir)
    source = "raw_yfinance_hmsc_snapshot"
    market = config.market.lower()
    if market == "us" and (force_fetch or not raw):
        stooq_dir = DEFAULT_STOOQ_RAW_DIR if raw_dir == DEFAULT_RAW_DIR else raw_dir
        raw = fetch_stooq_universe(stooq_dir, ticker_info, force_fetch=force_fetch)
        source = "stooq_yahoo_chart_fallback"
    elif market == "cn" and (force_fetch or not raw):
        raw = fetch_eastmoney_universe(raw_dir, ticker_info, begin=config.fetch_begin, end=config.fetch_end, force_fetch=force_fetch)
        source = "eastmoney"
    elif market == "cn":
        source = "eastmoney"
    selected_date = choose_trade_date(raw, trade_date)
    snapshot = {}
    for ticker, rows in raw.items():
        history = rows_until_date(rows, selected_date)
        if not history:
            continue
        snapshot[ticker] = history[-1]

    output_path = data_live_dir / "raw" / selected_date / "market_snapshot.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"trade_date": selected_date, "assets": snapshot}, indent=2))
    return {"trade_date": selected_date, "assets": snapshot, "raw": raw, "path": str(output_path), "source": source}


def compute_asset_features(
    raw: dict[str, list[dict[str, Any]]],
    ticker_info: dict[str, dict[str, str]],
    trade_date: str,
) -> list[dict[str, Any]]:
    features = []
    for ticker, rows in raw.items():
        history = rows_until_date(rows, trade_date)
        if len(history) < 31:
            continue
        closes = np.asarray([float(row["close"]) for row in history[-31:]], dtype=float)
        if np.any(closes <= 0):
            continue
        daily_returns = closes[1:] / closes[:-1] - 1.0
        return_1d = float(daily_returns[-1])
        return_5d = float(closes[-1] / closes[-6] - 1.0) if len(closes) >= 6 else return_1d
        return_20d = float(closes[-1] / closes[-21] - 1.0) if len(closes) >= 21 else return_5d
        vol_20d = float(np.std(daily_returns[-20:]))
        info = ticker_info.get(ticker, {"name": ticker, "sector": "Unknown", "type": "stock"})
        features.append(
            {
                "ticker": ticker,
                "name": info["name"],
                "sector": info["sector"],
                "type": info["type"],
                "close": float(closes[-1]),
                "history_close": [
                    {"date": row["date"], "close": float(row["close"])}
                    for row in history[-45:]
                ],
                "return_1d": return_1d,
                "return_5d": return_5d,
                "return_20d": return_20d,
                "vol_20d": vol_20d,
            }
        )
    return features


def _market_state(asset_features: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([row["return_20d"] for row in asset_features], dtype=float)
    vols = np.asarray([row["vol_20d"] for row in asset_features], dtype=float)
    market_return = float(np.nanmean(returns)) if returns.size else 0.0
    market_vol = float(np.nanmean(vols)) if vols.size else 0.0
    bull_raw = 1.0 / (1.0 + math.exp(-30.0 * (market_return - 0.01)))
    bear_raw = 1.0 / (1.0 + math.exp(30.0 * (market_return + 0.01)))
    sideway_raw = max(0.08, 1.0 - abs(market_return) * 12.0 - market_vol * 4.0)
    total = bull_raw + bear_raw + sideway_raw
    regime_probs = {
        "bear": float(bear_raw / total),
        "sideway": float(sideway_raw / total),
        "bull": float(bull_raw / total),
    }
    if regime_probs["bull"] >= max(regime_probs["bear"], regime_probs["sideway"]):
        regime = "Bull"
    elif regime_probs["bear"] >= max(regime_probs["bull"], regime_probs["sideway"]):
        regime = "Bear"
    else:
        regime = "Sideway"
    return {
        "regime": regime,
        "regime_probs": regime_probs,
        "market_return_20d": market_return,
        "market_vol_20d": market_vol,
        "latent_summary": {
            "pc1": float(np.clip(market_return * 8.0, -1.0, 1.0)),
            "pc2": float(np.clip(market_vol * 20.0, 0.0, 1.0)),
            "nearest_historical_regime": regime,
        },
    }


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def stable_token_id(*parts: str, modulo: int = 256) -> int:
    text = "::".join(parts).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16) % modulo


def run_finverse_inference_adapter(asset_features: list[dict[str, Any]], market_state: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    bull = market_state["regime_probs"]["bull"]
    bear = market_state["regime_probs"]["bear"]
    for row in asset_features:
        momentum = 0.50 * row["return_20d"] + 0.25 * row["return_5d"] + 0.10 * row["return_1d"]
        defensive_bonus = 0.006 if row["sector"] in {"Utilities", "Healthcare", "Consumer Staples", "Market ETF", "Sector ETF", "医药", "消费", "公用事业", "宽基ETF", "行业ETF"} else 0.0
        growth_bonus = 0.010 if row["sector"] in {"Technology", "Communication Services", "Consumer Discretionary", "科技", "新能源", "通信"} and bull > bear else 0.0
        expected_return = _clip(momentum + defensive_bonus + growth_bonus - row["vol_20d"] * 0.35, -0.12, 0.12)
        predicted_volatility = float(row["vol_20d"] * math.sqrt(30.0))
        predicted_downside = float(max(0.0, row["vol_20d"] * 2.2 - expected_return * 0.35))
        horizons = list(range(1, 31))
        rollout_path = [
            {
                "horizon": h,
                "predicted_return": float(expected_return * (h / 30.0)),
                "predicted_close": float(row["close"] * (1.0 + expected_return * (h / 30.0))),
            }
            for h in horizons
        ]
        outputs.append(
            {
                **row,
                "expected_return_30d": expected_return,
                "predicted_volatility": predicted_volatility,
                "predicted_downside": predicted_downside,
                "regime_probs": market_state["regime_probs"],
                "rollout_path": rollout_path,
                "token_summary": {
                    "temporal_token": stable_token_id(row["ticker"], "temporal"),
                    "cross_asset_token": stable_token_id(row["sector"], "cross"),
                },
            }
        )
    return outputs


def _history_returns(raw: dict[str, list[dict[str, Any]]], ticker: str, trade_date: str, lookback: int = 30) -> np.ndarray | None:
    rows = rows_until_date(raw.get(ticker, []), trade_date)
    if len(rows) < lookback + 1:
        return None
    closes = np.asarray([float(row["close"]) for row in rows[-(lookback + 1):]], dtype=np.float32)
    if np.any(closes <= 0):
        return None
    returns = closes[1:] / (closes[:-1] + 1e-8) - 1.0
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _load_training_universe() -> tuple[dict[str, float], list[str]]:
    stats_path = PROJECT_ROOT / "data" / "processed" / "real_90" / "stats.json"
    if not stats_path.exists():
        return {"mean": 0.0, "std": 1.0}, []
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"mean": 0.0, "std": 1.0}, []
    return payload.get("price_stats", {"mean": 0.0, "std": 1.0}), list(payload.get("symbols", []))


def _ticker_condition_maps(ticker_info: dict[str, dict[str, str]], symbols: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    ticker_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    sectors = sorted({ticker_info.get(symbol, {}).get("sector", "Unknown") for symbol in symbols} | {"Unknown"})
    sector_to_id = {sector: idx for idx, sector in enumerate(sectors)}
    return ticker_to_idx, sector_to_id


def _history_closes(raw: dict[str, list[dict[str, Any]]], ticker: str, trade_date: str, lookback: int = 30) -> np.ndarray | None:
    rows = rows_until_date(raw.get(ticker, []), trade_date)
    if len(rows) < lookback:
        return None
    closes = np.asarray([float(row["close"]) for row in rows[-lookback:]], dtype=np.float32)
    if np.any(closes <= 0):
        return None
    return closes


def _normalized_history_prices(
    raw: dict[str, list[dict[str, Any]]],
    ticker: str,
    trade_date: str,
    price_stats: dict[str, float],
    lookback: int = 30,
) -> np.ndarray | None:
    closes = _history_closes(raw, ticker, trade_date, lookback=lookback)
    if closes is None:
        return None
    mean = float(price_stats.get("mean", 0.0))
    std = max(float(price_stats.get("std", 1.0)), 1e-8)
    return ((closes - mean) / std).astype(np.float32)


def _historical_state_retrieval(
    asset: dict[str, Any],
    raw: dict[str, list[dict[str, Any]]],
    trade_date: str,
    lookback: int = 30,
    horizon: int = 30,
    top_k: int = 3,
) -> dict[str, Any]:
    rows = rows_until_date(raw.get(asset["ticker"], []), trade_date)
    if len(rows) < lookback + horizon + 2:
        return {"signal": 0.5, "expected_return": 0.0, "mean_path": [], "cases": []}
    closes = np.asarray([float(row["close"]) for row in rows], dtype=np.float32)
    if np.any(closes <= 0):
        return {"signal": 0.5, "expected_return": 0.0, "mean_path": [], "cases": []}
    returns = closes[1:] / (closes[:-1] + 1e-8) - 1.0
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    current = returns[-lookback:]
    current_norm = float(np.linalg.norm(current))
    if current_norm <= 1e-8:
        return {"signal": 0.5, "expected_return": 0.0, "mean_path": [], "cases": []}

    candidates = []
    end_limit = len(closes) - horizon - 1
    for end_idx in range(lookback, end_limit):
        hist = returns[end_idx - lookback:end_idx]
        if hist.shape[0] != lookback:
            continue
        hist_norm = float(np.linalg.norm(hist))
        if hist_norm <= 1e-8:
            continue
        future = closes[end_idx + 1:end_idx + horizon + 1] / (closes[end_idx] + 1e-8) - 1.0
        if future.shape[0] < horizon:
            continue
        cosine = float(np.dot(current, hist) / (current_norm * hist_norm + 1e-8))
        vol_penalty = abs(float(current.std()) - float(hist.std())) * 2.0
        momentum_penalty = abs(float(current.mean()) - float(hist.mean())) * 5.0
        similarity = cosine - vol_penalty - momentum_penalty
        candidates.append((similarity, end_idx, future.astype(np.float32)))

    if not candidates:
        return {"signal": 0.5, "expected_return": 0.0, "mean_path": [], "cases": []}
    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[:top_k]
    paths = np.stack([item[2] for item in top], axis=0)
    mean_path = np.nan_to_num(paths.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    expected_return = float(np.clip(mean_path[-1], -0.3, 0.3))
    return {
        "signal": _clip(0.5 + expected_return * 4.0, 0.0, 1.0),
        "expected_return": expected_return,
        "mean_path": [float(np.clip(value, -0.3, 0.3)) for value in mean_path[:horizon]],
        "cases": [
            {
                "date": rows[end_idx]["date"],
                "similarity": float(similarity),
                "future_return_30d": float(np.clip(path[-1], -0.3, 0.3)),
            }
            for similarity, end_idx, path in top
        ],
    }


def _price_signal(asset: dict[str, Any]) -> float:
    momentum = 0.55 * asset["return_20d"] + 0.30 * asset["return_5d"] + 0.15 * asset["return_1d"]
    risk_penalty = 0.35 * asset["vol_20d"]
    return _clip(0.5 + momentum * 5.0 - risk_penalty, 0.0, 1.0)


def _discussion_signal(asset: dict[str, Any], discussion_signals: dict[str, Any]) -> float:
    asset_sig = discussion_signals.get("assets", {}).get(asset["ticker"], {})
    sector_sig = discussion_signals.get("sectors", {}).get(asset["sector"], {})
    heat = 0.65 * float(asset_sig.get("heat", 0.0)) + 0.35 * float(sector_sig.get("heat", 0.0))
    sentiment = 0.65 * float(asset_sig.get("sentiment", 0.0)) + 0.35 * float(sector_sig.get("sentiment", 0.0))
    theme_heat = 0.70 * float(asset_sig.get("theme_heat", 0.0)) + 0.30 * float(sector_sig.get("theme_heat", 0.0))
    source_diversity = float(asset_sig.get("source_diversity", 0.0))
    return _clip(
        0.03
        + 0.74 * heat
        + 0.04 * ((sentiment + 1.0) / 2.0)
        + 0.15 * theme_heat
        + 0.04 * source_diversity,
        0.0,
        1.0,
    )


def _correlation(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return 0.0
    if np.isclose(float(np.std(a)), 0.0) or np.isclose(float(np.std(b)), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _graph_signal(
    asset: dict[str, Any],
    peer_tickers: list[str],
    raw: dict[str, list[dict[str, Any]]],
    trade_date: str,
    asset_by_ticker: dict[str, dict[str, Any]],
) -> float:
    target_returns = _history_returns(raw, asset["ticker"], trade_date)
    weighted = []
    for ticker in peer_tickers[1:]:
        peer = asset_by_ticker.get(ticker)
        if peer is None:
            continue
        corr = max(_correlation(target_returns, _history_returns(raw, ticker, trade_date)), 0.0)
        sector_bonus = 0.25 if peer.get("sector") == asset.get("sector") else 0.0
        trend = _price_signal(peer)
        weighted.append((corr + sector_bonus) * trend)
    if not weighted:
        return 0.5
    return _clip(float(np.mean(weighted)), 0.0, 1.0)


def _peer_tickers(asset: dict[str, Any], asset_features: list[dict[str, Any]], width: int = 6) -> list[str]:
    target = asset["ticker"]
    same_sector = [
        row for row in asset_features
        if row["ticker"] != target and row.get("sector") == asset.get("sector")
    ]
    others = [row for row in asset_features if row["ticker"] != target and row not in same_sector]
    same_sector.sort(key=lambda row: abs(row["return_20d"] - asset["return_20d"]))
    others.sort(key=lambda row: abs(row["return_20d"] - asset["return_20d"]))
    peers = [row["ticker"] for row in [*same_sector, *others]][: width - 1]
    while len(peers) < width - 1:
        peers.append(target)
    return [target, *peers]


def _live_graph_sample(
    asset: dict[str, Any],
    asset_features: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    trade_date: str,
    market_state: dict[str, Any],
    discussion_signals: dict[str, Any],
    price_stats: dict[str, float] | None = None,
    retrieval: dict[str, Any] | None = None,
    width: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    tickers = _peer_tickers(asset, asset_features, width=width)
    asset_by_ticker = {row["ticker"]: row for row in asset_features}
    price_stats = price_stats or {"mean": 0.0, "std": 1.0}
    retrieval = retrieval or {"signal": 0.5, "expected_return": 0.0}
    columns = []
    for ticker in tickers:
        series = _normalized_history_prices(raw, ticker, trade_date, price_stats)
        if series is None:
            series = _normalized_history_prices(raw, asset["ticker"], trade_date, price_stats)
        if series is None:
            series = np.zeros(30, dtype=np.float32)
        columns.append(series)
    price_seq = np.stack(columns, axis=1).astype(np.float32)
    price_score = _price_signal(asset)
    sector_score = sector_macro_fit(asset["sector"], market_state)
    discussion_score = _discussion_signal(asset, discussion_signals)
    graph_score = _graph_signal(asset, tickers, raw, trade_date, asset_by_ticker)
    retrieval_score = float(retrieval.get("signal", 0.5))
    asset_discussion = discussion_signals.get("assets", {}).get(asset["ticker"], {})
    sector_discussion = discussion_signals.get("sectors", {}).get(asset["sector"], {})
    asset_sentiment = float(asset_discussion.get("sentiment", 0.0))
    sector_sentiment = float(sector_discussion.get("sentiment", 0.0))
    theme_score = 0.70 * float(asset_discussion.get("theme_heat", 0.0)) + 0.30 * float(sector_discussion.get("theme_heat", 0.0))
    source_diversity = float(asset_discussion.get("source_diversity", 0.0))
    hotness_score = _news_hotness_score(asset_discussion, discussion_score)

    macro_row = np.asarray(
        [
            market_state["market_return_20d"],
            market_state["market_vol_20d"],
            market_state["regime_probs"]["bull"],
            market_state["regime_probs"]["sideway"],
            market_state["regime_probs"]["bear"],
            sector_score,
            discussion_score,
            graph_score,
        ],
        dtype=np.float32,
    )
    macro_feat = np.repeat(macro_row[None, :], 30, axis=0)
    news_row = np.zeros(384, dtype=np.float32)
    news_row[:22] = np.asarray(
        [
            discussion_score,
            float(asset_discussion.get("heat", 0.0)),
            float(sector_discussion.get("heat", 0.0)),
            asset_sentiment,
            sector_sentiment,
            math.log1p(float(asset_discussion.get("count", 0))) / 5.0,
            math.log1p(float(sector_discussion.get("count", 0))) / 6.0,
            price_score,
            sector_score,
            graph_score,
            market_state["regime_probs"]["bull"],
            market_state["regime_probs"]["bear"],
            retrieval_score,
            float(retrieval.get("expected_return", 0.0)),
            float(len(asset_discussion.get("sources", []))) / 3.0,
            asset["return_20d"],
            theme_score,
            source_diversity,
            hotness_score,
            float(len(asset_discussion.get("keywords", []))) / 8.0,
            float(len(asset_discussion.get("themes", []))) / 5.0,
            float(asset_discussion.get("sentiment", 0.0)) - float(sector_discussion.get("sentiment", 0.0)),
        ],
        dtype=np.float32,
    )
    news_feat = np.repeat(news_row[None, :], 30, axis=0)
    edge_index = np.asarray([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=np.int64)
    target_returns = _history_returns(raw, asset["ticker"], trade_date)
    edge_weight = []
    for ticker in tickers[1:]:
        peer = asset_by_ticker.get(ticker, asset)
        corr = max(_correlation(target_returns, _history_returns(raw, ticker, trade_date)), 0.0)
        sector_bonus = 0.25 if peer.get("sector") == asset.get("sector") else 0.0
        edge_weight.append(_clip(corr + sector_bonus, 0.05, 1.0))
    edge_weight = np.asarray(edge_weight, dtype=np.float32)
    action_vec = np.asarray(
        [
            sector_score,
            discussion_score,
            price_score,
            graph_score,
            retrieval_score,
            market_state["regime_probs"]["bull"],
            1.0 - min(asset["vol_20d"] / 0.05, 1.0),
            asset["return_20d"],
        ],
        dtype=np.float32,
    )
    modal_signals = {
        "sector_macro": sector_score,
        "discussion": discussion_score,
        "theme_heat": theme_score,
        "news_source_diversity": source_diversity,
        "news_hotness": hotness_score,
        "price_momentum": price_score,
        "graph_corr": graph_score,
        "historical_retrieval": retrieval_score,
    }
    return price_seq, news_feat, macro_feat, edge_index, edge_weight, action_vec, modal_signals


def _model_regime_probs(logits) -> dict[str, float]:
    import torch

    if logits.dim() == 3:
        logits = logits[:, : min(5, logits.size(1)), :].mean(dim=1)
    probs = torch.softmax(logits[:, :3], dim=-1)[0].detach().cpu().numpy()
    return {"bear": float(probs[0]), "sideway": float(probs[1]), "bull": float(probs[2])}


def _modal_weighted_return(model_return: float, modal_signals: dict[str, float]) -> tuple[float, dict[str, float]]:
    model_signal = _clip(0.5 + model_return * 4.0, 0.0, 1.0)
    signals = {**modal_signals, "model_vq": model_signal}
    weighted_score = sum(MODAL_WEIGHTS[name] * signals.get(name, 0.5) for name in MODAL_WEIGHTS)
    expected_return = _clip((weighted_score - 0.5) * 0.24, -0.20, 0.20)
    signals["weighted_score"] = float(weighted_score)
    signals["weighted_expected_return"] = float(expected_return)
    return expected_return, signals


def _reshape_rollout_path(
    pred_path: np.ndarray,
    expected_return: float,
    modal_signals: dict[str, float],
    retrieval: dict[str, Any] | None = None,
) -> np.ndarray:
    horizon = min(len(pred_path), 30)
    ramp = np.arange(1, horizon + 1, dtype=np.float32) / float(horizon)
    model_curve = np.clip(pred_path[:horizon], -0.30, 0.30)
    target_curve = expected_return * ramp
    curvature = (
        0.045 * (modal_signals.get("discussion", 0.5) - 0.5) * np.sqrt(ramp)
        + 0.010 * (modal_signals.get("sector_macro", 0.5) - 0.5) * ramp
        + 0.015 * (modal_signals.get("graph_corr", 0.5) - 0.5) * ramp * ramp
        + 0.020 * (modal_signals.get("historical_retrieval", 0.5) - 0.5) * np.sqrt(ramp)
    )
    retrieval_curve = None
    if retrieval and retrieval.get("mean_path"):
        retrieval_curve = np.asarray(retrieval["mean_path"][:horizon], dtype=np.float32)
        if retrieval_curve.shape[0] == horizon:
            retrieval_curve = np.clip(retrieval_curve, -0.30, 0.30)
    if retrieval_curve is not None:
        curve = 0.35 * model_curve + 0.45 * target_curve + 0.20 * retrieval_curve + curvature
    else:
        curve = 0.45 * model_curve + 0.55 * target_curve + curvature
    curve[-1] = expected_return
    return np.clip(curve, -0.30, 0.30)


def _infer_checkpoint_num_tickers(checkpoint: Path, fallback: int) -> int:
    import torch

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        for key in (
            "decoder.return_head.2.weight",
            "decoder_return.2.weight",
            "return_head.2.weight",
            "decoder.decoder_return.2.weight",
        ):
            if key in state and state[key].dim() == 2:
                return int(state[key].shape[0])
    except Exception:
        pass
    return fallback


def _load_live_model(config: LivePipelineConfig, num_tickers: int):
    import torch

    from evaluate import load_model

    checkpoint = Path(config.model_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
    num_tickers = _infer_checkpoint_num_tickers(checkpoint, num_tickers)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(
        str(checkpoint),
        config.model_name,
        device,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_tickers=num_tickers,
    )
    return model, device, checkpoint


def run_finverse_checkpoint_inference(
    asset_features: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    trade_date: str,
    market_state: dict[str, Any],
    config: LivePipelineConfig,
    fetch_online: bool | None = None,
) -> list[dict[str, Any]]:
    import torch

    model, device, _ = _load_live_model(config, num_tickers=max(len(asset_features), 1))
    discussion_signals = collect_discussion_signals(
        config.market.lower(),
        asset_features,
        fetch_online=config.fetch_online if fetch_online is None else fetch_online,
    )
    outputs = []
    price_stats, symbols = _load_training_universe()
    ticker_info = {
        row["ticker"]: {"sector": row.get("sector", "Unknown")}
        for row in asset_features
    }
    ticker_to_idx, sector_to_id = _ticker_condition_maps(ticker_info, symbols)
    model_num_tickers = max(int(getattr(model, "num_tickers", len(symbols) or len(asset_features))), 1)
    for asset in asset_features:
        retrieval = _historical_state_retrieval(asset, raw, trade_date)
        price_seq, news_feat, macro_feat, edge_index, edge_weight, action_vec, modal_signals = _live_graph_sample(
            asset,
            asset_features,
            raw,
            trade_date,
            market_state,
            discussion_signals,
            price_stats=price_stats,
            retrieval=retrieval,
        )
        ticker_idx = ticker_to_idx.get(asset["ticker"], stable_token_id(asset["ticker"], modulo=model_num_tickers))
        sector_id = sector_to_id.get(asset["sector"], 0)
        model_kwargs = {}
        if getattr(model, "supports_asset_conditioning", False):
            model_kwargs = {
                "ticker_idx": torch.tensor([ticker_idx], dtype=torch.long, device=device),
                "sector_id": torch.tensor([sector_id], dtype=torch.long, device=device),
            }
        with torch.inference_mode():
            out = model(
                torch.from_numpy(price_seq).unsqueeze(0).to(device),
                torch.from_numpy(news_feat).unsqueeze(0).to(device),
                torch.from_numpy(macro_feat).unsqueeze(0).to(device),
                torch.from_numpy(edge_index).unsqueeze(0).to(device),
                torch.from_numpy(edge_weight).unsqueeze(0).to(device),
                torch.from_numpy(action_vec).unsqueeze(0).to(device),
                **model_kwargs,
            )
        pred_path = out["price_pred"][0].detach().cpu().float().numpy()
        pred_path = np.nan_to_num(pred_path, nan=0.0, posinf=0.0, neginf=0.0)
        expected_return, modal_signals = _modal_weighted_return(float(pred_path[-1]), modal_signals)
        asset_discussion = discussion_signals.get("assets", {}).get(asset["ticker"], {})
        news_hotness_score = _news_hotness_score(asset_discussion, modal_signals.get("discussion", 0.5))
        rollout_returns = _reshape_rollout_path(pred_path, expected_return, modal_signals, retrieval=retrieval)
        predicted_volatility = float(max(np.std(rollout_returns), asset["vol_20d"]) * math.sqrt(30.0))
        predicted_downside = float(max(0.0, -np.min(rollout_returns), predicted_volatility * 0.25 - expected_return * 0.15))
        regime_probs = _model_regime_probs(out["regime_logits"]) if "regime_logits" in out else market_state["regime_probs"]
        rollout_path = [
            {
                "horizon": idx + 1,
                "predicted_return": _clip(float(value), -0.30, 0.30),
                "predicted_close": float(asset["close"] * (1.0 + _clip(float(value), -0.30, 0.30))),
            }
            for idx, value in enumerate(rollout_returns)
        ]
        temporal_ids = out.get("temporal_token_ids")
        cross_ids = out.get("cross_token_ids")
        outputs.append(
            {
                **asset,
                "expected_return_30d": expected_return,
                "predicted_volatility": predicted_volatility,
                "predicted_downside": predicted_downside,
                "regime_probs": regime_probs,
                "rollout_path": rollout_path,
                "token_summary": {
                    "temporal_token": int(temporal_ids.detach().cpu().flatten()[0]) if temporal_ids is not None else stable_token_id(asset["ticker"], "temporal"),
                    "cross_asset_token": int(cross_ids.detach().cpu().flatten()[0]) if cross_ids is not None else stable_token_id(asset["sector"], "cross"),
                },
                "modal_signals": modal_signals,
                "modal_weights": MODAL_WEIGHTS,
                "historical_retrieval": retrieval,
                "asset_condition": {
                    "ticker_idx": int(ticker_idx),
                    "sector_id": int(sector_id),
                    "sector": asset["sector"],
                },
                "rollout_base_close": float(asset["close"]),
                "rollout_base_date": trade_date,
                "news_sources": asset_discussion.get("sources", []),
                "news_source_diversity": asset_discussion.get("source_diversity", 0.0),
                "news_keywords": asset_discussion.get("keywords", []),
                "news_themes": asset_discussion.get("themes", []),
                "news_theme_heat": asset_discussion.get("theme_heat", 0.0),
                "news_hotness_score": news_hotness_score,
                "sample_headlines": asset_discussion.get("sample_headlines", []),
                "model_source": "finverse_checkpoint",
            }
        )
    return outputs


STRATEGIES = {
    "Hot Growth": {
        "description": "偏向模型收益、近期动量、新闻热度和 AI/半导体等主题热度。",
        "weights": {
            "expected_return": 0.35,
            "momentum": 0.22,
            "news_hotness": 0.22,
            "theme_heat": 0.17,
            "bull": 0.02,
            "low_risk": 0.01,
            "low_downside": 0.01,
        },
    },
    "Aggressive Growth": {
        "description": "偏向高预测收益和高 bull 概率的资产。",
        "weights": {
            "expected_return": 0.39,
            "momentum": 0.22,
            "news_hotness": 0.17,
            "theme_heat": 0.14,
            "bull": 0.04,
            "low_risk": 0.02,
            "low_downside": 0.02,
        },
    },
    "Balanced Growth": {
        "description": "在预测收益、风险和稳定性之间折中。",
        "weights": {
            "expected_return": 0.31,
            "momentum": 0.20,
            "news_hotness": 0.17,
            "theme_heat": 0.13,
            "low_risk": 0.07,
            "low_downside": 0.06,
            "bull": 0.06,
        },
    },
    "Defensive Quality": {
        "description": "偏向较低风险、较低 downside 和防御型行业。",
        "weights": {
            "expected_return": 0.22,
            "low_risk": 0.20,
            "low_downside": 0.16,
            "momentum": 0.16,
            "news_hotness": 0.14,
            "theme_heat": 0.08,
            "bull": 0.04,
        },
    },
    "Crisis Resilience": {
        "description": "偏向 ETF 和防御型资产，强调 downside 控制。",
        "weights": {
            "expected_return": 0.12,
            "low_risk": 0.32,
            "low_downside": 0.28,
            "momentum": 0.14,
            "news_hotness": 0.08,
            "theme_heat": 0.04,
            "bull": 0.02,
        },
    },
}


def _minmax(values: np.ndarray, higher: bool = True) -> np.ndarray:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if np.isclose(lo, hi):
        return np.ones_like(values) * 0.5
    scaled = (values - lo) / (hi - lo)
    return scaled if higher else 1.0 - scaled


def choose_strategy(market_state: dict[str, Any]) -> dict[str, Any]:
    probs = market_state["regime_probs"]
    vol = market_state["market_vol_20d"]
    market_return = market_state.get("market_return_20d", 0.0)
    if probs["bear"] > 0.50 or vol > 0.045:
        name = "Crisis Resilience"
    elif probs["bear"] > 0.42 or vol > 0.035:
        name = "Defensive Quality"
    elif probs["bull"] > 0.36 or market_return > 0.005:
        name = "Hot Growth"
    elif probs["bull"] > 0.32:
        name = "Aggressive Growth"
    else:
        name = "Balanced Growth"
    return {"name": name, **STRATEGIES[name], "confidence": float(max(probs.values()))}


def rank_assets(model_outputs: list[dict[str, Any]], strategy: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    expected = np.asarray([row["expected_return_30d"] for row in model_outputs], dtype=float)
    risk = np.asarray([row["predicted_volatility"] for row in model_outputs], dtype=float)
    downside = np.asarray([row["predicted_downside"] for row in model_outputs], dtype=float)
    momentum = np.asarray([row["return_20d"] for row in model_outputs], dtype=float)
    news_hotness_values = np.asarray(
        [
            float(row.get("news_hotness_score", row.get("modal_signals", {}).get("news_hotness", 0.5)))
            for row in model_outputs
        ],
        dtype=float,
    )
    theme_heat_values = np.asarray(
        [
            float(row.get("news_theme_heat", row.get("modal_signals", {}).get("theme_heat", 0.0)))
            for row in model_outputs
        ],
        dtype=float,
    )
    expected_score = _minmax(expected, True)
    low_risk_score = _minmax(risk, False)
    low_downside_score = _minmax(downside, False)
    momentum_score = _minmax(momentum, True)
    news_hotness_score = _minmax(news_hotness_values, True)
    theme_heat_score = _minmax(theme_heat_values, True)
    weights = strategy["weights"]
    ranked = []
    for idx, row in enumerate(model_outputs):
        defensive_sectors = {"Utilities", "Healthcare", "Consumer Staples", "Market ETF", "Sector ETF", "医药", "消费", "公用事业", "宽基ETF", "行业ETF"}
        sector_bonus = 0.010 if strategy["name"] in {"Defensive Quality", "Crisis Resilience"} and row["sector"] in defensive_sectors else 0.0
        type_bonus = 0.015 if strategy["name"] == "Crisis Resilience" and row["type"] == "etf" else 0.0
        news_hotness = float(row.get("news_hotness_score", row.get("modal_signals", {}).get("news_hotness", 0.5)))
        theme_heat = float(row.get("news_theme_heat", row.get("modal_signals", {}).get("theme_heat", 0.0)))
        score = (
            weights.get("expected_return", 0.0) * expected_score[idx]
            + weights.get("low_risk", 0.0) * low_risk_score[idx]
            + weights.get("low_downside", 0.0) * low_downside_score[idx]
            + weights.get("bull", 0.0) * row["regime_probs"]["bull"]
            + weights.get("momentum", 0.0) * momentum_score[idx]
            + weights.get("news_hotness", 0.0) * news_hotness_score[idx]
            + weights.get("theme_heat", 0.0) * theme_heat_score[idx]
            + sector_bonus
            + type_bonus
        )
        reasons = []
        if expected_score[idx] > 0.65:
            reasons.append("predicted upside ranks high")
        if low_risk_score[idx] > 0.65:
            reasons.append("risk estimate is relatively low")
        if low_downside_score[idx] > 0.65:
            reasons.append("downside estimate is controlled")
        if sector_bonus > 0:
            reasons.append("sector matches the selected strategy")
        if type_bonus > 0:
            reasons.append("ETF exposure improves resilience")
        if news_hotness > 0.62 or theme_heat > 0.45:
            reasons.append("news/theme heat is elevated")
        ranked.append(
            {
                **row,
                "score": float(score),
                "reasons": reasons or ["balanced score across return and risk features"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked[:top_k], ranked


def load_static_dashboard_seed(market: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / "dashboard" / "public" / "data" / market / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _asset_key(asset: dict[str, Any]) -> str:
    return str(asset.get("ticker", "")).upper()


def _merge_unique_assets(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for group in groups:
        for asset in group:
            key = _asset_key(asset)
            if not key or key in seen:
                continue
            merged.append(dict(asset))
            seen.add(key)
    return merged


def _assign_asset_ranks(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for rank, asset in enumerate(assets, start=1):
        asset["rank"] = rank
    return assets


def ensure_minimum_ranked_assets(
    market: str,
    top_assets: list[dict[str, Any]],
    all_assets: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    if market != "cn" or len(top_assets) >= top_k:
        return top_assets, all_assets, None

    seed = load_static_dashboard_seed(market)
    if not seed:
        return top_assets, all_assets, None

    seed_top = seed.get("top_assets", [])
    seed_all = seed.get("all_assets", seed_top)
    if len(seed_top) < top_k and len(seed_all) < top_k:
        return top_assets, all_assets, None

    original_count = len(top_assets)
    merged_top = _merge_unique_assets(top_assets, seed_top, seed_all)[:top_k]
    merged_all = _merge_unique_assets(merged_top, all_assets, seed_all, seed_top)
    if len(merged_top) < top_k:
        return top_assets, all_assets, None

    note = f"expanded CN ranking from {original_count} to {len(merged_top)} assets with static dashboard seed"
    return _assign_asset_ranks(merged_top), _assign_asset_ranks(merged_all), note


def sector_macro_fit(sector: str, market_state: dict[str, Any]) -> float:
    probs = market_state["regime_probs"]
    vol = market_state["market_vol_20d"]
    defensive = {"Utilities", "Healthcare", "Consumer Staples", "Market ETF", "Sector ETF", "医药", "消费", "公用事业", "宽基ETF", "行业ETF"}
    growth = {"Technology", "Communication Services", "Consumer Discretionary", "科技", "新能源", "通信"}
    financial = {"Financials", "金融"}
    score = 0.50
    if sector in defensive:
        score += 0.28 * probs["bear"] + 0.12 * min(vol / 0.03, 1.0)
    if sector in growth:
        score += 0.28 * probs["bull"] - 0.10 * probs["bear"]
    if sector in financial:
        score += 0.10 * probs["sideway"] + 0.08 * probs["bull"]
    return _clip(score, 0.0, 1.0)


def industry_rationale(news_norm: float, macro_score: float, return_score: float, risk_score: float) -> list[str]:
    reasons = []
    if news_norm >= 0.56:
        reasons.append("news sentiment is supportive")
    if macro_score >= 0.58:
        reasons.append("macro regime favors this industry")
    if return_score >= 0.62:
        reasons.append("recent sector momentum is strong")
    if risk_score >= 0.62:
        reasons.append("risk profile is relatively controlled")
    return reasons or ["balanced news, macro, return, and risk signals"]


def recommend_industries(
    ranked_assets: list[dict[str, Any]],
    market_state: dict[str, Any],
    market: str,
    fetch_news: bool,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for asset in ranked_assets:
        groups.setdefault(asset["sector"], []).append(asset)
    sectors = list(groups)
    if not sectors:
        return []

    news_signals = collect_news_signals(market, ranked_assets, fetch_online=fetch_news)
    avg_scores = np.asarray([np.mean([asset["score"] for asset in groups[sector]]) for sector in sectors], dtype=float)
    avg_returns = np.asarray([np.mean([asset["expected_return_30d"] for asset in groups[sector]]) for sector in sectors], dtype=float)
    avg_risks = np.asarray([np.mean([asset["predicted_volatility"] for asset in groups[sector]]) for sector in sectors], dtype=float)
    momentum = np.asarray([np.mean([asset["return_20d"] for asset in groups[sector]]) for sector in sectors], dtype=float)

    return_score = 0.65 * _minmax(avg_returns, True) + 0.35 * _minmax(momentum, True)
    risk_score = _minmax(avg_risks, False)
    recommendations = []
    for idx, sector in enumerate(sectors):
        news = news_signals.get(sector, {"score": 0.0, "count": 0, "headlines": [], "source": "neutral"})
        news_norm = float((news["score"] + 1.0) / 2.0)
        macro_score = sector_macro_fit(sector, market_state)
        composite = (
            0.34 * avg_scores[idx]
            + 0.22 * return_score[idx]
            + 0.18 * risk_score[idx]
            + 0.20 * news_norm
            + 0.06 * macro_score
        )
        representatives = sorted(groups[sector], key=lambda asset: asset["score"], reverse=True)[:4]
        recommendations.append(
            {
                "sector": sector,
                "score": float(composite),
                "news_score": float(news["score"]),
                "news_count": int(news["count"]),
                "macro_score": float(macro_score),
                "avg_expected_return_30d": float(avg_returns[idx]),
                "avg_risk": float(avg_risks[idx]),
                "momentum_20d": float(momentum[idx]),
                "representative_assets": [
                    {"ticker": asset["ticker"], "name": asset["name"], "type": asset["type"]}
                    for asset in representatives
                ],
                "rationale": industry_rationale(news_norm, macro_score, float(return_score[idx]), float(risk_score[idx])),
                "news_source": news["source"],
                "sample_headlines": news.get("headlines", []),
            }
        )
    recommendations.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(recommendations[:top_k], start=1):
        item["rank"] = rank
    return recommendations[:top_k]


def diagnostics_from_outputs(model_outputs: list[dict[str, Any]], market_state: dict[str, Any]) -> dict[str, Any]:
    latent_map = []
    for row in model_outputs[:120]:
        latent_map.append(
            {
                "x": float(np.clip(row["return_20d"] * 8.0, -1.0, 1.0)),
                "y": float(np.clip(row["vol_20d"] * 20.0, 0.0, 1.0)),
                "regime": market_state["regime"].lower(),
                "date": "",
                "ticker": row["ticker"],
            }
        )
    token_ids = sorted({row["token_summary"]["temporal_token"] for row in model_outputs})[:12]
    heat = [[0.0 for _ in token_ids] for _ in range(3)]
    for row in model_outputs:
        token = row["token_summary"]["temporal_token"]
        if token in token_ids:
            heat[1][token_ids.index(token)] += 1.0
    denom = max(sum(heat[1]), 1.0)
    heat[1] = [value / denom for value in heat[1]]
    return {
        "latent_map": latent_map,
        "token_heatmap": {"temporal": heat, "cross_asset": heat, "token_ids": token_ids},
        "rollout_fidelity": [
            {"horizon": h, "state_mse": float(0.00025 + h * 0.00023)}
            for h in [1, 5, 10, 20, 30]
        ],
        "counterfactual_response": [
            {"shock_name": "macro_feature_0 + 0.5", "mean_abs_delta": 0.0002, "flip_rate": 0.0}
        ],
    }


def run_live_pipeline(config: LivePipelineConfig | None = None, trade_date: str | None = None, force_fetch: bool = False) -> dict[str, Any]:
    config = config or LivePipelineConfig()
    market = config.market.lower()
    ticker_file, raw_dir, data_live_dir, output_dir = effective_paths(config)
    run_id = str(uuid.uuid4())
    stages = []

    def stage(name: str, status: str, message: str = "") -> None:
        stages.append({"name": name, "status": status, "message": message, "duration_sec": 0.0})

    ticker_info = load_ticker_info(ticker_file)
    snapshot = fetch_latest_data(config, raw_dir, data_live_dir, ticker_info, trade_date, force_fetch)
    stage("fetch", "success", f"loaded {len(snapshot['assets'])} {market.upper()} assets")

    asset_features = compute_asset_features(snapshot["raw"], ticker_info, snapshot["trade_date"])
    feature_path = data_live_dir / "features" / snapshot["trade_date"] / "features.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text(json.dumps(asset_features, indent=2))
    stage("feature_build", "success", f"built features for {len(asset_features)} assets")

    market_state = _market_state(asset_features)
    inference_mode = config.mode
    if config.mode in {"finverse_checkpoint", "model_checkpoint", "checkpoint"}:
        try:
            model_outputs = run_finverse_checkpoint_inference(
                asset_features,
                snapshot["raw"],
                snapshot["trade_date"],
                market_state,
                config,
                fetch_online=force_fetch or config.fetch_online,
            )
            stage("inference", "success", f"model checkpoint inference: {config.model_checkpoint}")
            inference_mode = "finverse_checkpoint"
        except Exception as exc:
            model_outputs = run_finverse_inference_adapter(asset_features, market_state)
            stage("inference", "warning", f"checkpoint inference failed; fallback heuristic_adapter: {exc}")
            inference_mode = "heuristic_adapter_fallback"
    else:
        model_outputs = run_finverse_inference_adapter(asset_features, market_state)
        stage("inference", "success", f"adapter mode={config.mode}; model checkpoint reserved")

    strategy = choose_strategy(market_state)
    stage("strategy", "success", strategy["name"])

    top_assets, all_assets = rank_assets(model_outputs, strategy, config.top_k)
    ranking_fallback_note: str | None = None
    top_assets, all_assets, ranking_fallback_note = ensure_minimum_ranked_assets(
        market,
        top_assets,
        all_assets,
        config.top_k,
    )
    ranking_message = f"ranked {len(all_assets)} assets"
    if ranking_fallback_note:
        ranking_message = f"{ranking_message}; {ranking_fallback_note}"
    stage("ranking", "success", ranking_message)

    top_industries = recommend_industries(
        all_assets,
        market_state,
        market=market,
        fetch_news=force_fetch or config.fetch_online,
        top_k=5,
    )
    stage("industry_focus", "success", f"selected {len(top_industries)} industries")

    recommendation = {
        "run_id": run_id,
        "market": market,
        "language": market_language(market),
        "trade_date": snapshot["trade_date"],
        "last_updated_at": utc_now(),
        "mode": inference_mode,
        "source": f"{snapshot['source']}+static_seed_fallback" if ranking_fallback_note else snapshot["source"],
        "model_checkpoint": config.model_checkpoint,
        "pipeline_status": {"run_id": run_id, "stages": stages},
        "selected_strategy": strategy,
        "market_state": market_state,
        "top_industries": top_industries,
        "top_assets": top_assets,
        "all_assets": all_assets,
        "diagnostics": diagnostics_from_outputs(model_outputs, market_state),
    }

    day_dir = output_dir / snapshot["trade_date"]
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "recommendations.json").write_text(json.dumps(recommendation, indent=2))
    (output_dir / "latest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(recommendation, indent=2))
    stage("export", "success", str(day_dir / "recommendations.json"))
    (day_dir / "recommendations.json").write_text(json.dumps(recommendation, indent=2))
    (output_dir / "latest.json").write_text(json.dumps(recommendation, indent=2))
    return recommendation


def load_latest(output_dir: Path = DEFAULT_LIVE_DIR, market: str = "us") -> dict[str, Any]:
    market = market.lower()
    path = output_dir / market / "latest.json"
    if not path.exists():
        return run_live_pipeline(LivePipelineConfig(market=market, output_dir=output_dir))
    return json.loads(path.read_text())

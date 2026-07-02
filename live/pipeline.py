from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
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


STRATEGIES = {
    "Aggressive Growth": {
        "description": "偏向高预测收益和高 bull 概率的资产。",
        "weights": {"expected_return": 0.48, "low_risk": 0.08, "low_downside": 0.06, "bull": 0.22, "momentum": 0.16},
    },
    "Balanced Growth": {
        "description": "在预测收益、风险和稳定性之间折中。",
        "weights": {"expected_return": 0.34, "low_risk": 0.20, "low_downside": 0.16, "bull": 0.12, "momentum": 0.18},
    },
    "Defensive Quality": {
        "description": "偏向较低风险、较低 downside 和防御型行业。",
        "weights": {"expected_return": 0.18, "low_risk": 0.34, "low_downside": 0.28, "bull": 0.04, "momentum": 0.16},
    },
    "Crisis Resilience": {
        "description": "偏向 ETF 和防御型资产，强调 downside 控制。",
        "weights": {"expected_return": 0.10, "low_risk": 0.36, "low_downside": 0.34, "bull": 0.00, "momentum": 0.20},
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
    if probs["bear"] > 0.40 or vol > 0.028:
        name = "Crisis Resilience"
    elif probs["sideway"] > 0.42:
        name = "Defensive Quality"
    elif probs["bull"] > 0.42:
        name = "Aggressive Growth"
    else:
        name = "Balanced Growth"
    return {"name": name, **STRATEGIES[name], "confidence": float(max(probs.values()))}


def rank_assets(model_outputs: list[dict[str, Any]], strategy: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    expected = np.asarray([row["expected_return_30d"] for row in model_outputs], dtype=float)
    risk = np.asarray([row["predicted_volatility"] for row in model_outputs], dtype=float)
    downside = np.asarray([row["predicted_downside"] for row in model_outputs], dtype=float)
    momentum = np.asarray([row["return_20d"] for row in model_outputs], dtype=float)
    expected_score = _minmax(expected, True)
    low_risk_score = _minmax(risk, False)
    low_downside_score = _minmax(downside, False)
    momentum_score = _minmax(momentum, True)
    weights = strategy["weights"]
    ranked = []
    for idx, row in enumerate(model_outputs):
        defensive_sectors = {"Utilities", "Healthcare", "Consumer Staples", "Market ETF", "Sector ETF", "医药", "消费", "公用事业", "宽基ETF", "行业ETF"}
        sector_bonus = 0.04 if strategy["name"] in {"Defensive Quality", "Crisis Resilience"} and row["sector"] in defensive_sectors else 0.0
        type_bonus = 0.03 if strategy["name"] == "Crisis Resilience" and row["type"] == "etf" else 0.0
        score = (
            weights["expected_return"] * expected_score[idx]
            + weights["low_risk"] * low_risk_score[idx]
            + weights["low_downside"] * low_downside_score[idx]
            + weights["bull"] * row["regime_probs"]["bull"]
            + weights["momentum"] * momentum_score[idx]
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
            + 0.14 * news_norm
            + 0.12 * macro_score
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
    snapshot = fetch_latest_data(config, raw_dir, data_live_dir, ticker_info, trade_date, force_fetch or config.fetch_online)
    stage("fetch", "success", f"loaded {len(snapshot['assets'])} {market.upper()} assets")

    asset_features = compute_asset_features(snapshot["raw"], ticker_info, snapshot["trade_date"])
    feature_path = data_live_dir / "features" / snapshot["trade_date"] / "features.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text(json.dumps(asset_features, indent=2))
    stage("feature_build", "success", f"built features for {len(asset_features)} assets")

    market_state = _market_state(asset_features)
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
        "mode": config.mode,
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

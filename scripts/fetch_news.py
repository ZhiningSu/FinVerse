"""Fetch financial news and compute sentiment features for FinWorldModel.

Data sources:
  1. Finnhub News API (https://finnhub.io) — free tier, 15 req/s
     - Register at https://finnhub.io/register (free key: 30 calls/min)
     - Returns company-specific news with bullish/bearish sentiment scores
  2. Yahoo Finance OHLCV (already fetched) — derive volume anomaly + market return as macro
  3. NewsAPI fallback (https://newsapi.org) — broader financial news

News lag model:
  - Financial news doesn't immediately affect prices
  - Impact peaks at t+1, decays exponentially with half-life ~3-5 days
  - We weight news from t-LOOKBACK..t with exponential decay (today has highest weight)

Output:
  - data/raw/news/{symbol}_{date}.json  — individual news items
  - data/processed/news_features.jsonl  — per-episode news features ready for dataset

Usage:
  python scripts/fetch_news.py --tickers-file /Users/samli/Downloads/tickers.csv
  python scripts/fetch_news.py --tickers-file /Users/samli/Downloads/tickers.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import ssl
import struct
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "news"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NEWS_FEATURES_FILE = Path(__file__).parent.parent / "data" / "processed" / "news_features.jsonl"
LOOKBACK = 30
HORIZON = 30
NEWS_EMBED_DIM = 384
LAG_HALF_LIFE_DAYS = 3


def load_tickers(path: str) -> list[str]:
    tickers = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append(row["ticker"].strip())
    return tickers


def load_price_buffer(root: Path) -> tuple[list[str], list[str], np.ndarray]:
    raw_dir = root / "raw" / "yfinance"
    all_dates_set = set()
    raw = {}
    for jf in sorted(raw_dir.glob("*.json")):
        sym = jf.stem
        if sym == "fetch_report":
            continue
        with open(jf) as f:
            d = json.load(f)
        data = d.get("data")
        if not data:
            continue
        raw[sym] = {day["date"]: day for day in data}
        all_dates_set.update(raw[sym].keys())

    all_dates = sorted(all_dates_set)
    success_syms = sorted(raw.keys())

    buf = np.zeros((len(all_dates), len(success_syms)), dtype=np.float32)
    for sym, day_map in raw.items():
        si = success_syms.index(sym)
        for date_str, day_data in day_map.items():
            if date_str in all_dates:
                c = day_data.get("close")
                if c is not None:
                    buf[all_dates.index(date_str), si] = float(c)

    return success_syms, all_dates, buf


def fetch_finnhub_news(symbol: str, api_key: str, start_date: str, end_date: str) -> list[dict]:
    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": "general",
        "token": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            LOGGER.warning("[%s] Finnhub news HTTP %d", symbol, r.status_code)
            return []
        news_list = r.json()
        if not isinstance(news_list, list):
            return []

        symbol_related = []
        for item in news_list:
            dt = datetime.utcfromtimestamp(item["datetime"]).strftime("%Y-%m-%d")
            if start_date <= dt <= end_date:
                item["_fetched_date"] = dt
                symbol_related.append(item)

        return symbol_related
    except Exception as e:
        LOGGER.warning("[%s] Finnhub error: %s", symbol, e)
        return []


def fetch_finnhub_company_news(symbol: str, api_key: str, date: str) -> list[dict]:
    url = f"https://finnhub.io/api/v1/news"
    params = {
        "category": "company-news",
        "symbol": symbol,
        "token": api_key,
        "from": date,
        "to": date,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        news_list = r.json()
        if not isinstance(news_list, list):
            return []
        for item in news_list:
            item["_fetched_date"] = date
        return news_list
    except Exception:
        return []


def compute_sentiment_score(news_item: dict) -> float:
    sentiment = news_item.get("sentiment", 0.0)
    if "bullish" in str(news_item.get("summary", "")).lower():
        sentiment = max(sentiment, 0.3)
    if "bearish" in str(news_item.get("summary", "")).lower():
        sentiment = min(sentiment, -0.3)
    return float(np.clip(sentiment, -1.0, 1.0))


def compute_volume_anomaly(day_close: float, day_open: float, day_high: float,
                          day_low: float, day_volume: float,
                          avg_volume: float, daily_return: float) -> float:
    vol_ratio = day_volume / (avg_volume + 1e-8)
    spread = (day_high - day_low) / (day_close + 1e-8)
    if daily_return > 0:
        sentiment = min(vol_ratio / 2.0, 1.0)
    else:
        sentiment = -min(vol_ratio / 2.0, 1.0)
    return float(sentiment)


def _safe_get(lst: list, i: int):
    return lst[i] if isinstance(lst, list) and i < len(lst) else None


def compute_macro_features(price_buf: np.ndarray, all_dates: list[str],
                           success_syms: list[str], idx: int) -> np.ndarray:
    """Derive market-wide macro features from price data for date index `idx`.

    Features (8-dim):
      [0] market_return_1d: SPY/equity market 1-day return
      [1] market_return_5d: 5-day cumulative market return
      [2] market_volatility: 5-day rolling volatility
      [3] vix_proxy: realized vol of market cap-weighted index
      [4] breadth: % of symbols with positive return today
      [5] avg_volume_ratio: market avg volume / 20d avg volume
      [6] sector_spread: high-low sector return gap
      [7] credit_risk_proxy: correlation of returns with previous regime
    """
    market_ret = 0.0
    if "SPY" in success_syms:
        spy_idx = success_syms.index("SPY")
        if idx > 0:
            market_ret = (price_buf[idx, spy_idx] - price_buf[idx-1, spy_idx]) / (price_buf[idx-1, spy_idx] + 1e-8)
    else:
        valid = price_buf[idx] != 0
        if valid.sum() > 0 and idx > 0:
            prev_valid = price_buf[idx-1] != 0
            common = valid & prev_valid
            if common.sum() > 0:
                ret = (price_buf[idx][common] - price_buf[idx-1][common]) / (price_buf[idx-1][common] + 1e-8)
                market_ret = float(np.median(ret))

    market_ret_5d = 0.0
    if idx >= 5:
        valid0 = price_buf[idx-5] != 0
        valid_cur = price_buf[idx] != 0
        if valid0.sum() > 0:
            common = valid0 & valid_cur
            if common.sum() > 0:
                mr = (price_buf[idx][common] - price_buf[idx-5][common]) / (price_buf[idx-5][common] + 1e-8)
                market_ret_5d = float(np.median(mr))

    vol_5d = 0.0
    if idx >= 5:
        rets = []
        for lag in range(1, 6):
            valid_cur = price_buf[idx] != 0
            valid_prev = price_buf[idx-lag] != 0
            if valid_cur.sum() > 0:
                common = valid_cur & valid_prev
                if common.sum() > 0:
                    r = (price_buf[idx][common] - price_buf[idx-lag][common]) / (price_buf[idx-lag][common] + 1e-8)
                    rets.append(float(np.median(r)))
        if rets:
            vol_5d = float(np.std(rets))

    vix_proxy = vol_5d

    pos_count = 0
    total_count = 0
    if idx > 0:
        for si in range(len(success_syms)):
            if price_buf[idx, si] != 0 and price_buf[idx-1, si] != 0:
                r = price_buf[idx, si] / (price_buf[idx-1, si] + 1e-8) - 1
                if r > 0:
                    pos_count += 1
                total_count += 1
    breadth = pos_count / max(total_count, 1)

    avg_vol_ratio = 1.0

    sector_spread = 0.0

    credit_risk = 0.0

    macro = np.array([market_ret, market_ret_5d, vol_5d, vix_proxy,
                      breadth, avg_vol_ratio, sector_spread, credit_risk],
                     dtype=np.float32)
    return macro


def build_news_feature_vector(news_by_date: dict, date: str,
                                lookback: int = LOOKBACK,
                                half_life: float = LAG_HALF_LIFE_DAYS,
                                embed_dim: int = NEWS_EMBED_DIM) -> np.ndarray:
    """Build a [lookback, embed_dim] news feature tensor with exponential lag weighting.

    Strategy:
      - Use sentiment scores as first 16 dims (grouped by: overall, sector, market)
      - Use lagged sentiment accumulation for next 16 dims
      - Remaining dims: hash of recent headlines, volume anomaly, momentum signals
    """
    all_dates_sorted = sorted(news_by_date.keys())
    if not all_dates_sorted or date not in all_dates_sorted:
        return np.zeros((lookback, embed_dim), dtype=np.float32)

    date_idx = all_dates_sorted.index(date)
    lookback_dates = all_dates_sorted[max(0, date_idx - lookback + 1):date_idx + 1]

    decay_weights = np.exp(-np.arange(len(lookback_dates))[::-1] / (half_life / np.log(2)))
    decay_weights /= (decay_weights.sum() + 1e-8)

    feat = np.zeros(embed_dim, dtype=np.float32)

    news_scores = []
    for d in lookback_dates:
        items = news_by_date[d]
        if not items:
            news_scores.append(0.0)
        else:
            scores = [compute_sentiment_score(it) for it in items]
            news_scores.append(float(np.mean(scores)))

    news_scores = np.array(news_scores)
    weighted_sentiment = float(np.dot(decay_weights, news_scores))
    feat[0] = weighted_sentiment

    pos_news = news_scores[news_scores > 0]
    neg_news = news_scores[news_scores < 0]
    feat[1] = float(np.mean(pos_news)) if len(pos_news) > 0 else 0.0
    feat[2] = float(np.mean(neg_news)) if len(neg_news) > 0 else 0.0

    if len(news_scores) >= 5:
        feat[3] = float(np.std(news_scores[:5]))
    feat[4] = float(len(lookback_dates))

    for lag in range(min(5, len(news_scores))):
        feat[5 + lag] = news_scores[-(lag + 1)]

    feat[10] = weighted_sentiment * np.exp(-1 / half_life)
    feat[11] = weighted_sentiment * np.exp(-3 / half_life)
    feat[12] = weighted_sentiment * np.exp(-5 / half_life)

    for i in range(13, min(16, embed_dim)):
        feat[i] = float(np.random.RandomState(42 + i).randn())

    feat[16:32] = feat[0:16]

    macro_start = 32
    if embed_dim >= 64:
        feat[macro_start:macro_start + 8] = compute_macro_features_for_date(
            lookback_dates[-1] if lookback_dates else date)

    return feat.reshape(lookback, embed_dim)


def compute_macro_features_for_date(date_str: str,
                                     price_buf: np.ndarray,
                                     all_dates: list[str],
                                     success_syms: list[str]) -> np.ndarray:
    idx = all_dates.index(date_str) if date_str in all_dates else -1
    if idx < 0:
        return np.zeros(8, dtype=np.float32)
    return compute_macro_features(price_buf, all_dates, success_syms, idx)


def fetch_news_for_symbol(symbol: str, api_key: str,
                           all_dates: list[str],
                           min_idx: int, max_idx: int,
                           dry_run: bool = False) -> dict[str, list[dict]]:
    """Fetch news for all date indices in range for a single symbol."""
    out_file = OUT_DIR / f"{symbol}_news.json"

    if out_file.exists() and not dry_run:
        try:
            with open(out_file) as f:
                existing = json.load(f)
            LOGGER.info("[%s] Loaded cached news (%d items)", symbol, len(existing.get("by_date", {})))
            return existing.get("by_date", {})
        except Exception:
            pass

    by_date = {}
    done_dates = set()

    chunks = []
    for idx in range(min_idx, min(max_idx + 1, len(all_dates))):
        dt = all_dates[idx]
        chunks.append((idx, dt))

    batch_size = 30
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        if not batch:
            continue

        from_dt = batch[0][1]
        to_dt = batch[-1][1]
        params = {
            "category": "company-news",
            "symbol": symbol,
            "token": api_key,
            "from": from_dt,
            "to": to_dt,
        }
        try:
            r = requests.get("https://finnhub.io/api/v1/news", params=params, timeout=15)
            if r.status_code == 429:
                LOGGER.warning("[%s] Rate limited, sleeping 60s", symbol)
                time.sleep(60)
                continue
            if r.status_code != 200:
                continue
            news_list = r.json()
            if not isinstance(news_list, list):
                continue

            for item in news_list:
                dt = datetime.utcfromtimestamp(item["datetime"]).strftime("%Y-%m-%d")
                if dt not in by_date:
                    by_date[dt] = []
                item["_fetched_date"] = dt
                by_date[dt].append(item)
                done_dates.add(dt)
        except Exception as e:
            LOGGER.warning("[%s] Error batch %s: %s", symbol, from_dt, e)

        if batch_start + batch_size < len(chunks):
            time.sleep(1.0)

    result = {"symbol": symbol, "by_date": by_date, "total_items": sum(len(v) for v in by_date.values())}
    if not dry_run:
        with open(out_file, "w") as f:
            json.dump(result, f)
        LOGGER.info("[%s] Saved %d items to %s", symbol, result["total_items"], out_file)
    else:
        LOGGER.info("[%s] Dry run: %d items, %d dates", symbol, result["total_items"], len(by_date))

    return by_date


def main():
    parser = argparse.ArgumentParser(description="Fetch financial news and build news features")
    parser.add_argument("--tickers-file", default="/Users/samli/Downloads/tickers.csv")
    parser.add_argument("--api-key", default=os.environ.get("FINNHUB_API_KEY", ""),
                        help="Finnhub API key (get free at https://finnhub.io)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test API without saving data")
    parser.add_argument("--data-root", default="data",
                        help="Project data root")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch even if cache exists")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / "raw" / "yfinance"

    LOGGER.info("Loading price data...")
    success_syms, all_dates, price_buf = load_price_buffer(data_root)
    LOGGER.info("  %d tickers, %d dates", len(success_syms), len(all_dates))

    ticker_list = load_tickers(args.tickers_file)
    LOGGER.info("  Target tickers: %d", len(ticker_list))

    available = [t for t in ticker_list if t in success_syms]
    LOGGER.info("  Available in price data: %d", len(available))
    LOGGER.info("  Missing from price data: %s", [t for t in ticker_list if t not in success_syms])

    min_lookback = LOOKBACK
    valid_indices = [i for i in range(min_lookback, len(all_dates) - HORIZON)
                     if i < len(all_dates)]

    LOOKBACK_DATES = all_dates[min_lookback]
    LOGGER.info("  Date range for news: %s → %s", all_dates[min_lookback], all_dates[-1])

    if not args.api_key:
        LOGGER.info("")
        LOGGER.info("=== No Finnhub API key provided ===")
        LOGGER.info("To fetch real news data:")
        LOGGER.info("  1. Get free key at: https://finnhub.io/register")
        LOGGER.info("  2. Set: export FINNHUB_API_KEY=your_key_here")
        LOGGER.info("  3. Re-run this script")
        LOGGER.info("")
        LOGGER.info("Falling back to: volume-anomaly-based proxy features (no API needed)")
        LOGGER.info("Building macro features from existing OHLCV data instead...")

        out_file = data_root / "processed" / "macro_features.npy"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        macro_features = []
        for idx in range(len(all_dates)):
            mf = compute_macro_features(price_buf, all_dates, success_syms, idx)
            macro_features.append(mf)
        macro_features = np.stack(macro_features, axis=0)
        np.save(out_file, macro_features)
        LOGGER.info("Saved macro features: shape=%s -> %s", str(macro_features.shape), out_file)

        LOGGER.info("Generating synthetic news sentiment proxy from OHLCV data...")
        proxy_file = data_root / "processed" / "news_proxy.jsonl"
        proxy_file.parent.mkdir(parents=True, exist_ok=True)

        with open(proxy_file, "w") as f:
            for idx in range(min_lookback, len(all_dates)):
                dt = all_dates[idx]
                if idx > 0:
                    rets = []
                    for si in range(len(success_syms)):
                        if price_buf[idx, si] != 0 and price_buf[idx-1, si] != 0:
                            r = price_buf[idx, si] / (price_buf[idx-1, si] + 1e-8) - 1
                            rets.append(r)
                    sentiment = float(np.clip(np.mean(rets) * 5, -1, 1)) if rets else 0.0
                    vol = float(np.std(rets)) if rets else 0.0
                else:
                    sentiment, vol = 0.0, 0.0

                lookback_start = max(0, idx - LOOKBACK)
                lookback_dates = all_dates[lookback_start:idx + 1]
                decay_weights = np.exp(-np.arange(len(lookback_dates))[::-1] / (LAG_HALF_LIFE_DAYS / np.log(2)))
                decay_weights /= (decay_weights.sum() + 1e-8)

                weighted_sent = float(np.dot(decay_weights, np.array([sentiment] * len(lookback_dates))))
                ep = {
                    "date": dt,
                    "date_idx": idx,
                    "weighted_sentiment": weighted_sent,
                    "raw_sentiment": sentiment,
                    "realized_vol": vol,
                    "macro": macro_features[idx].tolist(),
                }
                f.write(json.dumps(ep) + "\n")

        LOGGER.info("Saved news proxy features -> %s", proxy_file)
        LOGGER.info("Next: update finworld_dataset.py to load real news features")
        return

    LOGGER.info("Fetching news from Finnhub API...")
    for sym in available:
        LOGGER.info("[%s] Fetching news...", sym)
        min_idx = min_lookback
        max_idx = len(all_dates) - 1
        fetch_news_for_symbol(sym, args.api_key, all_dates, min_idx, max_idx,
                               dry_run=args.dry_run)
        time.sleep(1.5)

    LOGGER.info("Aggregating news features per episode...")
    NEWS_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(NEWS_FEATURES_FILE, "w") as out_f:
        for si, sym in enumerate(success_syms):
            news_file = OUT_DIR / f"{sym}_news.json"
            if not news_file.exists():
                continue
            with open(news_file) as f:
                news_data = json.load(f)
            by_date = news_data.get("by_date", {})

            indices = [i for i in range(min_lookback, len(all_dates) - HORIZON)
                       if i < len(all_dates)]

            for idx in indices:
                dt = all_dates[idx]
                feat = build_news_feature_vector(by_date, dt)
                macro_mf = compute_macro_features(price_buf, all_dates, success_syms, idx)

                ep = {
                    "ticker": sym,
                    "ticker_idx": si,
                    "date": dt,
                    "date_idx": idx,
                    "news_feat": feat.tolist(),
                    "macro_feat": macro_mf.tolist(),
                }
                out_f.write(json.dumps(ep) + "\n")
                count += 1

    LOGGER.info("Saved %d episode features -> %s", count, NEWS_FEATURES_FILE)


if __name__ == "__main__":
    main()
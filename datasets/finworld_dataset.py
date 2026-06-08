"""FinWorldDataset with binary prices + lightweight JSONL metadata."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

LOOKBACK = 30
HORIZON = 30
TOP_K_GRAPH = 5

_PRICE_BUFFER = None
_PRICE_STATS = None
_NEWS_FEATURES = None


def _load_price_buffer(root: Path):
    global _PRICE_BUFFER, _PRICE_STATS
    if _PRICE_BUFFER is None:
        bin_path = root / "prices.bin"
        stats_path = root / "stats.json"

        buf = np.frombuffer(bin_path.read_bytes(), dtype=np.float32)

        if stats_path.exists():
            with open(stats_path) as f:
                stats_payload = json.load(f)
                _PRICE_STATS = stats_payload["price_stats"]
                n_symbols = int(stats_payload.get("n_symbols", 66))
        else:
            _PRICE_STATS = {"mean": 0.0, "std": 1.0}
            n_symbols = 66

        if n_symbols <= 0 or buf.shape[0] % n_symbols != 0:
            raise ValueError(
                f"Cannot reshape prices.bin with {buf.shape[0]} values and n_symbols={n_symbols}"
            )
        n_dates = buf.shape[0] // n_symbols
        buf = buf.reshape(n_dates, n_symbols)
        _PRICE_BUFFER = buf
        LOGGER.info("Loaded price buffer: shape=%s, mean=%.2f, std=%.2f",
                    str(buf.shape), _PRICE_STATS["mean"], _PRICE_STATS["std"])
    return _PRICE_BUFFER


def _load_news_features(root: Path):
    global _NEWS_FEATURES
    if _NEWS_FEATURES is not None:
        return _NEWS_FEATURES

    proxy_path = root / "processed" / "news_proxy.jsonl"
    if proxy_path.exists():
        by_date = {}
        with open(proxy_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                dt = ep["date"]
                sent = ep.get("weighted_sentiment", 0.0)
                vol = ep.get("realized_vol", 0.0)
                macro = ep.get("macro", [0.0] * 8)
                by_date[dt] = {
                    "sentiment": float(sent),
                    "realized_vol": float(vol),
                    "macro": np.array(macro, dtype=np.float32),
                }
        LOGGER.info("Loaded news proxy features for %d dates from %s", len(by_date), proxy_path)
        _NEWS_FEATURES = by_date
        return _NEWS_FEATURES

    LOGGER.info("news_proxy.jsonl not found — using price-derived macro features")
    buf = _load_price_buffer(root)
    n_dates = buf.shape[0]
    by_date = {}
    for idx in range(n_dates):
        rets = []
        for si in range(buf.shape[1]):
            if idx > 0 and buf[idx, si] != 0 and buf[idx-1, si] != 0:
                r = float(buf[idx, si] / (buf[idx - 1, si] + 1e-8) - 1)
                rets.append(r)
        sent = float(np.clip(np.mean(rets) * 5, -1, 1)) if rets else 0.0
        vol = float(np.std(rets)) if rets else 0.0
        by_date[str(idx)] = {"sentiment": sent, "realized_vol": vol, "macro": np.zeros(8, dtype=np.float32)}
    _NEWS_FEATURES = by_date
    return _NEWS_FEATURES


def _make_news_seq(news_by_date: dict, date_str: str | int, lookback: int = LOOKBACK) -> np.ndarray:
    half_life = 3.0
    if isinstance(date_str, int):
        date_str = str(date_str)
    keys = sorted(news_by_date.keys())
    if date_str in keys:
        date_idx = keys.index(date_str)
    else:
        try:
            date_idx = keys.index(str(date_str))
        except ValueError:
            return np.zeros((lookback, 384), dtype=np.float32)
    relevant = keys[max(0, date_idx - lookback + 1):date_idx + 1]
    weights = np.exp(-np.arange(len(relevant))[::-1] / (half_life / np.log(2)))
    weights /= (weights.sum() + 1e-8)

    feat = np.zeros(384, dtype=np.float32)
    sents = []
    for d in relevant:
        item = news_by_date.get(d, {})
        if isinstance(item, dict) and "sentiment" in item:
            sents.append(item["sentiment"])
        else:
            sents.append(0.0)
    sents = np.array(sents, dtype=np.float32)
    feat[0] = float(np.dot(weights, sents))
    pos = sents[sents > 0]
    neg = sents[sents < 0]
    feat[1] = float(np.mean(pos)) if len(pos) else 0.0
    feat[2] = float(np.mean(neg)) if len(neg) else 0.0
    feat[3] = float(np.std(sents)) if len(sents) > 1 else 0.0
    for lag in range(min(5, len(sents))):
        feat[4 + lag] = sents[-(lag + 1)]
    feat[9] = feat[0] * np.exp(-1 / half_life)
    feat[10] = feat[0] * np.exp(-3 / half_life)
    feat[11] = feat[0] * np.exp(-5 / half_life)
    feat[12:] = np.random.RandomState(7).randn(384 - 12).astype(np.float32) * 0.05
    return np.stack([feat * (1.0 + i * 0.01) for i in range(lookback)], axis=0)


def _make_macro_seq(news_by_date: dict, date_str: str | int, lookback: int = LOOKBACK) -> np.ndarray:
    half_life = 5.0
    if isinstance(date_str, int):
        date_str = str(date_str)
    keys = sorted(news_by_date.keys())
    if date_str in keys:
        date_idx = keys.index(date_str)
    else:
        try:
            date_idx = keys.index(str(date_str))
        except ValueError:
            return np.zeros((lookback, 8), dtype=np.float32)
    relevant = keys[max(0, date_idx - lookback + 1):date_idx + 1]
    weights = np.exp(-np.arange(len(relevant))[::-1] / (half_life / np.log(2)))
    weights /= (weights.sum() + 1e-8)

    macro = np.zeros(8, dtype=np.float32)
    macro_seq = []
    for d in relevant:
        item = news_by_date.get(d, {})
        if isinstance(item, dict) and "macro" in item:
            macro_seq.append(item["macro"])
        else:
            macro_seq.append(np.zeros(8, dtype=np.float32))
    macro_seq = np.stack(macro_seq, axis=0)
    weighted = np.tensordot(weights, macro_seq, axes=([0], [0])).astype(np.float32)
    return np.stack([
        weighted + np.random.RandomState(i).randn(8).astype(np.float32) * 0.01
        for i in range(lookback)
    ], axis=0).astype(np.float32)


class FinWorldDataset(Dataset):

    def __init__(self, root: str | Path, split: str = "train",
                 price_stats: dict | None = None, macro_stats: dict | None = None,
                 max_episodes: int | None = None):
        self.root = Path(root)
        self.split = split
        self.episode_file = self.root / "episodes" / f"{split}.jsonl"
        self.price_stats = price_stats or {}
        self.macro_stats = macro_stats or {}
        self.max_episodes = max_episodes

        self.price_buffer = _load_price_buffer(self.root)
        self.news_by_date = _load_news_features(self.root)

        if not self.episode_file.exists():
            LOGGER.warning("Episode file not found: %s. Generating synthetic data.", self.episode_file)
            self._meta = []
            self._synthesize = True
        else:
            self._meta = []
            self._synthesize = False
            with self.episode_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._meta.append(json.loads(line))

        if max_episodes:
            self._meta = self._meta[:max_episodes]

        LOGGER.info("FinWorldDataset('%s'): %d episodes, synth=%s",
                    split, len(self._meta), self._synthesize)

    def __len__(self):
        return len(self._meta)

    def _build_edges(self, meta):
        n = meta["n_graph"]
        edges = []
        for j in range(1, n):
            edges.append([0, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weight = torch.ones(n - 1, dtype=torch.float32)
        return edge_index, edge_weight

    def __getitem__(self, idx: int) -> dict:
        meta = self._meta[idx]

        if self._synthesize:
            np.random.seed(42 + idx)
            price_seq = np.random.randn(LOOKBACK, 5).astype(np.float32)
            news_feat = np.random.randn(LOOKBACK, 384).astype(np.float32)
            macro_feat = np.random.randn(LOOKBACK, 8).astype(np.float32)
            edge_index = torch.zeros(2, TOP_K_GRAPH, dtype=torch.long)
            edge_weight = torch.rand(TOP_K_GRAPH)
            price_target = np.random.randn(HORIZON, 5).astype(np.float32)
            action = np.random.randn(8).astype(np.float32)
        else:
            si = meta["ticker_idx"]
            date_idx = meta["date_idx"]
            n_graph = meta["n_graph"]
            if "neighbor_indices" in meta:
                g_indices = [si] + [int(n) for n in meta["neighbor_indices"]]
            else:
                neighbors = meta.get("neighbors", [])
                g_indices = [si] + [self._ticker_to_idx(n) for n in neighbors]

            mu = _PRICE_STATS["mean"]
            sigma = _PRICE_STATS["std"]

            lookback_slice = self.price_buffer[date_idx - LOOKBACK:date_idx, g_indices]
            price_seq = ((lookback_slice - mu) / sigma).astype(np.float32)

            horizon_slice = self.price_buffer[date_idx:date_idx + HORIZON, g_indices]
            price_target = ((horizon_slice - mu) / sigma).astype(np.float32)

            date_str = str(date_idx)
            news_feat = _make_news_seq(self.news_by_date, date_str)
            macro_feat = _make_macro_seq(self.news_by_date, date_str)

            action = np.random.randn(8).astype(np.float32)

            edge_index, edge_weight = self._build_edges(meta)

        return {
            "price_seq": torch.from_numpy(price_seq),
            "news_feat": torch.from_numpy(news_feat),
            "macro_feat": torch.from_numpy(macro_feat),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "price_target": torch.from_numpy(price_target),
            "action": torch.from_numpy(action),
        }

    @staticmethod
    def _ticker_to_idx(ticker: str) -> int:
        from scripts.process_yahoo_data import SYM_TO_IDX
        return SYM_TO_IDX.get(ticker, 0)


def collate_fn(batch):
    if not batch:
        return {}
    first = batch[0]
    result = {}
    for key in first:
        if isinstance(first[key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            result[key] = [item[key] for item in batch]
    return result

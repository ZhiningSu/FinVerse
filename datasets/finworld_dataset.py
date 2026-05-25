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
HORIZON = 10
TOP_K_GRAPH = 5

_PRICE_BUFFER = None
_PRICE_STATS = None


def _load_price_buffer(root: Path):
    global _PRICE_BUFFER, _PRICE_STATS
    if _PRICE_BUFFER is None:
        bin_path = root / "prices.bin"
        stats_path = root / "stats.json"

        buf = np.frombuffer(bin_path.read_bytes(), dtype=np.float32)
        n_dates = buf.shape[0] // 66
        buf = buf.reshape(n_dates, 66)
        _PRICE_BUFFER = buf

        if stats_path.exists():
            with open(stats_path) as f:
                _PRICE_STATS = json.load(f)["price_stats"]
        else:
            _PRICE_STATS = {"mean": 0.0, "std": 1.0}
        LOGGER.info("Loaded price buffer: shape=%s, mean=%.2f, std=%.2f",
                    str(buf.shape), _PRICE_STATS["mean"], _PRICE_STATS["std"])
    return _PRICE_BUFFER


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
            neighbors = meta.get("neighbors", [])
            g_indices = [si] + [self._ticker_to_idx(n) for n in neighbors]

            mu = _PRICE_STATS["mean"]
            sigma = _PRICE_STATS["std"]

            lookback_slice = self.price_buffer[date_idx - LOOKBACK:date_idx, g_indices]
            price_seq = ((lookback_slice - mu) / sigma).astype(np.float32)

            horizon_slice = self.price_buffer[date_idx:date_idx + HORIZON, g_indices]
            price_target = ((horizon_slice - mu) / sigma).astype(np.float32)

            seed = meta.get("seed", si * 1000000 + date_idx)
            rng = np.random.RandomState(seed & 0xFFFFFFFF)
            news_feat = rng.randn(LOOKBACK, 384).astype(np.float32)
            macro_feat = rng.randn(LOOKBACK, 8).astype(np.float32)
            action = rng.randn(8).astype(np.float32)

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
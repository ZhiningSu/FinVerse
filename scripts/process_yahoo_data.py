"""Process raw Yahoo Finance JSON data into FinWorldDataset episode JSONL + binary format."""
from __future__ import annotations

import csv
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "yfinance_hmsc"
TICKER_CSV = PROJECT_ROOT / "data" / "tickers" / "hmsc_us_90.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "real_90"
BIN_FILE = OUT_DIR / "prices.bin"
STATS_FILE = OUT_DIR / "stats.json"

LOOKBACK = 30
HORIZON = 30
TOP_K_GRAPH = 5

SPLITS = {
    "train": ("2020-01-01", "2022-12-31"),
    "validation": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}

SECTOR_NEIGHBORS = {
    "Technology": ["Financials", "Communication Services"],
    "Financials": ["Technology", "Communication Services"],
    "Healthcare": ["Healthcare", "Consumer Staples"],
    "Energy": ["Energy", "Industrials"],
    "Consumer Discretionary": ["Consumer Discretionary", "Consumer Staples"],
    "Consumer Staples": ["Consumer Staples", "Consumer Discretionary"],
    "Industrials": ["Industrials", "Energy"],
    "Communication Services": ["Communication Services", "Technology"],
    "Utilities": ["Utilities", "Real Estate"],
    "Real Estate": ["Real Estate", "Utilities"],
}

ALL_DATES = []
SUCCESS_SYMS = []
SYM_TO_IDX = {}
LOOKBACK_BUFFER = None
NEIGHBOR_MAP = {}
SECTOR_MAP = {}


def load_tickers():
    sector_map = {}
    with open(TICKER_CSV, "r") as f:
        for row in csv.DictReader(f):
            sector_map[row["ticker"].strip()] = row["sector"].strip()
    return sector_map


def load_raw_data():
    raw = {}
    for jf in sorted(RAW_DIR.glob("*.json")):
        if jf.name == "fetch_report.json":
            continue
        with open(jf, "r") as f:
            d = json.load(f)
        sym = d.get("symbol", jf.stem)
        raw[sym] = {day["date"]: day for day in d["data"]}
    return raw


def build_lookback_buffer(raw):
    global LOOKBACK_BUFFER, ALL_DATES, SUCCESS_SYMS, SYM_TO_IDX

    SUCCESS_SYMS = sorted(raw.keys())
    SYM_TO_IDX = {s: i for i, s in enumerate(SUCCESS_SYMS)}

    all_dates_set = set()
    for day_map in raw.values():
        all_dates_set.update(day_map.keys())
    ALL_DATES = sorted(all_dates_set)

    n_dates = len(ALL_DATES)
    n_syms = len(SUCCESS_SYMS)
    LOOKBACK_BUFFER = np.zeros((n_dates, n_syms), dtype=np.float32)

    for sym, day_map in raw.items():
        si = SYM_TO_IDX[sym]
        for date_str, day_data in day_map.items():
            if date_str in ALL_DATES:
                c = day_data.get("close")
                if c is not None:
                    LOOKBACK_BUFFER[ALL_DATES.index(date_str), si] = float(c)

    LOG.info("Lookback buffer: shape=%s, syms=%d, dates=%d",
             str(LOOKBACK_BUFFER.shape), n_syms, n_dates)


def build_neighbor_cache():
    global NEIGHBOR_MAP
    for sym in SUCCESS_SYMS:
        ref_sec = SECTOR_MAP.get(sym, "")
        cross_secs = SECTOR_NEIGHBORS.get(ref_sec, [])
        neighbors = [s for s in SUCCESS_SYMS if s != sym and SECTOR_MAP.get(s, "") in cross_secs][:TOP_K_GRAPH]
        NEIGHBOR_MAP[sym] = neighbors


def compute_stats():
    vals = LOOKBACK_BUFFER[LOOKBACK:].flatten()
    vals = vals[vals != 0]
    if len(vals) == 0:
        return {"mean": 0.0, "std": 1.0}
    return {
        "mean": float(vals.mean()),
        "std": max(float(vals.std()), 1e-8),
    }


def normalize(buffer, stats):
    mu = stats["mean"]
    sigma = stats["std"]
    return (buffer - mu) / sigma


def write_prices_binary():
    with open(BIN_FILE, "wb") as f:
        LOOKBACK_BUFFER.tofile(f)
    LOG.info("Wrote binary prices: %s, %d bytes", str(LOOKBACK_BUFFER.shape), BIN_FILE.stat().st_size)


def write_split(split_name, indices, stats, out_file):
    mu = stats["mean"]
    sigma = stats["std"]

    with open(out_file, "w") as f:
        for si, sym in enumerate(SUCCESS_SYMS):
            neighbors = NEIGHBOR_MAP.get(sym, [])
            g_indices = [si] + [SYM_TO_IDX[n] for n in neighbors]
            n_graph = len(g_indices)

            graph_edges = [{"src_ticker": sym, "dst_ticker": neighbors[j], "weight": 1.0}
                           for j in range(len(neighbors))]
            neighbor_indices = [SYM_TO_IDX[n] for n in neighbors]

            for i in indices:
                lb = i - LOOKBACK

                price_rows = LOOKBACK_BUFFER[lb:i][:, g_indices]
                norm_prices = ((price_rows - mu) / sigma).tolist()

                rollout_rows = LOOKBACK_BUFFER[i:i + HORIZON][:, g_indices]
                norm_rollouts = ((rollout_rows - mu) / sigma).tolist()

                seed = (si * 1000000 + i) & 0xFFFFFFFF
                ep = {
                    "ticker_idx": si,
                    "date_idx": i,
                    "n_graph": n_graph,
                    "neighbors": neighbors,
                    "neighbor_indices": neighbor_indices,
                    "seed": seed,
                    "graph_structures": graph_edges,
                }
                f.write(json.dumps(ep) + "\n")


def main():
    global SECTOR_MAP, RAW_DIR, TICKER_CSV, OUT_DIR, BIN_FILE, STATS_FILE

    parser = argparse.ArgumentParser(description="Process raw market data into FinWorldDataset format.")
    parser.add_argument("--source", type=str, default="yfinance_hmsc", choices=["yfinance", "yfinance_hmsc", "eastmoney"])
    parser.add_argument("--ticker_file", type=Path, default=TICKER_CSV)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / args.source
    TICKER_CSV = args.ticker_file
    OUT_DIR = args.out_dir
    BIN_FILE = OUT_DIR / "prices.bin"
    STATS_FILE = OUT_DIR / "stats.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading ticker list")
    SECTOR_MAP = load_tickers()

    LOG.info("Loading raw %s data from %s", args.source, RAW_DIR)
    raw = load_raw_data()
    LOG.info("Loaded %d tickers", len(raw))

    build_lookback_buffer(raw)
    build_neighbor_cache()
    LOG.info("Neighbor cache built for %d tickers", len(NEIGHBOR_MAP))

    stats = compute_stats()
    LOG.info("Train stats: mean=%.2f, std=%.2f", stats["mean"], stats["std"])

    with open(STATS_FILE, "w") as f:
        json.dump({
            "source": args.source,
            "price_stats": stats,
            "macro_stats": {},
            "n_symbols": len(SUCCESS_SYMS),
            "symbols": SUCCESS_SYMS,
            "n_dates": len(ALL_DATES),
            "dates": ALL_DATES,
        }, f, indent=2)
    LOG.info("Wrote %s", STATS_FILE)

    write_prices_binary()

    valid_indices = {}
    for name, (s_str, e_str) in SPLITS.items():
        s_dt = datetime.strptime(s_str, "%Y-%m-%d")
        e_dt = datetime.strptime(e_str, "%Y-%m-%d")
        valid_indices[name] = [
            i for i, d_str in enumerate(ALL_DATES)
            if s_dt <= datetime.strptime(d_str, "%Y-%m-%d") <= e_dt
            and i >= LOOKBACK and i + HORIZON <= len(ALL_DATES)
        ]
        LOG.info("  %s: %d valid date indices", name, len(valid_indices[name]))

    episodes_dir = OUT_DIR / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_name in ["train", "validation", "test"]:
        out_file = episodes_dir / f"{split_name}.jsonl"
        write_split(split_name, valid_indices[split_name], stats, out_file)
        n_ep = sum(1 for _ in open(out_file))
        total += n_ep
        LOG.info("Split '%s': %d episodes -> %s (%.1f KB)",
                 split_name, n_ep, out_file, out_file.stat().st_size / 1024)

    LOG.info("Done. Total: %d episodes. Binary: %.1f MB. JSONL: %.1f MB",
             total, BIN_FILE.stat().st_size / 1e6,
             sum(f.stat().st_size for f in episodes_dir.glob("*.jsonl")) / 1e6)


if __name__ == "__main__":
    main()

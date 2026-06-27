from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.pipeline import LivePipelineConfig, run_live_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinVerse dynamic live pipeline.")
    parser.add_argument("--market", choices=["us", "cn"], default="us")
    parser.add_argument("--trade-date", default=None, help="Optional YYYY-MM-DD date. Defaults to latest raw date.")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/live")
    parser.add_argument("--data-live-dir", default="data/live")
    args = parser.parse_args()

    result = run_live_pipeline(
        LivePipelineConfig(
            market=args.market,
            output_dir=Path(args.output_dir),
            data_live_dir=Path(args.data_live_dir),
            top_k=args.top_k,
        ),
        trade_date=args.trade_date,
        force_fetch=args.force_fetch,
    )
    print(json.dumps({
        "trade_date": result["trade_date"],
        "market": result["market"],
        "strategy": result["selected_strategy"]["name"],
        "top_assets": [asset["ticker"] for asset in result["top_assets"]],
        "latest_path": str(Path(args.output_dir) / args.market / "latest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()

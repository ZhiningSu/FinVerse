from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.pipeline import LivePipelineConfig, load_latest, run_live_pipeline


app = FastAPI(
    title="FinVerse Dynamic Agent Dashboard API",
    version="0.1.0",
    description="World-model-powered dynamic asset-selection dashboard API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRunRequest(BaseModel):
    market: Optional[str] = "us"
    trade_date: Optional[str] = None
    force_fetch: bool = False


def _clean_market(market: str) -> str:
    market = (market or "us").lower()
    if market not in {"us", "cn"}:
        raise HTTPException(status_code=400, detail=f"Unsupported market: {market}")
    return market


def _latest(market: str = "us") -> Dict[str, Any]:
    return load_latest(PROJECT_ROOT / "outputs" / "live", market=_clean_market(market))


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/pipeline/status")
def pipeline_status(market: str = "us") -> Dict[str, Any]:
    payload = _latest(market)
    status = payload.get("pipeline_status", {"stages": []})
    return {
        "market": payload.get("market", _clean_market(market)),
        "language": payload.get("language", "en"),
        "trade_date": payload.get("trade_date"),
        "last_updated_at": payload.get("last_updated_at"),
        "stages": status.get("stages", []),
        "model_checkpoint": payload.get("model_checkpoint"),
        "mode": payload.get("mode"),
    }


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    def _run() -> None:
        market = _clean_market(request.market or "us")
        run_live_pipeline(
            LivePipelineConfig(
                market=market,
                output_dir=PROJECT_ROOT / "outputs" / "live",
                data_live_dir=PROJECT_ROOT / "data" / "live",
            ),
            trade_date=request.trade_date,
            force_fetch=request.force_fetch,
        )

    background_tasks.add_task(_run)
    return {"run_id": "local-background", "status": "queued"}


@app.get("/api/recommendations/latest")
def latest_recommendations(market: str = "us") -> Dict[str, Any]:
    payload = _latest(market)
    return {
        "market": payload.get("market", _clean_market(market)),
        "language": payload.get("language", "en"),
        "trade_date": payload["trade_date"],
        "last_updated_at": payload["last_updated_at"],
        "selected_strategy": payload["selected_strategy"],
        "market_state": payload["market_state"],
        "top_industries": payload.get("top_industries", []),
        "top_assets": payload["top_assets"],
        "mode": payload["mode"],
    }


@app.get("/api/assets/{ticker}")
def asset_detail(ticker: str, market: str = "us") -> Dict[str, Any]:
    payload = _latest(market)
    ticker = ticker.upper()
    for asset in payload.get("all_assets", []):
        if asset["ticker"].upper() == ticker:
            return {
                "ticker": asset["ticker"],
                "trade_date": payload["trade_date"],
                "history_close": asset.get("history_close", []),
                "rollout_path": asset.get("rollout_path", []),
                "features": {
                    "expected_return_30d": asset.get("expected_return_30d"),
                    "predicted_volatility": asset.get("predicted_volatility"),
                    "predicted_downside": asset.get("predicted_downside"),
                    "bull_prob": asset.get("regime_probs", {}).get("bull"),
                    "sideway_prob": asset.get("regime_probs", {}).get("sideway"),
                    "bear_prob": asset.get("regime_probs", {}).get("bear"),
                },
                "explanation": asset.get("reasons", []),
                "score": asset.get("score"),
                "rank": asset.get("rank"),
                "sector": asset.get("sector"),
                "type": asset.get("type"),
            }
    raise HTTPException(status_code=404, detail=f"Asset not found: {ticker}")


@app.get("/api/diagnostics/world-model")
def world_model_diagnostics(market: str = "us") -> Dict[str, Any]:
    payload = _latest(market)
    return payload.get("diagnostics", {})


@app.get("/api/history")
def recommendation_history() -> Dict[str, Any]:
    live_dir = PROJECT_ROOT / "outputs" / "live"
    entries = []
    for path in sorted(live_dir.glob("*/*/recommendations.json"), reverse=True):
        payload = __import__("json").loads(path.read_text())
        entries.append(
            {
                "trade_date": payload.get("trade_date"),
                "strategy": payload.get("selected_strategy", {}).get("name"),
                "top_assets": [asset["ticker"] for asset in payload.get("top_assets", [])[:5]],
                "updated_at": payload.get("last_updated_at"),
            }
        )
    return {"items": entries}

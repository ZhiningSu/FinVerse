from __future__ import annotations

import argparse
import json
import logging
import random
import ssl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.finworld_dataset import FinWorldDataset, collate_fn
from models.baselines import DreamerStyleRSSM, MultiModalNoRollout, NoGraphWorldModel, PriceOnlyGRU
from models.forecasting_baselines import (
    ChronosMiniForecaster,
    DLinearForecaster,
    GRUForecaster,
    ITransformerForecaster,
    KronosMiniForecaster,
    LSTMForecaster,
    PatchTSTForecaster,
    TimesFMStyleForecaster,
    TransformerForecaster,
)
from models.world_model import WorldModel
from trainers.trainer import BaselineLoss, WorldModelLoss

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "full": WorldModel,
    "price_only": PriceOnlyGRU,
    "multi_noroll": MultiModalNoRollout,
    "no_graph": NoGraphWorldModel,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "dlinear": DLinearForecaster,
    "transformer": TransformerForecaster,
    "patchtst": PatchTSTForecaster,
    "itransformer": ITransformerForecaster,
    "kronos_mini": KronosMiniForecaster,
    "chronos_mini": ChronosMiniForecaster,
    "timesfm": TimesFMStyleForecaster,
    "vanilla_rssm": WorldModel,
    "dreamer_rssm": DreamerStyleRSSM,
    "finverse": WorldModel,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_num_sectors(state: dict, fallback: int = 32) -> int:
    weight = state.get("sector_embedding.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        return int(weight.shape[0])
    return fallback


def _infer_hidden_dim(state: dict, fallback: int) -> int:
    weight = state.get("encoder.price_input.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        return int(weight.shape[0])
    weight = state.get("price_encoder.weight_ih_l0")
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        return int(weight.shape[0] // 3)
    return fallback


def _infer_latent_dim(state: dict, fallback: int) -> int:
    weight = state.get("decoder.state_norm.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() == 1:
        return int(weight.shape[0])
    weight = state.get("transition.rssm_transition.weight_hh")
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        return int(weight.shape[1])
    return fallback


def _shape_compatible_state(model: torch.nn.Module, state: dict) -> dict:
    model_state = model.state_dict()
    return {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }


def load_model(checkpoint_path: str, model_name: str, device: torch.device, hidden_dim=256, latent_dim=128, num_tickers=90, num_sectors=32):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model_cls = MODEL_REGISTRY[model_name]
    if model_name in {"full", "finverse", "vanilla_rssm"}:
        hidden_dim = _infer_hidden_dim(state, hidden_dim)
        latent_dim = _infer_latent_dim(state, latent_dim)

    extra = {"price_dim": 6, "news_dim": 384, "macro_dim": 8, "graph_dim": 5, "action_dim": 8, "num_tickers": num_tickers}
    world_extra = {**extra, "num_sectors": _infer_num_sectors(state, num_sectors)}

    if model_name in {"price_only", "lstm", "gru", "dlinear", "transformer", "patchtst", "itransformer", "kronos_mini", "chronos_mini", "timesfm"}:
        model = model_cls(price_dim=6, hidden_dim=hidden_dim, output_dim=6, num_steps=30).to(device)
    elif model_name == "vanilla_rssm":
        model = model_cls(latent_dim=latent_dim, hidden_dim=hidden_dim, use_dual_vq=False, **world_extra).to(device)
    elif model_name == "dreamer_rssm":
        model = model_cls(latent_dim=latent_dim, hidden_dim=hidden_dim, **extra).to(device)
    else:
        model = model_cls(latent_dim=latent_dim, hidden_dim=hidden_dim, **world_extra).to(device)

    model.load_state_dict(_shape_compatible_state(model, state), strict=False)
    model.eval()
    return model


def predict_at_horizon(
    model,
    price_seq,
    news_feat,
    macro_feat,
    edge_index,
    edge_weight,
    action,
    horizon: int,
    ticker_idx=None,
    sector_id=None,
):
    price_seq = price_seq.to(torch.float32)
    news_feat = news_feat.to(torch.float32)
    macro_feat = macro_feat.to(torch.float32)
    action = action.to(torch.float32)
    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float32)

    model_name = type(model).__name__

    if model_name == "PriceOnlyGRU":
        out = model(price_seq)
        pred = out["price_pred"]
        if pred.dim() == 3:
            pred = pred[:, horizon - 1, :]
        return pred

    elif model_name == "MultiModalNoRollout":
        out = model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action)
        pred = out["price_pred"]
        if pred.dim() == 2:
            return pred[:, horizon - 1]
        return pred[:, horizon - 1, :] if horizon <= pred.size(1) else pred[:, -1, :]

    else:
        kwargs = {}
        if getattr(model, "supports_asset_conditioning", False):
            kwargs = {"ticker_idx": ticker_idx, "sector_id": sector_id}
        out = model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action, **kwargs)
        pred = out["price_pred"]
        if pred.dim() == 2:
            return pred[:, horizon - 1]
        elif pred.dim() == 3:
            h_idx = horizon - 1
            return pred[:, h_idx, :] if h_idx < pred.size(1) else pred[:, -1, :]
        return pred


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if np.isclose(denom, 0.0):
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _long_short_portfolio_returns(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    top_k: int,
    return_clip: float | None = 0.2,
) -> np.ndarray:
    daily_returns = []
    for date in np.unique(dates):
        mask = dates == date
        date_pred = pred[mask]
        date_target = target[mask]
        if return_clip and return_clip > 0:
            date_target = np.clip(date_target, -return_clip, return_clip)
        if date_pred.size < 2:
            continue
        k = min(top_k, date_pred.size // 2)
        if k <= 0:
            continue
        order = np.argsort(date_pred)
        short_idx = order[:k]
        long_idx = order[-k:]
        long_ret = float(np.mean(date_target[long_idx]))
        short_ret = float(np.mean(date_target[short_idx]))
        daily_returns.append(long_ret - short_ret)
    return np.asarray(daily_returns, dtype=float)


def _buy_and_hold_returns(
    target: np.ndarray,
    dates: np.ndarray,
    return_clip: float | None = 0.2,
) -> np.ndarray:
    daily_returns = []
    for date in np.unique(dates):
        mask = dates == date
        date_target = target[mask]
        if return_clip and return_clip > 0:
            date_target = np.clip(date_target, -return_clip, return_clip)
        if date_target.size == 0:
            continue
        daily_returns.append(float(np.mean(date_target)))
    return np.asarray(daily_returns, dtype=float)


def _portfolio_metrics(daily_returns: np.ndarray) -> dict:
    if daily_returns.size == 0:
        return {
            "Daily_Mean_Return": 0.0,
            "Daily_Return_Std": 0.0,
            "IR_Daily": 0.0,
            "IR": 0.0,
            "IR_Annualized": 0.0,
            "AER": 0.0,
        }
    daily_mean = float(daily_returns.mean())
    daily_std = float(daily_returns.std())
    ir_daily = 0.0 if np.isclose(daily_std, 0.0) else float(daily_mean / daily_std)
    ir_annualized = float(ir_daily * np.sqrt(252.0))
    aer = float(daily_mean * 252.0)
    return {
        "Daily_Mean_Return": daily_mean,
        "Daily_Return_Std": daily_std,
        "IR_Daily": ir_daily,
        "IR": ir_annualized,
        "IR_Annualized": ir_annualized,
        "AER": aer,
    }


@torch.no_grad()
def evaluate_buy_and_hold(
    dataloader,
    portfolio_return_clip: float | None = 0.2,
):
    targets = []
    date_ids = []
    for batch in tqdm(dataloader, desc="Eval BUY&HOLD"):
        price_target = batch["price_target"]
        target_h = price_target[:, 0, :] if price_target.dim() == 3 else price_target
        if target_h.dim() == 1:
            target_h = target_h.unsqueeze(-1)
        targets.extend(target_h[:, :1].squeeze(-1).detach().cpu().numpy().tolist())
        date_ids.extend(batch["date_idx"].detach().cpu().numpy().tolist())

    target_array = np.asarray(targets, dtype=float)
    date_array = np.asarray(date_ids, dtype=int)
    strategy_returns = _buy_and_hold_returns(
        target_array,
        date_array,
        return_clip=portfolio_return_clip,
    )
    portfolio_metrics = _portfolio_metrics(strategy_returns)
    return {
        "MSE@1": None,
        "MSE@5": None,
        "MSE@10": None,
        "MSE@20": None,
        "MSE@30": None,
        "MAE@1": None,
        "MAE@5": None,
        "MAE@10": None,
        "MAE@20": None,
        "MAE@30": None,
        "IC@1": None,
        "IC@5": None,
        "IC@10": None,
        "IC@20": None,
        "IC@30": None,
        "IC_mean": None,
        "RankIC@1": None,
        "RankIC@5": None,
        "RankIC@10": None,
        "RankIC@20": None,
        "RankIC@30": None,
        "RankIC_mean": None,
        "Volatility_MAE": None,
        "Volatility_R2": None,
        "Portfolio_TopK": "equal_weight_long",
        "Portfolio_Return_Clip": portfolio_return_clip,
        "Portfolio_Days": int(strategy_returns.size),
        **portfolio_metrics,
        "n_samples": int(target_array.size),
    }


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    model_name: str,
    portfolio_top_k: int = 5,
    portfolio_return_clip: float | None = 0.2,
):
    model.eval()
    device = next(model.parameters()).device
    mse_1, mse_5, mse_10, mse_20, mse_30 = [], [], [], [], []
    mae_1, mae_5, mae_10, mae_20, mae_30 = [], [], [], [], []
    pred_by_h = {h: [] for h in [1, 5, 10, 20, 30]}
    target_by_h = {h: [] for h in [1, 5, 10, 20, 30]}
    pred_paths, target_paths = [], []
    date_ids = []

    for batch in tqdm(dataloader, desc=f"Eval {model_name}"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        price_seq = batch["price_seq"]
        news_feat = batch["news_feat"]
        macro_feat = batch["macro_feat"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        price_target = batch["price_target"]
        action = batch["action"]
        date_ids.extend(batch["date_idx"].detach().cpu().numpy().tolist())
        batch_pred_path = []
        batch_target_path = []

        for h in [1, 5, 10, 20, 30]:
            pred = predict_at_horizon(
                model,
                price_seq,
                news_feat,
                macro_feat,
                edge_index,
                edge_weight,
                action,
                h,
                ticker_idx=batch.get("ticker_idx"),
                sector_id=batch.get("sector_id"),
            )
            target_h = price_target[:, h - 1, :] if price_target.dim() == 3 else price_target
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            if target_h.dim() == 1:
                target_h = target_h.unsqueeze(-1)
            pred = pred.reshape(pred.size(0), -1)
            target_h = target_h.reshape(target_h.size(0), -1)
            pred = pred[:, :1]
            target_h = target_h[:, :1]
            pred_by_h[h].extend(pred.squeeze(-1).detach().cpu().numpy().tolist())
            target_by_h[h].extend(target_h.squeeze(-1).detach().cpu().numpy().tolist())
            batch_pred_path.append(pred.squeeze(-1).detach().cpu().numpy())
            batch_target_path.append(target_h.squeeze(-1).detach().cpu().numpy())
            mse_h = F.mse_loss(pred, target_h, reduction="none").mean(dim=1).tolist()
            mae_h = F.l1_loss(pred, target_h, reduction="none").mean(dim=1).tolist()
            if h == 1:
                mse_1.extend(mse_h); mae_1.extend(mae_h)
            elif h == 5:
                mse_5.extend(mse_h); mae_5.extend(mae_h)
            elif h == 10:
                mse_10.extend(mse_h); mae_10.extend(mae_h)
            elif h == 20:
                mse_20.extend(mse_h); mae_20.extend(mae_h)
            else:
                mse_30.extend(mse_h); mae_30.extend(mae_h)

        pred_paths.append(np.stack(batch_pred_path, axis=1))
        target_paths.append(np.stack(batch_target_path, axis=1))

    pred_arrays = {h: np.asarray(values, dtype=float) for h, values in pred_by_h.items()}
    target_arrays = {h: np.asarray(values, dtype=float) for h, values in target_by_h.items()}
    ic = {h: _corr(pred_arrays[h], target_arrays[h]) for h in pred_arrays}
    rank_ic = {
        h: _corr(_rankdata(pred_arrays[h]), _rankdata(target_arrays[h]))
        for h in pred_arrays
    }
    pred_path = np.concatenate(pred_paths, axis=0)
    target_path = np.concatenate(target_paths, axis=0)
    pred_vol = pred_path.std(axis=1)
    target_vol = target_path.std(axis=1)
    vol_mae = float(np.mean(np.abs(pred_vol - target_vol)))
    vol_r2 = _safe_r2(target_vol, pred_vol)
    date_array = np.asarray(date_ids, dtype=int)
    strategy_returns = _long_short_portfolio_returns(
        pred_arrays[1],
        target_arrays[1],
        date_array,
        top_k=portfolio_top_k,
        return_clip=portfolio_return_clip,
    )
    portfolio_metrics = _portfolio_metrics(strategy_returns)

    return {
        "MSE@1": float(np.mean(mse_1)),
        "MSE@5": float(np.mean(mse_5)),
        "MSE@10": float(np.mean(mse_10)),
        "MSE@20": float(np.mean(mse_20)),
        "MSE@30": float(np.mean(mse_30)),
        "MAE@1": float(np.mean(mae_1)),
        "MAE@5": float(np.mean(mae_5)),
        "MAE@10": float(np.mean(mae_10)),
        "MAE@20": float(np.mean(mae_20)),
        "MAE@30": float(np.mean(mae_30)),
        "IC@1": ic[1],
        "IC@5": ic[5],
        "IC@10": ic[10],
        "IC@20": ic[20],
        "IC@30": ic[30],
        "IC_mean": float(np.mean(list(ic.values()))),
        "RankIC@1": rank_ic[1],
        "RankIC@5": rank_ic[5],
        "RankIC@10": rank_ic[10],
        "RankIC@20": rank_ic[20],
        "RankIC@30": rank_ic[30],
        "RankIC_mean": float(np.mean(list(rank_ic.values()))),
        "Volatility_MAE": vol_mae,
        "Volatility_R2": vol_r2,
        "Portfolio_TopK": int(portfolio_top_k),
        "Portfolio_Return_Clip": portfolio_return_clip,
        "Portfolio_Days": int(strategy_returns.size),
        **portfolio_metrics,
        "n_samples": len(mse_1),
    }


def print_table(results: dict):
    header = f"{'Model':<20} | {'MSE@1':>8} | {'MSE@5':>8} | {'IC':>8} | {'RankIC':>8} | {'VolMAE':>8} | {'DMean':>8} | {'DStd':>8} | {'DIR':>8} | {'AnnIR':>8} | {'AER':>8}"
    sep = "-" * len(header)

    def cell(value):
        return "   N/A  " if value is None else f"{value:>8.4f}"

    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"{name:<20} | {cell(m['MSE@1'])} | {cell(m['MSE@5'])} | "
            f"{cell(m['IC_mean'])} | {cell(m['RankIC_mean'])} | "
            f"{cell(m['Volatility_MAE'])} | {cell(m.get('Daily_Mean_Return'))} | "
            f"{cell(m.get('Daily_Return_Std'))} | {cell(m.get('IR_Daily'))} | "
            f"{cell(m.get('IR_Annualized', m.get('IR')))} | {cell(m['AER'])}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinWorld experiments")
    parser.add_argument("--data-root", default="data/processed/real_90")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--target-mode", choices=["return", "price"], default="return")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--portfolio-top-k", type=int, default=5)
    parser.add_argument("--portfolio-return-clip", type=float, default=0.2)
    parser.add_argument("--include-buy-hold", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--checkpoints", nargs="+", default=[], help="List of 'name:path' pairs")
    args = parser.parse_args()
    set_seed(args.seed)

    dataset = FinWorldDataset(
        args.data_root,
        split=args.split,
        max_episodes=args.max_episodes,
        max_dates=args.max_dates,
        target_mode=args.target_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    LOGGER.info("Test set: %d episodes", len(dataset))
    num_tickers = int(getattr(dataset, "price_buffer").shape[1])
    num_sectors = max(len(getattr(dataset, "sector_vocab", [])), 1)

    results = {}
    device = torch.device(args.device)

    if args.checkpoints:
        for spec in args.checkpoints:
            name, path = spec.split(":", 1)
            LOGGER.info("Loading checkpoint: %s from %s", name, path)
            model = load_model(
                path,
                _name_to_key(name),
                device,
                args.hidden_dim,
                args.latent_dim,
                num_tickers=num_tickers,
                num_sectors=num_sectors,
            )
            results[name] = evaluate_model(
                model,
                loader,
                name,
                portfolio_top_k=args.portfolio_top_k,
                portfolio_return_clip=args.portfolio_return_clip,
            )
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    else:
        import os
        for model_name in ["full", "price_only", "multi_noroll", "no_graph"]:
            ckpt_dir = Path(f"outputs/{model_name}")
            for ckpt_file in ["best_checkpoint.pt", "last_checkpoint.pt"]:
                ckpt_path = ckpt_dir / ckpt_file
                if ckpt_path.exists():
                    LOGGER.info("Loading %s from %s", model_name, ckpt_path)
                    model = load_model(
                        str(ckpt_path),
                        model_name,
                        device,
                        args.hidden_dim,
                        args.latent_dim,
                        num_tickers=num_tickers,
                        num_sectors=num_sectors,
                    )
                    display_name = {
                        "full": "FinWorldModel",
                        "price_only": "PriceOnlyGRU",
                        "multi_noroll": "MultiModal-noRollout",
                        "no_graph": "NoGraph",
                    }[model_name]
                    results[display_name] = evaluate_model(
                        model,
                        loader,
                        display_name,
                        portfolio_top_k=args.portfolio_top_k,
                        portfolio_return_clip=args.portfolio_return_clip,
                    )
                    del model
                    break

    if args.include_buy_hold:
        results["BUY&HOLD"] = evaluate_buy_and_hold(
            loader,
            portfolio_return_clip=args.portfolio_return_clip,
        )

    print()
    print_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Results saved to %s", args.output)


def _name_to_key(name: str) -> str:
    if name.startswith("FinVerse-"):
        return "finverse"
    mapping = {
        "FinWorldModel": "full",
        "PriceOnlyGRU": "price_only",
        "Price-only GRU": "price_only",
        "PriceOnly": "price_only",
        "MultiModal-noRollout": "multi_noroll",
        "MultiModalNoRollout": "multi_noroll",
        "Multimodal-noRollout": "multi_noroll",
        "NoGraph": "no_graph",
        "NoGraphWorldModel": "no_graph",
        "w/o Graph": "no_graph",
        "w/o Cross-Asset Ctx": "no_graph",
        "w/o Dual VQ": "vanilla_rssm",
        "w/o Probabilistic WM": "multi_noroll",
        "Price Only": "price_only",
        "LSTM": "lstm",
        "GRU": "gru",
        "DLinear": "dlinear",
        "Transformer": "transformer",
        "PatchTST": "patchtst",
        "Kronos-mini": "kronos_mini",
        "KronosMini": "kronos_mini",
        "Chronos-mini": "chronos_mini",
        "ChronosMini": "chronos_mini",
        "TimesFM": "timesfm",
        "TimesFM-style": "timesfm",
        "Vanilla RSSM": "vanilla_rssm",
        "VanillaRSSM": "vanilla_rssm",
        "Dreamer-style RSSM": "dreamer_rssm",
        "DreamerRSSM": "dreamer_rssm",
        "FinVerse": "finverse",
        "Full FinVerse": "finverse",
        "iTransformer": "itransformer",
    }
    return mapping.get(name, name)


if __name__ == "__main__":
    main()

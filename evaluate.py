from __future__ import annotations

import argparse
import json
import logging
import ssl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.finworld_dataset import FinWorldDataset, collate_fn
from models.baselines import MultiModalNoRollout, NoGraphWorldModel, PriceOnlyGRU
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
}


def load_model(checkpoint_path: str, model_name: str, device: torch.device, hidden_dim=256, latent_dim=128):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cls = MODEL_REGISTRY[model_name]

    extra = {"price_dim": 6, "news_dim": 384, "macro_dim": 8, "graph_dim": 5, "action_dim": 8, "num_tickers": 66}

    if model_name == "price_only":
        model = model_cls(price_dim=6, hidden_dim=hidden_dim, output_dim=6, num_steps=30).to(device)
    else:
        model = model_cls(latent_dim=latent_dim, hidden_dim=hidden_dim, **extra).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_at_horizon(model, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, horizon: int):
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
        out = model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action)
        pred = out["price_pred"]
        if pred.dim() == 2:
            return pred[:, horizon - 1]
        elif pred.dim() == 3:
            h_idx = horizon - 1
            return pred[:, h_idx, :] if h_idx < pred.size(1) else pred[:, -1, :]
        return pred


@torch.no_grad()
def evaluate_model(model, dataloader, model_name: str):
    model.eval()
    mse_1, mse_5, mse_10, mse_20, mse_30 = [], [], [], [], []
    mae_1, mae_5, mae_10, mae_20, mae_30 = [], [], [], [], []

    for batch in tqdm(dataloader, desc=f"Eval {model_name}"):
        batch = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        price_seq = batch["price_seq"]
        news_feat = batch["news_feat"]
        macro_feat = batch["macro_feat"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        price_target = batch["price_target"]
        action = batch["action"]

        for h in [1, 5, 10, 20, 30]:
            pred = predict_at_horizon(model, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, h)
            target_h = price_target[:, h - 1, :] if price_target.dim() == 3 else price_target
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            if target_h.dim() == 1:
                target_h = target_h.unsqueeze(-1)
            pred = pred.reshape(pred.size(0), -1)
            target_h = target_h.reshape(target_h.size(0), -1)
            if pred.size(1) > target_h.size(1):
                pred = pred[:, :target_h.size(1)]
            else:
                target_h = target_h[:, :pred.size(1)]
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
        "n_samples": len(mse_1),
    }


def print_table(results: dict):
    header = f"{'Model':<20} | {'MSE@1':>8} | {'MSE@5':>8} | {'MSE@10':>8} | {'MAE@1':>8} | {'MAE@5':>8} | {'MAE@10':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for name, m in results.items():
        print(f"{name:<20} | {m['MSE@1']:>8.4f} | {m['MSE@5']:>8.4f} | {m['MSE@10']:>8.4f} | {m['MAE@1']:>8.4f} | {m['MAE@5']:>8.4f} | {m['MAE@10']:>8.4f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinWorld experiments")
    parser.add_argument("--data-root", default="data/processed/real")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--checkpoints", nargs="+", default=[], help="List of 'name:path' pairs")
    args = parser.parse_args()

    dataset = FinWorldDataset(args.data_root, split=args.split, max_episodes=args.max_episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    LOGGER.info("Test set: %d episodes", len(dataset))

    results = {}
    device = torch.device(args.device)

    if args.checkpoints:
        for spec in args.checkpoints:
            name, path = spec.split(":", 1)
            LOGGER.info("Loading checkpoint: %s from %s", name, path)
            model = load_model(path, _name_to_key(name), device, args.hidden_dim, args.latent_dim)
            results[name] = evaluate_model(model, loader, name)
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
                    model = load_model(str(ckpt_path), model_name, device, args.hidden_dim, args.latent_dim)
                    display_name = {
                        "full": "FinWorldModel",
                        "price_only": "PriceOnlyGRU",
                        "multi_noroll": "MultiModal-noRollout",
                        "no_graph": "NoGraph",
                    }[model_name]
                    results[display_name] = evaluate_model(model, loader, display_name)
                    del model
                    break

    print()
    print_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Results saved to %s", args.output)


def _name_to_key(name: str) -> str:
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
    }
    return mapping.get(name, name)


if __name__ == "__main__":
    main()
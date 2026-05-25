from __future__ import annotations

import argparse
import json
import logging
import ssl
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

PRICE_COLS = ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume", "return_1d", "market_cap"]


def denormalize_price(pred, price_stats):
    if not price_stats:
        return pred
    result = pred.copy()
    for i, col in enumerate(PRICE_COLS):
        if col in price_stats:
            m = price_stats[col]["mean"]
            s = price_stats[col]["std"] if price_stats[col]["std"] > 0 else 1.0
            result[..., i] = result[..., i] * s + m
    return result


def plot_kline(ax, pred_open, pred_high, pred_low, pred_close, ground_truth=None, horizon=10, title="K-line Chart"):
    T = min(horizon, pred_open.shape[0])
    dates = pd.date_range(start="2024-01-01", periods=T, freq="B")

    green = "#26a69a"
    red = "#ef5350"

    for t in range(T):
        o = float(pred_open[t])
        h = float(pred_high[t])
        l = float(pred_low[t])
        c = float(pred_close[t])
        color = green if c >= o else red

        body_bottom = min(o, c)
        body_height = abs(c - o)
        ax.add_patch(plt.Rectangle((t - 0.3, body_bottom), 0.6, max(body_height, 1e-6), facecolor=color, edgecolor="black", linewidth=0.5))
        ax.plot([t, t], [l, h], color=color, linewidth=1.0)

        if ground_truth is not None and t < len(ground_truth):
            gt_c = float(ground_truth[t])
            gt_color = green if gt_c >= float(pred_open[t]) else red
            ax.plot(t, gt_c, "o", color=gt_color, markersize=3, alpha=0.5)

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)


def plot_comparison(trajectory, ground_truth=None, output_path=None, price_stats=None, ticker_idx=0):
    horizon = len(trajectory)

    pred_open = []
    pred_high = []
    pred_low = []
    pred_close = []

    for step in trajectory:
        pred = step["price_pred"].numpy()
        bar = denormalize_price(pred[0], price_stats)
        pred_open.append(bar[0])
        pred_high.append(bar[1])
        pred_low.append(bar[2])
        pred_close.append(bar[3])

    pred_open = np.array(pred_open)
    pred_high = np.array(pred_high)
    pred_low = np.array(pred_low)
    pred_close = np.array(pred_close)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    dates = pd.date_range(start="2024-01-01", periods=horizon, freq="B")
    ax.plot(dates, pred_close, "b-", linewidth=2, label="FinWorld Predicted Close", marker="o", markersize=3)
    if ground_truth is not None:
        gt_close = np.array([g[3] for g in ground_truth])
        gt_dates = pd.date_range(start="2024-01-01", periods=len(gt_close), freq="B")
        ax.plot(gt_dates, gt_close, "g--", linewidth=2, label="Ground Truth", marker="s", markersize=3, alpha=0.7)
    ax.set_title("FinWorld Model Rollout: Price Trajectory", fontsize=12)
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    plot_kline(ax2, pred_open, pred_high, pred_low, pred_close, horizon=horizon, title="FinWorld Rollout K-line")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Chart saved to %s", output_path)
    else:
        plt.show()
    plt.close()


def plot_learning_curves(history, output_path=None):
    if not history or not history.get("train_loss"):
        LOGGER.warning("No learning curve data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if history.get("train_loss"):
        axes[0].plot(history["train_loss"], "b-", linewidth=2)
        if history.get("val_loss"):
            axes[0].plot(history["val_loss"], "r--", linewidth=2, label="val")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if history.get("kl"):
        axes[1].plot(history["kl"], "orange", linewidth=2)
        axes[1].set_title("KL Divergence")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("KL")
        axes[1].grid(True, alpha=0.3)

    if history.get("recon"):
        axes[2].plot(history["recon"], "green", linewidth=2)
        axes[2].set_title("Reconstruction Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MSE")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Learning curves saved to %s", output_path)
    else:
        plt.show()
    plt.close()


def plot_regime_distribution(trajectory, output_path=None):
    if not trajectory:
        return

    logits = torch.cat([step["regime_logits"].cpu() for step in trajectory])
    probs = torch.softmax(logits, dim=-1).numpy()
    regimes = ["Bull", "Bear", "Crisis", "Neutral"]
    horizon = probs.shape[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    dates = pd.date_range(start="2024-01-01", periods=horizon, freq="B")
    bottom = np.zeros(horizon)

    colors = ["#26a69a", "#ef5350", "#ff7043", "#7e57c2"]
    for i in range(probs.shape[1]):
        ax.fill_between(dates, bottom, bottom + probs[:, i], label=regimes[i] if i < len(regimes) else f"Regime {i}", color=colors[i % len(colors)], alpha=0.7)
        bottom = bottom + probs[:, i]

    ax.set_title("FinWorld Regime Probability Over Horizon")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Regime chart saved to %s", output_path)
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FinWorld Inference & Visualization")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--episode", type=str, default=None, help="Path to episode JSON")
    parser.add_argument("--data-root", default="data/processed", help="Root directory for episode data")
    parser.add_argument("--output-dir", default="outputs/inference", help="Output directory for generated charts")
    parser.add_argument("--horizon", type=int, default=10, help="Rollout horizon in days")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ticker-idx", type=int, default=0, help="Index of ticker to visualize")
    parser.add_argument("--price-stats", type=str, default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.world_model import WorldModel
    from datasets.finworld_dataset import FinWorldDataset, collate_fn
    from torch.utils.data import DataLoader

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = WorldModel(price_dim=7, news_dim=384, macro_dim=8, graph_dim=5, action_dim=8, latent_dim=128, hidden_dim=256, num_tickers=80).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    price_stats = json.load(open(args.price_stats)) if args.price_stats else {}

    if args.episode:
        with open(args.episode) as f:
            episode = json.load(f)
    else:
        dataset = FinWorldDataset(args.data_root, split=args.split, price_stats=price_stats)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        episode = next(iter(loader))
        episode = {k: v[0].tolist() if isinstance(v[0], torch.Tensor) else v[0] for k, v in episode.items()}

    model.eval()
    with torch.no_grad():
        price_seq = torch.tensor(episode["price_seq"] if "price_seq" in episode else episode["prices"], dtype=torch.float32).unsqueeze(0).to(device)
        news_feat = torch.tensor(episode["news_feat"], dtype=torch.float32).unsqueeze(0).to(device)
        macro_feat = torch.tensor(episode["macro_feat"], dtype=torch.float32).unsqueeze(0).to(device)
        edge_index = torch.tensor(episode["edge_index"], dtype=torch.long).to(device)
        edge_weight = torch.tensor(episode["edge_weight"], dtype=torch.float32).to(device)
        action = torch.zeros(1, 8, dtype=torch.float32, device=device)

        q_mu, q_logvar = model.encode(price_seq, news_feat, macro_feat, edge_index, edge_weight)
        z = q_mu
        trajectory = []

        for t in range(args.horizon):
            prior_h, prior_stats = model.transition(prev_latent=z, action=action)
            price_pred, return_pred, regime_logits = model.decoder(prior_h)
            trajectory.append({"latent": z.cpu(), "price_pred": price_pred.cpu(), "return_pred": return_pred.cpu(), "regime_logits": regime_logits.cpu()})
            action = model.action_net(prior_h)
            z = prior_h

    ground_truth = None
    if "price_target" in episode:
        gt = episode["price_target"]
        if isinstance(gt, list):
            gt = np.array(gt)
        T = min(args.horizon, gt.shape[0]) if hasattr(gt, "shape") else args.horizon
        ground_truth = gt[:T] if hasattr(gt, "shape") else None

    plot_comparison(trajectory, ground_truth=ground_truth, output_path=output_dir / "kline_rollout.png", price_stats=price_stats, ticker_idx=args.ticker_idx)
    plot_regime_distribution(trajectory, output_path=output_dir / "regime_probs.png")

    trajectory_data = []
    for t, step in enumerate(trajectory):
        trajectory_data.append({
            "step": t,
            "latent_norm": float(torch.norm(step["latent"]).item()),
            "return_pred_mean": float(step["return_pred"].mean().item()),
            "regime_probs": torch.softmax(step["regime_logits"], dim=-1).squeeze(0).tolist(),
        })
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory_data, f, indent=2)

    LOGGER.info("Inference complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
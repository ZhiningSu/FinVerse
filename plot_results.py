from __future__ import annotations

import argparse
import json
import logging
import ssl
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.finworld_dataset import FinWorldDataset, collate_fn
from evaluate import evaluate_model, load_model

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "full": "FinWorldModel",
    "price_only": "PriceOnlyGRU",
    "multi_noroll": "MultiModal-NoRollout",
    "no_graph": "NoGraph",
}
COLORS = {
    "full": "#2E86AB",
    "price_only": "#E94F37",
    "multi_noroll": "#F39237",
    "no_graph": "#1B998B",
}


def plot_learning_curves_by_model(checkpoint_root: Path, output_dir: Path):
    n_models = len(DISPLAY_NAMES)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Training Loss Curves", fontsize=13, fontweight="bold", y=1.02)

    for i, (model_key, name) in enumerate(DISPLAY_NAMES.items()):
        ax = axes[i]
        ckpt_dir = checkpoint_root / model_key
        ckpt_path = ckpt_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "last_checkpoint.pt"
        if not ckpt_path.exists():
            ax.set_title(name)
            ax.text(0.5, 0.5, "No checkpoint", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})

        epochs = np.arange(1, len(hist.get("train_loss", [])) + 1)
        if hist.get("train_loss"):
            ax.plot(epochs, hist["train_loss"], color=COLORS[model_key], linewidth=1.8, label="Train")
        if hist.get("val_loss") and len(hist["val_loss"]) == len(hist["train_loss"]):
            ax.plot(epochs, hist["val_loss"], color=COLORS[model_key], linewidth=1.8,
                    linestyle="--", marker="o", markersize=3, label="Val")

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "learning_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def plot_ablation_bars(results: dict, output_dir: Path):
    models = list(results.keys())
    metrics = ["MSE@1", "MSE@5", "MSE@10"]
    x = np.arange(len(metrics))
    width = 0.2
    n = len(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, name in enumerate(models):
        vals = [results[name][m] for m in metrics]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylabel("MSE (lower is better)", fontweight="bold")
    ax.set_title("Ablation Study: Prediction Error by Horizon", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(results[m][met] for met in metrics) for m in models) * 1.25)

    plt.tight_layout()
    out = output_dir / "ablation_bars.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def plot_horizon_comparison(results: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Error vs Prediction Horizon", fontsize=12, fontweight="bold")

    for metric, ax in zip(["MSE", "MAE"], axes):
        for model_key, name in DISPLAY_NAMES.items():
            if name not in results:
                continue
            r = results[name]
            vals = [r[f"{metric}@1"], r[f"{metric}@5"], r[f"{metric}@10"]]
            ax.plot([1, 5, 10], vals, marker="o", linewidth=2, markersize=7,
                    color=COLORS[model_key], label=name)
        ax.set_xlabel("Prediction Horizon (days)", fontweight="bold")
        ax.set_ylabel(f"{metric} (lower is better)", fontweight="bold")
        ax.set_title(f"{metric} vs Horizon", fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 5, 10])
        ax.set_xticklabels(["1", "5", "10"])

    plt.tight_layout()
    out = output_dir / "horizon_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def plot_time_series_predictions(checkpoint_dir: Path, model_key: str, dataset, device, n_episodes: int = 20, output_dir: Path = None):
    output_dir = Path(output_dir) if output_dir else Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = "best_checkpoint.pt" if (checkpoint_dir / "best_checkpoint.pt").exists() else "last_checkpoint.pt"
    ckpt_path = checkpoint_dir / ckpt_file
    if not ckpt_path.exists():
        LOGGER.warning("No checkpoint at %s", ckpt_path)
        return

    model = load_model(str(ckpt_path), model_key, device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    name = DISPLAY_NAMES.get(model_key, model_key)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Time Series Predictions: {name}", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    shown = 0
    for i, batch in enumerate(loader):
        if shown >= n_episodes:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        price_seq = batch["price_seq"]
        news_feat = batch["news_feat"]
        macro_feat = batch["macro_feat"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        action = batch["action"]
        target = batch["price_target"]

        if model_key == "price_only":
            out = model(price_seq)
        else:
            out = model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action)

        pred = out["price_pred"].detach().cpu().numpy()
        tgt = target.detach().cpu().numpy()

        if pred.ndim == 3:
            pred = pred[0]
        elif pred.ndim == 2:
            pred = pred[0]
        if tgt.ndim == 3:
            tgt = tgt[0]

        ax = axes[shown % 4]
        n_steps = min(len(pred), len(tgt))
        days = np.arange(n_steps)
        if pred.ndim == 2 and pred.shape[1] >= 4:
            p = pred[:n_steps, 3]
        else:
            p = pred[:n_steps]
        if tgt.ndim == 2 and tgt.shape[1] >= 4:
            t = tgt[:n_steps, 3]
        else:
            t = tgt[:n_steps]

        ax.plot(days, t, color="#2E86AB", linewidth=2, label="Ground Truth", alpha=0.9)
        ax.plot(days, p, color="#E94F37", linewidth=2, linestyle="--", label="Predicted", alpha=0.9)
        ax.set_title(f"Sample {i + 1}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalized Price")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        shown += 1
        if shown % 4 == 0 and shown < n_episodes:
            plt.tight_layout()
            out = output_dir / f"ts_{model_key}_batch{shown//4}.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            LOGGER.info("Saved: %s", out)
            plt.close()
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            fig.suptitle(f"Time Series Predictions: {name}", fontsize=13, fontweight="bold")

    if shown % 4 != 0 or shown == 0:
        plt.tight_layout()
        out = output_dir / f"ts_{model_key}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        LOGGER.info("Saved: %s", out)

    plt.close()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def plot_all_ablation_summary(results: dict, output_dir: Path):
    models = list(results.keys())
    metrics = ["MSE@1", "MSE@5", "MSE@10", "MAE@1", "MAE@5", "MAE@10"]
    n_models = len(models)
    width = 0.12
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, name in enumerate(models):
        model_key = next((k for k, v in DISPLAY_NAMES.items() if v == name), name)
        vals = [results[name][m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name, color=COLORS.get(model_key, None), alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylabel("Error (lower is better)", fontweight="bold")
    ax.set_title("Ablation Study: Full Model vs Baselines", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="gray", loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    for name in models:
        if "FinWorldModel" in name:
            best_baseline_mse = min(results[m]["MSE@1"] for m in models if m != name)
            finworld_mse = results[name]["MSE@1"]
            improvement = (best_baseline_mse - finworld_mse) / best_baseline_mse * 100
            ax.text(0.98, 0.95, f"FinWorldModel MSE@1 improvement vs best baseline: {improvement:.1f}%",
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))
            break

    plt.tight_layout()
    out = output_dir / "ablation_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def save_latex_table(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Prediction Error of FinWorldModel vs Baselines}")
    lines.append("\\begin{tabular}{l|ccc|ccc}")
    lines.append("\\hline \\hline")
    lines.append("Model & MSE@1 & MSE@5 & MSE@10 & MAE@1 & MAE@5 & MAE@10 \\\\ \\hline")

    for name, m in results.items():
        row = f"{name} & {m['MSE@1']:.4f} & {m['MSE@5']:.4f} & {m['MSE@10']:.4f} & {m['MAE@1']:.4f} & {m['MAE@5']:.4f} & {m['MAE@10']:.4f} \\\\"
        lines.append(row)

    lines.append("\\hline \\hline")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\end{table}")

    tex_path = output_dir / "ablation_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    LOGGER.info("Saved: %s", tex_path)


def main():
    parser = argparse.ArgumentParser(description="Generate paper plots for FinWorld experiments")
    parser.add_argument("--checkpoint-root", type=str, default="outputs",
        help="Root directory containing model subfolders with checkpoints")
    parser.add_argument("--data-root", default="data/processed/real")
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--max-test", type=int, default=500, help="Max test episodes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--skip-ts", action="store_true", help="Skip time series plots")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (skip plots)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)
    device = torch.device(args.device)

    dataset = FinWorldDataset(args.data_root, split="test", max_episodes=args.max_test)
    LOGGER.info("Test dataset: %d episodes", len(dataset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    results = {}
    for model_key, name in DISPLAY_NAMES.items():
        ckpt_dir = checkpoint_root / model_key
        ckpt_file = "best_checkpoint.pt" if (ckpt_dir / "best_checkpoint.pt").exists() else "last_checkpoint.pt"
        ckpt_path = ckpt_dir / ckpt_file
        if not ckpt_path.exists():
            LOGGER.warning("No checkpoint for %s at %s, skipping", model_key, ckpt_path)
            continue

        LOGGER.info("Evaluating %s...", name)
        model = load_model(str(ckpt_path), model_key, device, args.hidden_dim, args.latent_dim)
        results[name] = evaluate_model(model, loader, name)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    LOGGER.info("\n=== Results ===")
    for name, m in results.items():
        LOGGER.info("  %s: MSE@1=%.4f MSE@5=%.4f MSE@10=%.4f", name, m["MSE@1"], m["MSE@5"], m["MSE@10"])

    eval_out = output_dir / "eval_results.json"
    with open(eval_out, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Saved: %s", eval_out)

    if args.eval_only:
        return

    LOGGER.info("Generating plots...")
    plot_ablation_bars(results, output_dir)
    plot_horizon_comparison(results, output_dir)
    plot_all_ablation_summary(results, output_dir)
    save_latex_table(results, output_dir)

    if not args.skip_ts:
        for model_key, name in DISPLAY_NAMES.items():
            ckpt_dir = checkpoint_root / model_key
            if (ckpt_dir / "best_checkpoint.pt").exists() or (ckpt_dir / "last_checkpoint.pt").exists():
                LOGGER.info("Time series plot for %s...", name)
                ts_dataset = FinWorldDataset(args.data_root, split="test", max_episodes=50)
                plot_time_series_predictions(ckpt_dir, model_key, ts_dataset, device, n_episodes=50, output_dir=output_dir)

    plot_learning_curves_by_model(checkpoint_root, output_dir)

    LOGGER.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
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
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from datasets.finworld_dataset import FinWorldDataset, collate_fn
from evaluate import evaluate_model, load_model, MODEL_REGISTRY

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "full": "FinWorldModel",
    "price_only": "PriceOnlyGRU",
    "multi_noroll": "MultiModal-noRollout",
    "no_graph": "NoGraph",
}


def plot_learning_curves_by_model(output_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")

    for ax, (model_key, name) in zip(axes, DISPLAY_NAMES.items()):
        ckpt_dir = output_dir / model_key
        ckpt_path = ckpt_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "last_checkpoint.pt"
        if not ckpt_path.exists():
            ax.set_title(name)
            ax.text(0.5, 0.5, "No checkpoint", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Epoch")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})

        if hist.get("train_loss"):
            axes_actual = ax if model_key == "full" else fig.add_subplot(1, 4, list(DISPLAY_NAMES).index(model_key) + 1)
            ax.plot(hist["train_loss"], "b-", linewidth=2, label="Train")
            if hist.get("val_loss") and len(hist["val_loss"]) == len(hist["train_loss"]):
                ax.plot(hist["val_loss"], "r--", linewidth=2, label="Val")
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def plot_ablation_bars(results: dict, output_dir: Path):
    models = list(results.keys())
    metrics = ["MSE@1", "MSE@5", "MSE@10"]
    horizon_labels = ["H=1", "H=5", "H=10"]

    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(models):
        vals = [results[name][m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({l})" for m, l in zip(metrics, horizon_labels)])
    ax.set_ylabel("MSE (lower is better)")
    ax.set_title("Prediction Error by Horizon", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_dir / "ablation_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def plot_horizon_comparison(results: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Horizon Comparison", fontsize=14, fontweight="bold")

    for metric, ax in zip(["MSE", "MAE"], axes):
        for name in results:
            vals = [results[name][f"{metric}@1"], results[name][f"{metric}@5"], results[name][f"{metric}@10"]]
            ax.plot([1, 5, 10], vals, marker="o", linewidth=2, markersize=6, label=name)
        ax.set_xlabel("Prediction Horizon (days)")
        ax.set_ylabel(f"{metric} (lower is better)")
        ax.set_title(f"{metric} vs Horizon")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 5, 10])

    plt.tight_layout()
    out = output_dir / "horizon_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Time Series Predictions: {DISPLAY_NAMES.get(model_key, model_key)}", fontsize=14, fontweight="bold")
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

        pred = out["price_pred"].cpu().numpy()
        tgt = target.cpu().numpy()

        if pred.ndim == 3:
            pred = pred[0]
        elif pred.ndim == 2:
            pred = pred[0]
        if tgt.ndim == 3:
            tgt = tgt[0]

        ax = axes[shown % 4]
        days = np.arange(1, min(len(pred), len(tgt)) + 1)
        p = pred[:len(tgt), 3] if pred.shape[1] >= 4 else pred[:len(tgt), 0]
        t = tgt[:, 3] if tgt.shape[1] >= 4 else tgt[:, 0]

        ax.plot(days, t, "g-", linewidth=2, label="Ground Truth", alpha=0.8)
        ax.plot(days, p, "b--", linewidth=2, label="Predicted", alpha=0.8)
        ax.set_title(f"Episode {i + 1}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalized Close")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        shown += 1
        if shown % 4 == 0:
            plt.tight_layout()
            out = output_dir / f"ts_{model_key}_batch{shown//4}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            LOGGER.info("Saved: %s", out)
            plt.close()
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            fig.suptitle(f"Time Series Predictions: {DISPLAY_NAMES.get(model_key, model_key)}", fontsize=14, fontweight="bold")

    if shown % 4 != 0 or shown == 0:
        plt.tight_layout()
        out = output_dir / f"ts_{model_key}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, name in enumerate(models):
        vals = [results[name][m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("Ablation Study: Full Model vs Baselines", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    improvement = None
    for name in models:
        if "FinWorldModel" in name:
            best_baseline_mse = min(results[m]["MSE@1"] for m in models if m != name)
            finworld_mse = results[name]["MSE@1"]
            improvement = (best_baseline_mse - finworld_mse) / best_baseline_mse * 100
            ax.text(0.02, 0.95, f"FinWorld MSE@1 improvement vs best baseline: {improvement:.1f}%",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            break

    plt.tight_layout()
    out = output_dir / "ablation_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", out)
    plt.close()


def save_latex_table(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study: FinWorldModel vs Baselines}")
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
    parser.add_argument("--data-root", default="data/processed/real")
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--max-test", type=int, default=500, help="Max test episodes for evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--skip-ts", action="store_true", help="Skip time series plots")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (skip plots)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = FinWorldDataset(args.data_root, split="test", max_episodes=args.max_test)
    LOGGER.info("Test dataset: %d episodes", len(dataset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    results = {}
    for model_key, name in DISPLAY_NAMES.items():
        ckpt_dir = Path(f"outputs/{model_key}")
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
            ckpt_dir = Path(f"outputs/{model_key}")
            if (ckpt_dir / "best_checkpoint.pt").exists() or (ckpt_dir / "last_checkpoint.pt").exists():
                LOGGER.info("Time series plot for %s...", name)
                ts_dataset = FinWorldDataset(args.data_root, split="test", max_episodes=50)
                plot_time_series_predictions(ckpt_dir, model_key, ts_dataset, device, n_episodes=50, output_dir=output_dir)

    for model_key in DISPLAY_NAMES:
        plot_learning_curves_by_model(Path("outputs") / model_key)

    LOGGER.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
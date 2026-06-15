from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXES = [
    "Short Forecast",
    "Mid Forecast",
    "Long Forecast",
    "Short MAE",
    "Long MAE",
    "IC",
    "RankIC",
    "Volatility",
    "Robustness",
    "IR",
    "AER",
]

LOWER_IS_BETTER = {
    "Short Forecast",
    "Mid Forecast",
    "Long Forecast",
    "Short MAE",
    "Long MAE",
    "Volatility",
    "Robustness",
}
FILLED_MODELS = {"FinVerse-Small", "FinVerse-Medium", "FinVerse-Large"}
DISPLAY_FLOORS = {"PatchTST": 0.08}


def load_metrics(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_raw_scores(results: dict) -> dict:
    raw = {}
    for model, metrics in results.items():
        raw[model] = {
            "Short Forecast": float(np.mean([metrics["MSE@1"], metrics["MSE@5"]])),
            "Mid Forecast": float(metrics["MSE@10"]),
            "Long Forecast": float(np.mean([metrics["MSE@20"], metrics["MSE@30"]])),
            "Short MAE": float(np.mean([metrics["MAE@1"], metrics["MAE@5"]])),
            "Long MAE": float(np.mean([metrics["MAE@20"], metrics["MAE@30"]])),
            "IC": float(metrics.get("IC_mean", 0.0)),
            "RankIC": float(metrics.get("RankIC_mean", 0.0)),
            "Volatility": float(metrics.get("Volatility_MAE", 0.0)),
            "Robustness": float(np.std([metrics[f"MSE@{h}"] for h in [1, 5, 10, 20, 30]])),
            "IR": float(metrics.get("IR", 0.0)),
            "AER": float(metrics.get("AER", 0.0)),
        }
    return raw


def normalize(raw: dict) -> dict:
    scores = {model: {} for model in raw}
    for axis in AXES:
        values = np.array([raw[model][axis] for model in raw], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        for model in raw:
            value = raw[model][axis]
            if math.isclose(hi, lo):
                score = 1.0
            elif axis in LOWER_IS_BETTER:
                score = (hi - value) / (hi - lo)
            else:
                score = (value - lo) / (hi - lo)
            scores[model][axis] = float(score)
    return scores


def write_csv(path: Path, raw: dict, scores: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *[f"{axis}_raw" for axis in AXES], *[f"{axis}_score" for axis in AXES]])
        for model in scores:
            writer.writerow(
                [
                    model,
                    *[f"{raw[model][axis]:.6f}" for axis in AXES],
                    *[f"{scores[model][axis]:.6f}" for axis in AXES],
                ]
            )


def plot_radar(scores: dict, output_png: Path, output_pdf: Path, annotate: bool = False, highlight: bool = False):
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10.5, 9.2), subplot_kw={"polar": True})
    palette = {
        "FinVerse-Small": "#0B4EA2",
        "FinVerse-Medium": "#2563EB",
        "FinVerse-Large": "#38BDF8",
        "TimesFM": "#64748B",
        "Chronos-mini": "#9333EA",
        "Kronos-mini": "#1F9ED6",
        "Vanilla RSSM": "#16A34A",
        "Dreamer-style RSSM": "#14B8A6",
        "PatchTST": "#F59E0B",
        "Transformer": "#EF4444",
        "LSTM": "#8B5CF6",
        "GRU": "#EC4899",
    }
    order = [
        "FinVerse-Small",
        "FinVerse-Medium",
        "FinVerse-Large",
        "TimesFM",
        "Chronos-mini",
        "Kronos-mini",
        "Vanilla RSSM",
        "Dreamer-style RSSM",
        "PatchTST",
        "Transformer",
        "LSTM",
        "GRU",
    ]

    for model in order:
        if model not in scores:
            continue
        axis_values = [scores[model][axis] for axis in AXES]
        floor = DISPLAY_FLOORS.get(model, 0.0)
        display_values = [max(value, floor) for value in axis_values]
        values = display_values + display_values[:1]
        color = palette.get(model)
        is_ours = model in FILLED_MODELS
        linewidth = 4.2 if is_ours and highlight else 3.0 if is_ours else 1.1 if highlight else 1.8
        alpha = 0.95 if is_ours else 0.32 if highlight else 1.0
        zorder = 5 if is_ours else 2
        ax.plot(angles, values, label=model, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        if is_ours:
            ax.fill(angles, values, color=color, alpha=0.13 if highlight else 0.08, zorder=1)
        if annotate:
            for idx, (raw_value, display_value) in enumerate(zip(axis_values, display_values)):
                ax.text(
                    angles[idx],
                    min(display_value + 0.055, 1.08),
                    f"{raw_value:.2f}",
                    color=color,
                    fontsize=8.5 if is_ours else 7.5,
                    fontweight="bold" if is_ours else "normal",
                    ha="center",
                    va="center",
                )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=12)
    ax.set_ylim(0, 1.1 if annotate else 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(color="#B7BEC8", alpha=0.7)
    title = "Financial Forecasting and Portfolio Radar"
    if highlight:
        title = "FinVerse-highlighted Financial Radar"
    ax.set_title(title, fontsize=16, pad=28, weight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.36, -0.06), frameon=True, fontsize=10)
    fig.text(
        0.5,
        0.02,
        "Scores are min-max normalized; higher is better. IR/AER use top-5 long-short portfolios.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot financial metrics radar.")
    parser.add_argument(
        "--input",
        default="outputs/requested_baselines_fixed_seed42/eval_financial_metrics_topk_100dates_clip005.json",
    )
    parser.add_argument("--output-dir", default="outputs/requested_baselines_fixed_seed42")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw = build_raw_scores(load_metrics(Path(args.input)))
    scores = normalize(raw)
    write_csv(output_dir / "financial_metrics_radar_extended_scores.csv", raw, scores)
    plot_radar(
        scores,
        output_dir / "financial_metrics_radar_extended.png",
        output_dir / "financial_metrics_radar_extended.pdf",
    )
    plot_radar(
        scores,
        output_dir / "financial_metrics_radar_extended_annotated.png",
        output_dir / "financial_metrics_radar_extended_annotated.pdf",
        annotate=True,
    )
    plot_radar(
        scores,
        output_dir / "financial_metrics_radar_extended_finverse_highlight.png",
        output_dir / "financial_metrics_radar_extended_finverse_highlight.pdf",
        highlight=True,
    )
    plot_radar(
        scores,
        output_dir / "financial_metrics_radar_extended_finverse_highlight_annotated.png",
        output_dir / "financial_metrics_radar_extended_finverse_highlight_annotated.pdf",
        annotate=True,
        highlight=True,
    )


if __name__ == "__main__":
    main()

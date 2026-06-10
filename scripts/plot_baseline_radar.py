from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZONS = [1, 5, 10, 20, 30]
DISPLAY_FLOORS = {
    # Visual-only floor to keep very low PatchTST dimensions readable on radar.
    # Raw normalized scores in the CSV remain unchanged.
    "PatchTST": 0.08,
}


def mean(values):
    return float(sum(values) / len(values))


def load_metrics(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_raw_scores(results: dict) -> dict:
    raw = {}
    for model, metrics in results.items():
        mse = {h: metrics[f"MSE@{h}"] for h in HORIZONS}
        mae = {h: metrics[f"MAE@{h}"] for h in HORIZONS}
        raw[model] = {
            "Short Forecast": mean([mse[1], mse[5]]),
            "Mid Forecast": mse[10],
            "Long Forecast": mean([mse[20], mse[30]]),
            "Short MAE": mean([mae[1], mae[5]]),
            "Long MAE": mean([mae[20], mae[30]]),
            "Robustness": float(np.std([mse[h] for h in HORIZONS])),
        }
    return raw


def normalize_inverse(raw: dict) -> dict:
    axes = list(next(iter(raw.values())).keys())
    scores = {model: {} for model in raw}
    for axis in axes:
        values = np.array([raw[model][axis] for model in raw], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        for model in raw:
            value = raw[model][axis]
            if math.isclose(hi, lo):
                score = 1.0
            else:
                score = (hi - value) / (hi - lo)
            scores[model][axis] = float(score)
    return scores


def write_csv(path: Path, scores: dict):
    axes = list(next(iter(scores.values())).keys())
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *axes])
        for model, model_scores in scores.items():
            writer.writerow([model, *[f"{model_scores[axis]:.6f}" for axis in axes]])


def plot_radar(scores: dict, output_png: Path, output_pdf: Path, annotate: bool = False):
    axes = list(next(iter(scores.values())).keys())
    n = len(axes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"polar": True})
    palette = {
        "FinVerse": "#0B4EA2",
        "FinVerse-Small": "#0B4EA2",
        "FinVerse-Medium": "#2563EB",
        "FinVerse-Large": "#38BDF8",
        "Kronos-mini": "#1F9ED6",
        "Vanilla RSSM": "#16A34A",
        "PatchTST": "#F59E0B",
        "Transformer": "#EF4444",
        "LSTM": "#8B5CF6",
        "GRU": "#EC4899",
    }
    filled_models = {"FinVerse-Small", "FinVerse-Medium", "FinVerse-Large"}

    order = [
        "FinVerse-Small",
        "FinVerse-Medium",
        "FinVerse-Large",
        "FinVerse",
        "Kronos-mini",
        "Vanilla RSSM",
        "PatchTST",
        "Transformer",
        "LSTM",
        "GRU",
    ]
    for model in order:
        if model not in scores:
            continue
        axis_values = [scores[model][axis] for axis in axes]
        floor = DISPLAY_FLOORS.get(model, 0.0)
        display_axis_values = [max(value, floor) for value in axis_values]
        values = display_axis_values + display_axis_values[:1]
        color = palette.get(model, None)
        is_ours = model in filled_models
        linewidth = 3.0 if is_ours else 1.8
        ax.plot(angles, values, label=model, color=color, linewidth=linewidth)
        if is_ours:
            ax.fill(angles, values, color=color, alpha=0.08)
        if annotate:
            for idx, (value, display_value) in enumerate(zip(axis_values, display_axis_values)):
                radius = min(display_value + 0.055, 1.08)
                ax.text(
                    angles[idx],
                    radius,
                    f"{value:.2f}",
                    color=color,
                    fontsize=7.5 if not is_ours else 8.5,
                    fontweight="bold" if is_ours else "normal",
                    ha="center",
                    va="center",
                )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=12)
    ax.set_ylim(0, 1.1 if annotate else 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(color="#B7BEC8", alpha=0.7)
    ax.set_title("Baseline and FinVerse Scale Forecasting Radar", fontsize=16, pad=24, weight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.32, -0.05), frameon=True, fontsize=10)

    fig.text(
        0.5,
        0.02,
        "Scores are min-max normalized from MSE/MAE proxies; higher is better.",
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
    parser = argparse.ArgumentParser(description="Plot baseline radar from evaluation metrics.")
    parser.add_argument("--input", default="outputs/requested_baselines_small/eval_mainstream_500.json")
    parser.add_argument("--output-dir", default="outputs/requested_baselines_small")
    parser.add_argument("--rename", action="append", default=[], help="Rename a model as old=new.")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Add an extra result as label=path::source_key. If source_key is omitted, the first key is used.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results = load_metrics(input_path)
    for spec in args.rename:
        old, new = spec.split("=", 1)
        if old in results:
            results[new] = results.pop(old)
    for spec in args.extra:
        label, rest = spec.split("=", 1)
        if "::" in rest:
            path_text, source_key = rest.split("::", 1)
        else:
            path_text, source_key = rest, None
        extra_result = load_metrics(Path(path_text))
        if source_key is None:
            source_key = next(iter(extra_result))
        results[label] = extra_result[source_key]
    raw = build_raw_scores(results)
    scores = normalize_inverse(raw)

    write_csv(output_dir / "baseline_radar_proxy_scores.csv", scores)
    plot_radar(
        scores,
        output_dir / "baseline_radar_proxy.png",
        output_dir / "baseline_radar_proxy.pdf",
    )
    plot_radar(
        scores,
        output_dir / "baseline_radar_proxy_annotated.png",
        output_dir / "baseline_radar_proxy_annotated.pdf",
        annotate=True,
    )


if __name__ == "__main__":
    main()

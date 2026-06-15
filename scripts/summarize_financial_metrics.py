from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_ORDER = [
    "LSTM",
    "PatchTST",
    "TimesFM",
    "Chronos-mini",
    "Kronos-mini",
    "Vanilla RSSM",
    "Dreamer-style RSSM",
    "FinVerse-Small",
    "FinVerse-Medium",
    "FinVerse-Large",
    "GRU",
    "Transformer",
]

COLUMNS = [
    ("MSE@1", "MSE@1", "lower"),
    ("MSE@30", "MSE@30", "lower"),
    ("IC", "IC_mean", "higher"),
    ("RankIC", "RankIC_mean", "higher"),
    ("Vol. MAE", "Volatility_MAE", "lower"),
]


def add_derived(metrics: dict) -> None:
    if metrics.get("Daily_Mean_Return") is None and metrics.get("AER") is not None:
        metrics["Daily_Mean_Return"] = float(metrics["AER"]) / 252.0
    if metrics.get("IR_Daily") is None and metrics.get("IR") is not None:
        metrics["IR_Daily"] = float(metrics["IR"]) / (252.0 ** 0.5)
    if metrics.get("IR_Annualized") is None and metrics.get("IR") is not None:
        metrics["IR_Annualized"] = metrics["IR"]
    if metrics.get("Daily_Return_Std") is None:
        daily_mean = metrics.get("Daily_Mean_Return")
        daily_ir = metrics.get("IR_Daily")
        if daily_mean is None or daily_ir is None or abs(float(daily_ir)) < 1e-12:
            metrics["Daily_Return_Std"] = 0.0
        else:
            metrics["Daily_Return_Std"] = abs(float(daily_mean) / float(daily_ir))


def fmt(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def best_models(results: dict) -> dict[str, set[str]]:
    best = {}
    for _, key, direction in COLUMNS:
        values = {
            model: float(results[model][key])
            for model in MODEL_ORDER
            if model in results and results[model].get(key) is not None
        }
        if not values:
            best[key] = set()
            continue
        target = min(values.values()) if direction == "lower" else max(values.values())
        best[key] = {model for model, value in values.items() if abs(value - target) < 1e-12}
    return best


def write_tex(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = best_models(results)
    rows = []
    for model in MODEL_ORDER:
        if model not in results:
            continue
        cells = [model]
        for _, key, _ in COLUMNS:
            value = fmt(results[model].get(key))
            if model in best[key]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        rows.append(" & ".join(cells) + r" \\")
    header = "Model & " + " & ".join(label for label, _, _ in COLUMNS) + r" \\"
    body = "\n".join(rows)
    tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Financial prediction diagnostics on 100 complete test trading days. We report forecasting error, cross-sectional IC/RankIC, and volatility error, and exclude portfolio PnL from the main table because short-window daily portfolio returns are highly sensitive to ranking noise and annualization choices.}}
\label{{tab:financial_metrics}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lccccc}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    path.write_text(tex)


def write_csv(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *[label for label, _, _ in COLUMNS]])
        for model in MODEL_ORDER:
            if model in results:
                writer.writerow([model, *[fmt(results[model].get(key)) for _, key, _ in COLUMNS]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize financial metrics with transparent daily return fields.")
    parser.add_argument("--input", type=Path, default=Path("outputs/requested_baselines_fixed_seed42/eval_financial_metrics_topk_100dates_clip005.json"))
    parser.add_argument("--tex-output", type=Path, default=Path("paper/tables/financial_metrics_table.tex"))
    parser.add_argument("--csv-output", type=Path, default=Path("paper/tables/financial_metrics_table.csv"))
    args = parser.parse_args()

    with args.input.open() as f:
        results = json.load(f)
    for metrics in results.values():
        add_derived(metrics)
    write_tex(results, args.tex_output)
    write_csv(results, args.csv_output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_ORDER = [
    "Full FinVerse",
    "w/o Dual VQ",
    "w/o Cross-Asset Ctx",
    "w/o Probabilistic WM",
    "Price Only",
]

PORTFOLIO_MODEL_ORDER = [
    *MODEL_ORDER,
    "BUY&HOLD",
]

FORECAST_COLUMNS = [
    ("MSE@1", "MSE@1", "lower"),
    ("MSE@5", "MSE@5", "lower"),
    ("MSE@30", "MSE@30", "lower"),
    ("MAE@1", "MAE@1", "lower"),
    ("MAE@30", "MAE@30", "lower"),
    ("IC", "IC_mean", "higher"),
    ("RankIC", "RankIC_mean", "higher"),
    ("Vol. MAE", "Volatility_MAE", "lower"),
]

PORTFOLIO_COLUMNS = [
    ("MSE@1", "MSE@1", "lower"),
    ("MSE@30", "MSE@30", "lower"),
    ("IC", "IC_mean", "higher"),
    ("RankIC", "RankIC_mean", "higher"),
    ("Vol. MAE", "Volatility_MAE", "lower"),
    ("Daily Mean", "Daily_Mean_Return", "higher"),
    ("Daily Std", "Daily_Return_Std", "lower"),
    ("Daily IR", "IR_Daily", "higher"),
    ("Ann. Ret.", "AER", "higher"),
]


def load_json(path: Path) -> dict:
    with path.open() as f:
        payload = json.load(f)
    if "w/o Graph" in payload and "w/o Cross-Asset Ctx" not in payload:
        payload["w/o Cross-Asset Ctx"] = payload.pop("w/o Graph")
    for metrics in payload.values():
        add_portfolio_derived_metrics(metrics)
    return payload


def add_portfolio_derived_metrics(metrics: dict) -> None:
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


def fmt(value: float) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def best_models(results: dict, columns: list[tuple[str, str, str]]) -> dict[str, set[str]]:
    best: dict[str, set[str]] = {}
    for _, key, direction in columns:
        values = {
            model: float(results[model][key])
            for model in results
            if results[model].get(key) is not None
        }
        if not values:
            best[key] = set()
            continue
        target = min(values.values()) if direction == "lower" else max(values.values())
        best[key] = {model for model, value in values.items() if abs(value - target) < 1e-12}
    return best


def write_csv(path: Path, results: dict, columns: list[tuple[str, str, str]], model_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *[label for label, _, _ in columns]])
        for model in model_order:
            if model not in results:
                continue
            writer.writerow([model, *[fmt(results[model].get(key)) for _, key, _ in columns]])


def latex_table(
    caption: str,
    label: str,
    results: dict,
    columns: list[tuple[str, str, str]],
    note: str,
    model_order: list[str],
) -> str:
    best = best_models(results, columns)
    header = " & ".join(["Model", *[label for label, _, _ in columns]])
    rows = []
    for model in model_order:
        if model not in results:
            continue
        cells = [model]
        for _, key, _ in columns:
            raw_value = results[model].get(key)
            value = fmt(raw_value)
            if model in best[key]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        rows.append(" & ".join(cells) + r" \\")

    col_spec = "l" + "c" * len(columns)
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
{header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\vspace{{2pt}}
\footnotesize{{{note}}}
\end{{table*}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FinVerse ablation results.")
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    forecast = load_json(args.forecast)
    portfolio = load_json(args.portfolio)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "ablation_forecast_table.csv", forecast, FORECAST_COLUMNS, MODEL_ORDER)
    write_csv(args.output_dir / "ablation_topk_table.csv", portfolio, PORTFOLIO_COLUMNS, PORTFOLIO_MODEL_ORDER)

    forecast_note = (
        "All variants use the same training split, validation split, seed, return target, "
        "and 500 held-out test episodes. Lower is better for MSE, MAE, and Vol. MAE; "
        "higher is better for IC and RankIC."
    )
    portfolio_note = (
        "Neural model variants are evaluated on 100 complete test trading days with "
        "top-k=5 long-short portfolios and return clipping at 5\\%. BUY\\&HOLD is an "
        "equal-weight long-only market-basket baseline over the same dates. We report "
        "daily return statistics to avoid relying only on potentially misleading annualized IR."
    )

    tex = "\n\n".join(
        [
            latex_table(
                "Same-protocol ablation results for forecasting quality.",
                "tab:ablation_forecast",
                forecast,
                FORECAST_COLUMNS,
                forecast_note,
                MODEL_ORDER,
            ),
            latex_table(
                "Same-protocol ablation and buy-and-hold portfolio evaluation.",
                "tab:ablation_topk",
                portfolio,
                PORTFOLIO_COLUMNS,
                portfolio_note,
                PORTFOLIO_MODEL_ORDER,
            ),
        ]
    )
    (args.output_dir / "ablation_tables.tex").write_text(tex)


if __name__ == "__main__":
    main()

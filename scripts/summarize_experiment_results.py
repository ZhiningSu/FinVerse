from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORECAST_COLUMNS = [
    "MSE@1",
    "MSE@5",
    "MSE@10",
    "MSE@20",
    "MSE@30",
    "MAE@1",
    "IC_mean",
    "RankIC_mean",
    "Volatility_MAE",
    "IR_Daily",
    "IR_Annualized",
    "AER",
    "n_samples",
]

ROLLOUT_COLUMNS = [f"StateMSE@{h}" for h in [1, 5, 10, 20, 30]]
CRISIS_COLUMNS = [f"CrisisStateMSE@{h}" for h in [1, 5, 10, 20, 30]]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return "_No results found._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _forecast_rows(results: dict[str, Any]) -> list[list[str]]:
    rows = []
    for model, metrics in results.items():
        rows.append([model, *[_fmt(metrics.get(col)) for col in FORECAST_COLUMNS]])
    return rows


def _evidence_rows(evidence: dict[str, Any]) -> list[list[str]]:
    results = evidence.get("results", {})
    rows = []
    for model, payload in results.items():
        rollout = payload.get("rollout_fidelity", {})
        crisis = payload.get("crisis_simulation", {})
        shock = payload.get("counterfactual_macro_shock", {})
        rows.append(
            [
                model,
                *[_fmt(rollout.get(col)) for col in ROLLOUT_COLUMNS],
                _fmt(payload.get("regime_accuracy")),
                _fmt(payload.get("regime_macro_f1")),
                _fmt(shock.get("mean_abs_prediction_delta")),
                _fmt(shock.get("direction_flip_rate")),
                *[_fmt(crisis.get(col)) for col in CRISIS_COLUMNS],
                _fmt(crisis.get("n_crisis_samples")),
                _fmt(payload.get("n_samples")),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FinWorld experiment outputs as Markdown.")
    parser.add_argument("--core-json", default="outputs/final_results/eval_core_ablation_1000.json")
    parser.add_argument("--mainstream-json", default="outputs/final_results/eval_mainstream_all_1000.json")
    parser.add_argument("--evidence-json", default="outputs/paper_experiments/world_model_evidence_1000.json")
    parser.add_argument("--output", default="outputs/final_results/final_experiment_results.md")
    args = parser.parse_args()

    core = _read_json(Path(args.core_json))
    mainstream = _read_json(Path(args.mainstream_json))
    evidence = _read_json(Path(args.evidence_json))

    forecast_headers = ["Model", *FORECAST_COLUMNS]
    evidence_headers = [
        "Model",
        *ROLLOUT_COLUMNS,
        "RegimeAcc",
        "RegimeMacroF1",
        "ShockMeanAbsDelta",
        "ShockDirectionFlipRate",
        *CRISIS_COLUMNS,
        "CrisisN",
        "n_samples",
    ]

    parts = [
        "# Final Experiment Results",
        "",
        "## Mainstream Forecasting And Financial Diagnostics",
        "",
        _markdown_table(_forecast_rows(mainstream), forecast_headers),
        "## Core Ablation Forecasting And Financial Diagnostics",
        "",
        _markdown_table(_forecast_rows(core), forecast_headers),
        "## World Model Evidence",
        "",
        _markdown_table(_evidence_rows(evidence), evidence_headers),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

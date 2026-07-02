from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.finworld_dataset import FinWorldDataset, collate_fn
from evaluate import load_model, set_seed


HORIZONS = [1, 5, 10, 20, 30]


def _regime_labels(target: np.ndarray, bear_q: float, bull_q: float) -> np.ndarray:
    labels = np.ones_like(target, dtype=np.int64)
    labels[target <= bear_q] = 0
    labels[target >= bull_q] = 2
    return labels


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    scores = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = 0.0 if np.isclose(tp + fp, 0.0) else tp / (tp + fp)
        recall = 0.0 if np.isclose(tp + fn, 0.0) else tp / (tp + fn)
        f1 = 0.0 if np.isclose(precision + recall, 0.0) else 2 * precision * recall / (precision + recall)
        scores.append(f1)
    return float(np.mean(scores))


def _select_regime_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits[:, : min(5, logits.size(1)), :].mean(dim=1)
    return logits[:, :3]


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _best_by_metric(results: dict, metric: str) -> set[str]:
    values = {
        name: float(payload["rollout_fidelity"][metric])
        for name, payload in results.items()
        if metric in payload.get("rollout_fidelity", {})
    }
    if not values:
        return set()
    best = min(values.values())
    return {name for name, value in values.items() if abs(value - best) < 1e-12}


def _best_by_crisis_metric(results: dict, metric: str) -> set[str]:
    values = {
        name: float(payload["crisis_simulation"][metric])
        for name, payload in results.items()
        if metric in payload.get("crisis_simulation", {})
    }
    if not values:
        return set()
    best = min(values.values())
    return {name for name, value in values.items() if abs(value - best) < 1e-12}


def write_rollout_table(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [f"StateMSE@{h}" for h in HORIZONS]
    best = {metric: _best_by_metric(results, metric) for metric in metrics}
    rows = []
    for name, payload in results.items():
        cells = [name]
        for metric in metrics:
            value = _format_float(float(payload["rollout_fidelity"][metric]))
            if name in best[metric]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        rows.append(" & ".join(cells) + r" \\")

    header = "Model & " + " & ".join(metrics) + r" \\"
    body = "\n".join(rows)
    tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Rollout fidelity for imagined return trajectories. Lower StateMSE indicates that the imagined trajectory better matches the realized future trajectory.}}
\label{{tab:rollout_fidelity}}
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
    output_path.write_text(tex)


def write_rollout_csv(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [f"StateMSE@{h}" for h in HORIZONS] + [f"StateMAE@{h}" for h in HORIZONS]
    crisis_metrics = [f"CrisisStateMSE@{h}" for h in HORIZONS]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *metrics, *crisis_metrics, "CrisisN", "DirectionCorr@1", "RegimeAcc", "RegimeMacroF1", "ShockMeanAbsDelta", "ShockDirectionFlipRate", "n_samples"])
        for name, payload in results.items():
            rollout = payload["rollout_fidelity"]
            shock = payload["counterfactual_macro_shock"]
            crisis = payload["crisis_simulation"]
            writer.writerow(
                [
                    name,
                    *[_format_float(float(rollout[metric])) for metric in metrics],
                    *[_format_float(float(crisis[metric])) for metric in crisis_metrics],
                    int(crisis["n_crisis_samples"]),
                    _format_float(float(payload["one_step_direction_corr"])),
                    _format_float(float(payload["regime_accuracy"])),
                    _format_float(float(payload["regime_macro_f1"])),
                    _format_float(float(shock["mean_abs_prediction_delta"])),
                    _format_float(float(shock["direction_flip_rate"])),
                    int(payload["n_samples"]),
                ]
            )


def _best_by_scalar(results: dict, metric: str) -> set[str]:
    values = {
        name: float(payload[metric])
        for name, payload in results.items()
        if metric in payload
    }
    if not values:
        return set()
    best = max(values.values())
    return {name for name, value in values.items() if abs(value - best) < 1e-12}


def write_regime_table(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [("Regime Acc.", "regime_accuracy"), ("Macro-F1", "regime_macro_f1")]
    best = {key: _best_by_scalar(results, key) for _, key in metrics}
    rows = []
    for name, payload in results.items():
        cells = [name]
        for _, key in metrics:
            value = _format_float(float(payload.get(key, 0.0)))
            if name in best[key]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        rows.append(" & ".join(cells) + r" \\")
    header = "Model & " + " & ".join(label for label, _ in metrics) + r" \\"
    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Regime prediction diagnostics for supervised bull/sideway/bear labels. Higher is better.}}
\label{{tab:regime_prediction}}
\begin{{tabular}}{{lcc}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    output_path.write_text(tex)


def write_counterfactual_table(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, payload in results.items():
        shock = payload.get("counterfactual_macro_shock", {})
        rows.append(
            " & ".join(
                [
                    name,
                    _format_float(float(shock.get("mean_abs_prediction_delta", 0.0))),
                    _format_float(float(shock.get("direction_flip_rate", 0.0))),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Counterfactual macro-shock sensitivity diagnostics. Mean Abs. $\Delta$ measures prediction response magnitude; direction flip rate measures sign changes under the controlled shock.}}
\label{{tab:counterfactual_sensitivity}}
\begin{{tabular}}{{lcc}}
\toprule
Model & Mean Abs. $\Delta$ & Direction Flip Rate \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    output_path.write_text(tex)


def write_crisis_table(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [f"CrisisStateMSE@{h}" for h in [1, 5, 10, 20, 30]]
    best = {metric: _best_by_crisis_metric(results, metric) for metric in metrics}
    rows = []
    for name, payload in results.items():
        crisis = payload.get("crisis_simulation", {})
        cells = [name]
        for metric in metrics:
            value = _format_float(float(crisis.get(metric, 0.0)))
            if name in best[metric]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        cells.append(str(int(crisis.get("n_crisis_samples", 0))))
        rows.append(" & ".join(cells) + r" \\")
    header = "Model & " + " & ".join(metrics) + r" & Crisis N \\"
    body = "\n".join(rows)
    tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Crisis-window simulation diagnostics. Crisis windows are selected by realized short-horizon market stress, and lower CrisisStateMSE indicates better imagined trajectory fidelity under stress.}}
\label{{tab:crisis_simulation}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lcccccc}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    output_path.write_text(tex)


@torch.no_grad()
def collect_targets(loader: DataLoader) -> np.ndarray:
    targets = []
    for batch in loader:
        price_target = batch["price_target"]
        target_h = price_target[:, 0, :] if price_target.dim() == 3 else price_target
        if target_h.dim() == 1:
            target_h = target_h.unsqueeze(-1)
        targets.extend(target_h[:, 0].detach().cpu().numpy().tolist())
    return np.asarray(targets, dtype=float)


@torch.no_grad()
def evaluate_world_model(
    model,
    loader: DataLoader,
    device: torch.device,
    bear_q: float,
    bull_q: float,
    counterfactual_scale: float = 0.5,
    crisis_return_threshold: float = -0.02,
) -> dict:
    model.eval()
    rollout_mse = {h: [] for h in HORIZONS}
    rollout_mae = {h: [] for h in HORIZONS}
    crisis_mse = {h: [] for h in HORIZONS}
    preds_1, targets_1 = [], []
    regime_preds, regime_targets = [], []
    cf_abs_delta, cf_direction_flip = [], []

    for batch in tqdm(loader, desc="World-model evidence"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        price_seq = batch["price_seq"].to(torch.float32)
        news_feat = batch["news_feat"].to(torch.float32)
        macro_feat = batch["macro_feat"].to(torch.float32)
        edge_index = batch["edge_index"].to(torch.long)
        edge_weight = batch["edge_weight"].to(torch.float32)
        action = batch["action"].to(torch.float32)
        target = batch["price_target"].to(torch.float32)
        supervised_regime = batch.get("regime_target")
        if supervised_regime is not None:
            supervised_regime = supervised_regime.detach().cpu().numpy().astype(np.int64)

        out = model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action, target)
        pred_path = out["price_pred"]
        if pred_path.dim() == 2:
            pred_path = pred_path.unsqueeze(-1)
        if target.dim() == 2:
            target_path = target.unsqueeze(-1)
        else:
            target_path = target[:, :, :1]

        for h in HORIZONS:
            pred_h = pred_path[:, h - 1, :1]
            target_h = target_path[:, h - 1, :1]
            mse_each = F.mse_loss(pred_h, target_h, reduction="none").mean(dim=1)
            rollout_mse[h].extend(mse_each.cpu().numpy().tolist())
            rollout_mae[h].extend(F.l1_loss(pred_h, target_h, reduction="none").mean(dim=1).cpu().numpy().tolist())
            crisis_score = target_path[:, :5, 0].mean(dim=1)
            crisis_mask = crisis_score <= crisis_return_threshold
            if crisis_mask.any():
                crisis_mse[h].extend(mse_each[crisis_mask].cpu().numpy().tolist())
            if h == 1:
                preds_1.extend(pred_h.squeeze(-1).cpu().numpy().tolist())
                targets_1.extend(target_h.squeeze(-1).cpu().numpy().tolist())

        if "regime_logits" in out:
            logits = _select_regime_logits(out["regime_logits"])
            pred_regime = logits.argmax(dim=-1).cpu().numpy()
            if supervised_regime is None:
                realized = target_path[:, 0, 0].detach().cpu().numpy()
                true_regime = _regime_labels(realized, bear_q, bull_q)
            else:
                true_regime = supervised_regime
            regime_preds.extend(pred_regime.tolist())
            regime_targets.extend(true_regime.tolist())

        shocked_macro = macro_feat.clone()
        shocked_macro[..., 0] = shocked_macro[..., 0] + counterfactual_scale
        shocked_out = model(price_seq, news_feat, shocked_macro, edge_index, edge_weight, action, None)
        shocked_pred = shocked_out["price_pred"]
        if shocked_pred.dim() == 2:
            shocked_pred = shocked_pred.unsqueeze(-1)
        base_1 = pred_path[:, 0, 0]
        shock_1 = shocked_pred[:, 0, 0]
        delta = (shock_1 - base_1).detach().cpu().numpy()
        cf_abs_delta.extend(np.abs(delta).tolist())
        cf_direction_flip.extend(((base_1 * shock_1) < 0).detach().cpu().numpy().astype(float).tolist())

    preds_1_arr = np.asarray(preds_1, dtype=float)
    targets_1_arr = np.asarray(targets_1, dtype=float)
    rollout = {f"StateMSE@{h}": _safe_mean(rollout_mse[h]) for h in HORIZONS}
    rollout.update({f"StateMAE@{h}": _safe_mean(rollout_mae[h]) for h in HORIZONS})
    crisis = {f"CrisisStateMSE@{h}": _safe_mean(crisis_mse[h]) for h in HORIZONS}
    crisis["n_crisis_samples"] = len(crisis_mse[1])
    crisis["crisis_return_threshold"] = crisis_return_threshold
    regime_preds_arr = np.asarray(regime_preds, dtype=np.int64)
    regime_targets_arr = np.asarray(regime_targets, dtype=np.int64)
    if regime_targets_arr.size:
        regime_accuracy = float(np.mean(regime_preds_arr == regime_targets_arr))
        regime_macro_f1 = _macro_f1(regime_targets_arr, regime_preds_arr)
    else:
        regime_accuracy = 0.0
        regime_macro_f1 = 0.0
    return {
        "rollout_fidelity": rollout,
        "crisis_simulation": crisis,
        "one_step_direction_corr": _safe_corr(preds_1_arr, targets_1_arr),
        "regime_accuracy": regime_accuracy,
        "regime_macro_f1": regime_macro_f1,
        "counterfactual_macro_shock": {
            "shock_feature": "macro_feature_0",
            "shock_scale": counterfactual_scale,
            "mean_abs_prediction_delta": _safe_mean(cf_abs_delta),
            "direction_flip_rate": _safe_mean(cf_direction_flip),
        },
        "n_samples": int(len(targets_1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate world-model evidence for FinVerse.")
    parser.add_argument("--data-root", default="data/processed/real_90")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[],
        help="List of 'display_name:model_key:checkpoint_path' specs.",
    )
    parser.add_argument("--model-name", default="finverse")
    parser.add_argument("--output", default="outputs/world_model_evidence.json")
    parser.add_argument("--table-output", default=None)
    parser.add_argument("--regime-table-output", default=None)
    parser.add_argument("--counterfactual-table-output", default=None)
    parser.add_argument("--crisis-table-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--target-mode", choices=["return", "price"], default="return")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--counterfactual-scale", type=float, default=0.5)
    parser.add_argument("--crisis-return-threshold", type=float, default=-0.02)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = FinWorldDataset(
        args.data_root,
        split=args.split,
        max_episodes=args.max_episodes,
        target_mode=args.target_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    target_values = collect_targets(loader)
    bear_q, bull_q = np.quantile(target_values, [1 / 3, 2 / 3])
    num_tickers = int(getattr(dataset, "price_buffer").shape[1])

    specs = args.checkpoints
    if not specs:
        if not args.checkpoint:
            raise ValueError("Either --checkpoint or --checkpoints must be provided.")
        specs = [f"{args.model_name}:{args.model_name}:{args.checkpoint}"]

    device = torch.device(args.device)
    results = {}
    for spec in specs:
        display_name, model_key, checkpoint_path = spec.split(":", 2)
        model = load_model(checkpoint_path, model_key, device, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, num_tickers=num_tickers)
        payload = evaluate_world_model(
            model,
            loader,
            device,
            bear_q=float(bear_q),
            bull_q=float(bull_q),
            counterfactual_scale=args.counterfactual_scale,
            crisis_return_threshold=args.crisis_return_threshold,
        )
        payload["model_key"] = model_key
        results[display_name] = payload
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wrapped = {
        "regime_thresholds": {"bear_q": float(bear_q), "bull_q": float(bull_q)},
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(wrapped, indent=2))
    if args.table_output:
        write_rollout_table(results, Path(args.table_output))
    if args.regime_table_output:
        write_regime_table(results, Path(args.regime_table_output))
    if args.counterfactual_table_output:
        write_counterfactual_table(results, Path(args.counterfactual_table_output))
    if args.crisis_table_output:
        write_crisis_table(results, Path(args.crisis_table_output))
    if args.csv_output:
        write_rollout_csv(results, Path(args.csv_output))
    print(json.dumps(wrapped, indent=2))


if __name__ == "__main__":
    main()

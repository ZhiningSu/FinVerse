# FinVerse Experiment Protocol

This document freezes the experimental setup for the AAAI paper. It separates experiments that already have usable evidence from experiments that are configured but still require retraining.

## Core Claim To Test

FinVerse is a financial world model / cognitive core. The experiments should therefore test:

- market-state construction quality
- multi-step rollout fidelity
- cross-sectional financial prediction quality
- regime-awareness after supervised regime training
- counterfactual sensitivity under controlled market shocks
- crisis-window simulation under realized stress windows
- contribution of each module through same-protocol ablations

## Dataset

- Market: U.S. equities / ETFs.
- Window: 2020--2025.
- Active processed universe: 90 assets.
- Lookback length: 30 trading days.
- Prediction horizon: 30 trading days.
- Input features:
  - price/volume-derived sequence features
  - public event/news proxy features
  - price-derived macro proxy features
  - cross-asset neighborhood context
- Split: chronological train / validation / test.
- Target mode: future return.

## Model Variants

Main model:
- `finverse`: HMSC + Dual VQ + cross-asset context + recurrent probabilistic world model + regime auxiliary head.

Ablations:
- `vanilla_rssm`: remove Dual VQ tokenization.
- `no_graph`: remove cross-asset context branch.
- `multi_noroll`: remove probabilistic world-model dynamics.
- `price_only`: remove non-price modalities.

Mainstream baselines:
- LSTM
- GRU
- Transformer
- PatchTST
- TimesFM-style
- Chronos-mini
- Kronos-mini
- Vanilla RSSM
- Dreamer-style RSSM

Important:
- `TimesFM-style` and `Chronos-mini` are lightweight, train-from-scratch baselines under the current project data format, not official pretrained foundation-model checkpoints.
- `Dreamer-style RSSM` is a compact RSSM baseline with deterministic recurrent state and stochastic latent dynamics, trained from scratch.

Auxiliary sanity baseline:
- BUY&HOLD, reported only as a sanity-check diagnostic, not as primary evidence.

## Metrics

Forecasting:
- MSE@1/5/10/20/30
- MAE@1/5/10/20/30

Cross-sectional diagnostics:
- IC
- RankIC

World-model diagnostics:
- StateMSE@1/5/10/20/30
- StateMAE@1/5/10/20/30

Regime diagnostics:
- Regime Accuracy
- Regime Macro-F1
- Labels: bear / sideway / bull from the average realized return over the first 5 forecast steps.
- Thresholds: bear <= -1%, bull >= +1%, otherwise sideway.

Counterfactual diagnostics:
- Macro-shock mean absolute prediction delta
- Direction flip rate
- Shock feature: `macro_feature_0`
- Shock scale: `+0.5` normalized units by default

Crisis-window simulation diagnostics:
- CrisisStateMSE@1/5/10/20/30
- Crisis windows are selected from samples whose mean realized return over the first 5 forecast steps is below `-2%`.
- The diagnostic compares imagined trajectories against realized future trajectories only on these stress windows.
- This is a crisis-window fidelity test, not a claim that the model fully simulates every causal channel of historical crises.

Portfolio diagnostics:
- Daily mean return
- Daily return standard deviation
- Daily IR
- Diagnostic annualized return

Important:
- Portfolio diagnostics must not be used as the main evidence unless transaction costs, larger test windows, and stability checks are added.

## Current Paper Tables

Included in main paper:
- `main_baseline_table.tex`
- `rollout_fidelity_table.tex`
- `financial_metrics_table.tex`
- `ablation_tables.tex`

Configured but not included until retrained:
- `regime_prediction_table.tex`
- `counterfactual_sensitivity_table.tex`
- `crisis_simulation_table.tex`

## Recommended Next Runs

Run 1: Regime-supervised FinVerse

```bash
REGIME_WEIGHT=0.1 EPOCHS=10 TRAIN_EPISODES=3000 VAL_EPISODES=800 MODELS="finverse vanilla_rssm no_graph multi_noroll price_only" \
bash scripts/run_paper_experiments_remote.sh
```

Run 2: Longer FinVerse-only world-model training

```bash
REGIME_WEIGHT=0.1 EPOCHS=20 TRAIN_EPISODES=6000 VAL_EPISODES=1200 MODELS="finverse" \
bash scripts/run_paper_experiments_remote.sh
```

Run 3: Final evidence generation

```bash
bash scripts/run_paper_evidence_remote.sh
```

Run 4: Crisis-window diagnostic only

```bash
TEST_EPISODES=1500 bash scripts/run_paper_evidence_remote.sh
```

## Writing Rules

- Do not claim image modality unless an image encoder is implemented.
- Do not claim episodic memory retrieval unless a memory bank is implemented.
- Do not claim full GNN message passing unless GraphSAGE/GAT is added.
- Do not use portfolio PnL as the core result under the current short diagnostic window.
- Use rollout fidelity and regime diagnostics as the primary world-model evidence.
- Crisis simulation should be phrased as crisis-window rollout fidelity, not as full causal crisis simulation.

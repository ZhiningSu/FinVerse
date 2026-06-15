# FinVerse Paper: Today Finish Plan

## Core Positioning

This paper should be closed as a financial-agent cognitive core paper, not as a complete autonomous trading-agent paper.

One-sentence thesis:

FinVerse builds a hierarchical market-state world model that allows a financial agent to encode multi-modal market observations, imagine future market trajectories, and evaluate candidate strategies before real-world execution.

## What Is Already Done

- Main method: HMSC + dual VQ tokenization + recurrent probabilistic world model.
- Data pipeline: U.S. market data from 2020--2025 with OHLCV-derived sequences, macro/news proxies, and cross-asset graph features.
- Mainstream baselines: LSTM, GRU, Transformer, PatchTST, Kronos-mini, Vanilla RSSM.
- Financial metrics: IC, RankIC, volatility MAE/R2, IR, AER.
- Portfolio protocol: top-k long-short evaluation.
- Ablation variants: w/o Dual VQ, w/o Graph, w/o Probabilistic WM, Price Only.
- BUY&HOLD has been added as an investment baseline.
- Radar figures are available under `outputs/requested_baselines_fixed_seed42/`.

## What Must Be Finished Today

1. Freeze the story.
   - Do not add new modules today.
   - Do not claim this is already a complete financial agent.
   - Say it is the cognitive core / world model for a future financial agent.

2. Use conservative claims.
   - Strong claim: FinVerse improves short-horizon world-model forecasting and provides action/strategy evaluation machinery.
   - Moderate claim: graph and probabilistic dynamics help portfolio behavior in ablation diagnostics.
   - Do not claim universal dominance across every financial metric.

3. Insert the generated paper draft.
   - Main draft: `paper/overleaf_main.tex`
   - Ablation tables: `paper/tables/ablation_tables.tex`
   - Baseline table: `paper/tables/main_baseline_table.tex`

4. Upload figures to Overleaf.
   - `outputs/requested_baselines_fixed_seed42/financial_metrics_radar_extended.png`
   - `outputs/requested_baselines_fixed_seed42/financial_metrics_radar_extended_annotated.png`
   - Optional: `outputs/requested_baselines_fixed_seed42/baseline_radar_proxy.png`

5. Final paper checks.
   - Replace author placeholders.
   - Make sure every figure path exists in Overleaf.
   - Keep the paper within the target page limit.
   - Add real bibliography entries if the target venue requires them.

## If Time Remains

- Run one longer seed for Full FinVerse only.
- Add a limitation paragraph about noisy portfolio metrics and short training protocol.
- Add a short appendix describing how BUY&HOLD is computed.

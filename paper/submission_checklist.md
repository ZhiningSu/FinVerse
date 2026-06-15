# Submission Checklist

## Files Prepared

- `paper/overleaf_main.tex`: complete paper draft.
- `paper/overleaf_main.pdf`: locally compiled PDF.
- `paper/tables/main_baseline_table.tex`: mainstream forecasting comparison.
- `paper/tables/financial_metrics_table.tex`: IC/RankIC/volatility/IR/AER comparison.
- `paper/tables/ablation_tables.tex`: ablation tables with BUY&HOLD.
- `paper/today_finish_plan.md`: today-oriented completion plan.

## Figures To Upload To Overleaf

Use these existing generated figures:

- `outputs/requested_baselines_fixed_seed42/financial_metrics_radar_extended.png`
- `outputs/requested_baselines_fixed_seed42/financial_metrics_radar_extended_annotated.png`
- `outputs/requested_baselines_fixed_seed42/baseline_radar_proxy.png`

Optional PDF versions also exist in the same directory.

## Claims To Keep

- FinVerse is a financial world model / cognitive core, not a complete autonomous financial agent.
- HMSC improves short-horizon market-state construction.
- Dual VQ, graph structure, and probabilistic dynamics contribute complementary benefits.
- Portfolio metrics are diagnostic and should be interpreted conservatively.

## Claims To Avoid Today

- Do not claim state-of-the-art trading performance.
- Do not claim every module dominates every metric.
- Do not claim full news/LLM integration unless it is marked as future work.
- Do not claim the actor-critic component is fully trained unless additional experiments are added.

## Final Manual Edits Before Submission

1. Replace `Anonymous Authors`.
2. If using AAAI, paste the body into the official AAAI template or switch the document class.
3. Add real bibliography entries required by the venue.
4. Upload figures and adjust `\includegraphics` paths if figures are inserted.
5. Recompile twice in Overleaf.
6. Check page limit and remove the financial metrics table or move it to appendix if needed.

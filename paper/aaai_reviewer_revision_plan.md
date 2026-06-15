# AAAI Reviewer-Oriented Revision Plan

## Current Diagnosis

The reviewer critique is mostly correct. The core idea is strong, but the current evidence is not yet enough for a top-tier world-model paper. The immediate risk is not novelty; the risk is a mismatch between claims and evidence.

## What Must Be Fixed First

### 1. Claim-Evidence Alignment

Do immediately:
- Keep the paper positioned as a financial world-model / cognitive-core paper.
- Do not claim a complete financial agent.
- Do not claim trained actor-critic policy learning unless a real actor-critic experiment is added.
- Do not claim image modality unless an image encoder and image data are actually used.
- Do not claim memory retrieval unless a memory bank/retrieval module is implemented and ablated.

Current safe claim:

FinVerse constructs a hierarchical latent market state and learns recurrent probabilistic rollout dynamics for forecasting and diagnostic strategy evaluation.

### 2. Graph Wording

Current code uses a graph-context branch based on cross-asset neighbor inputs, but it is not a full GCN/GAT-style message-passing network.

Safe wording:
- cross-asset graph-context encoding
- graph-conditioned market context
- neighbor-aware cross-asset representation

Avoid:
- graph propagation
- GAT
- GCN
- explicit message passing

Unless the code is upgraded.

### 3. World-Model Evidence

We added `scripts/evaluate_world_model_evidence.py` to measure:
- rollout fidelity: `StateMSE@1/5/10/20/30`
- derived regime prediction
- counterfactual macro-shock sensitivity

Initial 200-sample diagnostic result:
- Rollout fidelity is measurable and usable.
- Derived regime accuracy is about random-level for three regimes.
- Counterfactual response is unstable and should not be presented as a strong result yet.

Therefore:
- Add rollout fidelity first.
- Treat regime/counterfactual as future work unless retrained with proper supervision.

## Two-Week Upgrade Plan

### Week 1: Make World-Model Evidence Real

1. Add supervised regime labels.
   - Bull / Bear / Sideway labels from realized market-index return or volatility-adjusted return.
   - Add regime loss to `WorldModelLoss`.
   - Report regime accuracy and macro-F1.
   - Status: implemented in code. Full paper-quality numbers still require retraining checkpoints with nonzero regime supervision.

2. Add rollout fidelity table.
   - Compare FinVerse vs Vanilla RSSM vs Transformer/PatchTST recursive forecasts.
   - Report `StateMSE@5`, `StateMSE@10`, `StateMSE@20`, `StateMSE@30`.

3. Fix portfolio metrics.
   - Report daily mean/std directly.
   - Report non-annualized Sharpe/IR or clipped annualized IR with clear definition.
   - Add transaction costs.
   - Avoid overclaiming high IR/AER.

### Week 2: Strengthen Novelty Evidence

4. Add counterfactual shock tests.
   - Rate shock.
   - Volatility shock.
   - Crisis-window stress test.
   - Report direction and magnitude consistency, not just return.

5. Upgrade graph module if graph is a main claim.
   - Implement a simple GraphSAGE/GAT layer using `edge_index` and `edge_weight`.
   - Re-run `w/o Graph` ablation.

6. Do not add image modality unless time permits.
   - If no chart-image encoder is implemented, remove all image claims.

## What Can Go Into the Current Paper Now

Safe additions:
- Rollout fidelity as a world-model diagnostic.
- Conservative graph-context wording.
- Explicit limitation that regime/counterfactual results require stronger supervision.
- A reviewer-aware paragraph saying portfolio metrics are diagnostic rather than definitive trading results.

Unsafe additions:
- Strong crisis simulation claims without crisis-window experiments.
- Strong memory claims without retrieval implementation.
- Strong multimodal image claims without image encoder.
- SOTA trading claims based on current IR/AER.

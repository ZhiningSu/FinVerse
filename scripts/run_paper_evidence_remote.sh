#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/wjt/anaconda3/envs/sft/bin/python}
DATA_ROOT=${DATA_ROOT:-data/processed/real_90}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/paper_experiments}
PAPER_TABLE_DIR=${PAPER_TABLE_DIR:-paper/tables}
DEVICE=${DEVICE:-cuda}
HIDDEN_DIM=${HIDDEN_DIM:-128}
LATENT_DIM=${LATENT_DIM:-128}
TEST_EPISODES=${TEST_EPISODES:-1000}
BATCH_SIZE=${BATCH_SIZE:-128}
TARGET_MODE=${TARGET_MODE:-return}
SEED=${SEED:-42}

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

mkdir -p "${OUTPUT_DIR}" "${PAPER_TABLE_DIR}"

"${PY}" -B scripts/evaluate_world_model_evidence.py \
  --data-root "${DATA_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-episodes "${TEST_EPISODES}" \
  --target-mode "${TARGET_MODE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --latent-dim "${LATENT_DIM}" \
  --output "${OUTPUT_DIR}/world_model_evidence_${TEST_EPISODES}.json" \
  --table-output "${PAPER_TABLE_DIR}/rollout_fidelity_table.tex" \
  --regime-table-output "${PAPER_TABLE_DIR}/regime_prediction_table.tex" \
  --counterfactual-table-output "${PAPER_TABLE_DIR}/counterfactual_sensitivity_table.tex" \
  --crisis-table-output "${PAPER_TABLE_DIR}/crisis_simulation_table.tex" \
  --csv-output "${PAPER_TABLE_DIR}/world_model_evidence_table.csv" \
  --checkpoints \
    "Full FinVerse:finverse:${OUTPUT_DIR}/finverse/best_checkpoint.pt" \
    "w/o Dual VQ:vanilla_rssm:${OUTPUT_DIR}/vanilla_rssm/best_checkpoint.pt" \
    "w/o Cross-Asset Ctx:no_graph:${OUTPUT_DIR}/no_graph/best_checkpoint.pt" \
    "w/o Probabilistic WM:multi_noroll:${OUTPUT_DIR}/multi_noroll/best_checkpoint.pt" \
    "Price Only:price_only:${OUTPUT_DIR}/price_only/best_checkpoint.pt"

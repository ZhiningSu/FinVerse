#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/wjt/anaconda3/envs/sft/bin/python}
DATA_ROOT=${DATA_ROOT:-data/processed/real_90}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/paper_experiments}
DEVICE=${DEVICE:-cuda}
HIDDEN_DIM=${HIDDEN_DIM:-128}
LATENT_DIM=${LATENT_DIM:-128}
EPOCHS=${EPOCHS:-10}
TRAIN_EPISODES=${TRAIN_EPISODES:-3000}
VAL_EPISODES=${VAL_EPISODES:-800}
TEST_EPISODES=${TEST_EPISODES:-1000}
BATCH_SIZE=${BATCH_SIZE:-128}
TARGET_MODE=${TARGET_MODE:-return}
KL_WEIGHT=${KL_WEIGHT:-0.001}
VQ_WEIGHT=${VQ_WEIGHT:-0.001}
REGIME_WEIGHT=${REGIME_WEIGHT:-0.1}
SEED=${SEED:-42}
MODELS=${MODELS:-"finverse vanilla_rssm no_graph multi_noroll price_only"}

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

mkdir -p "${OUTPUT_DIR}"

for model in ${MODELS}; do
  echo "===== Training ${model} ====="
  "${PY}" -B train.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${model}" \
    --num-epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-episodes "${TRAIN_EPISODES}" \
    --max-val-episodes "${VAL_EPISODES}" \
    --target-mode "${TARGET_MODE}" \
    --kl-weight "${KL_WEIGHT}" \
    --vq-weight "${VQ_WEIGHT}" \
    --regime-weight "${REGIME_WEIGHT}" \
    --seed "${SEED}" \
    --num-workers 0 \
    --device "${DEVICE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --latent-dim "${LATENT_DIM}" \
    --log-interval 20 \
    --save-interval 1
done

echo "===== Evaluating forecasting diagnostics ====="
"${PY}" -B evaluate.py \
  --data-root "${DATA_ROOT}" \
  --split test \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --num-workers 0 \
  --max-episodes "${TEST_EPISODES}" \
  --target-mode "${TARGET_MODE}" \
  --seed "${SEED}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --latent-dim "${LATENT_DIM}" \
  --output "${OUTPUT_DIR}/eval_forecasting_${TEST_EPISODES}.json" \
  --checkpoints \
    "Full FinVerse:${OUTPUT_DIR}/finverse/best_checkpoint.pt" \
    "w/o Dual VQ:${OUTPUT_DIR}/vanilla_rssm/best_checkpoint.pt" \
    "w/o Cross-Asset Ctx:${OUTPUT_DIR}/no_graph/best_checkpoint.pt" \
    "w/o Probabilistic WM:${OUTPUT_DIR}/multi_noroll/best_checkpoint.pt" \
    "Price Only:${OUTPUT_DIR}/price_only/best_checkpoint.pt"

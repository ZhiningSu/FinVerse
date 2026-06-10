#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/wjt/anaconda3/envs/sft/bin/python}
DATA_ROOT=${DATA_ROOT:-data/processed/real}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/mainstream_baselines_remote}
DEVICE=${DEVICE:-cuda}
HIDDEN_DIM=${HIDDEN_DIM:-128}
EPOCHS=${EPOCHS:-3}
TRAIN_EPISODES=${TRAIN_EPISODES:-1000}
VAL_EPISODES=${VAL_EPISODES:-300}
TEST_EPISODES=${TEST_EPISODES:-500}
BATCH_SIZE=${BATCH_SIZE:-128}
MODELS=${MODELS:-"lstm patchtst kronos_mini vanilla_rssm finverse gru transformer"}
TARGET_MODE=${TARGET_MODE:-return}
KL_WEIGHT=${KL_WEIGHT:-0.001}
VQ_WEIGHT=${VQ_WEIGHT:-0.001}
SEED=${SEED:-42}

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
    --seed "${SEED}" \
    --num-workers 0 \
    --device "${DEVICE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --log-interval 20 \
    --save-interval 1
done

echo "===== Evaluating ====="
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
  --output "${OUTPUT_DIR}/eval_mainstream_${TEST_EPISODES}.json" \
  --checkpoints \
    LSTM:${OUTPUT_DIR}/lstm/best_checkpoint.pt \
    PatchTST:${OUTPUT_DIR}/patchtst/best_checkpoint.pt \
    Kronos-mini:${OUTPUT_DIR}/kronos_mini/best_checkpoint.pt \
    "Vanilla RSSM:${OUTPUT_DIR}/vanilla_rssm/best_checkpoint.pt" \
    FinVerse:${OUTPUT_DIR}/finverse/best_checkpoint.pt \
    GRU:${OUTPUT_DIR}/gru/best_checkpoint.pt \
    Transformer:${OUTPUT_DIR}/transformer/best_checkpoint.pt

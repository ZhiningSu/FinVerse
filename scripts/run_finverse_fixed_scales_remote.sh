#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/wjt/anaconda3/envs/sft/bin/python}
DATA_ROOT=${DATA_ROOT:-data/processed/real}
DEVICE=${DEVICE:-cuda}
HIDDEN_DIM=${HIDDEN_DIM:-128}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_EPISODES=${TEST_EPISODES:-500}
TARGET_MODE=${TARGET_MODE:-return}
KL_WEIGHT=${KL_WEIGHT:-0.001}
VQ_WEIGHT=${VQ_WEIGHT:-0.001}
SEED=${SEED:-42}

run_scale() {
  local label=$1
  local epochs=$2
  local train_episodes=$3
  local val_episodes=$4
  local output_dir="outputs/finverse_fixed_${label}"

  mkdir -p "${output_dir}"

  echo "===== Training FinVerse ${label} ====="
  "${PY}" -B train.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --model finverse \
    --num-epochs "${epochs}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-episodes "${train_episodes}" \
    --max-val-episodes "${val_episodes}" \
    --target-mode "${TARGET_MODE}" \
    --kl-weight "${KL_WEIGHT}" \
    --vq-weight "${VQ_WEIGHT}" \
    --seed "${SEED}" \
    --num-workers 0 \
    --device "${DEVICE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --log-interval 20 \
    --save-interval 1

  echo "===== Evaluating FinVerse ${label} ====="
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
    --output "${output_dir}/eval_finverse_${TEST_EPISODES}.json" \
    --checkpoints \
      "FinVerse-${label}:${output_dir}/finverse/best_checkpoint.pt"
}

run_scale "small" 3 1000 300
run_scale "medium" 8 3000 800
run_scale "large" 15 6000 1200

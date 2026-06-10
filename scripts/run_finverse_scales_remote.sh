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
  local name=$1
  local epochs=$2
  local train_episodes=$3
  local val_episodes=$4
  local test_episodes=$5
  local output_dir="outputs/${name}"

  mkdir -p "${output_dir}"

  echo "===== Training FinVerse ${name} ====="
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

  echo "===== Evaluating FinVerse ${name} ====="
  "${PY}" -B evaluate.py \
    --data-root "${DATA_ROOT}" \
    --split test \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --num-workers 0 \
    --max-episodes "${test_episodes}" \
    --target-mode "${TARGET_MODE}" \
    --seed "${SEED}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --output "${output_dir}/eval_finverse_${test_episodes}.json" \
    --checkpoints \
      "FinVerse-${name}:${output_dir}/finverse/best_checkpoint.pt"
}

run_scale "finverse_medium" 8 3000 800 "${TEST_EPISODES}"
run_scale "finverse_large" 15 6000 1200 "${TEST_EPISODES}"

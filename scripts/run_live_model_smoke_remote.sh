#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/wjt/anaconda3/envs/sft/bin/python}
DATA_ROOT=${DATA_ROOT:-data/processed/real_90}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/live_model_full}
LIVE_OUTPUT_DIR=${LIVE_OUTPUT_DIR:-outputs/live}
LIVE_DATA_DIR=${LIVE_DATA_DIR:-data/live}
MARKET=${MARKET:-us}
DEVICE=${DEVICE:-cuda}
INFER_DEVICE=${INFER_DEVICE:-cpu}
HIDDEN_DIM=${HIDDEN_DIM:-128}
LATENT_DIM=${LATENT_DIM:-128}
EPOCHS=${EPOCHS:-12}
TRAIN_EPISODES=${TRAIN_EPISODES:-4096}
VAL_EPISODES=${VAL_EPISODES:-512}
BATCH_SIZE=${BATCH_SIZE:-64}
TOP_K=${TOP_K:-20}
SEED=${SEED:-42}

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

echo "===== Train stronger FinVerse checkpoint ====="
"${PY}" -B train.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model finverse \
  --num-epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-episodes "${TRAIN_EPISODES}" \
  --max-val-episodes "${VAL_EPISODES}" \
  --target-mode return \
  --kl-weight 0.001 \
  --vq-weight 0.001 \
  --regime-weight 0.1 \
  --action-weight 0.001 \
  --policy-weight 0.001 \
  --seed "${SEED}" \
  --num-workers 0 \
  --device "${DEVICE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --latent-dim "${LATENT_DIM}" \
  --log-interval 5 \
  --save-interval 1

CHECKPOINT="${OUTPUT_DIR}/finverse/best_checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${OUTPUT_DIR}/finverse/last_checkpoint.pt"
fi

echo "===== Generate live dashboard snapshot with model checkpoint ====="
"${PY}" -B scripts/run_live_pipeline.py \
  --market "${MARKET}" \
  --top-k "${TOP_K}" \
  --output-dir "${LIVE_OUTPUT_DIR}" \
  --data-live-dir "${LIVE_DATA_DIR}" \
  --mode finverse_checkpoint \
  --model-checkpoint "${CHECKPOINT}" \
  --model-name finverse \
  --hidden-dim "${HIDDEN_DIM}" \
  --latent-dim "${LATENT_DIM}" \
  --device "${INFER_DEVICE}" \
  --fetch-online

mkdir -p "dashboard/public/data/${MARKET}"
cp "${LIVE_OUTPUT_DIR}/${MARKET}/latest.json" "dashboard/public/data/${MARKET}/latest.json"

echo "Checkpoint: ${CHECKPOINT}"
echo "Live JSON: ${LIVE_OUTPUT_DIR}/${MARKET}/latest.json"
echo "Dashboard static JSON: dashboard/public/data/${MARKET}/latest.json"

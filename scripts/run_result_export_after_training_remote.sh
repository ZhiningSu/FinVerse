#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/home/wjt/anaconda3/envs/sft/bin/python}
DATA_ROOT=${DATA_ROOT:-data/processed/real_90}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_EPISODES=${TEST_EPISODES:-1000}
HIDDEN_DIM=${HIDDEN_DIM:-128}
LATENT_DIM=${LATENT_DIM:-128}
TARGET_MODE=${TARGET_MODE:-return}
SEED=${SEED:-42}
FINAL_DIR=${FINAL_DIR:-outputs/final_results}

wait_for_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "[$(date '+%F %T')] ${label} PID file not found, skipping wait: ${pid_file}"
    return
  fi
  local pid
  pid=$(cat "${pid_file}")
  if [[ -z "${pid}" ]]; then
    echo "[$(date '+%F %T')] ${label} PID file is empty, skipping wait: ${pid_file}"
    return
  fi
  echo "[$(date '+%F %T')] Waiting for ${label} PID=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] ${label} PID=${pid} finished"
}

add_ckpt() {
  local -n arr_ref=$1
  local display_name="$2"
  local checkpoint_path="$3"
  if [[ -f "${checkpoint_path}" ]]; then
    arr_ref+=("${display_name}:${checkpoint_path}")
  else
    echo "[$(date '+%F %T')] Missing checkpoint for ${display_name}: ${checkpoint_path}"
  fi
}

mkdir -p "${FINAL_DIR}" paper/tables

wait_for_pid_file outputs/paper_experiments/run.pid "paper_experiments"
wait_for_pid_file outputs/train_queue/mainstream_after_paper.pid "mainstream_baselines"

echo "[$(date '+%F %T')] Running world-model evidence tables"
bash scripts/run_paper_evidence_remote.sh

core_specs=()
add_ckpt core_specs "Full FinVerse" "outputs/paper_experiments/finverse/best_checkpoint.pt"
add_ckpt core_specs "w/o Dual VQ" "outputs/paper_experiments/vanilla_rssm/best_checkpoint.pt"
add_ckpt core_specs "w/o Cross-Asset Ctx" "outputs/paper_experiments/no_graph/best_checkpoint.pt"
add_ckpt core_specs "w/o Probabilistic WM" "outputs/paper_experiments/multi_noroll/best_checkpoint.pt"
add_ckpt core_specs "Price Only" "outputs/paper_experiments/price_only/best_checkpoint.pt"

if (( ${#core_specs[@]} > 0 )); then
  echo "[$(date '+%F %T')] Evaluating core ablations with BUY&HOLD"
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
    --include-buy-hold \
    --output "${FINAL_DIR}/eval_core_ablation_${TEST_EPISODES}.json" \
    --checkpoints "${core_specs[@]}"
fi

mainstream_specs=()
add_ckpt mainstream_specs "LSTM" "outputs/mainstream_baselines_remote/lstm/best_checkpoint.pt"
add_ckpt mainstream_specs "GRU" "outputs/mainstream_baselines_remote/gru/best_checkpoint.pt"
add_ckpt mainstream_specs "Transformer" "outputs/mainstream_baselines_remote/transformer/best_checkpoint.pt"
add_ckpt mainstream_specs "PatchTST" "outputs/mainstream_baselines_remote/patchtst/best_checkpoint.pt"
add_ckpt mainstream_specs "TimesFM" "outputs/mainstream_baselines_remote/timesfm/best_checkpoint.pt"
add_ckpt mainstream_specs "Chronos-mini" "outputs/mainstream_baselines_remote/chronos_mini/best_checkpoint.pt"
add_ckpt mainstream_specs "Kronos-mini" "outputs/mainstream_baselines_remote/kronos_mini/best_checkpoint.pt"
add_ckpt mainstream_specs "Vanilla RSSM" "outputs/mainstream_baselines_remote/vanilla_rssm/best_checkpoint.pt"
add_ckpt mainstream_specs "Dreamer-style RSSM" "outputs/mainstream_baselines_remote/dreamer_rssm/best_checkpoint.pt"
add_ckpt mainstream_specs "FinVerse" "outputs/mainstream_baselines_remote/finverse/best_checkpoint.pt"

if (( ${#mainstream_specs[@]} > 0 )); then
  echo "[$(date '+%F %T')] Evaluating mainstream baselines with BUY&HOLD"
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
    --include-buy-hold \
    --output "${FINAL_DIR}/eval_mainstream_all_${TEST_EPISODES}.json" \
    --checkpoints "${mainstream_specs[@]}"
fi

echo "[$(date '+%F %T')] Writing Markdown summary"
"${PY}" -B scripts/summarize_experiment_results.py \
  --core-json "${FINAL_DIR}/eval_core_ablation_${TEST_EPISODES}.json" \
  --mainstream-json "${FINAL_DIR}/eval_mainstream_all_${TEST_EPISODES}.json" \
  --evidence-json "outputs/paper_experiments/world_model_evidence_${TEST_EPISODES}.json" \
  --output "${FINAL_DIR}/final_experiment_results.md"

echo "[$(date '+%F %T')] Result export complete"
echo "Markdown: $(pwd)/${FINAL_DIR}/final_experiment_results.md"
echo "Core JSON: $(pwd)/${FINAL_DIR}/eval_core_ablation_${TEST_EPISODES}.json"
echo "Mainstream JSON: $(pwd)/${FINAL_DIR}/eval_mainstream_all_${TEST_EPISODES}.json"
echo "Evidence JSON: $(pwd)/outputs/paper_experiments/world_model_evidence_${TEST_EPISODES}.json"
echo "LaTeX tables: $(pwd)/paper/tables"

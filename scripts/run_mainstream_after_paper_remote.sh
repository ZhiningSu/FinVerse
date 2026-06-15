#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PAPER_PID_FILE=${PAPER_PID_FILE:-outputs/paper_experiments/run.pid}
LOG_PREFIX=${LOG_PREFIX:-outputs/train_queue}

mkdir -p "${LOG_PREFIX}"

if [[ ! -f "${PAPER_PID_FILE}" ]]; then
  echo "Paper experiment PID file not found: ${PAPER_PID_FILE}" >&2
  exit 1
fi

paper_pid=$(cat "${PAPER_PID_FILE}")
echo "[$(date '+%F %T')] Waiting for paper_experiments PID=${paper_pid}"

while kill -0 "${paper_pid}" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] paper_experiments finished; starting mainstream baselines"
bash scripts/run_mainstream_baselines_remote.sh
echo "[$(date '+%F %T')] mainstream baselines finished"

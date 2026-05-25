#!/bin/bash
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/processed/real}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"
DEVICE="${DEVICE:-cuda}"

NUM_GPUS=${NUM_GPUS:-1}
NUM_EPOCHS=${NUM_EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-3e-4}
LATENT_DIM=${LATENT_DIM:-128}
HIDDEN_DIM=${HIDDEN_DIM:-256}
KL_WEIGHT=${KL_WEIGHT:-0.1}

echo "=========================================="
echo "FinWorld Model - Remote Training Pipeline"
echo "=========================================="
echo "Data root:    $DATA_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "Device:       $DEVICE"
echo "Epochs:       $NUM_EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Latent dim:   $LATENT_DIM"
echo "Hidden dim:   $HIDDEN_DIM"
echo "KL weight:    $KL_WEIGHT"
echo "=========================================="

MODELS=("full" "price_only" "multi_noroll" "no_graph")
MODEL_LABELS=("FinWorldModel" "PriceOnlyGRU" "MultiModal-noRollout" "NoGraph")

train_model() {
    local model=$1
    local label=$2
    local ckpt_dir="$OUTPUT_DIR/$model"
    local log_file="$OUTPUT_DIR/logs/train_${model}.log"
    mkdir -p "$ckpt_dir" "$OUTPUT_DIR/logs"

    echo ""
    echo "=========================================="
    echo "Training [$model] $label"
    echo "Checkpoint dir: $ckpt_dir"
    echo "Log file: $log_file"
    echo "=========================================="

    python train.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --model "$model" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --latent-dim "$LATENT_DIM" \
        --hidden-dim "$HIDDEN_DIM" \
        --kl-weight "$KL_WEIGHT" \
        --device "$DEVICE" \
        --log-interval 5 \
        --val-interval 1 \
        --save-interval 5 \
        2>&1 | tee "$log_file"

    echo ""
    echo ">>> [$model] done. Checkpoints:"
    ls -lh "$ckpt_dir"/*.pt 2>/dev/null || echo "No checkpoints found"
}

for i in "${!MODELS[@]}"; do
    train_model "${MODELS[$i]}" "${MODEL_LABELS[$i]}"
done

echo ""
echo "=========================================="
echo "All training complete! Running evaluation"
echo "=========================================="

mkdir -p "$OUTPUT_DIR/plots"

python plot_results.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR/plots" \
    --max-test 1000 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hidden-dim "$HIDDEN_DIM" \
    --latent-dim "$LATENT_DIM" \
    2>&1 | tee "$OUTPUT_DIR/logs/plot_results.log"

echo ""
echo "=========================================="
echo "FINAL RESULTS SUMMARY"
echo "=========================================="
if [ -f "$OUTPUT_DIR/plots/eval_results.json" ]; then
    python -c "
import json, sys
with open('$OUTPUT_DIR/plots/eval_results.json') as f:
    results = json.load(f)
print(f'{'Model':<25} | {'MSE@1':>8} | {'MSE@5':>8} | {'MSE@10':>8} | {'MAE@1':>8} | {'MAE@5':>8} | {'MAE@10':>8}')
print('-' * 90)
for name, m in results.items():
    print(f'{name:<25} | {m[\"MSE@1\"]:>8.4f} | {m[\"MSE@5\"]:>8.4f} | {m[\"MSE@10\"]:>8.4f} | {m[\"MAE@1\"]:>8.4f} | {m[\"MAE@5\"]:>8.4f} | {m[\"MAE@10\"]:>8.4f}')
print('-' * 90)
"
fi

echo ""
echo "=========================================="
echo "Training outputs:"
for model in "${MODELS[@]}"; do
    echo "  $OUTPUT_DIR/$model/"
done
echo ""
echo "Plots:         $OUTPUT_DIR/plots/"
echo "Eval results:  $OUTPUT_DIR/plots/eval_results.json"
echo "LaTeX table:   $OUTPUT_DIR/plots/ablation_table.tex"
echo "=========================================="
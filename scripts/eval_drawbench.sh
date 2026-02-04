#!/bin/bash
set -euo pipefail

# Single-checkpoint DrawBench evaluation script, aligned with eval_genaibench.sh
# Override via environment variables or edit constants below.

# === Script paths and model configuration ===
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/evaluation/eval_drawbench_unifiedreward2.py"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"

# === Experiment and checkpoint settings (mirrors eval_genaibench style) ===
OUTPUT_BASE_DIR="/path/to/output"
EXPERIMENT_NAME="256-AdaFuseDiT-timewise-LNzero-4b"
CHECKPOINT_STEP="500000"
GEN_SCALE="7"     # must match generation guidance scale subdirectory
GEN_STEPS="28"    # must match generation steps subdirectory
METADATA_FILE="${ROOT_DIR}/configs/drawbench/metadata.json"
BATCH_SIZE=8

# === Path construction ===
IMAGES_DIR=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/draw-${GEN_SCALE}-${GEN_STEPS}
RESULT_FILE=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/drawbench-${GEN_SCALE}-${GEN_STEPS}-eval.jsonl

# Ensure result directory exists
mkdir -p "$(dirname "$RESULT_FILE")"

echo "Eval dir      : $IMAGES_DIR"
echo "Metadata      : $METADATA_FILE"
echo "Eval script   : $EVAL_SCRIPT"
echo "Server URL    : $SERVER_URL"
echo "Result file   : $RESULT_FILE"
echo "Batch Size   : $BATCH_SIZE"

python "$EVAL_SCRIPT" \
  --images-dir "$IMAGES_DIR" \
  --metadata-file "$METADATA_FILE" \
  --server-url "$SERVER_URL" \
  --output-file "$RESULT_FILE" \
  --batch-size "$BATCH_SIZE"

echo "Evaluation complete, results saved to: $RESULT_FILE"

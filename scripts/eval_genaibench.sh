#!/bin/bash

# Single-checkpoint GenAI-Bench evaluation script, same style as eval_geneval.sh

# === Script paths and model configuration ===
EVAL_SCRIPT="/path/to/t2v_metrics/genai_bench/custom_evaluate_local.py"
GPT_MODEL="Qwen3-VL-235B-A22B-Instruct"

# === Experiment and checkpoint settings ===
OUTPUT_BASE_DIR="/path/to/output"
EXPERIMENT_NAME="256-baseline-norm"
CHECKPOINT_STEP="500000"
GENAIBENCH_SCALE="6"   # must match generation scale to pick the right folder
STEPS="50"              # must match generation steps to pick the right folder

IMAGE_DIR=${OUTPUT_BASE_DIR}
RESULT_DIR=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/genaibench-${GENAIBENCH_SCALE}-${STEPS}-eval
NUM_THREADS=20
NUM_IMAGES_PER_PROMPT=1

# === Path construction ===
GEN_MODEL=${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/genaibench-${GENAIBENCH_SCALE}-${STEPS}
# === Evaluation output settings ===
# Ensure result directory exists
mkdir -p "$RESULT_DIR"

echo "Eval dir: $GEN_MODEL"

python "$EVAL_SCRIPT" \
  --output_dir "$IMAGE_DIR" \
  --gen_model "$GEN_MODEL" \
  --model "$GPT_MODEL" \
  --num_threads "$NUM_THREADS" \
  --result_dir "$RESULT_DIR" \
  --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"

echo "Evaluation complete."

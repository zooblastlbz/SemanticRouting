

#!/bin/bash

# Example multi-node training launcher using Accelerate + DeepSpeed.
# Customize the placeholders below before running.

# Required environment:
# - ACCELERATE_CONFIG: path to your accelerate_config.yaml
# - CONFIG_FILE: path to your training config YAML
# - PYTHON_BIN: path to your Python binary
# - MASTER_ADDR: IP of rank 0 host
# - MASTER_PORT: free port on rank 0 host

# Optional: WandB API key (set if you use WandB logging)
# export WANDB_API_KEY="YOUR_WANDB_KEY"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"./accelerate_config.yaml"}
CONFIG_FILE=${CONFIG_FILE:-"./configs/uniform.yaml"}
PYTHON_BIN=${PYTHON_BIN:-"python"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}

echo "Accelerate config : ${ACCELERATE_CONFIG}"
echo "Training config   : ${CONFIG_FILE}"
echo "Python binary     : ${PYTHON_BIN}"
echo "Master            : ${MASTER_ADDR}:${MASTER_PORT}"

${PYTHON_BIN} -m accelerate.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  train.py \
  -c "${CONFIG_FILE}"

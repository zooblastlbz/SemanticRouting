
vllm serve /path/to/Qwen3-VL-235B-A22B-Instruct/ \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --served-model-name Qwen3-VL-235B-A22B-Instruct \
  --async-scheduling \
  --enforce-eager \
  --allowed-local-media-path /path/to/local/media \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.80 \

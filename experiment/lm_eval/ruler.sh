#!/bin/bash
set -euo pipefail

MODEL_PATH="./qwen_mirror"
TASK_NAME="niah_single_1_local"
INCLUDE_PATH="./my_vllm/experiment/lm_eval/tasks"
MAX_LENGTH=32768
NUM_SAMPLE=50


echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK_NAME=${TASK_NAME}"
echo "INCLUDE_PATH=${INCLUDE_PATH}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "NUM_SAMPLE=${NUM_SAMPLE}"

lm-eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},max_length=${MAX_LENGTH}" \
  --tasks "${TASK_NAME}" \
  --include_path "${INCLUDE_PATH}" \
  --metadata '{"max_seq_lengths":[4096,8192,16384,32768],"num_samples_per_length":50}'
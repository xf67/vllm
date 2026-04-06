#!/bin/bash
set -euo pipefail

# 只用两张卡时改成 0,1；单卡就写 0
# export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./qwen_mirror"
TASK_NAME="ruler_cwe_local"
INCLUDE_PATH="./my_vllm/experiment/lm_eval/tasks"

# 评测配置
MAX_LENGTH=32768
NUM_SAMPLE=50
BATCH_SIZE="auto"

# RULER 长度档位
METADATA="{\"max_seq_lengths\":[4096,8192,16384,32768],\"num_samples_per_length\":${NUM_SAMPLE}}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK_NAME=${TASK_NAME}"
echo "INCLUDE_PATH=${INCLUDE_PATH}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "NUM_SAMPLE=${NUM_SAMPLE}"
echo "METADATA=${METADATA}"

lm-eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},max_length=${MAX_LENGTH},data_parallel_size=2" \
  --tasks "${TASK_NAME}" \
  --include_path "${INCLUDE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --metadata "${METADATA}"
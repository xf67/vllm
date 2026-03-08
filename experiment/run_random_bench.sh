#!/bin/bash
# Quick benchmark with random long prompts to verify k-segregation effect.
#
# Prerequisites: vllm server running with --no-enable-prefix-caching
#
# Usage:
#   bash test/normal/run_random_bench.sh [MODE_LABEL] [INPUT_LEN]

set -euo pipefail

MODEL=${MODEL:-"/home/xxf/models/olmoe-7B-A1B"}
ENDPOINT="/v1/completions"
PORT=${PORT:-8000}
SEED=42

MODE=${1:-"test"}
INPUT_LEN=${2:-2048}
OUTPUT_LEN=64
NUM_PROMPTS=256

RESULT_DIR="test/bench_results/random_${MODE}"
mkdir -p "$RESULT_DIR"

REQUEST_RATES=(2 4 8 12 16)

echo "================================================================"
echo "Random Benchmark — mode: $MODE  input_len: $INPUT_LEN"
echo "Num prompts: $NUM_PROMPTS  output_len: $OUTPUT_LEN"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Results: $RESULT_DIR"
echo "================================================================"
echo ""

for rate in "${REQUEST_RATES[@]}"; do
  echo "---- request-rate=$rate ----"

  rate_label=$(echo "$rate" | tr '.' '_')
  result_file="rate_${rate_label}_in${INPUT_LEN}.json"

  QOS_FILE="${QOS_FILE:-}" STATIC_QOS="${STATIC_QOS:--1}" \
    QOS_K_MEAN="${QOS_K_MEAN:-4}" QOS_K_STD="${QOS_K_STD:-2}" QOS_K_MAX="${QOS_K_MAX:-8}" \
    vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --endpoint "$ENDPOINT" \
    --dataset-name random \
    --num-prompts $NUM_PROMPTS \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --request-rate "$rate" \
    --seed $SEED \
    --port "$PORT" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_file"

  echo ""
done

echo "================================================================"
echo "Done. Results in $RESULT_DIR"
echo "================================================================"

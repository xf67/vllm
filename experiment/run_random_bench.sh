#!/bin/bash
# Quick benchmark with random / random2 (trace-driven) prompts.
#
# Prerequisites: vllm server running with --no-enable-prefix-caching
#
# Usage:
#   bash run_random_bench.sh [MODE_LABEL] [INPUT_LEN]
#
# Environment variables:
#   DATASET_NAME  : "random" (default) or "random2" (trace-driven)
#   TRACE_CSV     : path to trace CSV (required when DATASET_NAME=random2)
#   NUM_PROMPTS   : number of prompts (default 256)
#   OUTPUT_LEN    : output tokens per request (default 64, ignored for random2)
#   REQUEST_RATES : space-separated rates (default "2 4 8 12 16")

set -euo pipefail

MODEL=${MODEL:-"/home/xxf/models/olmoe-7B-A1B"}
ENDPOINT="/v1/completions"
PORT=${PORT:-8000}
SEED=42

MODE=${1:-"test"}
INPUT_LEN=${2:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-64}
NUM_PROMPTS=${NUM_PROMPTS:-256}
DATASET_NAME=${DATASET_NAME:-"random2"}
TRACE_CSV=${TRACE_CSV:-"/home/xxf/NewVLLM/AzureLLMInferenceTrace_filtered.csv"}

RESULT_DIR="test/bench_results/${DATASET_NAME}_${MODE}"
mkdir -p "$RESULT_DIR"

if [ "$MODE" = "ttft_agnostic" ]; then
  IFS=' ' read -ra REQUEST_RATES <<< "${REQUEST_RATES:-inf}"
else
  IFS=' ' read -ra REQUEST_RATES <<< "${REQUEST_RATES:-8 12 16}"
fi

echo "================================================================"
echo "Benchmark — dataset: $DATASET_NAME  mode: $MODE"
if [ "$DATASET_NAME" = "random2" ]; then
  echo "Trace CSV: $TRACE_CSV"
else
  echo "input_len: $INPUT_LEN  output_len: $OUTPUT_LEN"
fi
echo "Num prompts: $NUM_PROMPTS"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Results: $RESULT_DIR"
echo "================================================================"
echo ""

TRACE_ARGS=""
if [ "$DATASET_NAME" = "random2" ]; then
  if [ -z "$TRACE_CSV" ]; then
    echo "ERROR: TRACE_CSV is required for random2 dataset" >&2
    exit 1
  fi
  TRACE_ARGS="--trace-csv $TRACE_CSV"
fi

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
    --dataset-name "$DATASET_NAME" \
    --num-prompts $NUM_PROMPTS \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --request-rate "$rate" \
    --seed $SEED \
    --port "$PORT" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_file" \
    $TRACE_ARGS

  echo ""
done

echo "================================================================"
echo "Done. Results in $RESULT_DIR"
echo "================================================================"

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

MODEL=${MODEL:-"/home/xxf/NewVLLM/models_dpsk/uni_05_pa_001arc_8-24/checkpoint-2500"}
# /home/xxf/models/olmoe-7B-A1B
ENDPOINT="/v1/completions"
PORT=${PORT:-8000}
SEED=42

MODE=${1:-"test"}
INPUT_LEN=${2:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-64}
NUM_PROMPTS=${NUM_PROMPTS:-1024}
DATASET_NAME=${DATASET_NAME:-"random2"}
TRACE_CSV=${TRACE_CSV:-"/home/xxf/NewVLLM/vllm/experiment/AzureLLMInferenceTrace_filtered2.csv"}

RESULT_ROOT="${RESULT_ROOT:-test/bench_results}"
RESULT_DIR="${RESULT_ROOT}/${MODE}"
mkdir -p "$RESULT_DIR"

if [[ "$MODE" = "ttft_agnostic" || "$MODE" = "inf" ]]; then
  IFS=' ' read -ra REQUEST_RATES <<< "${REQUEST_RATES:-inf}"
else
  IFS=' ' read -ra REQUEST_RATES <<< "${REQUEST_RATES:-5}"
fi

# TTFT_MAX_STATIC   >0 → fixed value (seconds) for all requests
# PERF_MODEL_PATH   path to perf_model.json (enables calibrated mode)
# TTFT_MULTIPLIER   prefill time multiplier (default 3.0)
# TTFT_QUEUE_MS     extra queue budget in ms (default 5.0)
# TTFT_JITTER_LOW   random multiplier lower bound (default 0.8)
# TTFT_JITTER_HIGH  random multiplier upper bound (default 1.5)

unset TTFT_MAX_STATIC

export PERF_MODEL_PATH="${PERF_MODEL_PATH:-/home/xxf/NewVLLM/test/dpsk_perf_model_24.json}"
export TTFT_MULTIPLIER="${TTFT_MULTIPLIER:-3}"
export TTFT_QUEUE_MS="${TTFT_QUEUE_MS:-10}"
export TTFT_JITTER_LOW="${TTFT_JITTER_LOW:-1}"
export TTFT_JITTER_HIGH="${TTFT_JITTER_HIGH:-1.5}"

# default: normal 4 1 8
export KQOS_DIST="${KQOS_DIST:-normal}"
export QOS_K_MEAN="${QOS_K_MEAN:-8}"
export QOS_K_STD="${QOS_K_STD:-5}"
export QOS_K_MAX="${QOS_K_MAX:-24}"
export DIV_K="${DIV_K:-1}"

# 在uniform的时候，mean和std分别表示min和max
# export KQOS_DIST="uniform"
# export QOS_K_MEAN=1
# export QOS_K_STD=8
# export QOS_K_MAX=8

# export KQOS_LOCAL_DIST=1,1,1,1,8,8

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
    --metric-percentiles "75,90,95,99" \
    $TRACE_ARGS

  echo ""
done

echo "================================================================"
echo "Done. Results in $RESULT_DIR"
echo "================================================================"

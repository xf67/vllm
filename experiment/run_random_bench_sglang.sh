#!/bin/bash
# Quick benchmark for SGLang using vllm bench serve as the client.
#
# Usage:
#   bash run_sglang_bench.sh [MODE_LABEL] [INPUT_LEN]
#
# Environment variables:
#   MODEL_NAME     : must match SGLang's --served-model-name
#   HOST           : SGLang host
#   PORT           : SGLang port
#   API_BASE       : full OpenAI-compatible base url, e.g. http://127.0.0.1:30000/v1
#   DATASET_NAME   : "random" (default) or "random2" (trace-driven)
#   TRACE_CSV      : path to trace CSV (required when DATASET_NAME=random2)
#   NUM_PROMPTS    : number of prompts
#   OUTPUT_LEN     : output cap per request
#   REQUEST_RATES  : space-separated rates
#   BACKEND_KIND   : openai or openai-chat
#   ENDPOINT       : /v1/completions or /v1/chat/completions
#
# Notes:
#   - Default is completions mode to stay closer to your current script.
#   - If the tested model is chat/instruct-only and completions behaves oddly,
#     switch BACKEND_KIND=openai-chat and ENDPOINT=/v1/chat/completions.

set -euo pipefail

MODEL_NAME="${MODEL:-/home/xxf/NewVLLM/models/olmoe-7B-A1B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
API_BASE="${API_BASE:-http://${HOST}:${PORT}}"

SEED=42

MODE="${1:-sglang}"
INPUT_LEN="${2:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
DATASET_NAME="${DATASET_NAME:-random2}"
TRACE_CSV="${TRACE_CSV:-/home/xxf/NewVLLM/vllm/experiment/AzureLLMInferenceTrace_filtered2.csv}"

RESULT_ROOT="${RESULT_ROOT:-test/bench_results}"
RESULT_DIR="${RESULT_ROOT}/${MODE}"
mkdir -p "$RESULT_DIR"

BACKEND_KIND="${BACKEND_KIND:-openai}"
ENDPOINT="${ENDPOINT:-/v1/completions}"

if [[ "$MODE" = "inf" ]]; then
  IFS=' ' read -ra REQUEST_RATES <<< "${REQUEST_RATES:-inf}"
else
  IFS=' ' read -ra REQUEST_RATES <<< "${REQUEST_RATES:-5}"
fi

echo "================================================================"
echo "Benchmark — SGLang baseline"
echo "API base:      $API_BASE"
echo "Model name:    $MODEL_NAME"
echo "Backend:       $BACKEND_KIND"
echo "Endpoint:      $ENDPOINT"
echo "Dataset:       $DATASET_NAME"
if [[ "$DATASET_NAME" == "random2" ]]; then
  echo "Trace CSV:     $TRACE_CSV"
else
  echo "Input len:     $INPUT_LEN"
  echo "Output len:    $OUTPUT_LEN"
fi
echo "Num prompts:   $NUM_PROMPTS"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Results:       $RESULT_DIR"
echo "================================================================"
echo ""

TRACE_ARGS=()
if [[ "$DATASET_NAME" == "random2" ]]; then
  if [[ -z "$TRACE_CSV" ]]; then
    echo "ERROR: TRACE_CSV is required for random2 dataset" >&2
    exit 1
  fi
  TRACE_ARGS+=(--trace-csv "$TRACE_CSV")
fi

for rate in "${REQUEST_RATES[@]}"; do
  echo "---- request-rate=$rate ----"

  rate_label=$(echo "$rate" | tr '.' '_')
  result_file="rate_${rate_label}_in${INPUT_LEN}.json"

  vllm bench serve \
    --backend "$BACKEND_KIND" \
    --base-url "$API_BASE" \
    --model "$MODEL_NAME" \
    --endpoint "$ENDPOINT" \
    --dataset-name "$DATASET_NAME" \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --request-rate "$rate" \
    --seed "$SEED" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_file" \
    --metric-percentiles "75,90,95,99" \
    "${TRACE_ARGS[@]}"

  echo ""
done

echo "================================================================"
echo "Done. Results in $RESULT_DIR"
echo "================================================================"
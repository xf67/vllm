#!/bin/bash
# Benchmark with ShareGPT dataset: compare QOS_AWARE modes.
#
# Prerequisites: vllm server must be running.
#   Start server for each mode separately, then run this script.
#
# Usage:
#   # Mode 1: QoS-aware scheduler
#   QOS_K_LIST=r1,8 QOS_AWARE=1 PERF_MODEL_PATH=test/olmoe_perf_model.json \
#     vllm serve /home/xxf/models/olmoe-7B-A1B \
#     --no-enable-prefix-caching \
#     --compilation-config '{"cudagraph_mode":"PIECEWISE","share_attn_cudagraph_across_topk":true}'
#
#   # Then run:
#   bash test/normal/run_sharegpt_bench.sh [MODE_LABEL]
#
# MODE_LABEL is used for result filenames (e.g. "qos_aware", "baseline_static", "baseline_dynamic")

set -euo pipefail

MODEL="/home/xxf/models/olmoe-7B-A1B"
DATASET_PATH="/home/xxf/NewVLLM/ShareGPT_V3_unfiltered_cleaned_split.json"
ENDPOINT="/v1/completions"
PORT=${PORT:-8000}
SEED=42

MODE=${1:-"test"}
RESULT_DIR="test/bench_results/${MODE}"
mkdir -p "$RESULT_DIR"

NUM_PROMPTS=500

# Request rates to test: from light load to saturation
REQUEST_RATES=(2 4 8 16 inf)

echo "================================================================"
echo "ShareGPT Benchmark — mode: $MODE"
echo "Model: $MODEL"
echo "Num prompts: $NUM_PROMPTS"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Results: $RESULT_DIR"
echo "================================================================"
echo ""

for rate in "${REQUEST_RATES[@]}"; do
  echo "---- request-rate=$rate ----"

  rate_label=$(echo "$rate" | tr '.' '_')
  result_file="rate_${rate_label}.json"

  STATIC_QOS=-1 QOS_K_MEAN=4 QOS_K_STD=2 \
    vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --endpoint "$ENDPOINT" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" \
    --num-prompts $NUM_PROMPTS \
    --request-rate "$rate" \
    --seed $SEED \
    --port "$PORT" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_file"

  echo ""
done

echo "================================================================"
echo "All runs complete. Results in $RESULT_DIR"
echo ""
echo "To compare modes, run with different server configs:"
echo "  bash $0 qos_aware       # QOS_AWARE=1 + PERF_MODEL_PATH"
echo "  bash $0 baseline_dynamic # QOS_AWARE=2 (dynamic k, no scheduler)"
echo "  bash $0 baseline_static  # QOS_AWARE=0 (static max k)"
echo "================================================================"

#!/bin/bash
# Profile prefill time for (seq_len, k) combinations to build the
# PrefillPerfModel lookup table.
#
# Prerequisites: vllm server must be running with prefix caching DISABLED:
#   QOS_K_LIST=r1,8 QOS_AWARE=2 vllm serve $MODEL \
#     --no-enable-prefix-caching \
#     --compilation-config '{"cudagraph_mode":"PIECEWISE","share_attn_cudagraph_across_topk":true}'
#
# Usage:
#   bash profile_prefill.sh [MODEL_PATH] [OUTPUT_JSON]

set -euo pipefail

MODEL=${1:-"/home/xxf/models/olmoe-7B-A1B"}
OUTPUT=${2:-"/home/xxf/NewVLLM/test/olmoe_perf_model.json"}
ENDPOINT="/v1/completions"
NUM_PROMPTS=32
SEED=42
PORT=${PORT:-8000}

SEQ_LENS=(128 256 512 1024 2048)
K_VALUES=(1 2 3 4 5 6 7 8)

RESULT_DIR="$(dirname "$OUTPUT")/profile_results"
mkdir -p "$RESULT_DIR"

echo "================================================================"
echo "Profiling prefill time → $OUTPUT"
echo "Intermediate results → $RESULT_DIR"
echo "Model: $MODEL"
echo "Seq lens: ${SEQ_LENS[*]}"
echo "K values: ${K_VALUES[*]}"
echo "================================================================"
echo ""

for k in "${K_VALUES[@]}"; do
  for seq_len in "${SEQ_LENS[@]}"; do
    echo -n "  k=$k  seq_len=$seq_len ... "
    outfile="$RESULT_DIR/k${k}_s${seq_len}.json"

    STATIC_QOS=$k vllm bench serve \
      --backend vllm \
      --model "$MODEL" \
      --endpoint "$ENDPOINT" \
      --dataset-name random \
      --num-prompts $NUM_PROMPTS \
      --random-input-len $seq_len \
      --random-output-len 1 \
      --max-concurrency 1 \
      --seed $SEED \
      --port "$PORT" \
      --save-result \
      --result-dir "$RESULT_DIR" \
      --result-filename "k${k}_s${seq_len}.json" \
      2>/dev/null

    if [ -f "$outfile" ]; then
      mean_ttft=$(python3 "$(dirname "$0")/read_ttft.py" "$outfile")
      echo "${mean_ttft}ms"
    else
      echo "FAILED"
    fi
  done
done

# Aggregate results into the lookup table JSON
echo ""
echo "Building $OUTPUT ..."
python3 "$(dirname "$0")/aggregate_results.py" "$RESULT_DIR" "$OUTPUT" "${SEQ_LENS[*]}" "${K_VALUES[*]}"

echo ""
echo "Done."
echo "  Perf model:           $OUTPUT"
echo "  Intermediate results: $RESULT_DIR"
echo "  Set PERF_MODEL_PATH=$OUTPUT when starting the vllm server."

#!/bin/bash
# ============================================================
#  SGLang Server Launcher
# ============================================================
#
# Usage:
#   bash start_sglang_server.sh
#
# Optional env vars:
#   MODEL               model path or HF repo id
#   SERVED_MODEL_NAME   model name exposed via /v1/models
#   HOST                bind host
#   PORT                bind port
#   TP_SIZE             tensor parallel size
#   MEM_FRACTION_STATIC static memory fraction
#   CONTEXT_LENGTH      max context length
#   SCHEDULE_POLICY     fcfs / lpm / random / ...
#   DISABLE_RADIX_CACHE 1 to disable prefix cache for fairness
#   DISABLE_CUDA_GRAPH  1 to disable cuda graph
#   CHUNKED_PREFILL_SIZE
#   MAX_RUNNING_REQUESTS
#   LOG_LEVEL
#   API_KEY
#
# Notes:
#   - For paper baseline fairness, if your vLLM run disables prefix caching,
#     set DISABLE_RADIX_CACHE=1 here too.
#   - If your model path is local, SERVED_MODEL_NAME should be a clean stable
#     string used by the benchmark client.
# ============================================================

set -euo pipefail

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export NVCC_CCBIN=/usr/bin/g++-12

MODEL="${MODEL:-/home/xxf/NewVLLM/models/olmoe-7B-A1B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-/home/xxf/NewVLLM/models/olmoe-7B-A1B}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-fcfs}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"

LOG_LEVEL="${LOG_LEVEL:-info}"
API_KEY="${API_KEY:-}"

DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"

CMD=(
  python -m sglang.launch_server
  --model-path "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp "$TP_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --context-length "$CONTEXT_LENGTH"
  --schedule-policy "$SCHEDULE_POLICY"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --log-level "$LOG_LEVEL"
)

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api-key "$API_KEY")
fi

if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
  CMD+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
fi

if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  CMD+=(--disable-radix-cache)
fi

if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  CMD+=(--disable-cuda-graph)
fi

echo "============================================================"
echo "  SGLang Server"
echo "============================================================"
echo "  Model:               $MODEL"
echo "  Served model name:   $SERVED_MODEL_NAME"
echo "  Host:                $HOST"
echo "  Port:                $PORT"
echo "  TP size:             $TP_SIZE"
echo "  Mem fraction:        $MEM_FRACTION_STATIC"
echo "  Context length:      $CONTEXT_LENGTH"
echo "  Schedule policy:     $SCHEDULE_POLICY"
echo "  Chunked prefill:     $CHUNKED_PREFILL_SIZE"
echo "  Max running reqs:    ${MAX_RUNNING_REQUESTS:-<auto>}"
echo "  Disable radix cache: $DISABLE_RADIX_CACHE"
echo "  Disable cuda graph:  $DISABLE_CUDA_GRAPH"
echo "  Log level:           $LOG_LEVEL"
echo "============================================================"
echo ""

exec "${CMD[@]}"
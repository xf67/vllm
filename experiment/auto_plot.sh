#!/usr/bin/env bash

INPUT_PATH=/home/xxf/NewVLLM/vllm/test/bench_results
NAMES=(fifo fifo2 fifo4 fifo8 fifo16)

for name in "${NAMES[@]}"; do
  python /home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py \
    "${INPUT_PATH}/log-${name}" \
    -o "${INPUT_PATH}/${name}_plot"
done


ARGS=()
for name in "${NAMES[@]}"; do
  ARGS+=("${INPUT_PATH}/random2_${name}" "${name}")
done

python /home/xxf/NewVLLM/vllm/experiment/plot_bench_results.py \
  "${ARGS[@]}" \
  -o "${INPUT_PATH}/plot_compare"
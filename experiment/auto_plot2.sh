#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Post-process experiment results
#
# Usage:
#   bash process_all_results.sh
#   bash process_all_results.sh <group_name>
#
# Examples:
#   bash process_all_results.sh
#   bash process_all_results.sh normal_rr20_std0p5
#   bash process_all_results.sh uniform_rr20_min1_max8
# ============================================================

AUTO_RUNS_ROOT="${AUTO_RUNS_ROOT:-/home/xxf/NewVLLM/vllm/experiment/test/auto_runs}"
BENCH_RESULTS_ROOT="${BENCH_RESULTS_ROOT:-/home/xxf/NewVLLM/vllm/experiment/test/bench_results}"

PLOT_DISPATCH="${PLOT_DISPATCH:-/home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py}"
PLOT_BENCH="${PLOT_BENCH:-/home/xxf/NewVLLM/vllm/experiment/plot_bench_results.py}"

# 可选：只处理某一个 client group
TARGET_GROUP="${1:-}"

shopt -s nullglob

process_one_group() {
  local group_dir="$1"
  local group_name
  group_name="$(basename "$group_dir")"

  echo "============================================================"
  echo "[GROUP] $group_name"
  echo "============================================================"

  local ARGS=()

  for exp_dir in "${group_dir}"/*; do
    [[ -d "$exp_dir" ]] || continue

    local exp_name
    exp_name="$(basename "$exp_dir")"
    echo "[EXP] $exp_name"

    local dispatch_csv="${AUTO_RUNS_ROOT}/${group_name}/${exp_name}/dispatch.csv"
    local dispatch_out="${AUTO_RUNS_ROOT}/${group_name}/${exp_name}/dispatch_plot"

    if [[ -f "$dispatch_csv" ]]; then
      echo "  -> plot dispatch: $dispatch_csv"
      python "$PLOT_DISPATCH" \
        "$dispatch_csv" \
        -o "$dispatch_out"
    else
      echo "  -> skip dispatch plot, file not found: $dispatch_csv"
    fi

    ARGS+=("$exp_dir" "$exp_name")
  done

  if (( ${#ARGS[@]} > 0 )); then
    local compare_out="${group_dir}/plot_compare"
    echo "[GROUP] plot bench compare -> $compare_out"
    python "$PLOT_BENCH" \
      "${ARGS[@]}" \
      -o "$compare_out"
  else
    echo "[GROUP] no experiment dirs found under $group_dir"
  fi

  echo
}

if [[ -n "$TARGET_GROUP" ]]; then
  target_dir="${BENCH_RESULTS_ROOT}/${TARGET_GROUP}"
  if [[ ! -d "$target_dir" ]]; then
    echo "ERROR: target group not found: $target_dir" >&2
    exit 1
  fi
  process_one_group "$target_dir"
else
  for group_dir in "${BENCH_RESULTS_ROOT}"/*; do
    [[ -d "$group_dir" ]] || continue
    process_one_group "$group_dir"
  done
fi

echo "All post-processing done."
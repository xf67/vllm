#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Post-process all experiment results
# ============================================================

AUTO_RUNS_ROOT="${AUTO_RUNS_ROOT:-/home/xxf/NewVLLM/vllm/experiment/test/auto_runs}"
BENCH_RESULTS_ROOT="${BENCH_RESULTS_ROOT:-/home/xxf/NewVLLM/vllm/experiment/test/bench_results}"

PLOT_DISPATCH="${PLOT_DISPATCH:-/home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py}"
PLOT_BENCH="${PLOT_BENCH:-/home/xxf/NewVLLM/vllm/experiment/plot_bench_results.py}"

# 只处理一级 client group，如 rr20_std0p25
shopt -s nullglob

for group_dir in "${BENCH_RESULTS_ROOT}"/*; do
  [[ -d "$group_dir" ]] || continue

  group_name="$(basename "$group_dir")"
  echo "============================================================"
  echo "[GROUP] $group_name"
  echo "============================================================"

  ARGS=()

  # 遍历该 group 下每个实验目录
  for exp_dir in "${group_dir}"/*; do
    [[ -d "$exp_dir" ]] || continue

    exp_name="$(basename "$exp_dir")"
    echo "[EXP] $exp_name"

    dispatch_csv="${AUTO_RUNS_ROOT}/${group_name}/${exp_name}/dispatch.csv"
    dispatch_out="${AUTO_RUNS_ROOT}/${group_name}/${exp_name}/dispatch_plot"

    if [[ -f "$dispatch_csv" ]]; then
      echo "  -> plot dispatch: $dispatch_csv"
      python "$PLOT_DISPATCH" \
        "$dispatch_csv" \
        -o "$dispatch_out"
    else
      echo "  -> skip dispatch plot, file not found: $dispatch_csv"
    fi

    # bench compare 用的是 bench_results/<group>/<exp> 整个目录
    ARGS+=("$exp_dir" "$exp_name")
  done

  # 画这个 client group 下所有实验的 bench 对比图
  if (( ${#ARGS[@]} > 0 )); then
    compare_out="${group_dir}/plot_compare"
    echo "[GROUP] plot bench compare -> $compare_out"
    python "$PLOT_BENCH" \
      "${ARGS[@]}" \
      -o "$compare_out"
  else
    echo "[GROUP] no experiment dirs found under $group_dir"
  fi

  echo
done

echo "All post-processing done."
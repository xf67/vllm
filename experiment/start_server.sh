#!/bin/bash
# ============================================================
#  MoE QoS-aware vLLM Server Launcher
# ============================================================
#
#  Usage:
#    bash start_server.sh [MODE]
#
#  MODE = fifo (default) | edf | ttft_agnostic
#
#  Examples:
#    bash start_server.sh fifo
#    bash start_server.sh edf
#    bash start_server.sh ttft_agnostic
#
# ============================================================
set -euo pipefail

# -------------------- Model & Server -----------------------
MODEL="${MODEL:-/home/xxf/NewVLLM/models_dpsk/uni_05_pa_001arc_8-24/checkpoint-2500}" 
# /home/xxf/models/olmoe-7B-A1B
# ~/MoE-Prism/moe-gate-finetune-deepseek/uni_05_pa_001arc_8-24/checkpoint-2500
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

# -------------------- Scheduling Mode ----------------------
# fifo          : 纯FCFS，forward时k=batch中max(k_qos)
# edf           : Earliest-Deadline-First，按TTFT紧迫度排序
# ttft_agnostic : 离线最大吞吐，按k_qos分组batch
# fifo_safe_swap: fifo但是有个换序
SCHED_MODE="${SCHED_MODE:-fifo}"
VLLM_DP_K_AWARE_DISPATCH="${VLLM_DP_K_AWARE_DISPATCH:-0}"
# [a,b,c,d...]表示dp_rank0,1,2,3...分别属于a,b,c,d...lane
VLLM_DP_ENGINE_LANES="${VLLM_DP_ENGINE_LANES:-0,1}"
# 固定K边界模式开关；开启后:
#   k < lower 只走 lane0
#   k > upper 只走 lane1
#   中间区间按 4*waiting+running 做负载均衡
VLLM_DP_FIXED_K_BOUNDARY_DISPATCH="${VLLM_DP_FIXED_K_BOUNDARY_DISPATCH:-0}"
# 固定K边界，格式为 lower,upper
VLLM_DP_K_BOUNDARIES="${VLLM_DP_K_BOUNDARIES:-}"
# boundary覆盖的k宽度；=1时与原先单点boundary行为一致
VLLM_DP_K_BOUNDARY_WIDTH="${VLLM_DP_K_BOUNDARY_WIDTH:-20}"
# 动态boundary模式的初始topk boundary
VLLM_DP_K_THRESHOLD="${VLLM_DP_K_THRESHOLD-12}"
# boundary移动的条件是 running+waiting*4 作为pressure，pressure的差值超过这个hysteresis
VLLM_DP_K_HYSTERESIS="${VLLM_DP_K_HYSTERESIS-32}"
# cooldown是指变化boundary后几个step之内不能再变
VLLM_DP_K_COOLDOWN="${VLLM_DP_K_COOLDOWN-4}"
# lane内没有waiting请求时，允许跨lane挑选waiting=0的rank
MAYBE_OVERRIDE="${MAYBE_OVERRIDE:-0}"

# -------------------- QoS / K 相关 -------------------------
# QOS_AWARE: model runner层面是否将k_qos传给forward (bool)
#   0 = 不传k_qos（纯vllm默认，忽略所有QoS）
#   1 = 传k_qos，forward使用batch中max(k_qos)
QOS_AWARE="${QOS_AWARE:-1}"

# QOS_K_LIST: CUDA graph capture的k范围
#   格式: r<start>,<end>  (range) 或 k1,k2,k3 (list)
#   例: r1,8 表示 k=1..8;  2,4,6,8 表示捕获这4个k
QOS_K_LIST="${QOS_K_LIST:-r1,24}"

# -------------------- Perf Model (EDF) ---------------------
# prefill时间预测模型JSON路径，EDF模式必需
PERF_MODEL_PATH="${PERF_MODEL_PATH:-/home/xxf/NewVLLM/test/dpsk_perf_model_24.json}"

# -------------------- EDF 参数 -----------------------------
# TTFT安全系数: slack < est_prefill * factor 时紧急调度
TTFT_SAFETY_FACTOR="${TTFT_SAFETY_FACTOR:-1.5}"
# 前瞻步数: 预测多少步后高k decode任务可能结束
EDF_LOOKAHEAD_STEPS="${EDF_LOOKAHEAD_STEPS:-5}"
# k准入门控: 当request的k > 预测未来batch_k时，
# 只在 slack < ttft_max * urgency 时才放行（0.3 = 已消耗70%时间预算才放行）
EDF_K_GATE_URGENCY="${EDF_K_GATE_URGENCY:-0.3}"

# ---- FIFO_SWAP ---
FIFO_SAFE_SWAP_WINDOW="${FIFO_SAFE_SWAP_WINDOW:-8}"
FIFO_SWAP_KUP_RATIO="${FIFO_SWAP_KUP_RATIO:-0.9}"

# -------------------- TTFT_AGNOSTIC 参数 -------------------
# batch利用率阈值: 当已用token >= max_tokens * ratio时, 不再提升k等级
TTFT_AGNOSTIC_MIN_BATCH_RATIO="${TTFT_AGNOSTIC_MIN_BATCH_RATIO:-0.5}"

# -------------------- Compilation / CUDAGraph --------------
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
SHARE_ATTN_ACROSS_TOPK="${SHARE_ATTN_ACROSS_TOPK:-true}"

# -------------------- Dispatch Metrics Log -----------------
# 设置后，scheduler每步写一行CSV，记录dispatch K、队列K分布等
# 留空则不记录
DISPATCH_LOG="${DISPATCH_LOG:-/home/xxf/NewVLLM/vllm/test/log.csv}"

# -------------------- Debug & Profiling --------------------
# 取消注释以下行来启用
# export VLLM_TORCH_PROFILER_DIR=/home/xxf/NewVLLM/traces
# export RUN_DEBUG_PORT=5678
# export VLLM_DISABLE_COMPILE_CACHE=1
# export VLLM_LOGGING_LEVEL="DEBUG"


# ============================================================
#  Export all env vars
# ============================================================
export SCHED_MODE
export QOS_AWARE
export QOS_K_LIST
export PERF_MODEL_PATH
export TTFT_SAFETY_FACTOR
export EDF_LOOKAHEAD_STEPS
export EDF_K_GATE_URGENCY
export TTFT_AGNOSTIC_MIN_BATCH_RATIO
export DISPATCH_LOG
export FIFO_SAFE_SWAP_WINDOW
export FIFO_SWAP_KUP_RATIO
export VLLM_DP_K_AWARE_DISPATCH
export VLLM_DP_ENGINE_LANES
export VLLM_DP_FIXED_K_BOUNDARY_DISPATCH
export VLLM_DP_K_BOUNDARIES
export VLLM_DP_K_BOUNDARY_WIDTH
export VLLM_DP_K_THRESHOLD
export VLLM_DP_K_HYSTERESIS
export VLLM_DP_K_COOLDOWN
export MAYBE_OVERRIDE

# ============================================================
#  Print config summary
# ============================================================
echo "============================================================"
echo "  MoE QoS vLLM Server"
echo "============================================================"
echo "  Model:             $MODEL"
echo "  Port:              $PORT"
echo "  GPU Mem Util:      $GPU_MEM_UTIL"
echo "  Sched Mode:        $SCHED_MODE"
echo "  QOS_AWARE:         $QOS_AWARE"
echo "  QOS_K_LIST:        $QOS_K_LIST"
echo "  Perf Model:        $PERF_MODEL_PATH"
echo "  DP K-AWARE:        $VLLM_DP_K_AWARE_DISPATCH"
echo "  DP LANE CONFIG:    $VLLM_DP_ENGINE_LANES"
echo "  DP FIXED K MODE:   $VLLM_DP_FIXED_K_BOUNDARY_DISPATCH"
echo "  DP K BOUNDARIES:   $VLLM_DP_K_BOUNDARIES"
echo "  DP K BDR WIDTH:    $VLLM_DP_K_BOUNDARY_WIDTH"
echo "  DP K THRESHOLD:    $VLLM_DP_K_THRESHOLD"
echo "  MAYBE_OVERRIDE:    $MAYBE_OVERRIDE"
echo "  DP:                $DP_SIZE"
echo "  TP:                $TP_SIZE"
echo "  PP:                $PP_SIZE"
echo "------------------------------------------------------------"
echo "  EDF params:"
echo "    TTFT_SAFETY_FACTOR:       $TTFT_SAFETY_FACTOR"
echo "    EDF_LOOKAHEAD_STEPS:      $EDF_LOOKAHEAD_STEPS"
echo "    EDF_K_GATE_URGENCY:       $EDF_K_GATE_URGENCY"
echo "  TTFT_AGNOSTIC params:"
echo "    MIN_BATCH_RATIO:          $TTFT_AGNOSTIC_MIN_BATCH_RATIO"
echo "  FIFO_SWAP params:"
echo "    SWAP_WINDOW:              $FIFO_SAFE_SWAP_WINDOW"
echo "    FIFO_SWAP_KUP_RATIO:      $FIFO_SWAP_KUP_RATIO"
echo "  Dispatch Log:      ${DISPATCH_LOG:-<disabled>}"
echo "------------------------------------------------------------"
echo "  CUDAGraph Mode:    $CUDAGRAPH_MODE"
echo "  Share Attn TopK:   $SHARE_ATTN_ACROSS_TOPK"
echo "============================================================"
echo ""

# ============================================================
#  Launch
# ============================================================
exec vllm serve "$MODEL" \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\", \"share_attn_cudagraph_across_topk\": $SHARE_ATTN_ACROSS_TOPK}" \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE"

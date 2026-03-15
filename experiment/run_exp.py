#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Config
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = Path(os.environ.get("SERVER_SCRIPT", SCRIPT_DIR / "start_server.sh"))
BENCH_SCRIPT = Path(os.environ.get("BENCH_SCRIPT", SCRIPT_DIR / "run_random_bench.sh"))

PORT = os.environ.get("PORT", "8000")
INPUT_LEN = os.environ.get("INPUT_LEN", "2048")

MODEL = os.environ.get("MODEL", "/home/xxf/models/olmoe-7B-A1B")
TRACE_CSV = os.environ.get(
    "TRACE_CSV", "/home/xxf/NewVLLM/AzureLLMInferenceTrace_filtered2.csv"
)
DATASET_NAME = os.environ.get("DATASET_NAME", "random2")
NUM_PROMPTS = os.environ.get("NUM_PROMPTS", "2048")
OUTPUT_LEN = os.environ.get("OUTPUT_LEN", "64")

BASE_DIR = Path(os.environ.get("BASE_DIR", SCRIPT_DIR / "test" / "auto_runs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_RATES = ["20", "30", "40", "inf"]

# normal 分布下扫描的 std
NORMAL_QOS_K_STDS = ["3","2","1"]

# uniform 分布下的配置
# 注意：在你的 bench 逻辑里，uniform 时 mean=min, std=max
UNIFORM_CASES = [
    {
        "QOS_K_MEAN": "1",   # min
        "QOS_K_STD": "8",    # max
        "QOS_K_MAX": "8",
    }
]

FIFO_SAFE_SWAP_WINDOWS = ["2", "4", "6", "8", "12", "16"]
TTFT_AGNOSTIC_RATIOS = ["0.5", "0.75", "0.9"]

COMMON_ENV = {
    "MODEL": MODEL,
    "PORT": PORT,
    "TRACE_CSV": TRACE_CSV,
    "DATASET_NAME": DATASET_NAME,
    "NUM_PROMPTS": NUM_PROMPTS,
    "OUTPUT_LEN": OUTPUT_LEN,
    "GPU_MEM_UTIL": os.environ.get("GPU_MEM_UTIL", "0.9"),
    "QOS_AWARE": os.environ.get("QOS_AWARE", "1"),
    "QOS_K_LIST": os.environ.get("QOS_K_LIST", "r1,8"),
    "PERF_MODEL_PATH": os.environ.get(
        "PERF_MODEL_PATH", "/home/xxf/NewVLLM/test/olmoe_perf_model.json"
    ),
    "TTFT_SAFETY_FACTOR": os.environ.get("TTFT_SAFETY_FACTOR", "1.5"),
    "EDF_LOOKAHEAD_STEPS": os.environ.get("EDF_LOOKAHEAD_STEPS", "5"),
    "EDF_K_GATE_URGENCY": os.environ.get("EDF_K_GATE_URGENCY", "0.3"),
    "CUDAGRAPH_MODE": os.environ.get("CUDAGRAPH_MODE", "PIECEWISE"),
    "SHARE_ATTN_ACROSS_TOPK": os.environ.get("SHARE_ATTN_ACROSS_TOPK", "true"),
    "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "INFO"),
    "TTFT_MULTIPLIER": os.environ.get("TTFT_MULTIPLIER", "2"),
    "TTFT_QUEUE_MS": os.environ.get("TTFT_QUEUE_MS", "10"),
    "TTFT_JITTER_LOW": os.environ.get("TTFT_JITTER_LOW", "1"),
    "TTFT_JITTER_HIGH": os.environ.get("TTFT_JITTER_HIGH", "1.5"),
    "KQOS_DIST": os.environ.get("KQOS_DIST", "normal"),
    "QOS_K_MEAN": os.environ.get("QOS_K_MEAN", "4"),
    "QOS_K_MAX": os.environ.get("QOS_K_MAX", "8"),
}


# ============================================================
# Helpers
# ============================================================

def sanitize(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "p")
    )


def kill_process_group(proc: Optional[subprocess.Popen], name: str) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    print(f"[CLEANUP] stopping {name} pgid={pgid}")

    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return

    for _ in range(30):
        if proc.poll() is not None:
            print(f"[CLEANUP] {name} exited after SIGINT")
            return
        time.sleep(1)

    print(f"[CLEANUP] {name} SIGINT timeout, sending SIGTERM")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    for _ in range(10):
        if proc.poll() is not None:
            print(f"[CLEANUP] {name} exited after SIGTERM")
            return
        time.sleep(1)

    print(f"[CLEANUP] {name} SIGTERM timeout, sending SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def start_server(server_mode: str, exp_dir: Path, env: Dict[str, str]) -> subprocess.Popen:
    server_log = exp_dir / "server.log"
    f = open(server_log, "w")
    proc = subprocess.Popen(
        ["bash", str(SERVER_SCRIPT), server_mode],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
        cwd=str(SCRIPT_DIR),
        text=True,
    )
    print(f"[SERVER] started mode={server_mode}, pid={proc.pid}, log={server_log}")
    return proc


def run_bench(bench_label: str, exp_dir: Path, env: Dict[str, str]) -> int:
    bench_log = exp_dir / "bench.log"
    with open(bench_log, "w") as f:
        proc = subprocess.run(
            ["bash", str(BENCH_SCRIPT), bench_label, INPUT_LEN],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(SCRIPT_DIR),
            text=True,
        )
    print(
        f"[BENCH] finished label={bench_label}, "
        f"returncode={proc.returncode}, log={bench_log}"
    )
    return proc.returncode


def make_env(
    request_rate: str,
    exp_dir: Path,
    dist_env: Dict[str, str],
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)

    env["REQUEST_RATES"] = request_rate
    env["DISPATCH_LOG"] = str(exp_dir / "dispatch.csv")

    # 分布相关参数
    env.update(dist_env)

    if extra_env:
        env.update(extra_env)

    # 避免上一轮遗留变量污染
    if not extra_env or "FIFO_SAFE_SWAP_WINDOW" not in extra_env:
        env.pop("FIFO_SAFE_SWAP_WINDOW", None)
    if not extra_env or "TTFT_AGNOSTIC_MIN_BATCH_RATIO" not in extra_env:
        env.pop("TTFT_AGNOSTIC_MIN_BATCH_RATIO", None)

    return env


def run_one(
    server_mode: str,
    exp_name: str,
    request_rate: str,
    client_group: str,
    dist_env: Dict[str, str],
    extra_env: Optional[Dict[str, str]] = None,
) -> bool:
    rel_exp_name = f"{client_group}/{exp_name}"

    exp_dir = BASE_DIR / rel_exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    bench_label = rel_exp_name

    env = make_env(
        request_rate=request_rate,
        exp_dir=exp_dir,
        dist_env=dist_env,
        extra_env=extra_env,
    )

    print("\n" + "=" * 68)
    print(f"[RUN] client={client_group} exp={exp_name}")
    print(f"[RUN] exp_dir={exp_dir}")
    print(f"[RUN] bench_result_dir=test/bench_results/{rel_exp_name}")
    print(f"[RUN] dispatch={env['DISPATCH_LOG']}")
    print(f"[RUN] dist_env={dist_env}")
    if extra_env:
        print(f"[RUN] extra_env={extra_env}")
    print("=" * 68)

    server_proc = None
    bench_code = 1
    try:
        server_proc = start_server(server_mode, exp_dir, env)
        bench_code = run_bench(bench_label, exp_dir, env)
    finally:
        kill_process_group(server_proc, "server")

    if bench_code != 0:
        print(f"[FAIL] client={client_group} exp={exp_name} bench_exit={bench_code}")
        return False

    print(f"[DONE] client={client_group} exp={exp_name}")
    return True


def run_all_server_modes(
    request_rate: str,
    client_group: str,
    dist_env: Dict[str, str],
    failures: List[str],
) -> None:
    # 1) fifo
    ok = run_one(
        server_mode="fifo",
        exp_name="fifo",
        request_rate=request_rate,
        client_group=client_group,
        dist_env=dist_env,
        extra_env=None,
    )
    if not ok:
        failures.append(f"{client_group}/fifo")

    # 2) fifo_safe_swap
    for window in FIFO_SAFE_SWAP_WINDOWS:
        ok = run_one(
            server_mode="fifo_safe_swap",
            exp_name=f"fifo_safe_swap_w{sanitize(window)}",
            request_rate=request_rate,
            client_group=client_group,
            dist_env=dist_env,
            extra_env={"FIFO_SAFE_SWAP_WINDOW": window},
        )
        if not ok:
            failures.append(f"{client_group}/fifo_safe_swap_w{window}")

    # 3) ttft_agnostic
    for ratio in TTFT_AGNOSTIC_RATIOS:
        ok = run_one(
            server_mode="ttft_agnostic",
            exp_name=f"ttft_agnostic_ratio{sanitize(ratio)}",
            request_rate=request_rate,
            client_group=client_group,
            dist_env=dist_env,
            extra_env={"TTFT_AGNOSTIC_MIN_BATCH_RATIO": ratio},
        )
        if not ok:
            failures.append(f"{client_group}/ttft_agnostic_ratio{ratio}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    failures: List[str] = []

    try:
        # ----------------------------------------------------
        # normal groups
        # ----------------------------------------------------
        for rate in REQUEST_RATES:
            for std in NORMAL_QOS_K_STDS:
                dist_env = {
                    "KQOS_DIST": "normal",
                    "QOS_K_MEAN": "4",
                    "QOS_K_STD": std,
                    "QOS_K_MAX": "8",
                }
                client_group = f"normal_rr{sanitize(rate)}_std{sanitize(std)}"
                run_all_server_modes(
                    request_rate=rate,
                    client_group=client_group,
                    dist_env=dist_env,
                    failures=failures,
                )

        # ----------------------------------------------------
        # uniform groups
        # ----------------------------------------------------
        for rate in REQUEST_RATES:
            for case in UNIFORM_CASES:
                dist_env = {
                    "KQOS_DIST": "uniform",
                    "QOS_K_MEAN": case["QOS_K_MEAN"],   # min
                    "QOS_K_STD": case["QOS_K_STD"],     # max
                    "QOS_K_MAX": case["QOS_K_MAX"],
                }
                client_group = (
                    f"uniform_rr{sanitize(rate)}"
                    f"_min{sanitize(case['QOS_K_MEAN'])}"
                    f"_max{sanitize(case['QOS_K_STD'])}"
                )
                run_all_server_modes(
                    request_rate=rate,
                    client_group=client_group,
                    dist_env=dist_env,
                    failures=failures,
                )

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] user interrupted")
        return 130

    print("\n" + "=" * 68)
    print("All experiments finished.")
    print(f"Base dir: {BASE_DIR}")
    print("=" * 68)

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
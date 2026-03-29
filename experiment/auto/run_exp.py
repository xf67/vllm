#!/usr/bin/env python3
import os
import sys
import yaml
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

SERVER_SCRIPT = "/home/xxf/NewVLLM/vllm/test/start_server.sh"
BENCH_SCRIPT = "/home/xxf/NewVLLM/vllm/test/run_random_bench.sh"
EXPERIMENTS_YAML = "/home/xxf/NewVLLM/vllm/test/experiments.yaml"
WORKDIR = "/home/xxf/NewVLLM/vllm"

LOG_ROOT = Path("/home/xxf/NewVLLM/vllm/test/auto_logs")
COOLDOWN_SECONDS = 5
SERVER_STOP_GRACE = 20

# -------------------------
# Python 内统一默认值
# 实验 json 里没写的，就走这里
# -------------------------
DEFAULT_SERVER_ARGS = ["fifo"]
DEFAULT_BENCH_ARGS = ["fifo", "2048"]

DEFAULT_ENV: Dict[str, str] = {
    "PORT": "8000",
    "NUM_PROMPTS": "1024",
    "DATASET_NAME": "random2",
    "TRACE_CSV": "/home/xxf/NewVLLM/vllm/experiment/AzureLLMInferenceTrace_filtered2.csv",
    "MODEL": "/home/xxf/NewVLLM/models/olmoe-p",
    "RESULT_ROOT": "/home/xxf/NewVLLM/vllm/test/bench_results",

    "OUTPUT_LEN": "64",
    "QOS_K_MEAN": "8",
    "QOS_K_STD": "5",
    "QOS_K_MAX": "24",
    "DIV_K": "1",
    "KQOS_DIST": "normal",

    "PERF_MODEL_PATH": "/home/xxf/NewVLLM/test/dpsk_perf_model_24.json",
    "TTFT_MULTIPLIER": "3",
    "TTFT_QUEUE_MS": "10",
    "TTFT_JITTER_LOW": "1",
    "TTFT_JITTER_HIGH": "1.5",

    "GPU_MEM_UTIL": "0.9",
    "DP_SIZE": "2",
    "QOS_AWARE": "1",
    "QOS_K_LIST": "r1,24",
    "CUDAGRAPH_MODE": "PIECEWISE",
    "SHARE_ATTN_ACROSS_TOPK": "true",

    "VLLM_DP_K_AWARE_DISPATCH": "0",
    "VLLM_DP_ENGINE_LANES": "0,1",
    "VLLM_DP_FIXED_K_BOUNDARY_DISPATCH": "0",
    "VLLM_DP_K_BOUNDARIES": "",
    "VLLM_DP_K_BOUNDARY_WIDTH": "20",
    "VLLM_DP_K_THRESHOLD": "12",
    "VLLM_DP_K_HYSTERESIS": "32",
    "VLLM_DP_K_COOLDOWN": "16",
    "MAYBE_OVERRIDE": "0",

    "REQUEST_RATES": "5"
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_experiments(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError("experiments.yaml is empty")

    if isinstance(data, list):
        experiments = data
    elif isinstance(data, dict) and "experiments" in data:
        experiments = data["experiments"]
    else:
        raise ValueError("experiments.yaml must be either a list or a dict with key 'experiments'")

    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")

    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f"experiment #{i} is not a dict")
        if "name" not in exp:
            raise ValueError(f"experiment #{i} missing required field: name")

    return experiments


def build_final_env(exp_env: Optional[Dict[str, Any]]) -> Dict[str, str]:
    env = os.environ.copy()

    # 先叠默认值
    env.update(DEFAULT_ENV)

    # 再叠实验自己的值
    if exp_env:
        env.update({k: str(v) for k, v in exp_env.items()})

    return env


def build_final_server_args(exp: Dict[str, Any]) -> List[str]:
    return list(exp.get("server_args", DEFAULT_SERVER_ARGS))


def build_final_bench_args(exp: Dict[str, Any]) -> List[str]:
    return list(exp.get("bench_args", DEFAULT_BENCH_ARGS))


def popen_cmd(
    cmd: List[str],
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        preexec_fn=os.setsid,
        text=True,
    )
    proc._stdout_f = stdout_f
    proc._stderr_f = stderr_f
    return proc


def close_proc_files(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    for attr in ("_stdout_f", "_stderr_f"):
        f = getattr(proc, attr, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


def kill_process_group(proc: Optional[subprocess.Popen], name: str, grace_sec: int = 15) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        close_proc_files(proc)
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        close_proc_files(proc)
        return

    print(f"[INFO] stopping {name}, pid={proc.pid}, pgid={pgid}")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        close_proc_files(proc)
        return

    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            close_proc_files(proc)
            return
        time.sleep(1)

    print(f"[WARN] {name} did not exit after SIGTERM, force killing...")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    close_proc_files(proc)


def run_one_experiment(exp_idx: int, exp: Dict[str, Any]) -> Dict[str, Any]:
    name = exp["name"]
    exp_dir = LOG_ROOT / f"{exp_idx:03d}_{name}"
    ensure_dir(exp_dir)

    final_env = build_final_env(exp.get("env"))
    server_args = build_final_server_args(exp)
    bench_args = build_final_bench_args(exp)

    server_cmd = ["bash", SERVER_SCRIPT] + server_args
    bench_cmd = ["bash", BENCH_SCRIPT] + bench_args

    # 自动准备结果目录
    if "RESULT_ROOT" in final_env and final_env["RESULT_ROOT"]:
        ensure_dir(Path(final_env["RESULT_ROOT"]))

    # 自动准备并清空 dispatch log
    if "DISPATCH_LOG" in final_env and final_env["DISPATCH_LOG"]:
        dispatch_log = Path(final_env["DISPATCH_LOG"])
        ensure_dir(dispatch_log.parent)
        if dispatch_log.exists():
            dispatch_log.unlink()

    # 保存本轮实际配置
    config_to_save = {
        "name": name,
        "exp_idx": exp_idx,
        "server_cmd": server_cmd,
        "bench_cmd": bench_cmd,
        "default_env": DEFAULT_ENV,
        "override_env": exp.get("env", {}),
        "final_env": final_env,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"[INFO] experiment: {exp_idx:03d}_{name}")
    print(f"[INFO] log dir: {exp_dir}")
    print(f"[INFO] server cmd: {' '.join(server_cmd)}")
    print(f"[INFO] bench  cmd: {' '.join(bench_cmd)}")
    print("=" * 100)

    server_proc = None
    bench_proc = None
    status = {
        "name": name,
        "exp_idx": exp_idx,
        "log_dir": str(exp_dir),
        "bench_returncode": None,
        "ok": False,
    }

    try:
        server_proc = popen_cmd(
            cmd=server_cmd,
            env=final_env,
            stdout_path=exp_dir / "server.stdout.log",
            stderr_path=exp_dir / "server.stderr.log",
            cwd=WORKDIR,
        )

        bench_proc = popen_cmd(
            cmd=bench_cmd,
            env=final_env,
            stdout_path=exp_dir / "bench.stdout.log",
            stderr_path=exp_dir / "bench.stderr.log",
            cwd=WORKDIR,
        )

        bench_ret = bench_proc.wait()
        status["bench_returncode"] = bench_ret
        status["ok"] = (bench_ret == 0)
        return status

    except KeyboardInterrupt:
        print("[WARN] interrupted by user")
        status["bench_returncode"] = 130
        return status

    finally:
        kill_process_group(bench_proc, "bench", grace_sec=10)
        kill_process_group(server_proc, "server", grace_sec=SERVER_STOP_GRACE)
        print(f"[INFO] cooldown {COOLDOWN_SECONDS}s ...")
        time.sleep(COOLDOWN_SECONDS)


def main():
    ensure_dir(LOG_ROOT)

    experiments = load_experiments(EXPERIMENTS_JSON)
    if not experiments:
        print("[ERROR] no experiments found in json")
        sys.exit(1)

    summary = []
    for idx, exp in enumerate(experiments, start=1):
        result = run_one_experiment(idx, exp)
        summary.append(result)

    with open(LOG_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "#" * 100)
    print("[SUMMARY]")
    all_ok = True
    for item in summary:
        print(json.dumps(item, ensure_ascii=False))
        if not item["ok"]:
            all_ok = False
    print("#" * 100)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
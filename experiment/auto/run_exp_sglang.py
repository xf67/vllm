#!/usr/bin/env python3
import os
import sys
import yaml
import json
import time
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

SERVER_SCRIPT = "/home/xxf/NewVLLM/vllm/experiment/start_server_sglang.sh"
BENCH_SCRIPT = "/home/xxf/NewVLLM/vllm/experiment/run_random_bench_sglang.sh"
EXPERIMENTS_YAML = "/home/xxf/NewVLLM/vllm/experiment/auto/experiment_sglang.yaml"
WORKDIR = "/home/xxf/NewVLLM/vllm"

LOG_ROOT = Path("/home/xxf/NewVLLM/vllm/test/auto_logs_sglang")
COOLDOWN_SECONDS = 5
SERVER_STOP_GRACE = 20

DEFAULT_SERVER_ARGS: List[str] = []
DEFAULT_BENCH_ARGS: List[str] = ["sglang", "2048"]

DEFAULT_ENV: Dict[str, str] = {
    # -------------------- conda env --------------------
    "SERVER_CONDA_ENV": "sglang",
    "BENCH_CONDA_ENV": "newest",

    # -------------------- shared --------------------
    "HOST": "127.0.0.1",
    "PORT": "30000",
    "MODEL": "/home/xxf/NewVLLM/models/olmoe-p",
    "RESULT_ROOT": "/home/xxf/NewVLLM/vllm/test/bench_results",
    "NUM_PROMPTS": "1024",
    "DATASET_NAME": "random2",
    "TRACE_CSV": "/home/xxf/NewVLLM/vllm/experiment/AzureLLMInferenceTrace_filtered2.csv",
    "OUTPUT_LEN": "64",
    "REQUEST_RATES": "5",
    "SEED": "42",

    # -------------------- sglang server --------------------
    "SERVED_MODEL_NAME": "sglang-baseline",
    "TP_SIZE": "1",
    "DP_SIZE": "1",
    "MEM_FRACTION_STATIC": "0.85",
    "CONTEXT_LENGTH": "4096",
    "CHUNKED_PREFILL_SIZE": "2048",
    "MAX_RUNNING_REQUESTS": "",
    "SCHEDULE_POLICY": "fcfs",
    "DISABLE_RADIX_CACHE": "1",
    "DISABLE_CUDA_GRAPH": "0",

    # compile env for sglang jit
    # "CC_BIN": "/usr/bin/gcc-12",
    # "CXX_BIN": "/usr/bin/g++-12",

    # -------------------- bench --------------------
    "BACKEND_KIND": "openai",
    "ENDPOINT": "/completions",
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
    env.update(DEFAULT_ENV)

    if exp_env:
        env.update({k: str(v) for k, v in exp_env.items()})

    if not env.get("API_BASE"):
        host = env.get("HOST", "127.0.0.1")
        port = env.get("PORT", "30000")
        env["API_BASE"] = f"http://{host}:{port}/v1"

    return env


def build_final_server_args(exp: Dict[str, Any]) -> List[str]:
    return [str(x) for x in exp.get("server_args", DEFAULT_SERVER_ARGS)]


def build_final_bench_args(exp: Dict[str, Any]) -> List[str]:
    return [str(x) for x in exp.get("bench_args", DEFAULT_BENCH_ARGS)]


def find_conda_executable() -> str:
    conda = shutil.which("conda")
    if conda:
        return conda

    common_paths = [
        "/home/xxf/anaconda3/bin/conda",
        "/home/xxf/miniconda3/bin/conda",
        "/opt/conda/bin/conda",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Cannot find `conda`. Please add it to PATH or hardcode its path.")


CONDA_EXE = find_conda_executable()


def wrap_with_conda_run(cmd: List[str], conda_env: str) -> List[str]:
    return [CONDA_EXE, "run", "--no-capture-output", "-n", conda_env] + cmd


def popen_cmd(
    cmd: List[str],
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    cwd: Optional[str] = None,
    conda_env: Optional[str] = None,
) -> subprocess.Popen:
    real_cmd = cmd
    if conda_env:
        real_cmd = wrap_with_conda_run(cmd, conda_env)

    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        real_cmd,
        cwd=cwd,
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        preexec_fn=os.setsid,
        text=True,
    )
    proc._stdout_f = stdout_f
    proc._stderr_f = stderr_f
    proc._real_cmd = real_cmd
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

    server_conda_env = final_env["SERVER_CONDA_ENV"]
    bench_conda_env = final_env["BENCH_CONDA_ENV"]

    if "RESULT_ROOT" in final_env and final_env["RESULT_ROOT"]:
        ensure_dir(Path(final_env["RESULT_ROOT"]))

    config_to_save = {
        "name": name,
        "exp_idx": exp_idx,
        "server_cmd": server_cmd,
        "bench_cmd": bench_cmd,
        "server_conda_env": server_conda_env,
        "bench_conda_env": bench_conda_env,
        "default_env": DEFAULT_ENV,
        "override_env": exp.get("env", {}),
        "final_env": final_env,
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"[INFO] experiment: {exp_idx:03d}_{name}")
    print(f"[INFO] log dir: {exp_dir}")
    print(f"[INFO] server env: {server_conda_env}")
    print(f"[INFO] bench  env: {bench_conda_env}")
    print(f"[INFO] server cmd: {' '.join(wrap_with_conda_run(server_cmd, server_conda_env))}")
    print(f"[INFO] bench  cmd: {' '.join(wrap_with_conda_run(bench_cmd, bench_conda_env))}")
    print(f"[INFO] api_base: {final_env['API_BASE']}")
    print(f"[INFO] endpoint: {final_env.get('ENDPOINT', '')}")
    print("=" * 100)

    server_proc = None
    bench_proc = None
    status = {
        "name": name,
        "exp_idx": exp_idx,
        "log_dir": str(exp_dir),
        "server_conda_env": server_conda_env,
        "bench_conda_env": bench_conda_env,
        "server_started": False,
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
            conda_env=server_conda_env,
        )
        status["server_started"] = True

        bench_proc = popen_cmd(
            cmd=bench_cmd,
            env=final_env,
            stdout_path=exp_dir / "bench.stdout.log",
            stderr_path=exp_dir / "bench.stderr.log",
            cwd=WORKDIR,
            conda_env=bench_conda_env,
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

    experiments = load_experiments(EXPERIMENTS_YAML)
    if not experiments:
        print("[ERROR] no experiments found in yaml")
        sys.exit(1)

    total = len(experiments)
    print(f"[INFO] total experiments: {total}")

    summary = []
    for idx, exp in enumerate(experiments, start=1):
        print(f"[PROGRESS] {idx}/{total} experiments")
        result = run_one_experiment(idx, exp)
        summary.append(result)

    with open(LOG_ROOT / "summary.json", "w", encoding="utf-8") as f:
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
#!/usr/bin/env python3
"""Run compatible benchmarks against one long-lived vLLM server.

Experiments are grouped by the environment that affects server construction.
Only benchmark-side settings (request rate, QoS distribution, dataset, etc.)
may vary inside a group.  Each group starts the server once, waits for it to
become healthy, runs all of its benchmarks, and then stops the server.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SERVER_SCRIPT = "/home/xxf/NewVLLM/vllm/experiment/start_server.sh"
BENCH_SCRIPT = "/home/xxf/NewVLLM/vllm/experiment/run_random_bench.sh"
EXPERIMENTS_YAML = "/home/xxf/NewVLLM/vllm/experiment/auto/experiment_bu.yaml"
WORKDIR = "/home/xxf/NewVLLM/vllm"

DEFAULT_LOG_ROOT = Path("/home/xxf/NewVLLM/vllm/test/auto_logs_fast")
COOLDOWN_SECONDS = 5
BETWEEN_BENCH_SECONDS = 1
SERVER_STOP_GRACE = 20
SERVER_READY_TIMEOUT_SECONDS = 3600
SERVER_READY_POLL_SECONDS = 2

# The shell scripts accept positional arguments, although start_server.sh
# currently gets its scheduling mode from SCHED_MODE.
DEFAULT_SERVER_ARGS = ["fifo"]
DEFAULT_BENCH_ARGS = ["fifo", "2048"]

# Keep these defaults aligned with run_exp.py and start_server.sh.  Explicitly
# listing all server defaults makes grouping reflect the server that will
# actually be constructed, including values omitted from an experiment.
DEFAULT_ENV: dict[str, str] = {
    "PORT": "8000",
    "NUM_PROMPTS": "1024",
    "DATASET_NAME": "random2",
    "TRACE_CSV": (
        "/home/xxf/NewVLLM/vllm/experiment/AzureLLMInferenceTrace_filtered2.csv"
    ),
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
    "TP_SIZE": "1",
    "PP_SIZE": "1",
    "EP": "0",
    "SCHED_MODE": "ttft_agnostic",
    "QOS_AWARE": "1",
    "QOS_K_LIST": "r1,24",
    "CUDAGRAPH_MODE": "PIECEWISE",
    "SHARE_ATTN_ACROSS_TOPK": "true",
    "TTFT_SAFETY_FACTOR": "1.5",
    "EDF_LOOKAHEAD_STEPS": "5",
    "EDF_K_GATE_URGENCY": "0.3",
    "FIFO_SAFE_SWAP_WINDOW": "8",
    "FIFO_SWAP_KUP_RATIO": "0.9",
    "TTFT_AGNOSTIC_MIN_BATCH_RATIO": "0.5",
    "VLLM_DP_K_AWARE_DISPATCH": "0",
    "VLLM_DP_ENGINE_LANES": "0,1",
    "VLLM_DP_FIXED_K_BOUNDARY_DISPATCH": "0",
    "VLLM_DP_K_BOUNDARIES": "",
    "VLLM_DP_K_BOUNDARY_WIDTH": "20",
    "VLLM_DP_K_THRESHOLD": "12",
    "VLLM_DP_K_HYSTERESIS": "32",
    "VLLM_DP_K_COOLDOWN": "16",
    "MAYBE_OVERRIDE": "0",
    "REQUEST_RATES": "5",
}

# These variables are consumed only while constructing/sending benchmark
# requests.  Differences here do not require a server restart.  Any new YAML
# variable is conservatively treated as server-affecting unless added here.
BENCH_ONLY_ENV_KEYS = {
    "DATASET_NAME",
    "DIV_K",
    "KQOS_DIST",
    "KQOS_LOCAL_DIST",
    "NUM_PROMPTS",
    "OUTPUT_LEN",
    "QOS_FILE",
    "QOS_K_MAX",
    "QOS_K_MEAN",
    "QOS_K_STD",
    "REQUEST_RATES",
    "RESULT_ROOT",
    "STATIC_QOS",
    "TRACE_CSV",
    "TTFT_JITTER_HIGH",
    "TTFT_JITTER_LOW",
    "TTFT_MULTIPLIER",
    "TTFT_QUEUE_MS",
}

# The server opens DISPATCH_LOG once.  It is replaced with a session-level
# file and split back into each experiment's requested path after every bench.
VIRTUALIZED_SERVER_ENV_KEYS = {"DISPATCH_LOG"}


@dataclass
class PreparedExperiment:
    index: int
    spec: dict[str, Any]
    name: str
    log_dir: Path
    managed_env: dict[str, str]
    process_env: dict[str, str]
    server_args: list[str]
    bench_args: list[str]
    signature: tuple


@dataclass
class ExperimentGroup:
    group_id: str
    signature: tuple
    experiments: list[PreparedExperiment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run experiments while reusing a compatible vLLM server across "
            "multiple benchmarks."
        )
    )
    parser.add_argument(
        "--experiments",
        default=EXPERIMENTS_YAML,
        help=f"Experiment YAML path (default: {EXPERIMENTS_YAML})",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Log directory (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print server reuse groups without starting any process.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_experiments(yaml_path: str) -> list[dict[str, Any]]:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError("experiments YAML is empty")

    if isinstance(data, list):
        experiments = data
    elif isinstance(data, dict) and "experiments" in data:
        experiments = data["experiments"]
    else:
        raise ValueError(
            "experiments YAML must be a list or a dict with key 'experiments'"
        )

    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")

    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f"experiment #{i} is not a dict")
        if "name" not in exp:
            raise ValueError(f"experiment #{i} missing required field: name")

    return experiments


def stringify_env(values: dict[str, Any] | None) -> dict[str, str]:
    if not values:
        return {}
    return {key: str(value) for key, value in values.items()}


def build_managed_env(exp_env: dict[str, Any] | None) -> dict[str, str]:
    env = dict(DEFAULT_ENV)
    env.update(stringify_env(exp_env))
    return env


def build_process_env(managed_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(managed_env)
    return env


def build_final_server_args(exp: dict[str, Any]) -> list[str]:
    return [str(arg) for arg in exp.get("server_args", DEFAULT_SERVER_ARGS)]


def build_final_bench_args(exp: dict[str, Any]) -> list[str]:
    return [str(arg) for arg in exp.get("bench_args", DEFAULT_BENCH_ARGS)]


def server_env_keys(experiments: list[dict[str, Any]]) -> list[str]:
    configured_keys = set(DEFAULT_ENV)
    for exp in experiments:
        configured_keys.update((exp.get("env") or {}).keys())
    return sorted(configured_keys - BENCH_ONLY_ENV_KEYS - VIRTUALIZED_SERVER_ENV_KEYS)


def make_server_signature(
    managed_env: dict[str, str],
    server_args: list[str],
    signature_env_keys: list[str],
) -> tuple:
    env_signature = tuple((key, managed_env.get(key, "")) for key in signature_env_keys)
    return tuple(server_args), env_signature


def signature_id(signature: tuple) -> str:
    encoded = json.dumps(signature, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def prepare_experiments(
    experiments: list[dict[str, Any]],
    log_root: Path,
) -> tuple[list[PreparedExperiment], list[str]]:
    signature_env_keys = server_env_keys(experiments)
    prepared = []
    for index, exp in enumerate(experiments, start=1):
        managed_env = build_managed_env(exp.get("env"))
        server_args = build_final_server_args(exp)
        signature = make_server_signature(
            managed_env,
            server_args,
            signature_env_keys,
        )
        name = str(exp["name"])
        prepared.append(
            PreparedExperiment(
                index=index,
                spec=exp,
                name=name,
                log_dir=log_root / f"{index:03d}_{name}",
                managed_env=managed_env,
                process_env=build_process_env(managed_env),
                server_args=server_args,
                bench_args=build_final_bench_args(exp),
                signature=signature,
            )
        )
    return prepared, signature_env_keys


def group_experiments(
    prepared: list[PreparedExperiment],
) -> list[ExperimentGroup]:
    # Dict insertion order preserves the first appearance of each server
    # configuration while coalescing compatible experiments later in the YAML.
    grouped: dict[tuple, list[PreparedExperiment]] = {}
    for exp in prepared:
        grouped.setdefault(exp.signature, []).append(exp)

    groups = []
    for signature, members in grouped.items():
        groups.append(
            ExperimentGroup(
                group_id=signature_id(signature),
                signature=signature,
                experiments=members,
            )
        )
    return groups


def popen_cmd(
    cmd: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    cwd: str | None = None,
) -> subprocess.Popen:
    # These handles must stay open for the lifetime of the child process.
    stdout_f = open(stdout_path, "w", encoding="utf-8")  # noqa: SIM115
    stderr_f = open(stderr_path, "w", encoding="utf-8")  # noqa: SIM115

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            preexec_fn=os.setsid,
            text=True,
        )
    except Exception:
        stdout_f.close()
        stderr_f.close()
        raise

    proc._stdout_f = stdout_f
    proc._stderr_f = stderr_f
    return proc


def close_proc_files(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    for attr in ("_stdout_f", "_stderr_f"):
        f = getattr(proc, attr, None)
        if f is not None:
            with suppress(Exception):
                f.close()


def kill_process_group(
    proc: subprocess.Popen | None,
    name: str,
    grace_sec: int = 15,
) -> None:
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
    with suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)

    with suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)
    close_proc_files(proc)


def tail_file(path: Path, max_bytes: int = 8000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def server_is_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health",
            timeout=2,
        ) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def wait_for_server(
    proc: subprocess.Popen,
    port: int,
    stderr_path: Path,
) -> None:
    started_at = time.monotonic()
    next_progress = started_at
    while True:
        returncode = proc.poll()
        if returncode is not None:
            details = tail_file(stderr_path)
            raise RuntimeError(
                f"server exited before becoming healthy, returncode={returncode}"
                + (f"\n{details}" if details else "")
            )

        if server_is_healthy(port):
            elapsed = time.monotonic() - started_at
            print(f"[INFO] server healthy after {elapsed:.1f}s")
            return

        now = time.monotonic()
        if now - started_at >= SERVER_READY_TIMEOUT_SECONDS:
            raise TimeoutError(
                "server did not become healthy within "
                f"{SERVER_READY_TIMEOUT_SECONDS}s; see {stderr_path}"
            )
        if now >= next_progress:
            elapsed = now - started_at
            print(f"[INFO] waiting for server on port {port} ({elapsed:.0f}s)")
            next_progress = now + 30
        time.sleep(SERVER_READY_POLL_SECONDS)


def ranked_dispatch_paths(base_path: Path, dp_size: int) -> list[Path]:
    if dp_size <= 1:
        return [base_path]
    stem, _ = os.path.splitext(str(base_path))
    return [Path(f"{stem}_dp{rank}.csv") for rank in range(dp_size)]


class DispatchLogSlicer:
    """Split a long-lived server's dispatch log into per-benchmark files."""

    def __init__(self, session_base_path: Path, dp_size: int):
        self.session_base_path = session_base_path
        self.dp_size = dp_size
        self.source_paths = ranked_dispatch_paths(session_base_path, dp_size)
        self.offsets = {path: 0 for path in self.source_paths}
        self.headers: dict[Path, bytes] = {}

    def prepare(self) -> None:
        ensure_dir(self.session_base_path.parent)
        for path in self.source_paths:
            if path.exists():
                path.unlink()

    def _read_new_rows(self, source_path: Path) -> bytes:
        if not source_path.exists():
            print(f"[WARN] dispatch source not found: {source_path}")
            return b""

        with open(source_path, "rb") as f:
            if self.offsets[source_path] == 0:
                self.headers[source_path] = f.readline()
                self.offsets[source_path] = f.tell()
            f.seek(self.offsets[source_path])
            rows = f.read()
            self.offsets[source_path] = f.tell()
            return rows

    def snapshot(self, destination_base: str | None) -> list[str]:
        written = []
        destination_paths = (
            ranked_dispatch_paths(Path(destination_base), self.dp_size)
            if destination_base
            else [None] * len(self.source_paths)
        )

        for source_path, destination_path in zip(
            self.source_paths,
            destination_paths,
        ):
            rows = self._read_new_rows(source_path)
            if destination_path is None:
                continue

            ensure_dir(destination_path.parent)
            header = self.headers.get(source_path, b"")
            with open(destination_path, "wb") as f:
                f.write(header)
                f.write(rows)
            written.append(str(destination_path))
        return written


def save_experiment_config(
    exp: PreparedExperiment,
    group: ExperimentGroup,
    session_dir: Path,
    signature_env_keys: list[str],
) -> None:
    ensure_dir(exp.log_dir)
    server_cmd = ["bash", SERVER_SCRIPT] + exp.server_args
    bench_cmd = ["bash", BENCH_SCRIPT] + exp.bench_args
    config_to_save = {
        "name": exp.name,
        "exp_idx": exp.index,
        "server_group": group.group_id,
        "server_reused": len(group.experiments) > 1,
        "server_session_dir": str(session_dir),
        "server_cmd": server_cmd,
        "bench_cmd": bench_cmd,
        "default_env": DEFAULT_ENV,
        "override_env": exp.spec.get("env", {}),
        "managed_env": exp.managed_env,
        "server_signature_env": {
            key: exp.managed_env.get(key, "") for key in signature_env_keys
        },
    }
    with open(exp.log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)


def failed_status(
    exp: PreparedExperiment,
    group_id: str,
    error: str,
) -> dict[str, Any]:
    return {
        "name": exp.name,
        "exp_idx": exp.index,
        "server_group": group_id,
        "log_dir": str(exp.log_dir),
        "bench_returncode": None,
        "ok": False,
        "error": error,
    }


def run_benchmark(
    exp: PreparedExperiment,
    group: ExperimentGroup,
    session_dir: Path,
    signature_env_keys: list[str],
) -> dict[str, Any]:
    save_experiment_config(
        exp,
        group,
        session_dir,
        signature_env_keys,
    )

    result_root = exp.managed_env.get("RESULT_ROOT", "")
    if result_root:
        ensure_dir(Path(result_root))

    bench_cmd = ["bash", BENCH_SCRIPT] + exp.bench_args
    print("-" * 100)
    print(f"[INFO] experiment: {exp.index:03d}_{exp.name}")
    print(f"[INFO] bench cmd: {' '.join(bench_cmd)}")
    print(f"[INFO] bench log dir: {exp.log_dir}")

    bench_proc = None
    try:
        bench_proc = popen_cmd(
            cmd=bench_cmd,
            env=exp.process_env,
            stdout_path=exp.log_dir / "bench.stdout.log",
            stderr_path=exp.log_dir / "bench.stderr.log",
            cwd=WORKDIR,
        )
        bench_returncode = bench_proc.wait()
        return {
            "name": exp.name,
            "exp_idx": exp.index,
            "server_group": group.group_id,
            "log_dir": str(exp.log_dir),
            "bench_returncode": bench_returncode,
            "ok": bench_returncode == 0,
        }
    finally:
        kill_process_group(bench_proc, "bench", grace_sec=10)


def run_group(
    group_index: int,
    group: ExperimentGroup,
    total_groups: int,
    log_root: Path,
    signature_env_keys: list[str],
) -> list[dict[str, Any]]:
    first = group.experiments[0]
    model_name = Path(first.managed_env["MODEL"]).name or "model"
    session_dir = (
        log_root
        / "server_sessions"
        / f"{group_index:03d}_{model_name}_{group.group_id}"
    )
    ensure_dir(session_dir)

    server_env = first.process_env.copy()
    dp_size = int(first.managed_env.get("DP_SIZE", "1"))
    wants_dispatch_log = any(
        exp.managed_env.get("DISPATCH_LOG", "") for exp in group.experiments
    )
    dispatch_slicer = None
    if wants_dispatch_log:
        dispatch_slicer = DispatchLogSlicer(
            session_base_path=session_dir / "dispatch.csv",
            dp_size=dp_size,
        )
        dispatch_slicer.prepare()
        server_env["DISPATCH_LOG"] = str(dispatch_slicer.session_base_path)
    else:
        server_env["DISPATCH_LOG"] = ""

    server_cmd = ["bash", SERVER_SCRIPT] + first.server_args
    server_stdout = session_dir / "server.stdout.log"
    server_stderr = session_dir / "server.stderr.log"
    session_config = {
        "group_index": group_index,
        "group_id": group.group_id,
        "model": first.managed_env["MODEL"],
        "experiments": [
            {"index": exp.index, "name": exp.name} for exp in group.experiments
        ],
        "server_cmd": server_cmd,
        "server_signature_env": {
            key: first.managed_env.get(key, "") for key in signature_env_keys
        },
        "dispatch_session_base": (
            str(dispatch_slicer.session_base_path) if dispatch_slicer else None
        ),
    }
    with open(session_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(session_config, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(
        f"[SERVER GROUP] {group_index}/{total_groups} "
        f"id={group.group_id} model={first.managed_env['MODEL']}"
    )
    print(f"[INFO] compatible experiments: {len(group.experiments)}")
    print(f"[INFO] server cmd: {' '.join(server_cmd)}")
    print(f"[INFO] server log dir: {session_dir}")
    print("=" * 100)

    statuses = []
    server_proc = None
    active_exp_index = 0
    try:
        port = int(first.managed_env["PORT"])
        if port_is_open(port):
            raise RuntimeError(
                f"port {port} is already in use before starting server group"
            )

        server_proc = popen_cmd(
            cmd=server_cmd,
            env=server_env,
            stdout_path=server_stdout,
            stderr_path=server_stderr,
            cwd=WORKDIR,
        )
        wait_for_server(server_proc, port, server_stderr)

        for active_exp_index, exp in enumerate(group.experiments):
            if server_proc.poll() is not None:
                raise RuntimeError(
                    "shared server exited before benchmark "
                    f"{exp.index:03d}_{exp.name}, "
                    f"returncode={server_proc.returncode}"
                )

            status = run_benchmark(
                exp,
                group,
                session_dir,
                signature_env_keys,
            )
            if dispatch_slicer is not None:
                time.sleep(0.2)
                status["dispatch_logs"] = dispatch_slicer.snapshot(
                    exp.managed_env.get("DISPATCH_LOG") or None
                )
            statuses.append(status)

            if active_exp_index + 1 < len(group.experiments):
                print(
                    f"[INFO] keeping server loaded; next compatible benchmark "
                    f"in {BETWEEN_BENCH_SECONDS}s"
                )
                time.sleep(BETWEEN_BENCH_SECONDS)

    except KeyboardInterrupt:
        print("[WARN] interrupted by user")
        raise
    except Exception as exc:
        error = str(exc)
        print(f"[ERROR] server group {group.group_id}: {error}")
        completed_indices = {item["exp_idx"] for item in statuses}
        for exp in group.experiments:
            if exp.index not in completed_indices:
                ensure_dir(exp.log_dir)
                statuses.append(failed_status(exp, group.group_id, error))
    finally:
        kill_process_group(
            server_proc,
            f"server group {group.group_id}",
            grace_sec=SERVER_STOP_GRACE,
        )
        print(f"[INFO] cooldown {COOLDOWN_SECONDS}s before next server group")
        time.sleep(COOLDOWN_SECONDS)

    return statuses


def print_plan(groups: list[ExperimentGroup]) -> None:
    total_experiments = sum(len(group.experiments) for group in groups)
    print(f"[PLAN] experiments: {total_experiments}")
    print(f"[PLAN] server starts/model loads: {len(groups)}")
    print(f"[PLAN] avoided reloads: {max(0, total_experiments - len(groups))}")
    for index, group in enumerate(groups, start=1):
        first = group.experiments[0]
        print(
            f"\n[{index:03d}] group={group.group_id} "
            f"model={first.managed_env['MODEL']} "
            f"experiments={len(group.experiments)}"
        )
        for exp in group.experiments:
            print(f"      {exp.index:03d} {exp.name}")


def write_summary(log_root: Path, summary: list[dict[str, Any]]) -> bool:
    summary.sort(key=lambda item: item["exp_idx"])
    with open(log_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "#" * 100)
    print("[SUMMARY]")
    all_ok = True
    for item in summary:
        print(json.dumps(item, ensure_ascii=False))
        if not item["ok"]:
            all_ok = False
    print("#" * 100)
    return all_ok


def main() -> None:
    args = parse_args()
    experiments = load_experiments(args.experiments)
    if not experiments:
        print("[ERROR] no experiments found")
        sys.exit(1)

    prepared, signature_env_keys = prepare_experiments(
        experiments,
        args.log_root,
    )
    groups = group_experiments(prepared)
    print_plan(groups)
    if args.plan_only:
        return

    ensure_dir(args.log_root)
    summary = []
    try:
        for group_index, group in enumerate(groups, start=1):
            summary.extend(
                run_group(
                    group_index,
                    group,
                    len(groups),
                    args.log_root,
                    signature_env_keys,
                )
            )
    except KeyboardInterrupt:
        print("[WARN] experiment runner interrupted")
        if summary:
            write_summary(args.log_root, summary)
        sys.exit(130)

    all_ok = write_summary(args.log_root, summary)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

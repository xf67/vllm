#!/usr/bin/env python3
"""Generate paper-style matplotlib plots for offline benchmark results."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path("/home/xxf/NewVLLM/vllm/test-vllm-ok/bench_results")
DEFAULT_OUTPUT_DIR = Path("/home/xxf/NewVLLM/vllm/test-vllm-ok/paper_figs/offline")

EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<workload>.+)_offline-offline_(?P<strategy>[^-]+)-(?P<dist>m\d+s\d+)-rinf$"
)
DIST_RE = re.compile(r"^m(?P<mean>\d+)s(?P<std>\d+)$")

STRATEGY_LABELS = {
    "ours": "Ours",
    "vllmD": "vLLM",
    "vllmK": "vLLM + QoS",
}
STRATEGY_ORDER = ["ours", "vllmK", "vllmD"]
COLOR_MAP = {
    "ours": "#d62728",
    "vllmK": "#1f77b4",
    "vllmD": "#2ca02c",
}
BREAKDOWN_COLORS = {
    "prefill": "#9ecae1",
    "queue": "#6baed6",
    "overhead": "#3182bd",
}
MARKERS = ["o", "s", "^", "D", "P", "X"]


@dataclass(frozen=True)
class OfflineRun:
    workload: str
    strategy: str
    dist: str
    result_dir: Path
    metrics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot offline benchmark figures for paper usage."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Directory containing offline benchmark result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=None,
        help="Optional workload filter, e.g. dpsk_24 olmoe_32.",
    )
    return parser.parse_args()


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def dist_sort_key(dist: str) -> tuple[int, int]:
    match = DIST_RE.match(dist)
    if not match:
        return (10**9, 10**9)
    return (int(match.group("mean")), int(match.group("std")))


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    try:
        return (STRATEGY_ORDER.index(strategy), strategy)
    except ValueError:
        return (len(STRATEGY_ORDER), strategy)


def workload_sort_key(workload: str) -> tuple[int, str]:
    preferred = ["dpsk_24", "olmoe_32"]
    try:
        return (preferred.index(workload), workload)
    except ValueError:
        return (len(preferred), workload)


def dist_to_label(dist: str) -> str:
    match = DIST_RE.match(dist)
    if not match:
        return dist
    return f"mu={match.group('mean')}, sigma={match.group('std')}"


def metric_value(run: OfflineRun, metric_name: str) -> float:
    if metric_name not in run.metrics:
        available = ", ".join(sorted(run.metrics.keys()))
        raise KeyError(
            f"Metric '{metric_name}' not found in {run.result_dir}. "
            f"Available keys: {available}"
        )
    value = run.metrics[metric_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric '{metric_name}' in {run.result_dir} is not numeric: {type(value).__name__}"
        )
    return float(value)


def weighted_per_k_metric(run: OfflineRun, metric_name: str) -> float:
    per_k = run.metrics.get("per_k_qos")
    if not isinstance(per_k, dict) or not per_k:
        return 0.0

    total = 0
    weighted_sum = 0.0
    for stats in per_k.values():
        if not isinstance(stats, dict):
            continue
        count = int(stats.get("count", 0) or 0)
        value = float(stats.get(metric_name, 0) or 0)
        total += count
        weighted_sum += count * value

    if total <= 0:
        return 0.0
    return weighted_sum / total


def load_offline_runs(result_root: Path) -> list[OfflineRun]:
    runs: list[OfflineRun] = []
    for result_dir in sorted(result_root.iterdir()):
        if not result_dir.is_dir() or "_offline-" not in result_dir.name:
            continue

        match = EXPERIMENT_NAME_RE.match(result_dir.name)
        if not match:
            raise ValueError(f"Unexpected offline result directory name: {result_dir.name}")

        json_files = sorted(result_dir.glob("rate_*.json"))
        if not json_files:
            print(f"[WARN] Skip empty result directory: {result_dir}")
            continue
        if len(json_files) != 1:
            raise ValueError(
                f"Expected exactly one rate_*.json in {result_dir}, found {len(json_files)}"
            )

        with json_files[0].open("r", encoding="utf-8") as f:
            data = json.load(f)

        runs.append(
            OfflineRun(
                workload=match.group("workload"),
                strategy=match.group("strategy"),
                dist=match.group("dist"),
                result_dir=result_dir,
                metrics=data,
            )
        )

    if not runs:
        raise ValueError(f"No offline benchmark results found under {result_root}")
    return runs


def group_runs_by_workload(
    runs: list[OfflineRun],
) -> dict[str, dict[tuple[str, str], OfflineRun]]:
    grouped: dict[str, dict[tuple[str, str], OfflineRun]] = defaultdict(dict)
    for run in runs:
        grouped[run.workload][(run.strategy, run.dist)] = run
    return grouped


def plot_overall_metrics_per_workload(
    workload: str,
    series: dict[tuple[str, str], OfflineRun],
    output_dir: Path,
) -> None:
    dists = sorted({dist for _, dist in series}, key=dist_sort_key)
    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if any(key[0] == strategy for key in series)
    ]
    x = np.arange(len(dists))
    width = 0.8 / max(len(strategies), 1)

    metrics = [
        ("request_throughput", "Request Throughput (req/s)", 1.0),
        ("duration", "Total Completion Time (s)", 1.0),
        ("total_token_throughput", "Total Token Throughput (tok/s)", 1.0),
        ("mean_tpot_ms", "Mean TPOT (ms)", 1.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    for ax, (metric_name, ylabel, scale) in zip(axes.flat, metrics):
        for idx, strategy in enumerate(strategies):
            values = [
                metric_value(series[(strategy, dist)], metric_name) * scale
                for dist in dists
            ]
            ax.bar(
                x + idx * width - (len(strategies) - 1) * width / 2,
                values,
                width=width,
                color=COLOR_MAP.get(strategy, "#333333"),
                label=STRATEGY_LABELS.get(strategy, strategy),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(dists)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(workload, y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, f"offline_{workload}_overall_metrics")


def plot_queue_breakdown_per_workload(
    workload: str,
    series: dict[tuple[str, str], OfflineRun],
    output_dir: Path,
) -> None:
    dists = sorted({dist for _, dist in series}, key=dist_sort_key)
    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if any(key[0] == strategy for key in series)
    ]
    x = np.arange(len(dists))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for idx, strategy in enumerate(strategies):
        offsets = x + idx * width - (len(strategies) - 1) * width / 2
        prefill = []
        queue = []
        overhead = []
        for dist in dists:
            run = series[(strategy, dist)]
            prefill.append(weighted_per_k_metric(run, "prefill_mean_ms") / 1000.0)
            queue.append(weighted_per_k_metric(run, "queue_wait_mean_ms") / 1000.0)
            overhead.append(weighted_per_k_metric(run, "overhead_mean_ms") / 1000.0)

        ax.bar(
            offsets,
            prefill,
            width=width,
            color=BREAKDOWN_COLORS["prefill"],
            edgecolor="white",
            label="Prefill" if idx == 0 else None,
        )
        ax.bar(
            offsets,
            queue,
            bottom=prefill,
            width=width,
            color=BREAKDOWN_COLORS["queue"],
            edgecolor="white",
            label="Queue wait" if idx == 0 else None,
        )
        ax.bar(
            offsets,
            overhead,
            bottom=np.array(prefill) + np.array(queue),
            width=width,
            color=BREAKDOWN_COLORS["overhead"],
            edgecolor="white",
            label="Overhead" if idx == 0 else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dists)
    ax.set_ylabel("Weighted First-Service Delay Breakdown (s)")
    ax.set_title(workload)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    strategy_text = " | ".join(
        f"{strategy}: {STRATEGY_LABELS.get(strategy, strategy)}" for strategy in strategies
    )
    ax.text(
        0.0,
        -0.18,
        f"Within each distribution group, bars are ordered as {strategy_text}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.text(
        0.0,
        -0.27,
        "Under request_rate=inf, this highlights backlog-driven delay composition rather than online latency.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    fig.tight_layout()
    save_figure(fig, output_dir, f"offline_{workload}_first_service_breakdown")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_paper_style()

    runs = load_offline_runs(args.result_root)
    if args.workloads:
        allowed = set(args.workloads)
        runs = [run for run in runs if run.workload in allowed]
        if not runs:
            raise ValueError(f"No matching workloads found for filter: {sorted(allowed)}")

    grouped = group_runs_by_workload(runs)

    for workload in sorted(grouped, key=workload_sort_key):
        plot_overall_metrics_per_workload(workload, grouped[workload], args.output_dir)
        plot_queue_breakdown_per_workload(workload, grouped[workload], args.output_dir)

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot one merged offline throughput figure for merged-all results."""

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


RESULT_ROOT = Path("/home/xxf/NewVLLM/vllm/test-vllm-merged-all")
DEFAULT_OUTPUT_DIR = Path("/home/xxf/NewVLLM/vllm/test-vllm-merged-all/paper_figs/offline")

DEVICE_DIR_RE = re.compile(r"^bench_results-(?P<device>.+)$")
EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<workload>.+)_offline-offline_(?P<strategy>[^-]+)-(?P<dist>m\d+s\d+)-rinf$"
)
DIST_RE = re.compile(r"^m(?P<mean>\d+)s(?P<std>\d+)$")

STRATEGY_LABELS = {
    "ours": "Ours",
    "vllmK": "vLLM + QoS",
    "vllmD": "vLLM",
}
STRATEGY_ORDER = ["ours", "vllmK", "vllmD"]
COLOR_MAP = {
    "ours": "#d62728",
    "vllmK": "#1f77b4",
    "vllmD": "#2ca02c",
}
DEVICE_ORDER = ["a100", "a6000"]
WORKLOAD_ORDER = ["dpsk_24", "olmoe_32"]


@dataclass(frozen=True)
class OfflineRun:
    device: str
    workload: str
    strategy: str
    dist: str
    result_dir: Path
    metrics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one merged offline throughput figure for A100/A6000 results.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Directory containing bench_results-a100 and bench_results-a6000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the merged offline figure.",
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
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
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


def workload_sort_key(workload: str) -> tuple[int, str]:
    try:
        return (WORKLOAD_ORDER.index(workload), workload)
    except ValueError:
        return (len(WORKLOAD_ORDER), workload)


def device_sort_key(device: str) -> tuple[int, str]:
    try:
        return (DEVICE_ORDER.index(device), device)
    except ValueError:
        return (len(DEVICE_ORDER), device)


def strategies_for_runs(runs: dict[tuple[str, str], OfflineRun]) -> list[str]:
    known = [
        strategy
        for strategy in STRATEGY_ORDER
        if any(key[0] == strategy for key in runs)
    ]
    extras = sorted({key[0] for key in runs} - set(known))
    return known + extras


def metric_value(run: OfflineRun, metric_name: str) -> float:
    value = run.metrics.get(metric_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric '{metric_name}' in {run.result_dir} is not numeric: {type(value).__name__}"
        )
    return float(value)


def load_offline_runs(result_root: Path) -> list[OfflineRun]:
    runs: list[OfflineRun] = []

    for bench_dir in sorted(result_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        device_match = DEVICE_DIR_RE.match(bench_dir.name)
        if not device_match:
            continue
        device = device_match.group("device")

        for result_dir in sorted(bench_dir.iterdir()):
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
                    device=device,
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


def group_runs(
    runs: list[OfflineRun],
) -> dict[tuple[str, str], dict[tuple[str, str], OfflineRun]]:
    grouped: dict[tuple[str, str], dict[tuple[str, str], OfflineRun]] = defaultdict(dict)
    for run in runs:
        grouped[(run.device, run.workload)][(run.strategy, run.dist)] = run
    return grouped


def unique_legend_entries(
    axes: list[plt.Axes],
) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if not label or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def plot_request_throughput_merged(
    grouped: dict[tuple[str, str], dict[tuple[str, str], OfflineRun]],
    output_dir: Path,
) -> None:
    devices = sorted({device for device, _ in grouped}, key=device_sort_key)
    workloads = sorted({workload for _, workload in grouped}, key=workload_sort_key)

    fig, axes = plt.subplots(
        len(devices),
        len(workloads),
        figsize=(6.6, 3.9),
        sharey=True,
    )
    axes_array = np.atleast_2d(axes)
    axes_list = [ax for row in axes_array for ax in row]

    for row_idx, device in enumerate(devices):
        for col_idx, workload in enumerate(workloads):
            ax = axes_array[row_idx, col_idx]
            panel_runs = grouped.get((device, workload), {})
            if not panel_runs:
                ax.set_visible(False)
                continue

            dists = sorted({dist for _, dist in panel_runs}, key=dist_sort_key)
            strategies = strategies_for_runs(panel_runs)
            x = np.arange(len(dists))
            width = 0.8 / max(len(strategies), 1)

            for strategy_idx, strategy in enumerate(strategies):
                positions = []
                values = []
                for dist_idx, dist in enumerate(dists):
                    key = (strategy, dist)
                    if key not in panel_runs:
                        continue
                    positions.append(
                        x[dist_idx] + strategy_idx * width - (len(strategies) - 1) * width / 2
                    )
                    values.append(metric_value(panel_runs[key], "request_throughput"))

                if not values:
                    continue

                ax.bar(
                    positions,
                    values,
                    width=width,
                    color=COLOR_MAP.get(strategy, "#333333"),
                    label=STRATEGY_LABELS.get(strategy, strategy),
                )

            ax.set_xticks(x)
            ax.set_xticklabels(dists)
            ax.set_title(f"{workload} ({device.upper()})", pad=4)
            if row_idx == len(devices) - 1:
                ax.set_xlabel("Load (m/s)")
            if col_idx == 0:
                ax.set_ylabel("Request Throughput (req/s)")
            else:
                ax.tick_params(axis="y", labelleft=False)

    handles, labels = unique_legend_entries(axes_list)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(len(labels), 1),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        bottom=0.12,
        top=0.83,
        wspace=0.16,
        hspace=0.32,
    )
    save_figure(fig, output_dir, "offline_request_throughput_merged_all")


def main() -> None:
    args = parse_args()
    apply_paper_style()

    runs = load_offline_runs(args.result_root)
    if args.workloads:
        allowed = set(args.workloads)
        runs = [run for run in runs if run.workload in allowed]
        if not runs:
            raise ValueError(f"No matching workloads found for filter: {sorted(allowed)}")

    grouped = group_runs(runs)
    plot_request_throughput_merged(grouped, args.output_dir)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()

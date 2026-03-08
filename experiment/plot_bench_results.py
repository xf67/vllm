"""Visualize and compare benchmark results from two (or more) experiment runs.

Usage:
    python plot_bench_results.py DIR1 [LABEL1] DIR2 [LABEL2] [-o OUTPUT_DIR]

Example:
    python plot_bench_results.py \
        test/bench_results/old/random_qos_ver_1 ver_1 \
        test/bench_results/old/random_qos_ver_2 ver_2 \
        -o test/bench_results/old/plots
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RATE_SORT_KEY = lambda r: float("inf") if r == "inf" else float(r)


_skip_inf = True


def load_run(result_dir: str) -> dict[str, dict]:
    """Load all rate_*.json files from a result directory, keyed by rate label."""
    pattern = re.compile(r"^rate_(.+?)(?:_in\d+)?\.json$")
    runs = {}
    for fname in sorted(os.listdir(result_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        rate_label = m.group(1).replace("_", ".")
        if _skip_inf and rate_label == "inf":
            continue
        with open(os.path.join(result_dir, fname)) as f:
            runs[rate_label] = json.load(f)
    return dict(sorted(runs.items(), key=lambda kv: RATE_SORT_KEY(kv[0])))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark comparison")
    parser.add_argument("dirs", nargs="+",
                        help="Alternating: DIR [LABEL] DIR [LABEL] ...")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Save plots to this directory instead of showing")
    parser.add_argument("--skip-inf", action="store_true", default=True,
                        help="Skip rate=inf groups (default: True)")
    parser.add_argument("--no-skip-inf", action="store_false", dest="skip_inf",
                        help="Include rate=inf groups")
    args = parser.parse_args()

    pairs = []
    it = iter(args.dirs)
    for token in it:
        d = token
        try:
            maybe_label = next(it)
        except StopIteration:
            maybe_label = None
        if maybe_label and os.path.isdir(maybe_label):
            pairs.append((d, os.path.basename(d)))
            pairs.append((maybe_label, os.path.basename(maybe_label)))
        elif maybe_label:
            pairs.append((d, maybe_label))
        else:
            pairs.append((d, os.path.basename(d)))
    return pairs, args.output_dir, args.skip_inf


def save_or_show(fig, output_dir, name):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path)
        print(f"  Saved {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_overall_metrics(all_runs, output_dir):
    """Bar chart comparing overall TTFT / TPOT / throughput across rates."""
    metrics = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("p99_tpot_ms", "P99 TPOT (ms)"),
        ("output_throughput", "Output Throughput (tok/s)"),
        ("total_token_throughput", "Total Token Throughput (tok/s)"),
    ]
    rates = sorted(
        {r for _, runs in all_runs for r in runs},
        key=RATE_SORT_KEY,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Overall Metrics Comparison", fontsize=14, fontweight="bold")

    n_groups = len(rates)
    n_bars = len(all_runs)
    width = 0.8 / n_bars
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_bars, 3)))

    for ax, (key, title) in zip(axes.flat, metrics):
        x = np.arange(n_groups)
        for i, (label, runs) in enumerate(all_runs):
            vals = [runs.get(r, {}).get(key, 0) for r in rates]
            ax.bar(x + i * width - (n_bars - 1) * width / 2, vals,
                   width, label=label, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels([f"rate={r}" for r in rates])
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_or_show(fig, output_dir, "overall_metrics")


def plot_per_k_comparison(all_runs, output_dir):
    """Per-k SLO compliance and TTFT for each request rate."""
    rates = sorted(
        {r for _, runs in all_runs for r in runs},
        key=RATE_SORT_KEY,
    )
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(all_runs), 3)))

    for rate in rates:
        all_ks = sorted(
            {int(k)
             for _, runs in all_runs
             if rate in runs
             for k in runs[rate].get("per_k_qos", {})},
        )
        if not all_ks:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Per-k Metrics  (rate={rate})", fontsize=14, fontweight="bold")

        n_groups = len(all_ks)
        n_bars = len(all_runs)
        width = 0.8 / n_bars
        x = np.arange(n_groups)

        sub_metrics = [
            ("slo_compliance", "SLO Compliance (%)"),
            ("ttft_mean_ms", "Mean TTFT (ms)"),
            ("prefill_mean_ms", "Mean Prefill (ms)"),
            ("queue_wait_mean_ms", "Mean Queue Wait (ms)"),
        ]
        for ax, (mk, mtitle) in zip(axes.flat, sub_metrics):
            for i, (label, runs) in enumerate(all_runs):
                pk = runs.get(rate, {}).get("per_k_qos", {})
                vals = [pk.get(str(k), {}).get(mk, 0) for k in all_ks]
                ax.bar(x + i * width - (n_bars - 1) * width / 2, vals,
                       width, label=label, color=colors[i])
            ax.set_xticks(x)
            ax.set_xticklabels([f"k={k}" for k in all_ks])
            ax.set_title(mtitle)
            ax.legend(fontsize=8)

        fig.tight_layout()
        save_or_show(fig, output_dir, f"per_k_rate_{rate}")


def plot_ttft_breakdown(all_runs, output_dir):
    """Stacked bar: prefill + queue_wait + overhead = TTFT, per k, per rate."""
    rates = sorted(
        {r for _, runs in all_runs for r in runs},
        key=RATE_SORT_KEY,
    )

    for label, runs in all_runs:
        fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 5),
                                 sharey=True)
        if len(rates) == 1:
            axes = [axes]
        fig.suptitle(f"TTFT Breakdown — {label}", fontsize=14, fontweight="bold")

        for ax, rate in zip(axes, rates):
            pk = runs.get(rate, {}).get("per_k_qos", {})
            ks = sorted(pk.keys(), key=int)
            prefill = [pk[k].get("prefill_mean_ms", 0) for k in ks]
            queue = [pk[k].get("queue_wait_mean_ms", 0) for k in ks]
            overhead = [pk[k].get("overhead_mean_ms", 0) for k in ks]

            x = np.arange(len(ks))
            ax.bar(x, prefill, label="prefill")
            ax.bar(x, queue, bottom=prefill, label="queue_wait")
            bottom2 = [p + q for p, q in zip(prefill, queue)]
            ax.bar(x, overhead, bottom=bottom2, label="overhead")

            ax.set_xticks(x)
            ax.set_xticklabels([f"k={k}" for k in ks])
            ax.set_title(f"rate={rate}")
            ax.set_ylabel("Time (ms)")
            ax.legend(fontsize=8)

        fig.tight_layout()
        save_or_show(fig, output_dir, f"ttft_breakdown_{label}")


def main():
    pairs, output_dir, skip_inf = parse_args()

    global _skip_inf
    _skip_inf = skip_inf

    all_runs = []
    for d, label in pairs:
        runs = load_run(d)
        all_runs.append((label, runs))
        print(f"Loaded {label}: {list(runs.keys())} rates from {d}")

    if not all_runs:
        print("No data loaded.")
        sys.exit(1)

    print()
    plot_overall_metrics(all_runs, output_dir)
    plot_per_k_comparison(all_runs, output_dir)
    plot_ttft_breakdown(all_runs, output_dir)

    if output_dir:
        print(f"\nAll plots saved to {output_dir}")
    else:
        print("\nDisplaying plots...")


if __name__ == "__main__":
    main()

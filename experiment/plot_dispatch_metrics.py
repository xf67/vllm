"""Plot dispatch metrics from scheduler CSV logs.

Usage:
    python plot_dispatch_metrics.py LOG1.csv [LABEL1] [LOG2.csv LABEL2 ...] [-o OUTPUT_DIR]

Examples:
    # Single log
    python plot_dispatch_metrics.py dispatch_fifo.csv

    # Compare two scheduling modes
    python plot_dispatch_metrics.py \
        dispatch_fifo.csv FIFO \
        dispatch_edf.csv  EDF  \
        -o dispatch_plots/

    # Only show steps 1000~5000
    python plot_dispatch_metrics.py log.csv -o plots --start 1000 --end 5000
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

_ZERO_AS_NAN_COLS = {
    "dispatch_k", "running_mean_k", "running_weighted_mean_k",
    "waiting_mean_k", "waiting_weighted_mean_k",
}


def load_csv(path: str) -> dict[str, np.ndarray]:
    """Load a dispatch metrics CSV into {column_name: array}.

    For K-related columns, 0 is treated as NaN (no valid request has k_qos=0)
    so matplotlib skips empty-queue steps automatically.
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        cols: dict[str, list] = {h: [] for h in reader.fieldnames}
        for row in reader:
            for h in reader.fieldnames:
                val = row[h].strip()
                v = float(val) if val else np.nan
                if h in _ZERO_AS_NAN_COLS and v == 0:
                    v = np.nan
                cols[h].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dispatch metrics from scheduler CSV logs")
    parser.add_argument(
        "inputs", nargs="+",
        help="Pairs of CSV_PATH [LABEL]. "
             "If LABEL is omitted the filename stem is used.")
    parser.add_argument(
        "-o", "--output", default="dispatch_plots",
        help="Directory to save PNGs (default: dispatch_plots)")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't call plt.show() (useful for headless servers)")
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start step (inclusive), default: first step")
    parser.add_argument(
        "--end", type=int, default=None,
        help="End step (inclusive), default: last step")
    return parser.parse_args()


def pair_inputs(inputs: list[str]) -> list[tuple[str, str]]:
    """Parse positional args into (path, label) pairs."""
    pairs = []
    i = 0
    while i < len(inputs):
        path = inputs[i]
        i += 1
        if i < len(inputs) and not inputs[i].endswith(".csv"):
            label = inputs[i]
            i += 1
        else:
            label = os.path.splitext(os.path.basename(path))[0]
        pairs.append((path, label))
    return pairs


def plot_series(
    ax: plt.Axes,
    datasets: list[tuple[str, dict[str, np.ndarray]]],
    x_key: str,
    y_key: str,
    ylabel: str,
    title: str,
):
    for label, data in datasets:
        ax.plot(data[x_key], data[y_key], label=label,
                linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(datasets) > 1:
        ax.legend(fontsize=8)


def main():
    args = parse_args()
    pairs = pair_inputs(args.inputs)
    datasets = []
    for path, label in pairs:
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        datasets.append((label, load_csv(path)))

    if not datasets:
        print("No valid CSV files found.", file=sys.stderr)
        sys.exit(1)

    if args.start is not None or args.end is not None:
        filtered = []
        for label, data in datasets:
            steps = data["step"]
            mask = np.ones(len(steps), dtype=bool)
            if args.start is not None:
                mask &= steps >= args.start
            if args.end is not None:
                mask &= steps <= args.end
            filtered.append((label, {k: v[mask] for k, v in data.items()}))
        datasets = filtered

    os.makedirs(args.output, exist_ok=True)

    # --- Figure 1: dispatch_k over steps ---
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_series(ax, datasets, "step", "dispatch_k",
                "Dispatch K", "Dispatch K Over Steps")
    fig.savefig(os.path.join(args.output, "dispatch_k.png"))

    # --- Figure 2: running vs waiting mean_k ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    plot_series(axes[0], datasets, "step", "running_mean_k",
                "Mean K", "Running Requests: Mean K")
    plot_series(axes[1], datasets, "step", "waiting_mean_k",
                "Mean K", "Waiting Requests: Mean K")
    fig.suptitle("Mean K of Running vs Waiting Queues", fontsize=13)
    fig.savefig(os.path.join(args.output, "mean_k_queues.png"))

    # --- Figure 3: weighted mean_k (by tokens) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    plot_series(axes[0], datasets, "step", "running_weighted_mean_k",
                "Weighted Mean K", "Running: Token-Weighted Mean K")
    plot_series(axes[1], datasets, "step", "waiting_weighted_mean_k",
                "Weighted Mean K", "Waiting: Token-Weighted Mean K")
    fig.suptitle("Token-Weighted Mean K of Running vs Waiting Queues",
                 fontsize=13)
    fig.savefig(os.path.join(args.output, "weighted_mean_k_queues.png"))

    # --- Figure 4: queue sizes ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, data in datasets:
        x = data["step"]
        ax.plot(x, data["num_running"],
                label=f"{label} running", linewidth=0.8, alpha=0.8)
        ax.plot(x, data["num_waiting"],
                label=f"{label} waiting", linewidth=0.8, alpha=0.8,
                linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Queue Sizes Over Steps")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.output, "queue_sizes.png"))

    # --- Figure 5: total scheduled tokens per step ---
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_series(ax, datasets, "step", "total_scheduled_tokens",
                "Tokens", "Total Scheduled Tokens Per Step")
    fig.savefig(os.path.join(args.output, "scheduled_tokens.png"))

    # --- Figure 6: new prefills per step ---
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_series(ax, datasets, "step", "num_new_prefills",
                "New Prefills", "New Prefills Per Step")
    fig.savefig(os.path.join(args.output, "new_prefills.png"))

    # --- Figure 7: combined dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plot_series(axes[0, 0], datasets, "step", "dispatch_k",
                "K", "Dispatch K")
    plot_series(axes[0, 1], datasets, "step", "running_weighted_mean_k",
                "K", "Running Weighted Mean K")
    plot_series(axes[0, 2], datasets, "step", "waiting_weighted_mean_k",
                "K", "Waiting Weighted Mean K")
    for label, data in datasets:
        axes[1, 0].plot(data["step"], data["num_running"],
                        label=f"{label}", linewidth=0.8)
    axes[1, 0].set_title("Num Running")
    axes[1, 0].set_xlabel("Step")
    if len(datasets) > 1:
        axes[1, 0].legend(fontsize=8)
    plot_series(axes[1, 1], datasets, "step", "total_scheduled_tokens",
                "Tokens", "Scheduled Tokens")
    plot_series(axes[1, 2], datasets, "step", "num_new_prefills",
                "Count", "New Prefills")
    fig.suptitle("Dispatch Metrics Dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "dashboard.png"))

    print(f"Saved plots to {args.output}/")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()

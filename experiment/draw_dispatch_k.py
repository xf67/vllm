import csv
import os
import sys
import matplotlib.pyplot as plt


def load_curve(csv_file, min_step=30):
    timestamps = []
    dispatch_k = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            dispatch_k.append(float(row["dispatch_k"]))

    if len(timestamps) <= min_step:
        return None, None

    timestamps = timestamps[min_step:]
    dispatch_k = dispatch_k[min_step:]

    start_time = timestamps[0]
    elapsed = [t - start_time for t in timestamps]
    return elapsed, dispatch_k


def parse_args(argv):
    base_csv = None
    base_label = None
    base_value = None
    min_step = 30

    if "--min_step" in argv:
        idx = argv.index("--min_step")
        if idx + 1 >= len(argv):
            raise ValueError("--min_step requires one argument")
        min_step = int(argv[idx + 1])
        del argv[idx:idx + 2]

    if "--base" in argv:
        idx = argv.index("--base")
        if idx + 2 >= len(argv):
            raise ValueError("--base requires two arguments: <csv> <label>")
        base_csv = argv[idx + 1]
        base_label = argv[idx + 2]
        del argv[idx:idx + 3]

    if "--base_value" in argv:
        idx = argv.index("--base_value")
        if idx + 1 >= len(argv):
            raise ValueError("--base_value requires one argument: <value>")
        base_value = float(argv[idx + 1])
        del argv[idx:idx + 2]

    if len(argv) == 0 or len(argv) % 2 != 0:
        raise ValueError(
            "Usage: python plot_dispatch_k.py "
            "a.csv a_label [b.csv b_label ...] "
            "[--base x.csv x_label --base_value 32] "
            "[--min_step 600]"
        )

    curves = []
    for i in range(0, len(argv), 2):
        curves.append((argv[i], argv[i + 1]))

    if (base_csv is None) != (base_label is None):
        raise ValueError("--base must be followed by both <csv> and <label>")

    if base_csv is not None and base_value is None:
        raise ValueError("When using --base, you must also provide --base_value")

    return curves, base_csv, base_label, base_value, min_step


def main():
    argv = sys.argv[1:]

    try:
        curves, base_csv, base_label, base_value, min_step = parse_args(argv)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
        "lines.linewidth": 1.8,
    })

    highlight_color = "tab:orange"
    normal_colors = ["tab:blue", "tab:green", "tab:gray", "tab:purple", "tab:brown"]
    baseline_color = "tab:green"

    fig, ax = plt.subplots(figsize=(4.3, 1.8))

    valid_curve_files = []
    max_elapsed = 0.0

    for idx, (csv_file, label) in enumerate(curves):
        elapsed, dispatch_k = load_curve(csv_file, min_step=min_step)
        if elapsed is None:
            print(f"Warning: no data after min_step={min_step} in {csv_file}")
            continue

        if idx == 0:
            color = highlight_color
        else:
            color = normal_colors[(idx - 1) % len(normal_colors)]

        ax.plot(elapsed, dispatch_k, label=label, color=color)
        valid_curve_files.append(csv_file)
        max_elapsed = max(max_elapsed, elapsed[-1])

    if base_csv is not None:
        elapsed, _ = load_curve(base_csv, min_step=min_step)
        if elapsed is None:
            print(f"Warning: no data after min_step={min_step} in baseline csv {base_csv}")
        else:
            baseline_y = [base_value] * len(elapsed)
            ax.plot(elapsed, baseline_y, label=base_label, color=baseline_color)
            max_elapsed = max(max_elapsed, elapsed[-1])

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Dispatched Routing\nBudget ($k$)")
    ax.grid(True, alpha=0.3)
    if max_elapsed > 0:
        ax.set_xlim(0, max_elapsed * 1.15)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.98),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            ncol=1,
            handlelength=1.0,
            handletextpad=0.35,
            labelspacing=0.25,
            borderpad=0.25,
            columnspacing=0.0,
            prop={"size": 7},
        )

    plt.tight_layout(pad=0.2)

    if valid_curve_files:
        output_dir = os.path.dirname(os.path.abspath(valid_curve_files[0]))
    elif base_csv is not None:
        output_dir = os.path.dirname(os.path.abspath(base_csv))
    else:
        output_dir = os.getcwd()

    output_file = os.path.join(output_dir, "dispatch_k_compare.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved to {output_file}")


if __name__ == "__main__":
    main()

# python3 '/home/xxf/NewVLLM/vllm/experiment/draw_dispatch_k.py' '/home/xxf/NewVLLM/vllm/test2/bench_results-inf/inf-12-5/log-ttft-awe.csv' MoE-PRISM '/home/xxf/NewVLLM/vllm/test2/bench_results-inf/inf-12-5/log-fifo-awe.csv' vLLM+K  --base '/home/xxf/NewVLLM/vllm/test2/bench_results-inf/inf-12-5/log-fifo-agn.csv' vLLM --base_value 24 --min_step 50


# python3 '/home/xxf/NewVLLM/vllm/experiment/draw_dispatch_k.py' '/home/xxf/NewVLLM/vllm/test2/bench_results-inf/inf-16-6/log-ttft-awe.csv' MoE-PRISM '/home/xxf/NewVLLM/vllm/test2/bench_results-inf/inf-16-6/log-fifo-awe.csv' vLLM+K  --base '/home/xxf/NewVLLM/vllm/test2/bench_results-inf/inf-16-6/log-fifo-agn.csv' vLLM --base_value 32 --min_step 50

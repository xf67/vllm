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
    min_step = 30

    if "--min_step" in argv:
        idx = argv.index("--min_step")
        if idx + 1 >= len(argv):
            raise ValueError("--min_step requires one argument")
        min_step = int(argv[idx + 1])
        del argv[idx:idx + 2]

    if "--base" not in argv:
        raise ValueError(
            "Usage: python plot_dp_group_compare.py "
            "dp0.csv dp0_label dp1.csv dp1_label "
            "--base base_dp0.csv base_dp0_label base_dp1.csv base_dp1_label "
            "[--min_step 600]"
        )

    idx = argv.index("--base")
    main_args = argv[:idx]
    base_args = argv[idx + 1:]

    if len(main_args) != 4 or len(base_args) != 4:
        raise ValueError(
            "Usage: python plot_dp_group_compare.py "
            "dp0.csv dp0_label dp1.csv dp1_label "
            "--base base_dp0.csv base_dp0_label base_dp1.csv base_dp1_label "
            "[--min_step 600]"
        )

    dp0_csv, dp0_label, dp1_csv, dp1_label = main_args
    base0_csv, base0_label, base1_csv, base1_label = base_args

    return (
        dp0_csv, dp0_label, dp1_csv, dp1_label,
        base0_csv, base0_label, base1_csv, base1_label,
        min_step,
    )


def main():
    argv = sys.argv[1:]

    try:
        (
            dp0_csv, dp0_label, dp1_csv, dp1_label,
            base0_csv, base0_label, base1_csv, base1_label,
            min_step,
        ) = parse_args(argv)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
        "lines.linewidth": 0.7,   # 线细一点
    })

    # Ours: 深浅橙
    ours_colors = ["#d95f02", "#fdb462"]
    # Base: 深浅蓝
    base_colors = ["#1f77b4", "#9ecae1"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(4.6, 3.2),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.15,
        },
    )

    # ---------- top: Ours ----------
    top_curves = [
        (dp0_csv, dp0_label, ours_colors[0]),
        (dp1_csv, dp1_label, ours_colors[1]),
    ]

    valid_files = []
    max_elapsed = 0.0

    for csv_file, label, color in top_curves:
        elapsed, dispatch_k = load_curve(csv_file, min_step=min_step)
        if elapsed is None:
            print(f"Warning: no data after min_step={min_step} in {csv_file}")
            continue
        ax_top.plot(elapsed, dispatch_k, label=label, color=color)
        valid_files.append(csv_file)
        max_elapsed = max(max_elapsed, elapsed[-1])

    # ax_top.set_ylabel("Dispatched Routing\nBudget ($k$)")
    ax_top.grid(True, alpha=0.3)

    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        ax_top.legend(
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

    # ---------- bottom: Base ----------
    bot_curves = [
        (base0_csv, base0_label, base_colors[0]),
        (base1_csv, base1_label, base_colors[1]),
    ]

    for csv_file, label, color in bot_curves:
        elapsed, dispatch_k = load_curve(csv_file, min_step=min_step)
        if elapsed is None:
            print(f"Warning: no data after min_step={min_step} in {csv_file}")
            continue
        ax_bot.plot(elapsed, dispatch_k, label=label, color=color)
        valid_files.append(csv_file)
        max_elapsed = max(max_elapsed, elapsed[-1])

    ax_bot.set_xlabel("Elapsed Time (s)")
    # ax_bot.set_ylabel("Dispatched Routing\nBudget ($k$)")
    ax_bot.grid(True, alpha=0.3)

    fig.supylabel("Dispatched Routing Budget ($k$)", x=0.01)
    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        ax_bot.legend(
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

    # 可选：统一 y 范围，便于对比
    ax_top.set_ylim(20, 37)
    ax_bot.set_ylim(20, 37)
    if max_elapsed > 0:
        ax_top.set_xlim(0, max_elapsed * 1.1)

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.98)

    if valid_files:
        output_dir = os.path.dirname(os.path.abspath(valid_files[0]))
    else:
        output_dir = os.getcwd()

    output_file = os.path.join(output_dir, "dp_dispatch_k_split_compare.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved to {output_file}")


if __name__ == "__main__":
    main()


# python /home/xxf/NewVLLM/vllm/experiment/draw_dispatch_k_dp.py /home/xxf/NewVLLM/vllm/test2/bench_results-16-8-32/k-awe/log_dp0-30.csv MoE-PRISM-DP0 /home/xxf/NewVLLM/vllm/test2/bench_results-16-8-32/k-awe/log_dp1-30.csv MoE-PRISM-DP1 --base /home/xxf/NewVLLM/vllm/test2/bench_results-16-8-32/k-agn/log_dp0-30.csv vLLM+K-DP0 /home/xxf/NewVLLM/vllm/test2/bench_results-16-8-32/k-agn/log_dp1-30.csv vLLM+K-DP1 --min_step 50



 
# python /home/xxf/NewVLLM/vllm/experiment/draw_dispatch_k_dp.py /home/xxf/NewVLLM/vllm/test-vllm-ok/dispatch_logs/olmoe_32-online_ours-m16s5-r30_dp0.csv Ours-DP0 /home/xxf/NewVLLM/vllm/test-vllm-ok/dispatch_logs/olmoe_32-online_ours-m16s5-r30_dp1.csv Ours-DP1 --base /home/xxf/NewVLLM/vllm/test-vllm-ok/dispatch_logs/olmoe_32-online_vllmK-m16s5-r30_dp0.csv vLLM+K-DP0 /home/xxf/NewVLLM/vllm/test-vllm-ok/dispatch_logs/olmoe_32-online_vllmK-m16s5-r30_dp0.csv vLLM+K-DP1 --min_step 50

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

    # 单栏里上下两张图 + 右侧图例列
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(5.1, 3.2),
        sharex="col",
        gridspec_kw={
            "width_ratios": [4.6, 0.9],
            "height_ratios": [1, 1],
            "hspace": 0.15,
            "wspace": 0.05,
        },
    )

    ax_top, ax_top_leg = axes[0]
    ax_bot, ax_bot_leg = axes[1]

    # ---------- top: Ours ----------
    top_curves = [
        (dp0_csv, dp0_label, ours_colors[0]),
        (dp1_csv, dp1_label, ours_colors[1]),
    ]

    valid_files = []

    for csv_file, label, color in top_curves:
        elapsed, dispatch_k = load_curve(csv_file, min_step=min_step)
        if elapsed is None:
            print(f"Warning: no data after min_step={min_step} in {csv_file}")
            continue
        ax_top.plot(elapsed, dispatch_k, label=label, color=color)
        valid_files.append(csv_file)

    # ax_top.set_ylabel("Dispatched Routing\nBudget ($k$)")
    ax_top.grid(True, alpha=0.3)

    handles, labels = ax_top.get_legend_handles_labels()
    ax_top_leg.axis("off")
    if handles:
        ax_top_leg.legend(
            handles,
            labels,
            loc="center left",
            frameon=False,
            ncol=1,
            handlelength=1.0,
            handletextpad=0.35,
            labelspacing=0.25,
            borderpad=0.0,
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

    ax_bot.set_xlabel("Elapsed Time (s)")
    # ax_bot.set_ylabel("Dispatched Routing\nBudget ($k$)")
    ax_bot.grid(True, alpha=0.3)


    fig.supylabel("Dispatched Routing Budget ($k$)", x=0.01)
    handles, labels = ax_bot.get_legend_handles_labels()
    ax_bot_leg.axis("off")
    if handles:
        ax_bot_leg.legend(
            handles,
            labels,
            loc="center left",
            frameon=False,
            ncol=1,
            handlelength=1.0,
            handletextpad=0.35,
            labelspacing=0.25,
            borderpad=0.0,
            columnspacing=0.0,
            prop={"size": 7},
        )

    # 可选：统一 y 范围，便于对比
    ax_top.set_ylim(10, 25)
    ax_bot.set_ylim(10, 25)

    plt.tight_layout(pad=0.2)

    if valid_files:
        output_dir = os.path.dirname(os.path.abspath(valid_files[0]))
    else:
        output_dir = os.getcwd()

    output_file = os.path.join(output_dir, "dp_dispatch_k_split_compare.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved to {output_file}")


if __name__ == "__main__":
    main()


# python /home/xxf/NewVLLM/vllm/experiment/draw_dispatch_k_dp.py /home/xxf/NewVLLM/vllm/test/bench_results-12-5/k-awe/log_dp0-15.csv Ours-DP0 /home/xxf/NewVLLM/vllm/test/bench_results-12-5/k-awe/log_dp1-15.csv Ours-DP1 --base /home/xxf/NewVLLM/vllm/test/bench_results-12-5/k-agn/log_dp0-15.csv LB-K_aware-DP0 /home/xxf/NewVLLM/vllm/test/bench_results-12-5/k-agn/log_dp1-15.csv LB-K_aware-DP1 --min_step 50
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
            step = int(row["step"])
            if step < min_step:
                continue
            timestamps.append(float(row["timestamp"]))
            dispatch_k.append(float(row["dispatch_k"]))

    if not timestamps:
        return None, None

    start_time = timestamps[0]
    elapsed = [t - start_time for t in timestamps]
    return elapsed, dispatch_k


def parse_args(argv):
    base_csv = None
    base_label = None
    base_value = None

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
            "[--base x.csv x_label --base_value 32]"
        )

    curves = []
    for i in range(0, len(argv), 2):
        curves.append((argv[i], argv[i + 1]))

    if (base_csv is None) != (base_label is None):
        raise ValueError("--base must be followed by both <csv> and <label>")

    if base_csv is not None and base_value is None:
        raise ValueError("When using --base, you must also provide --base_value")

    return curves, base_csv, base_label, base_value


def main():
    argv = sys.argv[1:]

    try:
        curves, base_csv, base_label, base_value = parse_args(argv)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    plt.figure(figsize=(8, 4))

    valid_curve_files = []

    for csv_file, label in curves:
        elapsed, dispatch_k = load_curve(csv_file, min_step=30)
        if elapsed is None:
            print(f"Warning: no data with step >= 30 in {csv_file}")
            continue

        plt.plot(elapsed, dispatch_k, label=label)
        valid_curve_files.append(csv_file)

    if base_csv is not None:
        elapsed, _ = load_curve(base_csv, min_step=30)
        if elapsed is None:
            print(f"Warning: no data with step >= 30 in baseline csv {base_csv}")
        else:
            baseline_y = [base_value] * len(elapsed)
            plt.plot(elapsed, baseline_y, label=base_label) #, linestyle="--"

    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Dispatched Routing Budget ($k$)")
    # plt.title("dispatch_k over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if valid_curve_files:
        output_dir = os.path.dirname(os.path.abspath(valid_curve_files[0]))
    elif base_csv is not None:
        output_dir = os.path.dirname(os.path.abspath(base_csv))
    else:
        output_dir = os.getcwd()

    output_file = os.path.join(output_dir, "dispatch_k_compare.png")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"saved to {output_file}")


if __name__ == "__main__":
    main()


# python3 '/home/xxf/NewVLLM/vllm/experiment/draw_dispatch_k.py' '/home/xxf/NewVLLM/vllm/test/bench_results-inf/inf-8-5/log-ttft-awe.csv' Ours '/home/xxf/NewVLLM/vllm/test/bench_results-inf/inf-8-5/log-fifo-awe.csv' FIFO+k-aware  --base '/home/xxf/NewVLLM/vllm/test/bench_results-inf/inf-8-5/log-fifo-agn.csv' FIFO+k-agno --base_value 24
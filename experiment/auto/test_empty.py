#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import yaml


DEFAULT_RESULT_ROOT = "/home/xxf/NewVLLM/vllm/test/bench_results"
DEFAULT_EXPERIMENT_YAML = "/home/xxf/NewVLLM/vllm/experiment/auto/experiment.yaml"
DEFAULT_OUTPUT = "failed_exp.yaml"


def load_experiments(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"{yaml_path} is empty")

    if isinstance(data, list):
        experiments = data
        wrapped = False
    elif isinstance(data, dict) and "experiments" in data:
        experiments = data["experiments"]
        wrapped = True
    else:
        raise ValueError(
            "experiment.yaml must be either:\n"
            "1) a list of experiments, or\n"
            "2) a dict with key 'experiments'"
        )

    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")

    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f"experiment #{i} is not a dict")
        if "name" not in exp:
            raise ValueError(f"experiment #{i} missing required field: name")

    return experiments, wrapped


def find_empty_dirs(result_root: Path):
    empty_names = []
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")

    for child in sorted(result_root.iterdir()):
        if not child.is_dir():
            continue

        try:
            is_empty = next(child.iterdir(), None) is None
        except PermissionError:
            print(f"[WARN] skip no-permission dir: {child}", file=sys.stderr)
            continue

        if is_empty:
            empty_names.append(child.name)

    return empty_names


def main():
    parser = argparse.ArgumentParser(
        description="Find empty experiment result dirs and generate failed_exp.yaml from experiment.yaml"
    )
    parser.add_argument(
        "--result-root",
        default=DEFAULT_RESULT_ROOT,
        help=f"bench result root directory (default: {DEFAULT_RESULT_ROOT})",
    )
    parser.add_argument(
        "--experiment-yaml",
        default=DEFAULT_EXPERIMENT_YAML,
        help=f"experiment yaml path (default: {DEFAULT_EXPERIMENT_YAML})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output failed yaml path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    experiment_yaml = Path(args.experiment_yaml)
    output_path = Path(args.output)

    experiments, wrapped = load_experiments(experiment_yaml)
    empty_names = find_empty_dirs(result_root)
    empty_name_set = set(empty_names)

    failed_experiments = [exp for exp in experiments if exp["name"] in empty_name_set]

    missing_in_yaml = [name for name in empty_names if name not in {exp["name"] for exp in experiments}]

    if wrapped:
        out_data = {"experiments": failed_experiments}
    else:
        out_data = failed_experiments

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            out_data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    print(f"[INFO] scanned result root: {result_root}")
    print(f"[INFO] empty result dirs: {len(empty_names)}")
    print(f"[INFO] matched failed experiments: {len(failed_experiments)}")
    print(f"[INFO] output written to: {output_path}")

    if empty_names:
        print("[INFO] empty dirs:")
        for name in empty_names:
            print(f"  - {name}")

    if missing_in_yaml:
        print("[WARN] these empty dirs were not found in experiment.yaml:")
        for name in missing_in_yaml:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
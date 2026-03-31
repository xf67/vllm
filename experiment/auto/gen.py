#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def format_num(x: float | int) -> str:
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return str(xf).replace(".", "")


def yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if (
        s == ""
        or any(ch in s for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|", ">", "%", "@", "`"])
        or s.strip() != s
    ):
        return f'"{s}"'
    return s


def yaml_inline_list(xs: list[Any]) -> str:
    return "[" + ", ".join(yaml_scalar(x) for x in xs) + "]"


def extract_metadata_text(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        if line.strip() == "experiments:" and not line.startswith((" ", "\t")):
            break
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def make_case_key(scheduler: str, mean: float | int, std: float | int, rate: float | int) -> str:
    return f"{scheduler}|{mean}|{std}|{rate}"


def resolve_schedulers(model_cfg: dict[str, Any], scheduler_groups: dict[str, list[str]]) -> list[str]:
    sched = model_cfg["schedulers"]
    if isinstance(sched, str):
        if sched in scheduler_groups:
            return scheduler_groups[sched]
        return [sched]
    if isinstance(sched, list):
        return sched
    raise ValueError(f"Invalid schedulers field in model {model_cfg.get('name_prefix')}: {sched}")


def build_experiment_block(
    *,
    exp_name: str,
    default_env_anchor: str,
    model_anchor: str,
    scheduler_anchor: str,
    qos_k_mean: float | int,
    qos_k_std: float | int,
    request_rate: float | int,
    dispatch_log_dir: str,
    extra_env: dict[str, Any] | None = None,
    bench_args_mode: str = "name",
    server_args_map: dict[str, list[str]] | None = None,
) -> str:
    server_args_map = server_args_map or {}
    server_args = server_args_map.get(scheduler_anchor, ["fifo"])

    lines = []
    lines.append(f"  - name: {exp_name}")
    lines.append(f"    server_args: {yaml_inline_list(server_args)}")

    if bench_args_mode == "name":
        lines.append(f"    bench_args: [{exp_name}]")
    elif bench_args_mode == "server_args":
        lines.append(f"    bench_args: {yaml_inline_list(server_args)}")
    else:
        raise ValueError(f"Unsupported bench_args_mode: {bench_args_mode}")

    lines.append("    env:")
    lines.append(f"      <<: *{default_env_anchor}")
    lines.append(f"      <<: *{model_anchor}")
    lines.append(f"      <<: *{scheduler_anchor}")
    lines.append(f"      QOS_K_MEAN: {yaml_scalar(qos_k_mean)}")
    lines.append(f"      QOS_K_STD: {yaml_scalar(qos_k_std)}")
    lines.append(f"      REQUEST_RATES: {yaml_scalar(request_rate)}")
    lines.append(f"      DISPATCH_LOG: {dispatch_log_dir.rstrip('/')}/{exp_name}.csv")

    if extra_env:
        for k, v in extra_env.items():
            lines.append(f"      {k}: {yaml_scalar(v)}")

    return "\n".join(lines)


def generate_experiments_text(matrix_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    default_env_anchor = matrix_cfg.get("default_env_anchor", "defaults_env")
    bench_args_mode = matrix_cfg.get("bench_args_mode", "name")
    dispatch_log_dir = matrix_cfg["dispatch_log_dir"]
    scheduler_groups = matrix_cfg.get("scheduler_groups") or {}
    server_args_map = matrix_cfg.get("server_args_map") or {}
    models = matrix_cfg["models"]

    blocks = ["experiments:", ""]

    total_count = 0
    per_model_counter = Counter()
    per_scheduler_counter = Counter()

    for model_cfg in models:
        model_name = model_cfg["name_prefix"]
        model_anchor = model_cfg["env_anchor"]
        mean_std_list = model_cfg.get("mean_std") or []
        rates = model_cfg.get("rates") or []
        schedulers = resolve_schedulers(model_cfg, scheduler_groups)

        raw_skip_cases = model_cfg.get("skip_cases") or []
        skip_cases = {tuple(item) for item in raw_skip_cases}
        extra_env_per_case = model_cfg.get("extra_env_per_case") or {}

        if not mean_std_list:
            raise ValueError(f"model {model_name} has empty mean_std")
        if not rates:
            raise ValueError(f"model {model_name} has empty rates")
        if not schedulers:
            raise ValueError(f"model {model_name} has empty schedulers")

        for (mean, std), scheduler, rate in itertools.product(mean_std_list, schedulers, rates):
            case_tuple = (scheduler, mean, std, rate)
            if case_tuple in skip_cases:
                continue

            exp_name = (
                f"{model_name}-{scheduler}-"
                f"m{format_num(mean)}s{format_num(std)}-"
                f"r{format_num(rate)}"
            )

            block = build_experiment_block(
                exp_name=exp_name,
                default_env_anchor=default_env_anchor,
                model_anchor=model_anchor,
                scheduler_anchor=scheduler,
                qos_k_mean=mean,
                qos_k_std=std,
                request_rate=rate,
                dispatch_log_dir=dispatch_log_dir,
                extra_env=extra_env_per_case.get(make_case_key(scheduler, mean, std, rate)),
                bench_args_mode=bench_args_mode,
                server_args_map=server_args_map,
            )
            blocks.append(block)
            blocks.append("")

            total_count += 1
            per_model_counter[model_name] += 1
            per_scheduler_counter[scheduler] += 1

    summary = {
        "total": total_count,
        "per_model": dict(per_model_counter),
        "per_scheduler": dict(per_scheduler_counter),
    }

    return "\n".join(blocks).rstrip() + "\n", summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Total experiments: {summary['total']}")

    print("\nBy model:")
    for model, cnt in sorted(summary["per_model"].items()):
        print(f"  {model}: {cnt}")

    print("\nBy scheduler:")
    for scheduler, cnt in sorted(summary["per_scheduler"].items()):
        print(f"  {scheduler}: {cnt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="metadata-only yaml path")
    parser.add_argument("--matrix", required=True, help="matrix yaml path")
    parser.add_argument("--output", required=True, help="output full yaml path")
    args = parser.parse_args()

    metadata_text = Path(args.metadata).read_text(encoding="utf-8")
    metadata_text = extract_metadata_text(metadata_text)

    with open(args.matrix, "r", encoding="utf-8") as f:
        matrix_cfg = yaml.safe_load(f)

    experiments_text, summary = generate_experiments_text(matrix_cfg)

    final_text = metadata_text.rstrip() + "\n\n" + experiments_text
    Path(args.output).write_text(final_text, encoding="utf-8")

    print(f"Generated: {args.output}")
    print_summary(summary)


if __name__ == "__main__":
    main()

# python '/home/xxf/NewVLLM/vllm/experiment/auto/gen.py' --metadata /home/xxf/NewVLLM/vllm/experiment/auto/metadata.yaml --matrix /home/xxf/NewVLLM/vllm/experiment/auto/matrix.yaml --output /home/xxf/NewVLLM/vllm/experiment/auto/experiment.yaml

# python '/home/xxf/NewVLLM/vllm/experiment/auto/gen.py' --metadata /home/xxf/NewVLLM/vllm/experiment/auto/metadata.yaml --matrix /home/xxf/NewVLLM/vllm/experiment/auto/matrix_ep.yaml --output /home/xxf/NewVLLM/vllm/experiment/auto/experiment_ep.yaml
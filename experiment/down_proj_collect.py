#!/usr/bin/env python3
"""Profile activation concentration across all routed MoE experts.

The script hooks the input of every expert's ``down_proj`` layer.  For each
routed token, it measures how much L1 activation mass is carried by the
largest 10% and 25% of intermediate neurons.  These scale-free metrics are
aggregated online, so profiling all layers and experts does not require
retaining the activation tensors.

Example:
    python down_proj_collect.py \
        --model /home/xxf/NewVLLM/models/deepseek-v2-lite \
        --heatmap-output mod-bg-right-heatmap.png \
        --depth-output mod-bg-right-depth.png \
        --stats-output mod-bg-right.json

To redraw a figure without loading the model:
    python src/down_proj_collect.py \
        --load-stats src/mod-bg-right.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MASS_FRACTIONS = (0.10, 0.25)


@dataclass
class ExpertAccumulator:
    """Token-weighted activation-mass statistics for one layer/expert pair."""

    count: int = 0
    mass_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(len(MASS_FRACTIONS), dtype=np.float64)
    )

    def update(self, mass_share: torch.Tensor) -> None:
        self.count += mass_share.shape[0]
        self.mass_sum += mass_share.sum(dim=0).double().cpu().numpy()

    @property
    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.full(len(MASS_FRACTIONS), np.nan)
        return self.mass_sum / self.count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model name or local path")
    source.add_argument(
        "--load-stats",
        type=Path,
        help="Redraw from a JSON file produced by --stats-output",
    )
    data = parser.add_mutually_exclusive_group()
    data.add_argument(
        "--text-file",
        type=Path,
        help="Local UTF-8 calibration text (default: WikiText-2 train split)",
    )
    data.add_argument(
        "--dataset",
        default="Salesforce/wikitext",
        help="Hugging Face dataset name (default: Salesforce/wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--text-column", default="text", help="Dataset column containing text"
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=32,
        help="Number of max-length calibration sequences",
    )
    parser.add_argument(
        "--max-tokens-per-call",
        type=int,
        default=4096,
        help="Maximum routed tokens sampled in each expert invocation",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=Path("pic/mod-bg-right-heatmap.png"),
    )
    parser.add_argument(
        "--depth-output",
        type=Path,
        default=Path("pic/mod-bg-right-depth.png"),
    )
    parser.add_argument(
        "--stats-output", type=Path, default=Path("pic/mod-bg-right.json")
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def get_model_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """Return decoder layers for common Hugging Face causal-LM layouts."""
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        current = model
        for name in path:
            if not hasattr(current, name):
                break
            current = getattr(current, name)
        else:
            return current
    raise ValueError("Cannot locate decoder layers in this model")


def get_experts(layer: torch.nn.Module) -> Iterable[torch.nn.Module] | None:
    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    return experts


def make_hook(
    layer_idx: int,
    expert_idx: int,
    accumulators: dict[tuple[int, int], ExpertAccumulator],
    max_tokens: int,
):
    def hook(_module, inputs, _output) -> None:
        activation = inputs[0].detach()
        if activation.ndim == 1:
            activation = activation.unsqueeze(0)
        else:
            activation = activation.reshape(-1, activation.shape[-1])

        if activation.shape[0] > max_tokens:
            indices = torch.randperm(
                activation.shape[0], device=activation.device
            )[:max_tokens]
            activation = activation.index_select(0, indices)

        magnitude = torch.nan_to_num(activation.float().abs())
        total_mass = magnitude.sum(dim=-1)
        valid = total_mass > 0
        if not torch.any(valid):
            return
        magnitude = magnitude[valid]
        total_mass = total_mass[valid]

        neuron_count = magnitude.shape[-1]
        top_counts = [max(1, math.ceil(f * neuron_count)) for f in MASS_FRACTIONS]
        top_values = torch.topk(
            magnitude, k=max(top_counts), dim=-1, largest=True, sorted=True
        ).values
        cumulative_mass = top_values.cumsum(dim=-1)
        mass_share = torch.stack(
            [cumulative_mass[:, count - 1] / total_mass for count in top_counts],
            dim=-1,
        )
        accumulators[(layer_idx, expert_idx)].update(mass_share)

    return hook


def load_calibration_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for WikiText calibration. "
            "Install it or pass --text-file."
        ) from exc

    dataset = load_dataset(
        args.dataset, args.dataset_config, split=args.dataset_split
    )
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Column {args.text_column!r} is absent; available columns: "
            f"{dataset.column_names}"
        )
    documents = (str(text).strip() for text in dataset[args.text_column])
    return "\n\n".join(text for text in documents if text)


def model_dtype(name: str):
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def collect_stats(args: argparse.Namespace) -> dict:
    global torch
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model profiling requires torch and transformers. Install them in "
            "the model environment, or use --load-stats to redraw an existing run."
        ) from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=model_dtype(args.dtype),
        device_map="auto",
        offload_buffers=True,
    )
    model.eval()
    model.config.use_cache = False

    accumulators: dict[tuple[int, int], ExpertAccumulator] = defaultdict(
        ExpertAccumulator
    )
    handles = []
    registered_layers: list[int] = []
    expert_counts: dict[int, int] = {}
    for layer_idx, layer in enumerate(get_model_layers(model)):
        experts = get_experts(layer)
        if experts is None:
            continue
        registered_layers.append(layer_idx)
        expert_counts[layer_idx] = len(experts)
        for expert_idx, expert in enumerate(experts):
            down_proj = getattr(expert, "down_proj", None)
            if down_proj is None:
                raise ValueError(
                    f"Layer {layer_idx}, expert {expert_idx} has no down_proj"
                )
            handles.append(
                down_proj.register_forward_hook(
                    make_hook(
                        layer_idx,
                        expert_idx,
                        accumulators,
                        args.max_tokens_per_call,
                    )
                )
            )

    if not handles:
        raise ValueError("No routed experts with down_proj modules were found")

    text = load_calibration_text(args)
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0]
    required = args.max_length * args.num_sequences
    if token_ids.numel() < required:
        print(
            f"Warning: requested {required:,} tokens but calibration data contains "
            f"only {token_ids.numel():,}; using all available tokens."
        )

    input_device = model.get_input_embeddings().weight.device
    processed = 0
    try:
        with torch.inference_mode():
            stops = range(0, min(required, token_ids.numel()), args.max_length)
            for sequence_idx, start in enumerate(stops):
                sequence = token_ids[start : start + args.max_length].unsqueeze(0)
                if sequence.shape[-1] < 2:
                    break
                sequence = sequence.to(input_device)
                model(input_ids=sequence, use_cache=False)
                processed += sequence.numel()
                print(
                    f"Profiled sequence {sequence_idx + 1}/{args.num_sequences} "
                    f"({processed:,} calibration tokens)"
                )
    finally:
        for handle in handles:
            handle.remove()

    pairs = []
    for (layer_idx, expert_idx), accumulator in sorted(accumulators.items()):
        pairs.append(
            {
                "layer": layer_idx,
                "expert": expert_idx,
                "sampled_tokens": accumulator.count,
                "mass_share": accumulator.mean.tolist(),
            }
        )

    return {
        "model": args.model,
        "calibration_tokens": processed,
        "mass_fractions": list(MASS_FRACTIONS),
        "registered_layers": registered_layers,
        "expert_counts": {str(k): v for k, v in expert_counts.items()},
        "pairs": pairs,
    }


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats_arrays(stats: dict) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.asarray(stats["mass_fractions"], dtype=float)
    observed_layers = {int(pair["layer"]) for pair in stats["pairs"]}
    layers = sorted(int(layer) for layer in stats.get("registered_layers", []))
    if not layers:
        layers = sorted(observed_layers)
    configured_counts = [int(value) for value in stats.get("expert_counts", {}).values()]
    observed_count = max(
        (int(pair["expert"]) + 1 for pair in stats["pairs"]), default=0
    )
    max_experts = max(configured_counts + [observed_count])
    values = np.full((len(layers), max_experts, len(fractions)), np.nan)
    layer_to_row = {layer: row for row, layer in enumerate(layers)}
    counts = np.zeros((len(layers), max_experts), dtype=int)
    for pair in stats["pairs"]:
        row = layer_to_row[int(pair["layer"])]
        column = int(pair["expert"])
        values[row, column] = np.asarray(pair["mass_share"], dtype=float)
        counts[row, column] = int(pair["sampled_tokens"])
    return layers, fractions, values, counts


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_stats(
    stats: dict, heatmap_output: Path, depth_output: Path, dpi: int
) -> None:
    layers, fractions, values, counts = stats_arrays(stats)
    if values.size == 0 or not np.any(np.isfinite(values)):
        raise ValueError("The statistics file contains no observed experts")

    fraction_to_index = {round(value, 6): idx for idx, value in enumerate(fractions)}
    for required in MASS_FRACTIONS:
        if round(required, 6) not in fraction_to_index:
            raise ValueError(f"Statistics do not contain fraction {required}")
    top10 = values[:, :, fraction_to_index[0.10]]
    top25 = values[:, :, fraction_to_index[0.25]]

    configure_plot_style()

    heatmap_fig, heatmap_ax = plt.subplots(figsize=(4.5, 2.65))
    finite_top25 = 100 * top25[np.isfinite(top25)]
    heatmap_min = 5 * math.floor(np.min(finite_top25) / 5)
    heatmap_max = 5 * math.ceil(np.max(finite_top25) / 5)
    if heatmap_max <= heatmap_min:
        heatmap_max = heatmap_min + 5
    image = heatmap_ax.imshow(
        100 * top25,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=heatmap_min,
        vmax=heatmap_max,
    )
    heatmap_ax.set_xlabel("Expert index")
    heatmap_ax.set_ylabel("MoE layer")
    expert_ticks = sorted({0, top25.shape[1] // 4, top25.shape[1] // 2,
                           3 * top25.shape[1] // 4, top25.shape[1] - 1})
    heatmap_ax.set_xticks(expert_ticks)
    row_ticks = np.linspace(0, len(layers) - 1, 5, dtype=int)
    heatmap_ax.set_yticks(row_ticks, [layers[row] for row in row_ticks])
    colorbar = heatmap_fig.colorbar(
        image, ax=heatmap_ax, fraction=0.045, pad=0.025
    )
    colorbar.set_ticks(np.arange(heatmap_min, heatmap_max + 1, 10))
    heatmap_fig.tight_layout(pad=0.25)
    heatmap_output.parent.mkdir(parents=True, exist_ok=True)
    heatmap_fig.savefig(heatmap_output, dpi=dpi, bbox_inches="tight")
    plt.close(heatmap_fig)

    depth_fig, layer_ax = plt.subplots(figsize=(3.0, 2.65))
    colors = ("#D55E00", "#0072B2")
    for fraction, metric, color in zip(MASS_FRACTIONS, (top10, top25), colors):
        layer_median = 100 * np.nanmedian(metric, axis=1)
        layer_p10 = 100 * np.nanpercentile(metric, 10, axis=1)
        layer_p90 = 100 * np.nanpercentile(metric, 90, axis=1)
        layer_ax.fill_between(
            layers,
            layer_p10,
            layer_p90,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        layer_ax.plot(
            layers,
            layer_median,
            linewidth=2.0,
            color=color,
            label=f"Top {100 * fraction:.0f}%",
        )

    layer_ax.set_xlabel("MoE layer")
    layer_ax.set_ylabel("Activation mass (%)")
    layer_ax.set_xlim(min(layers), max(layers))
    layer_ax.set_ylim(25, 85)
    depth_tick_indices = np.linspace(0, len(layers) - 1, 4, dtype=int)
    layer_ax.set_xticks([layers[row] for row in depth_tick_indices])
    layer_ax.grid(axis="both", alpha=0.25, linewidth=0.6)
    layer_ax.legend(loc="lower right", frameon=False)

    observed = np.isfinite(top25)
    missing = counts == 0
    if np.any(missing):
        print(f"Warning: {missing.sum()} layer-expert pairs received no sampled tokens.")
    depth_fig.tight_layout(pad=0.25)
    depth_output.parent.mkdir(parents=True, exist_ok=True)
    depth_fig.savefig(depth_output, dpi=dpi, bbox_inches="tight")
    plt.close(depth_fig)

    print("\nPaper-ready summary")
    print("-------------------")
    print(
        f"Observed {observed.sum()} layer-expert pairs across {len(layers)} MoE "
        f"layers ({stats.get('calibration_tokens', 0):,} calibration tokens)."
    )
    for fraction, metric in zip(MASS_FRACTIONS, (top10, top25)):
        finite = 100 * metric[np.isfinite(metric)]
        p10, median, p90 = np.percentile(finite, (10, 50, 90))
        print(
            f"Top {100 * fraction:.0f}% neurons carry median {median:.1f}% "
            f"activation mass across layer-expert pairs "
            f"(10th-90th percentile: {p10:.1f}%-{p90:.1f}%)."
        )
    print(f"Heatmap written to {heatmap_output}")
    print(f"Depth summary written to {depth_output}")


def main() -> None:
    args = parse_args()
    if args.load_stats is not None:
        stats = load_stats(args.load_stats)
    else:
        stats = collect_stats(args)
        save_stats(stats, args.stats_output)
        print(f"Statistics written to {args.stats_output}")
    plot_stats(stats, args.heatmap_output, args.depth_output, args.dpi)


if __name__ == "__main__":
    main()

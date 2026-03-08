"""
Pre-generate QoS assignments (k_qos, ttft_max) for benchmark requests.

This ensures the SAME QoS values are used across different server configs
(QOS_AWARE=0/1/2) for fair comparison.

Usage:
    python generate_qos.py \
        --dataset sharegpt \
        --dataset-path /path/to/ShareGPT.json \
        --model /path/to/model \
        --num-prompts 500 \
        --output qos_assignments.json

    # Then run benchmark with:
    QOS_FILE=qos_assignments.json bash run_sharegpt_bench.sh qos_aware

Env vars (same as benchmark):
    PERF_MODEL_PATH   path to perf_model.json (calibrated ttft_max)
    TTFT_MULTIPLIER   prefill time multiplier (default 3.0)
    TTFT_QUEUE_MS     queue budget in ms (default 50.0)
    TTFT_JITTER_LOW   jitter lower bound (default 0.8)
    TTFT_JITTER_HIGH  jitter upper bound (default 1.5)
    TTFT_MAX_STATIC   >0 → fixed ttft_max for all (seconds)
    STATIC_QOS        >0 → fixed k for all, -1 → random
    QOS_K_MEAN        random k mean (default 4.0)
    QOS_K_STD         random k std (default 1.0)
    QOS_K_MAX         max k (default 8)
"""

import argparse
import json
import os
import random

import numpy as np


def get_perf_model():
    path = os.environ.get("PERF_MODEL_PATH", "")
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    import bisect
    table = {}
    for seq_str, k_dict in raw.items():
        table[int(seq_str)] = {int(k): float(v) for k, v in k_dict.items()}
    buckets = sorted(table.keys())
    return {"table": table, "buckets": buckets}


def pm_predict(pm, prompt_len, k):
    import bisect
    table, buckets = pm["table"], pm["buckets"]

    idx = bisect.bisect_right(buckets, prompt_len)
    if idx == 0:
        lo_seq = hi_seq = buckets[0]
    elif idx >= len(buckets):
        lo_seq = hi_seq = buckets[-1]
    else:
        lo_seq, hi_seq = buckets[idx - 1], buckets[idx]

    def lookup_k(seq_len, k_val):
        k_dict = table[seq_len]
        if k_val in k_dict:
            return k_dict[k_val]
        ks = sorted(k_dict.keys())
        i = bisect.bisect_right(ks, k_val)
        if i == 0:
            return k_dict[ks[0]]
        if i >= len(ks):
            return k_dict[ks[-1]]
        lo_k, hi_k = ks[i - 1], ks[i]
        frac = (k_val - lo_k) / (hi_k - lo_k)
        return k_dict[lo_k] + frac * (k_dict[hi_k] - k_dict[lo_k])

    lo_time = lookup_k(lo_seq, k)
    if lo_seq == hi_seq:
        return lo_time
    hi_time = lookup_k(hi_seq, k)
    frac = (prompt_len - lo_seq) / (hi_seq - lo_seq)
    return lo_time + frac * (hi_time - lo_time)


def gen_k_qos():
    static = int(os.environ.get("STATIC_QOS", -1))
    if static > 0:
        return static
    mean = float(os.environ.get("QOS_K_MEAN", 4.0))
    std = float(os.environ.get("QOS_K_STD", 1.0))
    k_max = int(os.environ.get("QOS_K_MAX", 8))
    k = int(np.random.normal(mean, std))
    return max(1, min(k, k_max))


def gen_ttft_max(prompt_len, k_qos, pm):
    static = float(os.environ.get("TTFT_MAX_STATIC", 0))
    if static > 0:
        return static

    jitter_lo = float(os.environ.get("TTFT_JITTER_LOW", 0.8))
    jitter_hi = float(os.environ.get("TTFT_JITTER_HIGH", 1.5))
    jitter = np.random.uniform(jitter_lo, jitter_hi)

    if pm is not None and k_qos > 0:
        prefill_ms = pm_predict(pm, prompt_len, k_qos)
        multiplier = float(os.environ.get("TTFT_MULTIPLIER", 1.5))
        queue_ms = float(os.environ.get("TTFT_QUEUE_MS", 50.0))
        ttft_ms = (prefill_ms * multiplier + queue_ms) * jitter
    else:
        base_ms = float(os.environ.get("TTFT_BASE_MS", 30.0))
        ms_per_token = float(os.environ.get("TTFT_MS_PER_TOKEN", 0.03))
        ttft_ms = (base_ms + prompt_len * ms_per_token) * jitter

    return round(ttft_ms / 1000.0, 6)


def load_sharegpt_prompt_lens(dataset_path, model_path, num_prompts, seed):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    data = [e for e in data if "conversations" in e and len(e["conversations"]) >= 2]
    random.seed(seed)
    random.shuffle(data)

    results = []
    for entry in data:
        if len(results) >= num_prompts:
            break
        prompt = entry["conversations"][0]["value"]
        prompt_len = len(tokenizer(prompt).input_ids)
        if prompt_len < 4 or prompt_len > 100000:
            continue
        output_text = entry["conversations"][1]["value"]
        output_len = len(tokenizer(output_text).input_ids)
        results.append({"prompt_len": prompt_len, "output_len": output_len})
    return results


def load_random_prompt_lens(num_prompts, input_len, output_len):
    return [{"prompt_len": input_len, "output_len": output_len}
            for _ in range(num_prompts)]


def main():
    parser = argparse.ArgumentParser(description="Generate QoS assignments")
    parser.add_argument("--dataset", default="sharegpt",
                        choices=["sharegpt", "random"])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--model", default="/home/xxf/models/olmoe-7B-A1B")
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-input-len", type=int, default=2048)
    parser.add_argument("--random-output-len", type=int, default=64)
    parser.add_argument("--output", default="qos_assignments.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    pm = get_perf_model()
    if pm:
        print(f"Perf model loaded: {len(pm['buckets'])} seq_len buckets")
    else:
        print("No perf model, using formula-based ttft_max")

    if args.dataset == "sharegpt":
        if not args.dataset_path:
            raise ValueError("--dataset-path required for sharegpt")
        print(f"Loading ShareGPT from {args.dataset_path} ...")
        entries = load_sharegpt_prompt_lens(
            args.dataset_path, args.model, args.num_prompts, args.seed)
    else:
        entries = load_random_prompt_lens(
            args.num_prompts, args.random_input_len, args.random_output_len)

    print(f"Generating QoS for {len(entries)} requests ...")

    assignments = []
    for i, entry in enumerate(entries):
        k = gen_k_qos()
        ttft = gen_ttft_max(entry["prompt_len"], k, pm)
        assignments.append({
            "index": i,
            "prompt_len": entry["prompt_len"],
            "output_len": entry["output_len"],
            "k_qos": k,
            "ttft_max": ttft,
        })

    with open(args.output, "w") as f:
        json.dump(assignments, f, indent=2)

    # Print summary
    ks = [a["k_qos"] for a in assignments]
    ttfts = [a["ttft_max"] * 1000 for a in assignments]
    plens = [a["prompt_len"] for a in assignments]

    print(f"\nSaved {len(assignments)} assignments → {args.output}")
    print(f"\n  prompt_len: min={min(plens)} median={sorted(plens)[len(plens)//2]}"
          f" max={max(plens)} mean={sum(plens)/len(plens):.0f}")
    print(f"  k_qos:      min={min(ks)} max={max(ks)}"
          f" mean={sum(ks)/len(ks):.1f}"
          f" distribution: {dict(sorted(((k, ks.count(k)) for k in set(ks))))}")
    print(f"  ttft_max:   min={min(ttfts):.1f}ms median={sorted(ttfts)[len(ttfts)//2]:.1f}ms"
          f" max={max(ttfts):.1f}ms mean={sum(ttfts)/len(ttfts):.1f}ms")


if __name__ == "__main__":
    main()

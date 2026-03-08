"""Aggregate per-(k, seq_len) benchmark results into a prefill perf-model
lookup table JSON."""
import json
import os
import sys

result_dir = sys.argv[1]
outpath = sys.argv[2]
seq_lens = [int(x) for x in sys.argv[3].split()]
k_vals = [int(x) for x in sys.argv[4].split()]

table = {}
for seq_len in seq_lens:
    table[str(seq_len)] = {}
    for k in k_vals:
        fpath = os.path.join(result_dir, f"k{k}_s{seq_len}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        mean_ttft = data.get("mean_ttft_ms", 0)
        if mean_ttft and mean_ttft > 0:
            table[str(seq_len)][str(k)] = round(mean_ttft, 2)

with open(outpath, "w") as f:
    json.dump(table, f, indent=2)
print(f"Written {outpath}")
print(json.dumps(table, indent=2))

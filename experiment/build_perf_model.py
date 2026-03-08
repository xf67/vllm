"""
Read profile_results/ directory and build perf_model.json.

Usage:
    python build_perf_model.py [RESULT_DIR] [OUTPUT_JSON]

Each file in RESULT_DIR is named k{k}_s{seq_len}.json and contains
vllm bench serve output with a "mean_ttft_ms" field.
"""

import json
import os
import re
import sys

result_dir = sys.argv[1] if len(sys.argv) > 1 else "test/profile_results"
output_path = sys.argv[2] if len(sys.argv) > 2 else "test/olmoe_perf_model.json"

pattern = re.compile(r"^k(\d+)_s(\d+)\.json$")

table: dict[str, dict[str, float]] = {}

for fname in sorted(os.listdir(result_dir)):
    m = pattern.match(fname)
    if not m:
        continue
    k, seq_len = m.group(1), m.group(2)
    fpath = os.path.join(result_dir, fname)
    with open(fpath) as f:
        data = json.load(f)

    mean_ttft = data.get("mean_ttft_ms")
    if mean_ttft is None or mean_ttft <= 0:
        print(f"  SKIP {fname}: mean_ttft_ms={mean_ttft}")
        continue

    if seq_len not in table:
        table[seq_len] = {}
    table[seq_len][k] = round(mean_ttft, 2)
    print(f"  k={k}  seq_len={seq_len}  →  {mean_ttft:.2f} ms")

sorted_table = {
    sl: dict(sorted(ks.items(), key=lambda x: int(x[0])))
    for sl, ks in sorted(table.items(), key=lambda x: int(x[0]))
}

with open(output_path, "w") as f:
    json.dump(sorted_table, f, indent=2)

print(f"\nWritten {output_path}")
print(json.dumps(sorted_table, indent=2))

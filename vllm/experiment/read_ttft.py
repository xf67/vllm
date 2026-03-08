"""Read mean_ttft_ms from a benchmark result JSON file."""
import json
import sys

fpath = sys.argv[1]
with open(fpath) as f:
    d = json.load(f)
v = d.get("mean_ttft_ms", 0)
print(f"{v:.2f}" if v else "0")

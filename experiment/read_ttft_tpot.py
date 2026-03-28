import json
import sys

json_file = sys.argv[1]

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

mean_ttft = round(float(data["mean_ttft_ms"]), 2)
p99_ttft = round(float(data["p99_ttft_ms"]), 2)
mean_tpot = round(float(data["mean_tpot_ms"]), 2)

print(f"{mean_ttft:.2f} & {p99_ttft:.2f} & {mean_tpot:.2f}")
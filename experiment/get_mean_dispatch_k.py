import csv

csv_file = "/home/xxf/NewVLLM/vllm/test/log.csv"

weighted_sum = 0.0
total_tokens = 0

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        k = float(row["dispatch_k"])
        tokens = int(row["total_scheduled_tokens"])
        weighted_sum += k * tokens
        total_tokens += tokens

if total_tokens == 0:
    print("total_scheduled_tokens is 0, cannot compute weighted average.")
else:
    weighted_avg_k = weighted_sum / total_tokens
    print(f"weighted average dispatch_k = {weighted_avg_k:.6f}")
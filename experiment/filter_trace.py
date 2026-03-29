"""
Filter a trace CSV, keeping only rows where:
1. ContextTokens >= min_context_len
2. ContextTokens + GeneratedTokens <= max_len

Usage:
    python filter_trace.py INPUT.csv OUTPUT.csv [--max-len 2048] [--min-context-len 1024]
"""

import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Filter trace CSV by context length and total token length")
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Max total tokens (input+output), default 2048")
    parser.add_argument("--min-context-len", type=int, default=1024,
                        help="Min context tokens, default 1024")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max number of samples to keep (0 = no limit)")
    args = parser.parse_args()

    kept = 0
    dropped = 0
    with open(args.input, newline="") as fin, \
         open(args.output, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            ctx = int(row["ContextTokens"])
            gen = int(row["GeneratedTokens"])

            if ctx < args.min_context_len:
                dropped += 1
                continue

            if ctx + gen <= args.max_len:
                writer.writerow(row)
                kept += 1
                if args.max_samples > 0 and kept >= args.max_samples:
                    break
            else:
                dropped += 1

    total = kept + dropped
    if total > 0:
        print(f"Scanned: {total}  Kept: {kept} ({100*kept/total:.1f}%)  "
              f"Dropped: {dropped} ({100*dropped/total:.1f}%)")
    else:
        print("Scanned: 0  Kept: 0 (0.0%)  Dropped: 0 (0.0%)")

    print(f"Filter conditions: ContextTokens >= {args.min_context_len}, "
          f"ContextTokens + GeneratedTokens <= {args.max_len}")

    if args.max_samples > 0:
        print(f"(capped at --max-samples {args.max_samples})")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

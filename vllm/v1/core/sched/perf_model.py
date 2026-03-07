"""
Prefill performance model for QoS-aware scheduling.

Provides estimated prefill time given (seq_len, k) via a lookup table
loaded from a JSON file. Used by the scheduler to predict whether a
request can still meet its TTFT deadline.

JSON format example:
{
  "128":  {"2": 5.2,  "4": 8.3,  "8": 14.5},
  "256":  {"2": 9.8,  "4": 13.2, "8": 22.7},
  "1024": {"2": 31.8, "4": 43.7, "8": 75.3},
  "2048": {"2": 60.2, "4": 82.4, "8": 142.6}
}
Values are prefill_time in milliseconds.
"""

import bisect
import json
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


class PrefillPerfModel:

    def __init__(self, path: str | None = None):
        self.seq_len_buckets: list[int] = []
        self.table: dict[int, dict[int, float]] = {}
        self.enabled = False

        if path is None:
            path = os.environ.get("PERF_MODEL_PATH", "")
        if path and os.path.isfile(path):
            self._load(path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            raw: dict[str, dict[str, float]] = json.load(f)

        for seq_str, k_dict in raw.items():
            seq_len = int(seq_str)
            self.table[seq_len] = {int(k): float(v) for k, v in k_dict.items()}

        self.seq_len_buckets = sorted(self.table.keys())
        if self.seq_len_buckets:
            self.enabled = True
            logger.info(
                "PrefillPerfModel loaded from %s: %d seq_len buckets, k values %s",
                path,
                len(self.seq_len_buckets),
                sorted(next(iter(self.table.values())).keys()),
            )

    def predict(self, prompt_len: int, k: int) -> float:
        """Estimate prefill time in milliseconds via linear interpolation."""
        if not self.enabled:
            return 0.0

        lo_seq, hi_seq = self._bracket_seq_len(prompt_len)
        lo_time = self._lookup_k(lo_seq, k)
        if lo_seq == hi_seq:
            return lo_time
        hi_time = self._lookup_k(hi_seq, k)
        frac = (prompt_len - lo_seq) / (hi_seq - lo_seq)
        return lo_time + frac * (hi_time - lo_time)

    def _bracket_seq_len(self, prompt_len: int) -> tuple[int, int]:
        idx = bisect.bisect_right(self.seq_len_buckets, prompt_len)
        if idx == 0:
            return self.seq_len_buckets[0], self.seq_len_buckets[0]
        if idx >= len(self.seq_len_buckets):
            return self.seq_len_buckets[-1], self.seq_len_buckets[-1]
        return self.seq_len_buckets[idx - 1], self.seq_len_buckets[idx]

    def _lookup_k(self, seq_len: int, k: int) -> float:
        k_dict = self.table[seq_len]
        if k in k_dict:
            return k_dict[k]
        ks = sorted(k_dict.keys())
        idx = bisect.bisect_right(ks, k)
        if idx == 0:
            return k_dict[ks[0]]
        if idx >= len(ks):
            return k_dict[ks[-1]]
        lo_k, hi_k = ks[idx - 1], ks[idx]
        frac = (k - lo_k) / (hi_k - lo_k)
        return k_dict[lo_k] + frac * (k_dict[hi_k] - k_dict[lo_k])

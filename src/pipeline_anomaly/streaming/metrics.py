"""In-process monitoring metrics for the stream processor.

All latencies are measured with ``time.perf_counter`` in the running process;
nothing here is estimated. Percentiles are computed over a sliding window of
recent observations so the snapshot stays cheap.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

COUNTERS = [
    "messages_received", "observations_processed", "observations_scored", "observations_flagged",
    "alerts_created", "rejected_messages", "duplicates_dropped", "late_dropped", "gap_rows_inserted",
    "values_forward_filled", "warming_up", "in_settling", "not_usable", "processing_errors", "batches",
]
LATENCIES = ["feature_ms", "inference_ms", "processing_ms", "end_to_end_ms", "publish_ms"]


def latency_summary(values) -> dict:
    """p50 / p95 / p99 / mean / max in milliseconds (None when there is no data)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "count": int(arr.size),
        "p50": round(float(p50), 3), "p95": round(float(p95), 3), "p99": round(float(p99), 3),
        "mean": round(float(arr.mean()), 3), "max": round(float(arr.max()), 3),
    }


class StreamMetrics:
    def __init__(self, window: int = 5000, clock=time.time):
        self._clock = clock
        self.started_at = clock()
        self.counts = {name: 0 for name in COUNTERS}
        self.rejections: dict[str, int] = {}
        self.latency = {name: deque(maxlen=window) for name in LATENCIES}
        self.batch_sizes: deque = deque(maxlen=window)
        self.busy_s = 0.0                       # total time spent inside process_batch
        self._recent: deque = deque(maxlen=200)  # (wall time, cumulative processed)
        self.first_message_at: float | None = None
        self.last_message_at: float | None = None

    def incr(self, name: str, n: int = 1) -> None:
        self.counts[name] += n

    def reject(self, reason: str) -> None:
        self.counts["rejected_messages"] += 1
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def observe(self, name: str, value_ms: float) -> None:
        self.latency[name].append(float(value_ms))

    def batch_done(self, n_messages: int, busy_s: float) -> None:
        now = self._clock()
        self.counts["batches"] += 1
        self.batch_sizes.append(n_messages)
        self.busy_s += busy_s
        self.first_message_at = self.first_message_at or now
        self.last_message_at = now
        self._recent.append((now, self.counts["messages_received"]))

    def throughput(self) -> dict:
        """Messages per second: over the active period, over the recent window, and at full load.

        ``capacity`` = messages / time actually spent processing, i.e. the rate
        the processor could sustain if input never ran dry.
        """
        n = self.counts["messages_received"]
        active = (self.last_message_at - self.first_message_at) if self.first_message_at and self.last_message_at else 0.0
        recent = 0.0
        if len(self._recent) >= 2:
            (t0, n0), (t1, n1) = self._recent[0], self._recent[-1]
            recent = (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0
        return {
            "overall_msgs_per_s": round(n / active, 2) if active > 0 else None,
            "recent_msgs_per_s": round(recent, 2),
            "capacity_msgs_per_s": round(n / self.busy_s, 1) if self.busy_s > 0 else None,
            "mean_cost_ms_per_message": round(1000 * self.busy_s / n, 3) if n else None,
        }

    def snapshot(self, status: str = "running", lag: int | None = None) -> dict:
        now = self._clock()
        sizes = np.asarray(self.batch_sizes, dtype=float)
        return {
            "updated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
            "updated_at_epoch": now,
            "status": status,
            "uptime_s": round(now - self.started_at, 1),
            "counts": dict(self.counts),
            "rejections_by_reason": dict(self.rejections),
            "throughput": self.throughput(),
            "latency_ms": {name: latency_summary(vals) for name, vals in self.latency.items()},
            "batch_size": {"mean": round(float(sizes.mean()), 2) if sizes.size else None,
                           "max": int(sizes.max()) if sizes.size else None},
            "consumer_lag": lag,
            "simulated": True,
        }

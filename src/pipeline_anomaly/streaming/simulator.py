"""Sensor simulator for the stream, plus optional transport-fault injection.

The stream reuses the batch project's simulator (``data_generation``), run
with a different seed and a later start date than the training data, so the
streaming system scores observations it has never seen. Rows are emitted in
time order, one message per segment per 5-minute step.

Scenarios produced by the simulator (see ``data_generation`` for details):
normal operation in four operating states, gradual seasonal and daily change,
autocorrelated random variation, and six labelled anomaly patterns (spike,
pressure-flow deviation, drift, flatline, noise burst, temperature excursion).
These are simulated patterns for analysis; none represents a real leak,
failure or incident.

``FaultInjector`` separately corrupts a small share of *messages* to exercise
the pipeline's error handling. Those are transport faults, not process
anomalies, and the detector is not expected to flag them.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data_generation import SENTINEL_VALUE, generate_dataset
from .messages import make_reading_message

FAULT_KINDS = [
    "null_value",        # a sensor value is null            -> accepted, forward-filled
    "missing_field",     # a sensor field is absent          -> accepted, forward-filled
    "sentinel_value",    # -999 written by a "faulty" device -> accepted, value discarded
    "non_numeric",       # "ERR" instead of a number         -> accepted, value discarded
    "bad_timestamp",     # unparseable timestamp             -> dead-letter
    "corrupt_json",      # truncated payload                 -> dead-letter
    "duplicate",         # the same message sent twice       -> second copy dropped
    "dropped",           # message never arrives             -> gap row inserted
]


class SensorSimulator:
    """Iterates ``(reading_row, truth_row)`` pairs in time order."""

    def __init__(self, days: int = 7, seed: int = 2026, start: str = "2025-07-01"):
        if seed == cfg.RANDOM_SEED:
            raise ValueError("Use a stream seed different from the training seed, or the stream is not unseen data.")
        df, self.manifest = generate_dataset(days=days, seed=seed, start=start, with_defects=False)
        self.data = df.sort_values(["timestamp", "segment_id"], kind="stable").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[tuple[dict, dict]]:
        for row in self.data.to_dict(orient="records"):
            reading = {k: row[k] for k in ["timestamp", "segment_id", "sensor_id", *cfg.SENSOR_COLS, "operating_state"]}
            truth = {
                "timestamp": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "segment_id": row["segment_id"],
                "is_anomaly": int(row["is_anomaly"]),
                "anomaly_type": row["anomaly_type"],
                "anomaly_event_id": row["anomaly_event_id"],
                "anomaly_sensor": row["anomaly_sensor"],
            }
            yield reading, truth


class FaultInjector:
    """Corrupts a configurable share of outgoing messages (seeded, reproducible)."""

    def __init__(self, rate: float = 0.01, seed: int = 0, kinds: list[str] | None = None):
        if not 0.0 <= rate < 1.0:
            raise ValueError("fault rate must be in [0, 1)")
        self.rate = rate
        self.kinds = kinds or FAULT_KINDS
        self.rng = np.random.default_rng(seed)
        self.counts: dict[str, int] = {k: 0 for k in self.kinds}

    def apply(self, message: dict) -> list:
        """Return the payload(s) to publish for this message (0, 1 or 2 items)."""
        if self.rate == 0.0 or self.rng.random() >= self.rate:
            return [message]
        kind = str(self.rng.choice(self.kinds))
        self.counts[kind] += 1
        sensor = str(self.rng.choice(cfg.SENSOR_COLS))
        msg = dict(message)
        if kind == "null_value":
            msg[sensor] = None
        elif kind == "missing_field":
            msg.pop(sensor)
        elif kind == "sentinel_value":
            msg[sensor] = SENTINEL_VALUE
        elif kind == "non_numeric":
            msg[sensor] = "ERR"
        elif kind == "bad_timestamp":
            msg["timestamp"] = "not-a-timestamp"
        elif kind == "corrupt_json":
            return ['{"schema_version": 1, "segment_id": "' + message["segment_id"] + '", "pressure_kpa": 41']
        elif kind == "duplicate":
            return [message, dict(message)]
        elif kind == "dropped":
            return []
        return [msg]


def build_messages(reading: dict, seq: int, produced_at: float, injector: FaultInjector | None) -> list:
    message = make_reading_message(reading, seq, produced_at)
    return injector.apply(message) if injector else [message]

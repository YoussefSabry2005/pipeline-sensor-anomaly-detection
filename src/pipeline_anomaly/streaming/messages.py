"""Message schema and validation for sensor readings.

Validation distinguishes two outcomes:

* **Rejected** - the message cannot be placed (no usable timestamp / segment,
  not a JSON object, unknown operating state). It goes to the dead-letter topic.
* **Accepted with issues** - a sensor value is missing, non-numeric or outside
  its physical range. The value is treated as missing and handled by the same
  forward-fill rule as the batch pipeline; the issue is recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config as cfg
from .settings import SCHEMA_VERSION

REQUIRED_FIELDS = ["timestamp", "segment_id", "sensor_id", "operating_state"]
GRID = pd.Timedelta(minutes=cfg.SAMPLE_MINUTES)


@dataclass
class Reading:
    timestamp: pd.Timestamp
    segment_id: str
    sensor_id: str
    operating_state: str
    values: dict[str, float]              # NaN where missing / invalid
    message_id: str | None = None
    produced_at: float | None = None      # wall-clock epoch seconds set by the producer
    issues: list[str] = field(default_factory=list)


class MessageRejected(ValueError):
    """The message cannot be processed; ``reason`` is a short machine-readable code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def make_reading_message(row: dict, seq: int, produced_at: float) -> dict:
    """Wire format of one observation. Contains NO label / ground-truth fields."""
    ts = pd.Timestamp(row["timestamp"])
    return {
        "schema_version": SCHEMA_VERSION,
        "message_id": f"{row['segment_id']}-{ts:%Y%m%dT%H%M}-{seq}",
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "segment_id": row["segment_id"],
        "sensor_id": row["sensor_id"],
        "pressure_kpa": _json_number(row["pressure_kpa"]),
        "flow_m3h": _json_number(row["flow_m3h"]),
        "temperature_c": _json_number(row["temperature_c"]),
        "operating_state": row["operating_state"],
        "produced_at": produced_at,
        "simulated": True,
    }


def _json_number(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _parse_timestamp(raw) -> pd.Timestamp | None:
    """Parse the wire format directly (fast path); fall back to pandas for other formats."""
    if isinstance(raw, str):
        try:
            return pd.Timestamp(datetime.strptime(raw, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            pass
    ts = pd.to_datetime(raw, errors="coerce")
    return ts if isinstance(ts, pd.Timestamp) and not pd.isna(ts) else None


def validate_message(value, known_segments: set[str] | None = None) -> Reading:
    """Turn a decoded message into a ``Reading`` or raise ``MessageRejected``."""
    if not isinstance(value, dict):
        raise MessageRejected("not_a_json_object")
    missing = [f for f in REQUIRED_FIELDS if value.get(f) in (None, "")]
    if missing:
        raise MessageRejected("missing_required_field", ",".join(missing))

    ts = _parse_timestamp(value["timestamp"])
    if ts is None:
        raise MessageRejected("invalid_timestamp", str(value["timestamp"])[:40])
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    if ts != ts.floor(f"{cfg.SAMPLE_MINUTES}min"):
        raise MessageRejected("off_grid_timestamp", str(ts))

    segment = str(value["segment_id"])
    if known_segments is not None and segment not in known_segments:
        raise MessageRejected("unknown_segment", segment[:40])
    state = str(value["operating_state"])
    if state not in cfg.OPERATING_STATES:
        raise MessageRejected("unknown_operating_state", state[:40])

    values, issues = {}, []
    for col in cfg.SENSOR_COLS:
        raw = value.get(col)
        if raw is None:
            values[col] = np.nan
            issues.append(f"{col}:missing")
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            values[col] = np.nan
            issues.append(f"{col}:not_numeric")
            continue
        lo, hi = cfg.VALID_RANGES[col]
        if not math.isfinite(num) or num < lo or num > hi:
            values[col] = np.nan
            issues.append(f"{col}:out_of_range")
        else:
            values[col] = num

    produced_at = value.get("produced_at")
    return Reading(
        timestamp=ts,
        segment_id=segment,
        sensor_id=str(value["sensor_id"]),
        operating_state=state,
        values=values,
        message_id=str(value["message_id"]) if value.get("message_id") is not None else None,
        produced_at=float(produced_at) if isinstance(produced_at, (int, float)) else None,
        issues=issues,
    )

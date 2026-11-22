"""Loading of the raw sensor file with schema enforcement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as cfg


class SchemaError(ValueError):
    """Raised when a file does not have the expected columns."""


def load_raw(path: Path | str = cfg.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV and coerce column types.

    * ``timestamp`` is parsed to ``datetime64`` (unparseable values -> NaT).
    * sensor columns are coerced to float (unparseable values -> NaN).
    * label columns get explicit defaults so downstream code never sees NaN
      in a label.

    No rows are dropped here; cleaning decisions belong to ``validation``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m pipeline_anomaly.data_generation` first."
        )
    df = pd.read_csv(path)

    missing = [c for c in cfg.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Raw file is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in cfg.SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ["segment_id", "sensor_id", "operating_state"]:
        df[c] = df[c].astype("string").astype(object)
    df["is_anomaly"] = pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).astype(int)
    df["anomaly_type"] = df["anomaly_type"].fillna("none").astype(str)
    df["anomaly_event_id"] = df["anomaly_event_id"].fillna("").astype(str)
    df["anomaly_sensor"] = df["anomaly_sensor"].fillna("").astype(str)
    return df[cfg.RAW_COLUMNS]

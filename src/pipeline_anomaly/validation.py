"""Data validation (inspect, never modify) and cleaning (documented fixes).

``validate`` produces a report dictionary; ``clean`` applies the decisions
documented in ``docs/methodology.md`` and returns a log of what it changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

FREQ = pd.Timedelta(minutes=cfg.SAMPLE_MINUTES)


def _invalid_mask(df: pd.DataFrame, col: str) -> pd.Series:
    lo, hi = cfg.VALID_RANGES[col]
    return df[col].notna() & ((df[col] < lo) | (df[col] > hi))


def validate(df: pd.DataFrame) -> dict:
    """Inspect a raw frame and return a JSON-serialisable quality report."""
    report: dict = {"n_rows": int(len(df)), "n_columns": int(df.shape[1])}
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(v) for c, v in df.isna().sum().items()}
    report["exact_duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_keys"] = int(df.duplicated(subset=cfg.KEY_COLS).sum())
    report["unparseable_timestamps"] = int(df["timestamp"].isna().sum())
    report["invalid_values"] = {c: int(_invalid_mask(df, c).sum()) for c in cfg.SENSOR_COLS}
    report["unknown_operating_states"] = int((~df["operating_state"].isin(cfg.OPERATING_STATES)).sum())

    labelled = df["is_anomaly"] == 1
    typed = df["anomaly_type"] != "none"
    report["label_inconsistencies"] = int((labelled != typed).sum())
    report["unknown_anomaly_types"] = int(
        (~df["anomaly_type"].isin(cfg.ANOMALY_TYPES + ["none"])).sum()
    )

    segments = {}
    for seg, g in df.dropna(subset=["timestamp"]).groupby("segment_id"):
        ts = g["timestamp"].drop_duplicates().sort_values()
        expected = int((ts.iloc[-1] - ts.iloc[0]) / FREQ) + 1
        deltas = ts.diff().dropna()
        segments[str(seg)] = {
            "start": str(ts.iloc[0]),
            "end": str(ts.iloc[-1]),
            "rows": int(len(g)),
            "unique_timestamps": int(len(ts)),
            "expected_timestamps": expected,
            "missing_timestamps": int(expected - len(ts)),
            "coverage_pct": round(100.0 * len(ts) / expected, 3),
            "is_sorted": bool(g["timestamp"].is_monotonic_increasing),
            "off_grid_intervals": int(((deltas % FREQ) != pd.Timedelta(0)).sum()),
            "largest_gap_minutes": float(deltas.max() / pd.Timedelta(minutes=1)),
        }
    report["segments"] = segments
    report["sensor_summary"] = {
        c: {k: round(float(v), 3) for k, v in df.loc[~_invalid_mask(df, c), c].describe().items()}
        for c in cfg.SENSOR_COLS
    }
    return report


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning decisions and return ``(clean_df, log)``.

    Decisions (in order):

    1. Drop rows whose timestamp could not be parsed (cannot be placed in time).
    2. Drop exact duplicate rows; for any remaining duplicate
       (segment, timestamp) keep the first occurrence.
    3. Replace physically impossible readings with NaN (they are instrument /
       telemetry artefacts, not process behaviour).
    4. Re-index each segment onto the full 5-minute grid so that gaps become
       explicit rows (``row_inserted = True``).
    5. Forward-fill sensor gaps for at most ``MAX_FFILL_SAMPLES`` samples.
       Forward-fill is used instead of interpolation because interpolation
       needs the *next* reading, which a live monitor would not have yet.
    6. Flag every row that still lacks a sensor value as ``usable = False``.

    Rows are flagged rather than silently deleted so the time grid stays
    regular, which the rolling-window features rely on.
    """
    log: dict = {"rows_in": int(len(df))}
    out = df.copy()

    bad_ts = out["timestamp"].isna()
    log["dropped_unparseable_timestamps"] = int(bad_ts.sum())
    out = out[~bad_ts]

    n = len(out)
    out = out.drop_duplicates()
    log["dropped_exact_duplicates"] = int(n - len(out))
    n = len(out)
    out = out.drop_duplicates(subset=cfg.KEY_COLS, keep="first")
    log["dropped_conflicting_duplicate_keys"] = int(n - len(out))

    log["invalid_values_set_missing"] = {}
    for c in cfg.SENSOR_COLS:
        mask = _invalid_mask(out, c)
        out.loc[mask, c] = np.nan
        log["invalid_values_set_missing"][c] = int(mask.sum())

    frames = []
    inserted_total = 0
    for seg, g in out.groupby("segment_id", sort=True):
        g = g.sort_values("timestamp").set_index("timestamp")
        grid = pd.date_range(g.index.min(), g.index.max(), freq=FREQ, name="timestamp")
        inserted = ~grid.isin(g.index)
        g = g.reindex(grid)
        g["row_inserted"] = inserted
        inserted_total += int(inserted.sum())

        g["segment_id"] = seg
        g["sensor_id"] = g["sensor_id"].ffill().bfill()
        g["operating_state"] = g["operating_state"].ffill().bfill()
        g["is_anomaly"] = g["is_anomaly"].fillna(0).astype(int)
        g["anomaly_type"] = g["anomaly_type"].fillna("none")
        g["anomaly_event_id"] = g["anomaly_event_id"].fillna("")
        g["anomaly_sensor"] = g["anomaly_sensor"].fillna("")

        was_missing = g[cfg.SENSOR_COLS].isna()
        g[cfg.SENSOR_COLS] = g[cfg.SENSOR_COLS].ffill(limit=cfg.MAX_FFILL_SAMPLES)
        g["n_filled"] = (was_missing & g[cfg.SENSOR_COLS].notna()).sum(axis=1).astype(int)
        g["usable"] = g[cfg.SENSOR_COLS].notna().all(axis=1)
        frames.append(g.reset_index())

    clean_df = pd.concat(frames, ignore_index=True)
    log["inserted_rows_for_missing_timestamps"] = inserted_total
    log["sensor_cells_forward_filled"] = int(clean_df["n_filled"].sum())
    log["rows_not_usable"] = int((~clean_df["usable"]).sum())
    log["rows_out"] = int(len(clean_df))

    cols = cfg.RAW_COLUMNS + ["row_inserted", "n_filled", "usable"]
    return clean_df[cols], log

"""Chronological train / validation / test split.

Time-series data must be split by time, not at random: a random split would
put samples from the same anomaly event (and overlapping rolling windows) on
both sides of the split and leak information.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg


def split_boundaries(
    timestamps: pd.Series, train_frac: float = cfg.TRAIN_FRAC, val_frac: float = cfg.VAL_FRAC
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the timestamps at which validation and test periods begin."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Fractions must be in (0, 1) and sum to less than 1.")
    start, end = timestamps.min(), timestamps.max()
    span = end - start
    freq = f"{cfg.SAMPLE_MINUTES}min"
    val_start = (start + span * train_frac).ceil(freq)
    test_start = (start + span * (train_frac + val_frac)).ceil(freq)
    return val_start, test_start


def assign_split(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Label every row ``train``, ``val`` or ``test`` using shared time cut-offs."""
    val_start, test_start = split_boundaries(df["timestamp"], **kwargs)
    split = pd.Series("train", index=df.index, name="split", dtype=object)
    split[df["timestamp"] >= val_start] = "val"
    split[df["timestamp"] >= test_start] = "test"
    return split

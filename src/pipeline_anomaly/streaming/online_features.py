"""Incremental (one observation at a time) cleaning and feature computation.

The batch project computes features with pandas over the whole history. Doing
that per message costs tens of milliseconds, so the stream keeps a small
fixed-size buffer per segment and computes only the model features for the
newest observation with NumPy.

**Consistency with training is the requirement.** The definitions below mirror
``validation.clean`` and ``features.build_features`` line by line (same
windows, same minimum counts, same forward-fill limit, same floors), and
``tests/test_streaming_features.py`` asserts that streaming and batch produce
the same feature values on the same data. Because only past observations are
in the buffer, look-ahead is impossible by construction.

Differences from batch, by necessity:

* A message older than the newest one already processed for its segment
  cannot be inserted into the past; it is reported as ``late`` and skipped.
* State lives in memory, so a restarted processor warms up again (~6 h of
  simulated data per segment) before it can score.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as cfg
from ..features import MODEL_FEATURES, SHORT_NAMES, PressureFlowModel
from .messages import Reading

BUFFER = 256                               # samples kept per segment (~21 h); longest window needs 74
STEP = pd.Timedelta(minutes=cfg.SAMPLE_MINUTES)
SHORT, LONG, HALF, TREND = cfg.SHORT_WINDOW, cfg.LONG_WINDOW, cfg.SHORT_WINDOW // 2, cfg.TREND_LAG
ALPHA = 2.0 / (SHORT + 1.0)                # pandas ewm(span=SHORT)
EWM_WEIGHTS = (1.0 - ALPHA) ** np.arange(BUFFER - 2, -1, -1)   # oldest ... newest past sample
SETTLING_STEPS = cfg.SETTLING_MINUTES // cfg.SAMPLE_MINUTES


def _stat(values: np.ndarray, fn, min_count: int) -> float:
    """``fn`` over the non-missing values, or NaN if there are too few (pandas min_periods)."""
    valid = values[~np.isnan(values)]
    return float(fn(valid)) if len(valid) >= min_count else np.nan


@dataclass
class FeatureResult:
    status: str                      # ok | duplicate | late
    features: np.ndarray | None      # ordered as MODEL_FEATURES; None unless status == "ok"
    feature_ready: bool = False      # usable row past warm-up with every feature available
    in_settling: bool = False        # within the settling window after a state change
    usable: bool = False             # all three sensor values present after forward-fill
    n_filled: int = 0                # sensor values forward-filled for this observation
    rows_inserted: int = 0           # gap rows inserted before this observation
    stream_reset: bool = False       # gap too long: buffers were cleared
    values: dict | None = None       # cleaned sensor values actually used

    @property
    def monitored(self) -> bool:
        return self.feature_ready and not self.in_settling


class SegmentState:
    """Rolling state for one segment."""

    def __init__(self, segment_id: str, coef: np.ndarray, resid_scale: float):
        self.segment_id, self.coef, self.resid_scale = segment_id, coef, resid_scale
        self.reset()

    def reset(self) -> None:
        self.x = np.full((BUFFER, 3), np.nan)        # cleaned sensor values, newest last
        self.pfz = np.full(BUFFER, np.nan)           # pressure-flow residual history
        self.last_ts: pd.Timestamp | None = None
        self.last_valid = np.full(3, np.nan)
        self.fill_run = np.zeros(3, dtype=int)
        self.state: str | None = None
        self.steps_in_state: int | None = None       # None until the first observed change
        self.seen: deque = deque(maxlen=BUFFER)      # recent timestamps, for duplicate detection

    def push(self, raw: np.ndarray, state: str) -> tuple[bool, int]:
        """Append one grid row (``raw`` may contain NaN). Returns (usable, n_filled)."""
        values = raw.astype(float).copy()
        n_filled = 0
        for j in range(3):
            if np.isnan(values[j]):
                # forward-fill from the past, for at most MAX_FFILL_SAMPLES consecutive samples
                if not np.isnan(self.last_valid[j]) and self.fill_run[j] < cfg.MAX_FFILL_SAMPLES:
                    values[j] = self.last_valid[j]
                    n_filled += 1
                self.fill_run[j] += 1
            else:
                self.last_valid[j] = values[j]
                self.fill_run[j] = 0

        self.x[:-1] = self.x[1:]
        self.x[-1] = values
        self.pfz[:-1] = self.pfz[1:]
        self.pfz[-1] = (values[0] - np.polyval(self.coef, values[1])) / self.resid_scale

        if self.state is not None and state != self.state:
            self.steps_in_state = 0
        elif self.steps_in_state is not None:
            self.steps_in_state += 1
        self.state = state
        return bool(not np.isnan(values).any()), n_filled

    @property
    def in_settling(self) -> bool:
        return self.steps_in_state is not None and self.steps_in_state < SETTLING_STEPS

    def compute(self) -> dict[str, float]:
        """Model features for the newest row (same definitions as ``features.py``)."""
        f: dict[str, float] = {}
        for j, (col, name) in enumerate(SHORT_NAMES.items()):
            x = self.x[:, j]
            floor = cfg.SIGMA_FLOOR[col]
            step = np.abs(np.diff(x))                                   # step[i] <-> x[i+1]
            vol_usual = _stat(step[-(LONG + 1):-1], np.median, LONG // 2)
            base_median = _stat(x[-(SHORT + 1):-1], np.median, HALF)
            scale = 1.4826 * vol_usual
            f[f"{name}_rz_1h"] = (x[-1] - base_median) / (max(scale, floor) if not np.isnan(scale) else np.nan)
            f[f"{name}_diff_1"] = x[-1] - x[-2]

            past = x[:-1]
            ok = ~np.isnan(past)
            ewma = float(np.dot(EWM_WEIGHTS[ok], past[ok]) / EWM_WEIGHTS[ok].sum()) if ok.sum() >= HALF else np.nan
            f[f"{name}_ewma_resid"] = x[-1] - ewma

            vol_recent = _stat(step[-SHORT:], np.mean, HALF)
            eps = 0.25 * floor
            f[f"{name}_vol_ratio"] = np.log((vol_recent + eps) / (vol_usual + eps))

        mean_now = _stat(self.pfz[-SHORT:], np.mean, HALF)
        mean_then = _stat(self.pfz[-(SHORT + TREND):-TREND], np.mean, HALF)
        f["pf_resid_z"] = float(self.pfz[-1])
        f["pf_resid_mean_1h"] = mean_now
        f["pf_resid_trend_3h"] = mean_now - mean_then
        t = self.x[:, 2]
        f["temperature_trend_3h"] = _stat(t[-SHORT:], np.mean, HALF) - _stat(t[-(SHORT + TREND):-TREND], np.mean, HALF)
        return f


class OnlineFeatureEngine:
    """Per-segment streaming cleaner + feature calculator."""

    def __init__(self, pf_model: PressureFlowModel):
        self.pf_model = pf_model
        self.segments: dict[str, SegmentState] = {
            seg: SegmentState(seg, coef, pf_model.resid_scale[seg]) for seg, coef in pf_model.coefs.items()
        }

    @property
    def known_segments(self) -> set[str]:
        return set(self.segments)

    def update(self, reading: Reading) -> FeatureResult:
        st = self.segments[reading.segment_id]
        ts = reading.timestamp

        rows_inserted, stream_reset = 0, False
        if st.last_ts is not None:
            if ts <= st.last_ts:
                return FeatureResult("duplicate" if ts in st.seen else "late", None)
            gap = int((ts - st.last_ts) / STEP) - 1
            if gap >= BUFFER:
                st.reset()                      # history is too old to be useful: start again
                stream_reset = True
            else:
                for _ in range(gap):            # make missing timestamps explicit, as batch cleaning does
                    st.push(np.full(3, np.nan), st.state)
                rows_inserted = gap

        raw = np.array([reading.values[c] for c in cfg.SENSOR_COLS], dtype=float)
        usable, n_filled = st.push(raw, reading.operating_state)
        st.last_ts = ts
        st.seen.append(ts)

        feats = st.compute()
        vector = np.array([feats[name] for name in MODEL_FEATURES], dtype=float)
        ready = usable and bool(np.isfinite(vector).all())
        return FeatureResult(
            "ok", vector, feature_ready=ready, in_settling=st.in_settling, usable=usable, n_filled=n_filled,
            rows_inserted=rows_inserted, stream_reset=stream_reset,
            values=dict(zip(cfg.SENSOR_COLS, st.x[-1].tolist())),
        )

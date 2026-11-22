"""Causal feature engineering for streaming-style monitoring.

Design rule: **a feature at time t may only use readings at or before t.**
Every rolling window is trailing, every "baseline" statistic is shifted by one
sample so the current reading is compared against the past only, and the one
fitted component (the pressure-flow curve) is fitted on the training period
only. ``tests/test_features.py`` checks the no-look-ahead property directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

SHORT_NAMES = {"pressure_kpa": "pressure", "flow_m3h": "flow", "temperature_c": "temperature"}

# Features handed to the multivariate detectors. Raw levels, lags and rolling
# means are deliberately excluded: they mostly encode the operating state
# (e.g. "flow is 0"), and an unsupervised model would then flag every rare but
# legitimate state such as a shutdown.
MODEL_FEATURES: list[str] = (
    [f"{n}_rz_1h" for n in SHORT_NAMES.values()]
    + [f"{n}_diff_1" for n in SHORT_NAMES.values()]
    + [f"{n}_ewma_resid" for n in SHORT_NAMES.values()]
    + [f"{n}_vol_ratio" for n in SHORT_NAMES.values()]
    + ["pf_resid_z", "pf_resid_mean_1h", "pf_resid_trend_3h", "temperature_trend_3h"]
)
# Per-sensor robust z-scores used by the statistical baseline detector.
ZSCORE_FEATURES = [f"{n}_rz_1h" for n in SHORT_NAMES.values()]
# Classic mean/std z-scores, kept in the feature table for comparison only.
CLASSIC_ZSCORE_FEATURES = [f"{n}_z_6h" for n in SHORT_NAMES.values()]


@dataclass
class PressureFlowModel:
    """Per-segment quadratic fit ``pressure ~ a*flow^2 + b*flow + c``.

    Encodes the engineering expectation that pressure and flow move together.
    The residual (observed minus expected pressure) is a feature that stays
    near zero in *every* operating state, which rolling statistics cannot do.
    """

    coefs: dict[str, np.ndarray] = field(default_factory=dict)
    resid_scale: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "PressureFlowModel":
        model = cls()
        for seg, g in train.groupby("segment_id"):
            g = g.dropna(subset=["pressure_kpa", "flow_m3h"])
            coef = np.polyfit(g["flow_m3h"], g["pressure_kpa"], deg=2)
            resid = g["pressure_kpa"] - np.polyval(coef, g["flow_m3h"])
            # Robust scale (MAD) so labelled-or-not anomalies in the training
            # period do not inflate the denominator.
            mad = float(np.median(np.abs(resid - np.median(resid))))
            model.coefs[str(seg)] = coef
            model.resid_scale[str(seg)] = max(1.4826 * mad, 1e-6)
        return model

    def residual_z(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for seg, g in df.groupby("segment_id"):
            if str(seg) not in self.coefs:
                raise KeyError(f"Segment {seg!r} was not seen when fitting PressureFlowModel")
            expected = np.polyval(self.coefs[str(seg)], g["flow_m3h"])
            out.loc[g.index] = (g["pressure_kpa"] - expected) / self.resid_scale[str(seg)]
        return out


def _segment_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features for one segment, already sorted by time on a regular grid."""
    feats = pd.DataFrame(index=g.index)
    short, long_, half = cfg.SHORT_WINDOW, cfg.LONG_WINDOW, cfg.SHORT_WINDOW // 2

    for col, name in SHORT_NAMES.items():
        x = g[col]
        past = x.shift(1)  # everything strictly before the current sample
        floor = cfg.SIGMA_FLOOR[col]

        # Lagged readings and changes
        feats[f"{name}_lag_1"] = past
        feats[f"{name}_lag_1h"] = x.shift(short)
        feats[f"{name}_diff_1"] = x - past
        feats[f"{name}_diff_1h"] = x - x.shift(short)

        # Rolling level and spread (trailing, include the current sample)
        feats[f"{name}_roll_mean_1h"] = x.rolling(short, min_periods=half).mean()
        feats[f"{name}_roll_std_1h"] = x.rolling(short, min_periods=half).std()
        feats[f"{name}_roll_mean_6h"] = x.rolling(long_, min_periods=long_ // 2).mean()
        feats[f"{name}_roll_std_6h"] = x.rolling(long_, min_periods=long_ // 2).std()

        # Classic rolling z-score: current reading versus the mean / std of the
        # previous 6 h. Kept for comparison: a state change inflates the 6 h
        # std for hours, which both triggers and then hides alerts.
        base_mean = past.rolling(long_, min_periods=long_ // 2).mean()
        base_std = past.rolling(long_, min_periods=long_ // 2).std().clip(lower=floor)
        feats[f"{name}_z_6h"] = (x - base_mean) / base_std

        # Typical sample-to-sample movement: rolling *median* of |change| over
        # the previous 6 h. A median ignores the few large steps of a state
        # change, so this scale stays honest right after a transition.
        step = (x - past).abs()
        vol_usual = step.shift(1).rolling(long_, min_periods=long_ // 2).median()

        # Robust z-score: deviation from the median of the previous hour,
        # scaled by the typical step size (1.4826 converts a median absolute
        # value to a standard-deviation-like unit).
        base_median = past.rolling(short, min_periods=half).median()
        feats[f"{name}_rz_1h"] = (x - base_median) / (1.4826 * vol_usual).clip(lower=floor)

        # Deviation from an exponentially weighted forecast of the past
        feats[f"{name}_ewma_resid"] = x - past.ewm(span=short, min_periods=half).mean()

        # Volatility ratio: recent movement versus its usual level. Strongly
        # negative = unusually quiet (stuck sensor); positive = unusually noisy.
        vol_recent = step.rolling(short, min_periods=half).mean()
        eps = 0.25 * floor
        feats[f"{name}_vol_ratio"] = np.log((vol_recent + eps) / (vol_usual + eps))

        # Trend indicator: change in the 1 h mean over the last 3 h
        mean_1h = feats[f"{name}_roll_mean_1h"]
        feats[f"{name}_trend_3h"] = mean_1h - mean_1h.shift(cfg.TREND_LAG)

    # Operating-context features
    changed = g["operating_state"] != g["operating_state"].shift(1)
    changed.iloc[0] = False
    run_id = changed.cumsum()
    steps_in_state = g.groupby(run_id).cumcount()
    # Before the first observed change we do not know how long the state has lasted.
    never_changed = run_id == 0
    feats["minutes_since_state_change"] = (steps_in_state * cfg.SAMPLE_MINUTES).where(
        ~never_changed, np.nan
    )
    feats["is_shutdown"] = (g["operating_state"] == "SHUTDOWN").astype(int)

    # Time-of-day / calendar features (cyclical encoding)
    hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feats["day_of_week"] = g["timestamp"].dt.dayofweek
    return feats


def build_features(clean: pd.DataFrame, pf_model: PressureFlowModel) -> pd.DataFrame:
    """Return ``clean`` with engineered feature columns appended.

    ``clean`` must be the output of ``validation.clean`` (regular time grid per
    segment). ``pf_model`` must have been fitted on training-period rows only.
    """
    clean = clean.sort_values(cfg.KEY_COLS, kind="stable").reset_index(drop=True)
    parts = [_segment_features(g) for _, g in clean.groupby("segment_id", sort=False)]
    feats = pd.concat(parts).loc[clean.index]

    out = pd.concat([clean, feats], axis=1)
    out["pf_resid_z"] = pf_model.residual_z(out)
    out["pf_resid_mean_1h"] = out.groupby("segment_id")["pf_resid_z"].transform(
        lambda s: s.rolling(cfg.SHORT_WINDOW, min_periods=cfg.SHORT_WINDOW // 2).mean()
    )
    # Trend of the residual: unlike a raw sensor trend it is not triggered by
    # a legitimate change of operating state.
    out["pf_resid_trend_3h"] = out["pf_resid_mean_1h"] - out.groupby("segment_id")[
        "pf_resid_mean_1h"
    ].shift(cfg.TREND_LAG)

    # A row can be scored only if it is a real, usable observation and every
    # model feature exists (i.e. it is past the warm-up period).
    out["feature_ready"] = (
        out["usable"] & ~out["row_inserted"] & out[MODEL_FEATURES].notna().all(axis=1)
    )
    # Settling window after a commanded state change: scored rows that are
    # expected to move and are therefore not monitored (see config).
    out["in_settling"] = out["minutes_since_state_change"].fillna(np.inf) < cfg.SETTLING_MINUTES
    out["monitored"] = out["feature_ready"] & ~out["in_settling"]
    return out

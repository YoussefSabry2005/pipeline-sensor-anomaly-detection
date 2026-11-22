"""Structured alerts for flagged observations.

An alert is an **analytical notification for review**. It says that an
observation scored as statistically unusual relative to the (simulated)
training period and lists the features that deviated most. It never states
that a leak, failure or safety event occurred, and nothing acts on it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config as cfg
from ..features import MODEL_FEATURES
from .settings import EPISODE_GAP_MINUTES, SCHEMA_VERSION, SEVERITY_BANDS

DISCLAIMER = (
    "Simulated data. Analytical notification for human review only; "
    "not an indication of a leak, equipment failure or safety incident."
)

_SENSOR_WORDS = {"pressure": "pressure", "flow": "flow rate", "temperature": "temperature"}


def severity_for(percentile: float) -> str:
    """Map a training-score percentile to a documented severity band (see settings)."""
    for name, lower in SEVERITY_BANDS:
        if percentile >= lower:
            return name
    return SEVERITY_BANDS[-1][0]


def describe_feature(name: str, deviation: float) -> str:
    """Plain-language description of one feature deviating in one direction."""
    up = deviation > 0
    if name == "pf_resid_z":
        return f"pressure is {'higher' if up else 'lower'} than expected for the current flow rate"
    if name == "pf_resid_mean_1h":
        return f"pressure has been {'higher' if up else 'lower'} than expected for the flow rate over the last hour"
    if name == "pf_resid_trend_3h":
        return f"the pressure-flow relationship has shifted {'upward' if up else 'downward'} over the last 3 hours"
    prefix, _, kind = name.partition("_")
    sensor = _SENSOR_WORDS.get(prefix, prefix)
    if kind == "rz_1h":
        return f"{sensor} is {'above' if up else 'below'} its level over the previous hour"
    if kind == "diff_1":
        return f"{sensor} {'rose' if up else 'fell'} sharply since the previous reading"
    if kind == "ewma_resid":
        return f"{sensor} is {'above' if up else 'below'} its recent smoothed trend"
    if kind == "vol_ratio":
        return (f"{sensor} is fluctuating more than usual" if up
                else f"{sensor} is fluctuating less than usual (values barely changing)")
    if kind == "trend_3h":
        return f"{sensor} has been {'rising' if up else 'falling'} over the last 3 hours"
    return f"{name} deviates from its typical value"


def top_deviations(features: np.ndarray, center: np.ndarray, scale: np.ndarray, k: int = 3) -> list[dict]:
    """The k features furthest from their training median, in robust (IQR) units."""
    deviation = (np.asarray(features, dtype=float) - center) / scale
    order = np.argsort(-np.abs(deviation))[:k]
    return [
        {
            "feature": MODEL_FEATURES[i],
            "value": round(float(features[i]), 4),
            "robust_deviation": round(float(deviation[i]), 2),
            "description": describe_feature(MODEL_FEATURES[i], float(deviation[i])),
        }
        for i in order
    ]


class EpisodeTracker:
    """Groups flagged observations that are close in (simulated) time into one episode."""

    def __init__(self, gap_minutes: int = EPISODE_GAP_MINUTES):
        self.gap = pd.Timedelta(minutes=gap_minutes)
        self._last: dict[str, tuple[pd.Timestamp, str, int]] = {}

    def assign(self, segment_id: str, ts: pd.Timestamp) -> tuple[str, int]:
        last = self._last.get(segment_id)
        if last is not None and ts - last[0] <= self.gap:
            episode_id, n = last[1], last[2] + 1
        else:
            episode_id, n = f"EP-{segment_id}-{ts:%Y%m%dT%H%M}", 1
        self._last[segment_id] = (ts, episode_id, n)
        return episode_id, n


def build_alert(
    *,
    timestamp: pd.Timestamp,
    segment_id: str,
    sensor_id: str,
    operating_state: str,
    readings: dict,
    detector_results: dict[str, dict],
    deviations: list[dict],
    episode: tuple[str, int],
    latency_ms: dict,
) -> dict:
    """Assemble the JSON-serialisable alert record."""
    fired = [name for name, r in detector_results.items() if r["flagged"]]
    percentile = max(r["percentile"] for r in detector_results.values())
    lead = max(fired, key=lambda n: detector_results[n]["percentile"])
    reasons = "; ".join(d["description"] for d in deviations)
    explanation = (
        f"{lead} scored this observation at the {100 * detector_results[lead]['percentile']:.1f}th percentile "
        f"of training-period scores (alert threshold: {100 * (1 - cfg.ALERT_BUDGET):.0f}th). "
        f"Largest deviations from typical values: {reasons}."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "alert_id": f"ALT-{segment_id}-{timestamp:%Y%m%dT%H%M}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "observation_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "segment_id": segment_id,
        "sensor_id": sensor_id,
        "operating_state": operating_state,
        "severity": severity_for(percentile),
        "anomaly_score_percentile": round(float(percentile), 4),
        "detection_methods": fired,
        "detector_scores": {
            name: {"score": round(float(r["score"]), 5), "percentile": round(float(r["percentile"]), 4),
                   "threshold": round(float(r["threshold"]), 5), "flagged": bool(r["flagged"])}
            for name, r in detector_results.items()
        },
        "readings": {k: (None if v is None or np.isnan(v) else round(float(v), 3)) for k, v in readings.items()},
        "explanation": explanation,
        "top_deviations": deviations,
        "episode_id": episode[0],
        "episode_alert_number": episode[1],
        "latency_ms": latency_ms,
        "simulated": True,
        "disclaimer": DISCLAIMER,
    }

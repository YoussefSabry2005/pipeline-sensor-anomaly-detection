"""Evaluation against the *synthetic* labels.

All numbers produced here measure how well a detector recovers the anomaly
patterns that the simulator injected. They are not estimates of performance
on a real pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from . import config as cfg


def _check(y_true, other) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    o = np.asarray(other)
    if y.shape != o.shape or y.ndim != 1:
        raise ValueError(f"Shape mismatch: labels {y.shape} vs predictions {o.shape}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary (0/1)")
    return y, o


def classification_metrics(y_true, flags) -> dict:
    """Point-wise precision, recall, F1, false-positive rate and confusion counts."""
    y, f = _check(y_true, flags)
    f = f.astype(int)
    if not np.isin(f, [0, 1]).all():
        raise ValueError("Flags must be binary (0/1)")
    tn, fp, fn, tp = confusion_matrix(y, f, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "alert_rate": float(f.mean()) if len(f) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def ranking_metrics(y_true, scores) -> dict:
    """Threshold-free metrics. PR-AUC is the more informative one when
    anomalies are rare; its no-skill baseline equals the anomaly prevalence."""
    y, s = _check(y_true, scores)
    if y.min() == y.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "prevalence": float(y.mean())}
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "prevalence": float(y.mean()),
    }


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Threshold maximising F1 on the supplied (validation) data.

    This uses labels, so it is reported separately as a *label-informed*
    reference point and is never tuned on the test period.
    """
    y, s = _check(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    # `flag` uses a strict ">" comparison, so step just below the chosen score.
    return float(np.nextafter(thresholds[best], -np.inf)), float(f1[best])


def event_metrics(frame: pd.DataFrame, flags) -> pd.DataFrame:
    """One row per labelled event: was it caught, and how quickly?

    ``frame`` needs ``anomaly_event_id``, ``anomaly_type`` and ``timestamp``.
    An event counts as detected if at least one of its samples is flagged.
    ``delay_minutes`` is the time from the event's first sample to the first
    flagged sample.
    """
    f = np.asarray(flags).astype(int)
    if len(f) != len(frame):
        raise ValueError("flags and frame must have the same length")
    ev = frame.loc[frame["anomaly_event_id"] != "", ["anomaly_event_id", "anomaly_type", "timestamp"]]
    ev = ev.assign(flag=f[(frame["anomaly_event_id"] != "").to_numpy()])
    rows = []
    for event_id, g in ev.groupby("anomaly_event_id", sort=True):
        hit = g.loc[g["flag"] == 1, "timestamp"]
        rows.append(
            {
                "anomaly_event_id": event_id,
                "anomaly_type": g["anomaly_type"].iat[0],
                "n_samples": int(len(g)),
                "n_flagged": int(g["flag"].sum()),
                "detected": bool(len(hit) > 0),
                "delay_minutes": (
                    float((hit.min() - g["timestamp"].min()) / pd.Timedelta(minutes=1))
                    if len(hit)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["anomaly_event_id", "anomaly_type", "n_samples", "n_flagged", "detected", "delay_minutes"],
    )


def recall_by_type(frame: pd.DataFrame, flags) -> pd.DataFrame:
    """Point-wise and event-level recall for each simulated anomaly type."""
    f = np.asarray(flags).astype(int)
    events = event_metrics(frame, f)
    rows = []
    for kind in cfg.ANOMALY_TYPES:
        mask = (frame["anomaly_type"] == kind).to_numpy()
        ev = events[events["anomaly_type"] == kind]
        rows.append(
            {
                "anomaly_type": kind,
                "n_points": int(mask.sum()),
                "point_recall": float(f[mask].mean()) if mask.any() else np.nan,
                "n_events": int(len(ev)),
                "event_recall": float(ev["detected"].mean()) if len(ev) else np.nan,
                "median_delay_minutes": float(ev["delay_minutes"].median()) if len(ev) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def false_positive_breakdown(frame: pd.DataFrame, flags) -> dict:
    """Where do the false alarms occur?

    Categories are mutually exclusive and checked in this order: inside a
    settling window, during shutdown, within one hour after a labelled event
    (the rolling windows are still "recovering"), otherwise steady operation.
    """
    f = np.asarray(flags).astype(int)
    fp = (f == 1) & (frame["is_anomaly"].to_numpy() == 0)
    settling = frame["in_settling"].to_numpy()
    shutdown = (frame["operating_state"] == "SHUTDOWN").to_numpy()
    recent_anomaly = (
        frame.groupby("segment_id")["is_anomaly"]
        .transform(lambda s: s.shift(1).rolling(cfg.SHORT_WINDOW, min_periods=1).max())
        .fillna(0)
        .to_numpy()
        > 0
    )
    total = int(fp.sum())
    cats = {
        "in_settling_window": int((fp & settling).sum()),
        "during_shutdown": int((fp & shutdown & ~settling).sum()),
        "within_1h_after_event": int((fp & recent_anomaly & ~settling & ~shutdown).sum()),
    }
    cats["steady_operation"] = total - sum(cats.values())
    cats["total_false_positives"] = total
    return cats

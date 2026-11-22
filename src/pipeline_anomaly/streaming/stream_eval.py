"""Evaluate a finished stream against the simulator's ground-truth topic.

The processor never reads ``sensor.truth``; it is joined here, after the fact,
purely to measure how the streaming system did on the simulated anomalies.
"""

from __future__ import annotations

import pandas as pd

from .. import evaluation as ev
from .broker import Broker
from .settings import TOPIC_SCORED, TOPIC_TRUTH


def read_all(broker: Broker, topic: str, page: int = 5000) -> list[dict]:
    out, offset = [], 0
    while True:
        chunk = broker.read(topic, offset, page)
        if not chunk:
            return out
        out.extend(m.value for m in chunk if isinstance(m.value, dict))
        offset = chunk[-1].offset


def scored_frame(broker: Broker) -> pd.DataFrame:
    rows = []
    for r in read_all(broker, TOPIC_SCORED):
        row = {"timestamp": r["timestamp"], "segment_id": r["segment_id"], "status": r["status"],
               "flagged": int(bool(r["flagged"])), "anomaly_score_percentile": r["anomaly_score_percentile"],
               "severity": r["severity"], "operating_state": r["operating_state"]}
        for name, d in (r.get("detectors") or {}).items():
            row[f"flag::{name}"] = int(bool(d["flagged"]))
            row[f"pct::{name}"] = d["percentile"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def evaluate_stream(broker: Broker) -> dict:
    """Point-wise and event-level metrics for the stream, against simulated labels.

    Evaluated rows are those the processor emitted with status ``scored`` or
    ``settling`` (settling rows cannot alert, so anomalies inside them count as
    missed, exactly as in the batch evaluation). Warm-up rows are excluded, as
    in batch.
    """
    scored = scored_frame(broker)
    truth = pd.DataFrame(read_all(broker, TOPIC_TRUTH))
    if scored.empty or truth.empty:
        return {"evaluated_rows": 0}
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    df = scored.merge(truth, on=["timestamp", "segment_id"], how="left", validate="one_to_one")
    df = df[df["status"].isin(["scored", "settling"])].sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)
    df["anomaly_event_id"] = df["anomaly_event_id"].fillna("")
    df["anomaly_type"] = df["anomaly_type"].fillna("none")

    def block(flags) -> dict:
        out = ev.classification_metrics(df["is_anomaly"], flags)
        events = ev.event_metrics(df, flags)
        out["event_recall"] = float(events["detected"].mean()) if len(events) else None
        out["events"] = int(len(events))
        return out

    result = {
        "evaluated_rows": int(len(df)),
        "anomalous_rows": int(df["is_anomaly"].sum()),
        "prevalence": float(df["is_anomaly"].mean()),
        "combined_alert_rule": block(df["flagged"].to_numpy()),
        "by_detector": {},
        "recall_by_anomaly_type": ev.recall_by_type(df, df["flagged"].to_numpy()).to_dict(orient="records"),
        "alerts_by_severity": df.loc[df["flagged"] == 1, "severity"].value_counts().to_dict(),
    }
    for col in [c for c in df.columns if c.startswith("flag::")]:
        result["by_detector"][col.split("::", 1)[1]] = block(df[col].fillna(0).astype(int).to_numpy())
    return result

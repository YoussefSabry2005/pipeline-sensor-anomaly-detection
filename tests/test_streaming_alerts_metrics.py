import json

import numpy as np
import pandas as pd
import pytest

from pipeline_anomaly.features import MODEL_FEATURES
from pipeline_anomaly.streaming.alerts import (
    DISCLAIMER, EpisodeTracker, build_alert, describe_feature, severity_for, top_deviations,
)
from pipeline_anomaly.streaming.metrics import StreamMetrics, latency_summary


@pytest.mark.parametrize("pct, expected", [(0.9701, "low"), (0.9899, "low"), (0.99, "medium"), (0.9969, "medium"),
                                           (0.997, "high"), (1.0, "high")])
def test_severity_bands(pct, expected):
    assert severity_for(pct) == expected


def test_top_deviations_are_ranked_and_described():
    features = np.zeros(len(MODEL_FEATURES))
    features[MODEL_FEATURES.index("pf_resid_mean_1h")] = -30.0
    features[MODEL_FEATURES.index("flow_vol_ratio")] = -8.0
    out = top_deviations(features, np.zeros(len(features)), np.ones(len(features)), k=2)
    assert [d["feature"] for d in out] == ["pf_resid_mean_1h", "flow_vol_ratio"]
    assert out[0]["robust_deviation"] == -30.0
    assert "lower than expected" in out[0]["description"]
    assert "less than usual" in out[1]["description"]


def test_every_model_feature_has_a_description():
    for name in MODEL_FEATURES:
        for direction in (1.0, -1.0):
            text = describe_feature(name, direction)
            assert text and name not in text            # plain language, not the raw column name


def test_episode_grouping():
    ep = EpisodeTracker(gap_minutes=30)
    t = pd.Timestamp("2025-07-01 10:00")
    first = ep.assign("SEG-A", t)
    assert first[1] == 1
    assert ep.assign("SEG-A", t + pd.Timedelta(minutes=20)) == (first[0], 2)
    assert ep.assign("SEG-B", t + pd.Timedelta(minutes=20))[1] == 1          # other segment: own episode
    new = ep.assign("SEG-A", t + pd.Timedelta(minutes=120))
    assert new[0] != first[0] and new[1] == 1


def _alert(**overrides):
    det = {
        "Isolation Forest": {"score": 0.61, "percentile": 0.9985, "threshold": 0.48, "flagged": True},
        "Local Outlier Factor": {"score": 1.2, "percentile": 0.80, "threshold": 1.44, "flagged": False},
    }
    kwargs = dict(
        timestamp=pd.Timestamp("2025-07-03 08:15"), segment_id="SEG-B", sensor_id="STN-102", operating_state="NORMAL",
        readings={"pressure_kpa": 3400.123456, "flow_m3h": 801.0, "temperature_c": float("nan")},
        detector_results=det,
        deviations=top_deviations(np.arange(len(MODEL_FEATURES), dtype=float), np.zeros(16), np.ones(16)),
        episode=("EP-SEG-B-20250703T0815", 1),
        latency_ms={"feature_ms": 0.3, "inference_ms": 9.1, "processing_ms": 11.0, "end_to_end_ms": 25.0, "batch_size": 4},
    )
    kwargs.update(overrides)
    return build_alert(**kwargs)


def test_alert_structure():
    a = _alert()
    for key in ["alert_id", "created_at", "observation_timestamp", "segment_id", "sensor_id", "severity",
                "anomaly_score_percentile", "detection_methods", "detector_scores", "readings", "explanation",
                "top_deviations", "episode_id", "latency_ms", "simulated", "disclaimer"]:
        assert key in a, key
    assert a["alert_id"] == "ALT-SEG-B-20250703T0815"
    assert a["severity"] == "high" and a["anomaly_score_percentile"] == 0.9985
    assert a["detection_methods"] == ["Isolation Forest"]            # only detectors that fired
    assert a["readings"]["temperature_c"] is None                    # NaN is not valid JSON
    assert a["readings"]["pressure_kpa"] == 3400.123
    assert a["simulated"] is True and a["disclaimer"] == DISCLAIMER
    json.dumps(a, allow_nan=False)                                   # strictly serialisable


def test_alert_wording_makes_no_incident_claim():
    a = _alert()
    text = (a["explanation"] + " " + " ".join(d["description"] for d in a["top_deviations"])).lower()
    for word in ["leak", "failure", "incident", "rupture", "unsafe", "danger", "emergency"]:
        assert word not in text
    assert "99.9th percentile" in a["explanation"] and "Isolation Forest" in a["explanation"]
    assert "not an indication" in a["disclaimer"]


def test_latency_summary_values():
    s = latency_summary(range(1, 101))                               # 1..100 ms
    assert s["count"] == 100 and s["max"] == 100.0 and s["mean"] == 50.5
    assert s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05) and s["p99"] == pytest.approx(99.01)
    empty = latency_summary([])
    assert empty["count"] == 0 and empty["p95"] is None


def test_metrics_counts_throughput_and_window():
    clock = {"t": 1000.0}
    m = StreamMetrics(window=3, clock=lambda: clock["t"])
    for _ in range(4):                                               # four batches of 50, one per second
        m.incr("messages_received", 50)
        m.batch_done(50, busy_s=0.1)
        clock["t"] += 1.0
    for v in (1.0, 2.0, 3.0, 4.0):
        m.observe("processing_ms", v)
    m.reject("invalid_json"); m.reject("invalid_json"); m.reject("off_grid_timestamp")
    snap = m.snapshot("running", lag=7)
    assert snap["counts"]["messages_received"] == 200 and snap["counts"]["batches"] == 4
    assert snap["counts"]["rejected_messages"] == 3
    assert snap["rejections_by_reason"] == {"invalid_json": 2, "off_grid_timestamp": 1}
    thr = snap["throughput"]
    assert thr["overall_msgs_per_s"] == pytest.approx(200 / 3, abs=0.01)       # 200 messages over 3 s of activity
    assert thr["recent_msgs_per_s"] == pytest.approx(50.0)
    assert thr["capacity_msgs_per_s"] == pytest.approx(500.0)        # 200 messages in 0.4 s of work
    assert thr["mean_cost_ms_per_message"] == pytest.approx(2.0)
    assert snap["latency_ms"]["processing_ms"]["count"] == 3         # sliding window keeps the last 3
    assert snap["latency_ms"]["processing_ms"]["max"] == 4.0
    assert snap["consumer_lag"] == 7 and snap["simulated"] is True
    json.dumps(snap, allow_nan=False)

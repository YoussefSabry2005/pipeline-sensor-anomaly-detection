import numpy as np
import pandas as pd
import pytest

from pipeline_anomaly import evaluation as ev


def test_classification_metrics_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    f = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    m = ev.classification_metrics(y, f)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 7)
    assert m["alert_rate"] == pytest.approx(0.3)


def test_classification_metrics_edge_cases():
    assert ev.classification_metrics([0, 0, 1], [0, 0, 0])["precision"] == 0.0   # no alerts
    assert ev.classification_metrics([0, 0, 0], [0, 1, 0])["recall"] == 0.0      # no positives
    with pytest.raises(ValueError):
        ev.classification_metrics([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        ev.classification_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError):
        ev.classification_metrics([0, 1], [0, 3])


def test_ranking_metrics():
    y = [0, 0, 0, 1, 1]
    perfect = ev.ranking_metrics(y, [0.1, 0.2, 0.3, 0.8, 0.9])
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["prevalence"] == pytest.approx(0.4)
    inverted = ev.ranking_metrics(y, [0.9, 0.8, 0.7, 0.2, 0.1])
    assert inverted["roc_auc"] == 0.0
    assert np.isnan(ev.ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])["roc_auc"])


def test_best_f1_threshold_separates_perfectly():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    thr, f1 = ev.best_f1_threshold(y, s)
    assert f1 == pytest.approx(1.0)
    assert ((s > thr).astype(int) == y).all()


def _event_frame():
    ts = pd.date_range("2025-01-01", periods=10, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "segment_id": "S",
            "is_anomaly": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            "anomaly_type": ["none", "drift", "drift", "drift", "none", "none", "spike", "spike", "none", "none"],
            "anomaly_event_id": ["", "E1", "E1", "E1", "", "", "E2", "E2", "", ""],
        }
    )


def test_event_metrics_detection_and_delay():
    frame = _event_frame()
    flags = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    out = ev.event_metrics(frame, flags).set_index("anomaly_event_id")
    assert bool(out.loc["E1", "detected"]) and out.loc["E1", "delay_minutes"] == 10.0
    assert out.loc["E1", "n_flagged"] == 1 and out.loc["E1", "n_samples"] == 3
    assert not bool(out.loc["E2", "detected"]) and np.isnan(out.loc["E2", "delay_minutes"])
    with pytest.raises(ValueError):
        ev.event_metrics(frame, flags[:-1])


def test_recall_by_type():
    out = ev.recall_by_type(_event_frame(), [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]).set_index("anomaly_type")
    assert out.loc["drift", "point_recall"] == pytest.approx(1 / 3)
    assert out.loc["drift", "event_recall"] == 1.0
    assert out.loc["spike", "event_recall"] == 0.0
    assert np.isnan(out.loc["flatline", "point_recall"]) and out.loc["flatline", "n_events"] == 0


def test_false_positive_breakdown_sums_to_total():
    frame = _event_frame().assign(
        in_settling=[True] + [False] * 9, operating_state=["NORMAL"] * 8 + ["SHUTDOWN"] * 2
    )
    flags = [1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    out = ev.false_positive_breakdown(frame, flags)
    assert out["total_false_positives"] == 5
    assert out["in_settling_window"] == 1
    assert out["during_shutdown"] == 2
    assert out["within_1h_after_event"] == 2
    assert out["steady_operation"] == 0

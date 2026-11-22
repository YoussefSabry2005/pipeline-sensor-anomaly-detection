import json

import numpy as np
import pandas as pd
import pytest

from pipeline_anomaly import config as cfg
from pipeline_anomaly.features import MODEL_FEATURES, build_features
from pipeline_anomaly.streaming import processor as proc
from pipeline_anomaly.streaming.broker import InMemoryBroker, SQLiteBroker
from pipeline_anomaly.streaming.logging_utils import setup_logging
from pipeline_anomaly.streaming.producer import run_producer
from pipeline_anomaly.streaming.settings import (
    CONSUMER_GROUP, TOPIC_ALERTS, TOPIC_DEADLETTER, TOPIC_METRICS, TOPIC_READINGS, TOPIC_SCORED, TOPIC_TRUTH,
)
from pipeline_anomaly.streaming.simulator import SensorSimulator
from pipeline_anomaly.streaming.stream_eval import evaluate_stream, read_all
from pipeline_anomaly.validation import clean


def drain(broker, processor, batch=32):
    while True:
        msgs = broker.poll(TOPIC_READINGS, CONSUMER_GROUP, batch)
        if not msgs:
            return
        broker.publish_and_commit(processor.process_batch(msgs), TOPIC_READINGS, CONSUMER_GROUP, msgs[-1].offset)


@pytest.fixture(scope="module")
def run(bundle, tmp_path_factory):
    """One complete in-memory stream: 3 simulated days with 3 % injected message faults."""
    from pipeline_anomaly.streaming.settings import StreamSettings

    s = StreamSettings()
    s.stream_days, s.stream_seed, s.stream_rate, s.fault_rate = 3, 99, 0.0, 0.03
    s.alerts_file = tmp_path_factory.mktemp("stream") / "alerts.jsonl"
    broker = InMemoryBroker()
    stats = run_producer(broker, s)
    processor = proc.StreamProcessor(bundle, s)
    drain(broker, processor)
    return {"broker": broker, "stats": stats, "processor": processor, "settings": s,
            "scored": read_all(broker, TOPIC_SCORED), "alerts": read_all(broker, TOPIC_ALERTS),
            "dead": read_all(broker, TOPIC_DEADLETTER), "counts": processor.metrics.counts}


def test_every_message_is_accounted_for(run):
    c, faults = run["counts"], run["stats"]["faults"]
    assert c["messages_received"] == run["stats"]["messages_published"]
    assert c["messages_received"] == (
        c["observations_processed"] + c["rejected_messages"] + c["duplicates_dropped"] + c["late_dropped"]
    )
    assert c["processing_errors"] == 0
    assert len(run["scored"]) == c["observations_processed"]
    assert run["broker"].lag(TOPIC_READINGS, CONSUMER_GROUP) == 0


def test_injected_faults_are_handled_as_documented(run):
    c, faults = run["counts"], run["stats"]["faults"]
    assert sum(faults.values()) > 20
    assert c["rejected_messages"] == faults["bad_timestamp"] + faults["corrupt_json"] == len(run["dead"])
    assert c["duplicates_dropped"] == faults["duplicate"]
    assert c["gap_rows_inserted"] >= faults["dropped"] - 4          # a drop at the very start/end leaves no gap
    assert c["values_forward_filled"] > 0
    reasons = {d["reason"] for d in run["dead"]}
    assert reasons == {"invalid_timestamp", "invalid_json"}
    assert all(d["simulated"] and "payload" in d for d in run["dead"])


def test_scored_records_have_no_labels_and_valid_latency(run):
    statuses = {r["status"] for r in run["scored"]}
    assert "scored" in statuses and "warming_up" in statuses and "settling" in statuses
    for r in run["scored"]:
        assert not any(k in r for k in cfg.LABEL_COLS)
        lat = r["latency_ms"]
        assert lat["feature_ms"] >= 0 and lat["processing_ms"] >= lat["feature_ms"] and lat["end_to_end_ms"] >= 0
        if r["status"] != "scored":
            assert r["flagged"] is False and r["alert_id"] is None and r["detectors"] == {}
        else:
            assert 0.0 <= r["anomaly_score_percentile"] <= 1.0
            assert set(r["detectors"]) == set(run["settings"].detectors)
            assert lat["inference_ms"] > 0


def test_alerts_match_flagged_observations(run):
    flagged = [r for r in run["scored"] if r["flagged"]]
    assert len(flagged) == len(run["alerts"]) == run["counts"]["alerts_created"] > 0
    assert {r["alert_id"] for r in flagged} == {a["alert_id"] for a in run["alerts"]}
    assert len({a["alert_id"] for a in run["alerts"]}) == len(run["alerts"])       # unique ids
    for a in run["alerts"]:
        assert a["severity"] in {"low", "medium", "high"} and a["simulated"] is True
        assert a["detection_methods"] and a["explanation"] and len(a["top_deviations"]) == 3
        assert a["anomaly_score_percentile"] > 1 - cfg.ALERT_BUDGET - 0.005
        json.dumps(a, allow_nan=False)


def test_streaming_scores_equal_batch_scores(bundle, pf_model, stream_settings):
    """The same data through the batch code path must give the same anomaly scores."""
    stream_settings.fault_rate = 0.0                      # fault-free, so both paths see identical data
    broker = InMemoryBroker()
    run_producer(broker, stream_settings)
    processor = proc.StreamProcessor(bundle, stream_settings)
    drain(broker, processor)
    scored = read_all(broker, TOPIC_SCORED)

    sim = SensorSimulator(stream_settings.stream_days, stream_settings.stream_seed, stream_settings.stream_start)
    batch = build_features(clean(sim.data)[0], pf_model)
    batch = batch[batch["monitored"]].reset_index(drop=True)
    stream = {(r["segment_id"], r["timestamp"]): r for r in scored if r["status"] == "scored"}
    assert len(stream) == len(batch) > 2000              # exactly the same rows are scorable

    keys = list(zip(batch["segment_id"], batch["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")))
    for name in stream_settings.detectors:
        reference = bundle["detectors"][name].score(batch)
        streamed = np.array([stream[k]["detectors"][name]["score"] for k in keys])
        np.testing.assert_allclose(streamed, reference, rtol=1e-6, atol=1e-9)
        flags = np.array([stream[k]["detectors"][name]["flagged"] for k in keys])
        assert (flags == (reference > bundle["thresholds"][name])).mean() > 0.999


def test_processing_error_does_not_stop_the_stream(bundle, stream_settings, monkeypatch):
    broker = InMemoryBroker()
    stream_settings.fault_rate = 0.0
    run_producer(broker, stream_settings, max_messages=40)
    processor = proc.StreamProcessor(bundle, stream_settings)
    original, calls = processor.features.update, {"n": 0}

    def flaky(reading):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("boom")
        return original(reading)

    monkeypatch.setattr(processor.features, "update", flaky)
    drain(broker, processor)
    c = processor.metrics.counts
    assert c["processing_errors"] == 1 and c["observations_processed"] == 39
    dead = read_all(broker, TOPIC_DEADLETTER)
    assert len(dead) == 1 and dead[0]["reason"] == "processing_error" and dead[0]["detail"] == "RuntimeError"


def test_inference_failure_is_reported_not_fatal(bundle, stream_settings, monkeypatch):
    broker = InMemoryBroker()
    stream_settings.fault_rate, stream_settings.stream_days = 0.0, 1
    run_producer(broker, stream_settings)
    processor = proc.StreamProcessor(bundle, stream_settings)

    def broken(_X):
        raise ValueError("model unavailable")

    monkeypatch.setattr(processor.detectors["Isolation Forest"], "score", broken)
    drain(broker, processor)
    statuses = pd.Series([r["status"] for r in read_all(broker, TOPIC_SCORED)]).value_counts()
    assert statuses.get("inference_error", 0) > 0 and "scored" not in statuses
    assert processor.metrics.counts["processing_errors"] > 0
    assert broker.count(TOPIC_ALERTS) == 0


def test_unknown_detector_is_rejected(bundle, stream_settings):
    stream_settings.detectors = ("Not A Detector",)
    with pytest.raises(KeyError):
        proc.StreamProcessor(bundle, stream_settings)


def test_run_loop_with_sqlite_broker_and_logs(bundle, stream_settings):
    """Full loop on a real broker file: offsets, alert file, metrics topic and JSON logs."""
    setup_logging(stream_settings.log_dir, "INFO", "processor", console=False)
    stream_settings.stream_days = 1
    broker = SQLiteBroker(stream_settings.broker_path)
    stats = run_producer(broker, stream_settings)
    snap = proc.run_processor(broker, proc.StreamProcessor(bundle, stream_settings), idle_timeout_s=0.3)

    assert snap["status"] == "stopped" and snap["consumer_lag"] == 0
    assert snap["counts"]["messages_received"] == stats["messages_published"]
    assert broker.count(TOPIC_METRICS) >= 1
    lat = snap["latency_ms"]
    assert lat["processing_ms"]["p50"] > 0 and lat["inference_ms"]["p95"] >= lat["inference_ms"]["p50"]
    assert snap["throughput"]["capacity_msgs_per_s"] > 0

    alert_lines = [json.loads(line) for line in stream_settings.alerts_file.read_text().splitlines()]
    assert len(alert_lines) == broker.count(TOPIC_ALERTS)

    # a second run finds nothing new: outputs and offsets were committed together
    again = proc.run_processor(broker, proc.StreamProcessor(bundle, stream_settings), idle_timeout_s=0.3)
    assert again["counts"]["messages_received"] == 0
    assert broker.count(TOPIC_SCORED) == snap["counts"]["observations_processed"]

    logs = [json.loads(line) for line in (stream_settings.log_dir / "processor.jsonl").read_text().splitlines()]
    events = {entry["event"] for entry in logs}
    assert {"processor_started", "observation_scored", "processor_stopped"} <= events
    scored_log = next(e for e in logs if e["event"] == "observation_scored")
    assert {"ts", "level", "component", "segment_id", "anomaly_score_percentile", "inference_ms", "processing_ms"} <= set(scored_log)
    if snap["counts"]["rejected_messages"]:
        assert "message_rejected" in events
    setup_logging(None, "INFO", console=False)


def test_stream_evaluation_against_truth(run):
    result = evaluate_stream(run["broker"])
    assert result["evaluated_rows"] > 2000 and result["anomalous_rows"] > 0
    for block in [result["combined_alert_rule"], *result["by_detector"].values()]:
        for key in ("precision", "recall", "f1", "false_positive_rate"):
            assert 0.0 <= block[key] <= 1.0
    assert set(result["by_detector"]) == set(run["settings"].detectors)
    assert run["broker"].count(TOPIC_TRUTH) == run["stats"]["simulated_rows"]

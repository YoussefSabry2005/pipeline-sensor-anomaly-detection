import pandas as pd
import pytest

from pipeline_anomaly import config as cfg
from pipeline_anomaly.streaming.broker import InMemoryBroker
from pipeline_anomaly.streaming.producer import run_producer
from pipeline_anomaly.streaming.settings import TOPIC_READINGS, TOPIC_TRUTH
from pipeline_anomaly.streaming.simulator import FAULT_KINDS, FaultInjector, SensorSimulator


@pytest.fixture(scope="module")
def sim():
    return SensorSimulator(days=2, seed=99, start="2025-07-01")


def test_simulator_is_time_ordered_and_complete(sim):
    rows = list(sim)
    assert len(rows) == len(sim) == 2 * 24 * 12 * 4
    ts = [pd.Timestamp(r["timestamp"]) for r, _ in rows]
    assert ts == sorted(ts)
    reading, truth = rows[0]
    assert set(reading) == {"timestamp", "segment_id", "sensor_id", *cfg.SENSOR_COLS, "operating_state"}
    assert {"is_anomaly", "anomaly_type"} <= set(truth)
    assert not any(k in reading for k in cfg.LABEL_COLS)        # labels travel separately


def test_simulator_is_deterministic_and_contains_anomalies(sim):
    again = SensorSimulator(days=2, seed=99, start="2025-07-01")
    pd.testing.assert_frame_equal(sim.data, again.data)
    assert sim.data["is_anomaly"].sum() > 0
    assert set(sim.data["anomaly_type"]) - {"none"} <= set(cfg.ANOMALY_TYPES)


def test_stream_seed_must_differ_from_training_seed():
    with pytest.raises(ValueError):
        SensorSimulator(days=1, seed=cfg.RANDOM_SEED)


def test_fault_injector_off_passes_messages_through():
    inj = FaultInjector(rate=0.0)
    m = {"segment_id": "SEG-A", "pressure_kpa": 1.0}
    assert inj.apply(m) == [m] and sum(inj.counts.values()) == 0


def test_fault_injector_kinds():
    base = {"segment_id": "SEG-A", "timestamp": "2025-07-01 00:00:00", "pressure_kpa": 1.0, "flow_m3h": 2.0,
            "temperature_c": 3.0}
    expected_len = {"duplicate": 2, "dropped": 0}
    for kind in FAULT_KINDS:
        inj = FaultInjector(rate=0.999999, seed=1, kinds=[kind])
        out = inj.apply(dict(base))
        assert len(out) == expected_len.get(kind, 1), kind
        assert inj.counts[kind] == 1
        if kind == "corrupt_json":
            assert isinstance(out[0], str)
        if kind == "bad_timestamp":
            assert out[0]["timestamp"] != base["timestamp"]
    with pytest.raises(ValueError):
        FaultInjector(rate=1.5)


def test_fault_injection_is_reproducible():
    a, b = FaultInjector(0.2, seed=5), FaultInjector(0.2, seed=5)
    msgs = [{"segment_id": "SEG-A", "timestamp": "t", "pressure_kpa": float(i), "flow_m3h": 1.0, "temperature_c": 1.0}
            for i in range(200)]
    assert [a.apply(dict(m)) for m in msgs] == [b.apply(dict(m)) for m in msgs]
    assert 10 < sum(a.counts.values()) < 80


def test_producer_publishes_readings_and_truth(stream_settings):
    broker = InMemoryBroker()
    stream_settings.stream_days = 1
    stats = run_producer(broker, stream_settings)
    n = 24 * 12 * 4
    assert stats["simulated_rows"] == n == broker.count(TOPIC_TRUTH)
    f = stats["faults"]
    assert broker.count(TOPIC_READINGS) == stats["messages_published"] == n + f["duplicate"] - f["dropped"]


def test_producer_resumes_the_timeline(stream_settings):
    broker = InMemoryBroker()
    stream_settings.stream_days, stream_settings.fault_rate = 1, 0.0
    run_producer(broker, stream_settings)
    run_producer(broker, stream_settings)                       # e.g. a restarted producer
    ts = pd.to_datetime([m.value["timestamp"] for m in broker.read(TOPIC_TRUTH, 0, 10**6)])
    assert ts.is_monotonic_increasing and ts.nunique() == 2 * 24 * 12
    assert ts.max() == pd.Timestamp(stream_settings.stream_start) + pd.Timedelta(days=2) - pd.Timedelta(minutes=5)


def test_producer_respects_max_messages(stream_settings):
    broker = InMemoryBroker()
    stats = run_producer(broker, stream_settings, max_messages=100)
    assert stats["simulated_rows"] == 100

"""The streaming feature engine must reproduce the batch features exactly."""

import numpy as np
import pandas as pd
import pytest

from pipeline_anomaly import config as cfg
from pipeline_anomaly.data_generation import generate_dataset
from pipeline_anomaly.features import MODEL_FEATURES, build_features
from pipeline_anomaly.streaming.messages import make_reading_message, validate_message
from pipeline_anomaly.streaming.online_features import BUFFER, OnlineFeatureEngine
from pipeline_anomaly.validation import clean


def feed(engine, rows):
    out = []
    for r in rows:
        reading = validate_message(make_reading_message(r, 0, 0.0), engine.known_segments)
        out.append((r["segment_id"], pd.Timestamp(r["timestamp"]), engine.update(reading)))
    return out


@pytest.fixture(scope="module")
def defective_stream():
    """Six days WITH missing cells, sentinel values and missing timestamps."""
    df, _ = generate_dataset(days=6, seed=11, start="2025-07-01", with_defects=True)
    return df.drop_duplicates().sort_values(["timestamp", "segment_id"]).reset_index(drop=True)


def test_streaming_features_match_batch(defective_stream, pf_model):
    batch = build_features(clean(defective_stream)[0], pf_model).set_index(["segment_id", "timestamp"])
    results = feed(OnlineFeatureEngine(pf_model), defective_stream.to_dict("records"))

    compared = 0
    for seg, ts, res in results:
        ref = batch.loc[(seg, ts)]
        assert res.feature_ready == bool(ref["feature_ready"]), (seg, ts)
        assert res.in_settling == bool(ref["in_settling"]), (seg, ts)
        assert res.usable == bool(ref["usable"])
        assert res.n_filled == int(ref["n_filled"])
        if res.feature_ready:
            np.testing.assert_allclose(res.features, ref[MODEL_FEATURES].to_numpy(float), rtol=1e-7, atol=1e-7)
            compared += 1
    assert compared > 5000
    assert sum(r.rows_inserted for _, _, r in results) == int(batch["row_inserted"].sum())


def _rows(n, start="2025-07-01", seed=3):
    df, _ = generate_dataset(days=2, seed=seed, start=start, with_defects=False)
    return df[df["segment_id"] == "SEG-A"].head(n).to_dict("records")


def test_warm_up_then_ready(pf_model):
    results = feed(OnlineFeatureEngine(pf_model), _rows(120))
    ready = [r.feature_ready for _, _, r in results]
    assert not any(ready[: cfg.LONG_WINDOW // 2])       # cannot score without history
    assert all(ready[-20:])


def test_duplicate_and_late_messages(pf_model):
    engine, rows = OnlineFeatureEngine(pf_model), _rows(20)
    feed(engine, rows[:10])
    assert feed(engine, [rows[9]])[0][2].status == "duplicate"
    assert feed(engine, [rows[4]])[0][2].status == "duplicate"
    engine2 = OnlineFeatureEngine(pf_model)
    feed(engine2, rows[:5] + rows[8:12])                # 5, 6, 7 never arrived in time
    assert feed(engine2, [rows[6]])[0][2].status == "late"
    assert feed(engine2, [rows[12]])[0][2].status == "ok"


def test_gap_rows_are_inserted(pf_model):
    engine, rows = OnlineFeatureEngine(pf_model), _rows(30)
    feed(engine, rows[:10])
    res = feed(engine, [rows[14]])[0][2]
    assert res.status == "ok" and res.rows_inserted == 4 and not res.stream_reset


def test_very_long_gap_resets_state(pf_model):
    engine, rows = OnlineFeatureEngine(pf_model), _rows(100)
    feed(engine, rows)
    later = dict(rows[-1])
    later["timestamp"] = pd.Timestamp(later["timestamp"]) + pd.Timedelta(minutes=5 * (BUFFER + 10))
    res = feed(engine, [later])[0][2]
    assert res.stream_reset and not res.feature_ready     # history discarded: warming up again


def test_forward_fill_limit(pf_model):
    engine, rows = OnlineFeatureEngine(pf_model), _rows(40)
    feed(engine, rows[:10])
    blanks = [dict(r, pressure_kpa=None) for r in rows[10:10 + cfg.MAX_FFILL_SAMPLES + 2]]
    results = [r for _, _, r in feed(engine, blanks)]
    assert all(r.usable and r.n_filled == 1 for r in results[: cfg.MAX_FFILL_SAMPLES])
    assert all(r.values["pressure_kpa"] == rows[9]["pressure_kpa"] for r in results[: cfg.MAX_FFILL_SAMPLES])
    assert not results[cfg.MAX_FFILL_SAMPLES].usable       # beyond the limit: left missing, not guessed
    assert not results[cfg.MAX_FFILL_SAMPLES].feature_ready


def test_settling_window_after_state_change(pf_model):
    engine, rows = OnlineFeatureEngine(pf_model), _rows(120)
    for r in rows[100:]:
        r["operating_state"] = "HIGH_FLOW" if rows[99]["operating_state"] != "HIGH_FLOW" else "LOW_FLOW"
    results = [r for _, _, r in feed(engine, rows)]
    n = cfg.SETTLING_MINUTES // cfg.SAMPLE_MINUTES
    assert not results[99].in_settling
    assert all(r.in_settling and not r.monitored for r in results[100:100 + n])
    assert not results[100 + n].in_settling

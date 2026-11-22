import numpy as np
import pandas as pd

from pipeline_anomaly import config as cfg
from pipeline_anomaly.data_generation import SENTINEL_VALUE, generate_dataset


def test_schema_and_segments(generated):
    df, manifest = generated
    assert list(df.columns) == cfg.RAW_COLUMNS
    assert sorted(df["segment_id"].unique()) == manifest["segments"]
    assert manifest["synthetic"] is True
    assert set(df["operating_state"].unique()) <= set(cfg.OPERATING_STATES)


def test_generation_is_deterministic():
    a, _ = generate_dataset(days=5, seed=123)
    b, _ = generate_dataset(days=5, seed=123)
    c, _ = generate_dataset(days=5, seed=124)
    pd.testing.assert_frame_equal(a, b)
    assert not a["pressure_kpa"].equals(c["pressure_kpa"])


def test_row_count_matches_manifest(generated):
    df, manifest = generated
    d = manifest["defects"]
    expected = manifest["rows_before_defects"] - d["dropped_rows"] + d["duplicate_rows"]
    assert len(df) == expected == manifest["rows_written"]


def test_labels_are_consistent(generated):
    df, _ = generated
    anomalous = df["is_anomaly"] == 1
    assert set(df["is_anomaly"].unique()) <= {0, 1}
    assert (df.loc[anomalous, "anomaly_type"].isin(cfg.ANOMALY_TYPES)).all()
    assert (df.loc[~anomalous, "anomaly_type"] == "none").all()
    assert (df.loc[anomalous, "anomaly_event_id"] != "").all()
    assert (df.loc[anomalous, "anomaly_sensor"].isin(cfg.SENSOR_COLS)).all()
    # every anomaly type is represented, and anomalies stay a small minority
    assert set(df.loc[anomalous, "anomaly_type"]) == set(cfg.ANOMALY_TYPES)
    assert 0.005 < anomalous.mean() < 0.15


def test_no_anomalies_during_shutdown(generated):
    df, _ = generated
    assert not ((df["operating_state"] == "SHUTDOWN") & (df["is_anomaly"] == 1)).any()


def test_each_event_has_one_type_and_segment(generated):
    df, _ = generated
    ev = df[df["anomaly_event_id"] != ""].groupby("anomaly_event_id")
    assert (ev["anomaly_type"].nunique() == 1).all()
    assert (ev["segment_id"].nunique() == 1).all()


def test_clean_run_has_no_defects():
    df, manifest = generate_dataset(days=3, seed=1, with_defects=False)
    assert "defects" not in manifest
    assert not df[cfg.SENSOR_COLS].isna().any().any()
    assert not df.duplicated().any()
    assert (df["flow_m3h"] >= 0).all()
    assert not np.isclose(df[cfg.SENSOR_COLS], SENTINEL_VALUE).any()
    # exactly one row per segment per 5-minute step
    assert (df.groupby("segment_id")["timestamp"].diff().dropna() == pd.Timedelta(minutes=5)).all()

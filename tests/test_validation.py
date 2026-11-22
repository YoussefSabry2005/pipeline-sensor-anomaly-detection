import numpy as np
import pandas as pd

from pipeline_anomaly import config as cfg
from pipeline_anomaly.validation import clean, validate


def test_validate_finds_injected_defects(raw, manifest):
    report = validate(raw)
    d = manifest["defects"]
    assert report["n_rows"] == len(raw)
    assert report["exact_duplicate_rows"] == d["duplicate_rows"]
    assert sum(s["missing_timestamps"] for s in report["segments"].values()) == d["dropped_rows"]
    # sentinels on rows that were later dropped/duplicated shift the count slightly
    n_invalid = sum(report["invalid_values"].values())
    assert 0 < n_invalid <= d["sentinel_cells"] + d["duplicate_rows"]
    assert sum(report["missing_values"][c] for c in cfg.SENSOR_COLS) > 0
    assert report["label_inconsistencies"] == 0
    assert report["unknown_operating_states"] == 0
    assert report["unparseable_timestamps"] == 0


def test_validate_does_not_modify_input(raw):
    before = raw.copy()
    validate(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_produces_regular_grid(cleaned):
    df, log = cleaned
    assert not df.duplicated(subset=cfg.KEY_COLS).any()
    for _, g in df.groupby("segment_id"):
        assert (g["timestamp"].diff().dropna() == pd.Timedelta(minutes=cfg.SAMPLE_MINUTES)).all()
    assert log["rows_out"] == len(df)
    assert log["inserted_rows_for_missing_timestamps"] == int(df["row_inserted"].sum())


def test_clean_removes_invalid_values(cleaned):
    df, log = cleaned
    for c, (lo, hi) in cfg.VALID_RANGES.items():
        vals = df[c].dropna()
        assert ((vals >= lo) & (vals <= hi)).all()
    assert sum(log["invalid_values_set_missing"].values()) > 0


def test_clean_validates_clean(cleaned):
    report = validate(cleaned[0][cfg.RAW_COLUMNS])
    assert report["exact_duplicate_rows"] == 0
    assert report["duplicate_keys"] == 0
    assert sum(report["invalid_values"].values()) == 0
    assert all(s["missing_timestamps"] == 0 for s in report["segments"].values())


def _toy(n=40):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "segment_id": "SEG-X",
            "sensor_id": "STN-X",
            "pressure_kpa": np.linspace(4000, 4039, n),
            "flow_m3h": 800.0,
            "temperature_c": 10.0,
            "operating_state": "NORMAL",
            "is_anomaly": 0,
            "anomaly_type": "none",
            "anomaly_event_id": "",
            "anomaly_sensor": "",
        }
    )


def test_forward_fill_is_causal_and_limited():
    df = _toy()
    df.loc[5, "pressure_kpa"] = np.nan                      # short gap -> filled from the past
    df.loc[10:10 + cfg.MAX_FFILL_SAMPLES + 2, "pressure_kpa"] = np.nan  # long gap -> partly unusable
    out, log = clean(df)
    assert out.loc[5, "pressure_kpa"] == df.loc[4, "pressure_kpa"]  # past value, not an interpolation
    assert out.loc[5, "n_filled"] == 1 and out.loc[5, "usable"]
    assert out.loc[10 + cfg.MAX_FFILL_SAMPLES - 1, "usable"]
    assert not out.loc[10 + cfg.MAX_FFILL_SAMPLES, "usable"]
    assert log["rows_not_usable"] == 3


def test_missing_rows_are_inserted_and_flagged():
    df = _toy().drop(index=[7, 8])
    out, log = clean(df)
    assert len(out) == 40
    assert out["row_inserted"].sum() == 2 == log["inserted_rows_for_missing_timestamps"]
    assert (out.loc[out["row_inserted"], "is_anomaly"] == 0).all()


def test_sentinel_and_duplicates_handled():
    df = _toy()
    df.loc[3, "flow_m3h"] = -999.0
    df = pd.concat([df, df.iloc[[12]]], ignore_index=True)
    out, log = clean(df)
    assert log["dropped_exact_duplicates"] == 1
    assert log["invalid_values_set_missing"]["flow_m3h"] == 1
    assert out.loc[3, "flow_m3h"] == 800.0

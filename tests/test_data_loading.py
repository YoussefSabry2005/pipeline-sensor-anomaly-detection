import pandas as pd
import pytest

from pipeline_anomaly import config as cfg
from pipeline_anomaly.data_loading import SchemaError, load_raw


def test_loaded_types(raw):
    assert list(raw.columns) == cfg.RAW_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(raw["timestamp"])
    for c in cfg.SENSOR_COLS:
        assert pd.api.types.is_float_dtype(raw[c])
    assert pd.api.types.is_integer_dtype(raw["is_anomaly"])
    # label columns never contain NaN after loading
    assert not raw[cfg.LABEL_COLS].isna().any().any()


def test_round_trip_preserves_rows(raw, generated):
    assert len(raw) == len(generated[0])
    assert raw["is_anomaly"].sum() == generated[0]["is_anomaly"].sum()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.csv")


def test_missing_column_raises(tmp_path, generated):
    path = tmp_path / "bad.csv"
    generated[0].head(50).drop(columns=["flow_m3h"]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="flow_m3h"):
        load_raw(path)


def test_unparseable_values_become_missing(tmp_path, generated):
    df = generated[0].head(20).copy().astype({"pressure_kpa": object, "timestamp": object})
    df.loc[df.index[3], "pressure_kpa"] = "ERR"
    df.loc[df.index[5], "timestamp"] = "not a date"
    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False)
    out = load_raw(path)
    assert pd.isna(out.loc[3, "pressure_kpa"])
    assert pd.isna(out.loc[5, "timestamp"])

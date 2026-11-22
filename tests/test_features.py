import numpy as np
import pandas as pd
import pytest

from pipeline_anomaly import config as cfg
from pipeline_anomaly.features import MODEL_FEATURES, PressureFlowModel, build_features
from pipeline_anomaly.splits import assign_split, split_boundaries


def test_feature_columns_exist(features):
    for f in MODEL_FEATURES:
        assert f in features.columns
    for name in ["pressure", "flow", "temperature"]:
        for suffix in ["lag_1", "lag_1h", "diff_1", "roll_mean_1h", "roll_std_1h", "roll_mean_6h", "z_6h"]:
            assert f"{name}_{suffix}" in features.columns
    assert {"hour_sin", "hour_cos", "day_of_week", "minutes_since_state_change"} <= set(features.columns)


def test_row_count_and_order_preserved(features, cleaned):
    assert len(features) == len(cleaned[0])
    assert features.groupby("segment_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


def test_ready_rows_have_finite_features(model_frame):
    assert len(model_frame) > 0
    assert np.isfinite(model_frame[MODEL_FEATURES].to_numpy()).all()


def test_warmup_rows_are_not_ready(features):
    first = features.groupby("segment_id").head(cfg.SHORT_WINDOW)
    assert not first["feature_ready"].any()


def test_no_look_ahead(cleaned, pf_model):
    """Changing the future must not change any feature in the past."""
    df = cleaned[0]
    seg = df[df["segment_id"] == "SEG-A"].reset_index(drop=True)
    cut = len(seg) // 2
    altered = seg.copy()
    rng = np.random.default_rng(0)
    for c in cfg.SENSOR_COLS:
        altered.loc[cut:, c] = altered.loc[cut:, c] * 1.5 + rng.normal(0, 50, len(seg) - cut)
    altered.loc[cut:, "operating_state"] = "HIGH_FLOW"

    a = build_features(seg, pf_model).iloc[:cut]
    b = build_features(altered, pf_model).iloc[:cut]
    feature_cols = [c for c in a.columns if c not in seg.columns]
    pd.testing.assert_frame_equal(a[feature_cols], b[feature_cols])


def test_lag_and_diff_definitions(features):
    g = features[features["segment_id"] == "SEG-B"].reset_index(drop=True)
    i = 500
    assert g.loc[i, "pressure_lag_1"] == g.loc[i - 1, "pressure_kpa"]
    assert g.loc[i, "flow_lag_1h"] == g.loc[i - cfg.SHORT_WINDOW, "flow_m3h"]
    assert g.loc[i, "temperature_diff_1"] == pytest.approx(
        g.loc[i, "temperature_c"] - g.loc[i - 1, "temperature_c"]
    )
    window = g.loc[i - cfg.SHORT_WINDOW + 1 : i, "pressure_kpa"]
    assert g.loc[i, "pressure_roll_mean_1h"] == pytest.approx(window.mean())


def test_settling_flag_follows_state_changes(features):
    g = features[features["segment_id"] == "SEG-A"].reset_index(drop=True)
    change = g.index[(g["operating_state"] != g["operating_state"].shift()) & (g.index > 0)][0]
    n_settle = cfg.SETTLING_MINUTES // cfg.SAMPLE_MINUTES
    assert g.loc[change, "minutes_since_state_change"] == 0
    assert g.loc[change : change + n_settle - 1, "in_settling"].all()
    assert not g.loc[change + n_settle, "in_settling"]
    assert not g.loc[change - 1, "in_settling"] or change < n_settle
    assert not (g["in_settling"] & g["monitored"]).any()


def test_flatline_lowers_volatility_ratio(features):
    flat = features[(features["anomaly_type"] == "flatline") & features["feature_ready"]]
    normal = features[(features["is_anomaly"] == 0) & features["monitored"]]
    # use the volatility ratio of whichever sensor was frozen, late in each event
    last = flat.groupby("anomaly_event_id").tail(1)
    vals = [r[f"{r['anomaly_sensor'].split('_')[0]}_vol_ratio"] for _, r in last.iterrows()]
    typical = normal[[f"{n}_vol_ratio" for n in ["pressure", "flow", "temperature"]]].median().min()
    assert np.median(vals) < typical - 0.5


def test_pressure_flow_residual_centred_on_normal_data(features):
    normal = features[(features["is_anomaly"] == 0) & features["monitored"]]
    assert abs(normal["pf_resid_z"].median()) < 0.25
    dev = features[(features["anomaly_type"] == "pressure_flow_deviation") & features["monitored"]]
    assert dev["pf_resid_z"].median() < -2.0


def test_pressure_flow_model_unknown_segment_raises(pf_model, cleaned):
    df = cleaned[0].head(10).assign(segment_id="SEG-UNKNOWN")
    with pytest.raises(KeyError):
        pf_model.residual_z(df)


def test_pressure_flow_model_recovers_quadratic():
    q = np.linspace(100, 1000, 200)
    df = pd.DataFrame({"segment_id": "S", "flow_m3h": q, "pressure_kpa": 1000 + 0.003 * q**2})
    model = PressureFlowModel.fit(df)
    assert model.coefs["S"][0] == pytest.approx(0.003, rel=1e-6)
    assert model.coefs["S"][2] == pytest.approx(1000, rel=1e-6)


def test_split_is_chronological(features):
    val_start, test_start = split_boundaries(features["timestamp"])
    assert val_start < test_start
    assert features.loc[features["split"] == "train", "timestamp"].max() < val_start
    assert features.loc[features["split"] == "val", "timestamp"].max() < test_start
    assert features.loc[features["split"] == "test", "timestamp"].min() == test_start
    assert set(assign_split(features).unique()) == {"train", "val", "test"}


def test_split_rejects_bad_fractions(features):
    with pytest.raises(ValueError):
        split_boundaries(features["timestamp"], train_frac=0.9, val_frac=0.2)

import numpy as np
import pytest

from pipeline_anomaly import detectors as det
from pipeline_anomaly.features import MODEL_FEATURES


@pytest.fixture(scope="module")
def train(model_frame):
    return model_frame[(model_frame["split"] == "train") & model_frame["monitored"]]


@pytest.mark.parametrize("factory", [
    det.RollingZScoreDetector, det.IsolationForestDetector,
    lambda: det.LOFDetector(max_train_rows=3000), det.PCAReconstructionDetector,
])
def test_detector_contract(factory, train, model_frame):
    d = factory().fit(train)
    sample = model_frame.iloc[:2000]
    scores = d.score(sample)
    assert scores.shape == (len(sample),)
    assert np.isfinite(scores).all()

    # An observation that is extreme in every feature must outrank typical ones.
    extreme = train.iloc[[0]].copy()
    extreme[MODEL_FEATURES] = train[MODEL_FEATURES].median().to_numpy() + 50 * train[MODEL_FEATURES].std().to_numpy()
    assert d.score(extreme)[0] > np.quantile(d.score(train.iloc[:2000]), 0.99)


def test_isolation_forest_is_reproducible(train):
    a = det.IsolationForestDetector().fit(train).score(train.iloc[:500])
    b = det.IsolationForestDetector().fit(train).score(train.iloc[:500])
    np.testing.assert_allclose(a, b)


def test_missing_feature_column_raises(train):
    with pytest.raises(KeyError):
        det.IsolationForestDetector().fit(train.drop(columns=[MODEL_FEATURES[0]]))


def test_nan_features_rejected(train):
    bad = train.iloc[:100].copy()
    bad.iloc[0, bad.columns.get_loc(MODEL_FEATURES[0])] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        det.PCAReconstructionDetector().fit(train).score(bad)


def test_validate_scores():
    assert det.validate_scores([1, 2, 3]).dtype == float
    with pytest.raises(ValueError):
        det.validate_scores([1.0, np.nan])
    with pytest.raises(ValueError):
        det.validate_scores([1.0, np.inf])
    with pytest.raises(ValueError):
        det.validate_scores(np.ones((2, 2)))
    with pytest.raises(ValueError):
        det.validate_scores([1.0, 2.0], n_expected=3)


def test_threshold_matches_alert_budget():
    scores = np.random.default_rng(0).normal(size=50_000)
    thr = det.threshold_from_budget(scores, 0.03)
    assert det.flag(scores, thr).mean() == pytest.approx(0.03, abs=0.001)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            det.threshold_from_budget(scores, bad)


def test_flag_is_strict_and_binary():
    flags = det.flag([0.1, 0.5, 0.9], threshold=0.5)
    assert flags.tolist() == [0, 0, 1]
    assert flags.dtype.kind == "i"


def test_percentile_mapping():
    ref = np.arange(100, dtype=float)
    pct = det.to_percentile(ref, [-5, 0, 49.5, 99, 500])
    assert pct.min() >= 0 and pct.max() <= 1
    assert (np.diff(pct) >= 0).all()          # monotone in the score
    assert pct[0] == 0 and pct[-1] == 1


def test_ensemble_is_mean_of_percentiles():
    out = det.ensemble_scores({"a": np.array([0.0, 1.0]), "b": np.array([1.0, 1.0])})
    np.testing.assert_allclose(out, [0.5, 1.0])
    with pytest.raises(ValueError):
        det.ensemble_scores({})

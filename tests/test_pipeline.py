import numpy as np

from pipeline_anomaly import config as cfg
from pipeline_anomaly import pipeline


def test_scored_frame_has_all_models(scored, model_frame):
    assert len(scored.frame) == len(model_frame)
    for name, key in pipeline.MODEL_KEYS.items():
        for prefix in ("rawscore__", "score__", "pct__", "flag__"):
            assert f"{prefix}{key}" in scored.frame.columns
        assert np.isfinite(scored.frame[f"score__{key}"]).all()
        assert scored.frame[f"pct__{key}"].between(0, 1).all()
        assert set(scored.frame[f"flag__{key}"].unique()) <= {0, 1}
        assert name in scored.thresholds


def test_settling_rows_never_alert(scored):
    settling = scored.frame[~scored.frame["monitored"]]
    assert len(settling) > 0
    for key in pipeline.MODEL_KEYS.values():
        assert settling[f"flag__{key}"].sum() == 0


def test_training_alert_rate_equals_budget(scored):
    train = scored.frame[(scored.frame["split"] == "train") & scored.frame["monitored"]]
    for key in pipeline.MODEL_KEYS.values():
        assert abs(train[f"flag__{key}"].mean() - cfg.ALERT_BUDGET) < 0.005


def test_evaluate_outputs(scored):
    results = pipeline.evaluate(scored)
    comp = results["comparison"]
    assert list(comp["model"]) == list(pipeline.MODEL_KEYS)
    for col in ["precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc", "event_recall"]:
        assert comp[col].between(0, 1).all()
    assert (comp["tp"] + comp["fn"]).nunique() == 1          # same positives for every model
    assert (results["test"]["split"] == "test").all()        # evaluated on the test period only
    # detectors must beat random ranking on the simulated anomalies
    assert (comp.set_index("model").loc[["Isolation Forest", "Local Outlier Factor"], "roc_auc"] > 0.6).all()
    assert set(results["recall_by_type"]) == set(pipeline.MODEL_KEYS)
    assert len(results["feature_auc"]) > 0

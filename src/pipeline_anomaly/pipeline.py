"""End-to-end pipeline: data -> validation -> features -> detectors -> evaluation.

Run everything with::

    python -m pipeline_anomaly.pipeline

Each stage is also a plain function so the notebook and the tests can call
the stages individually.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg
from . import evaluation as ev
from .data_generation import write_dataset
from .data_loading import load_raw
from .detectors import (
    ENSEMBLE_NAME,
    BaseDetector,
    default_detectors,
    ensemble_scores,
    flag,
    threshold_from_budget,
    to_percentile,
)
from .features import CLASSIC_ZSCORE_FEATURES, MODEL_FEATURES, PressureFlowModel, build_features
from .splits import assign_split, split_boundaries
from .validation import clean, validate

MODEL_KEYS = {
    "Rolling z-score": "zscore",
    "Isolation Forest": "iforest",
    "Local Outlier Factor": "lof",
    "PCA reconstruction": "pca",
    ENSEMBLE_NAME: "ensemble",
}


@dataclass
class PreparedData:
    raw: pd.DataFrame
    validation_report: dict
    cleaning_log: dict
    features: pd.DataFrame          # every row of the regular grid
    pf_model: PressureFlowModel
    val_start: pd.Timestamp
    test_start: pd.Timestamp

    @property
    def model_frame(self) -> pd.DataFrame:
        """Rows that can be scored (real, complete, past warm-up)."""
        return self.features[self.features["feature_ready"]].reset_index(drop=True)


@dataclass
class ScoredData:
    frame: pd.DataFrame             # model_frame + score__/pct__/flag__ columns
    detectors: dict[str, BaseDetector]
    thresholds: dict[str, float]    # label-free alert-budget thresholds


# --------------------------------------------------------------------------
# Stage 1-3: data, validation, features
# --------------------------------------------------------------------------
def prepare_data(regenerate: bool = False) -> PreparedData:
    if regenerate or not cfg.RAW_FILE.exists():
        write_dataset(cfg.RAW_FILE)
    raw = load_raw(cfg.RAW_FILE)
    report = validate(raw)
    clean_df, log = clean(raw)

    split = assign_split(clean_df)
    val_start, test_start = split_boundaries(clean_df["timestamp"])
    train_rows = clean_df[(split == "train") & clean_df["usable"] & ~clean_df["row_inserted"]]
    pf_model = PressureFlowModel.fit(train_rows)  # fitted on the training period only

    feats = build_features(clean_df, pf_model)
    feats["split"] = assign_split(feats)
    return PreparedData(raw, report, log, feats, pf_model, val_start, test_start)


# --------------------------------------------------------------------------
# Stage 4: detectors
# --------------------------------------------------------------------------
def fit_and_score(model_frame: pd.DataFrame, alert_budget: float = cfg.ALERT_BUDGET) -> ScoredData:
    """Fit every detector on the monitored training rows and score all rows.

    Columns added per detector ``<key>``:

    * ``rawscore__<key>`` - the detector's score for every row.
    * ``score__<key>``    - the same, but rows inside a settling window are set
      to the lowest training score so they can never raise an alert.
    * ``pct__<key>``      - ``score`` as a percentile of the training scores.
    * ``flag__<key>``     - 1 if ``score`` exceeds the alert-budget threshold.
    """
    frame = model_frame.copy()
    monitored = frame["monitored"].to_numpy()
    fit_rows = ((frame["split"] == "train") & frame["monitored"]).to_numpy()
    train = frame[fit_rows]

    def _apply_mask(raw: np.ndarray) -> np.ndarray:
        return np.where(monitored, raw, raw[fit_rows].min())

    detectors, thresholds, percentiles, raw_percentiles = {}, {}, {}, {}
    for det in default_detectors():
        key = MODEL_KEYS[det.name]
        det.fit(train)
        raw = det.score(frame)
        scores = _apply_mask(raw)
        thresholds[det.name] = threshold_from_budget(raw[fit_rows], alert_budget)
        percentiles[det.name] = to_percentile(raw[fit_rows], scores)
        raw_percentiles[det.name] = to_percentile(raw[fit_rows], raw)
        frame[f"rawscore__{key}"] = raw
        frame[f"score__{key}"] = scores
        frame[f"pct__{key}"] = percentiles[det.name]
        frame[f"flag__{key}"] = flag(scores, thresholds[det.name])
        detectors[det.name] = det

    raw_ens = ensemble_scores(raw_percentiles)
    ens = _apply_mask(raw_ens)
    thresholds[ENSEMBLE_NAME] = threshold_from_budget(raw_ens[fit_rows], alert_budget)
    frame["rawscore__ensemble"] = raw_ens
    frame["score__ensemble"] = ens
    frame["pct__ensemble"] = to_percentile(raw_ens[fit_rows], ens)
    frame["flag__ensemble"] = flag(ens, thresholds[ENSEMBLE_NAME])
    return ScoredData(frame, detectors, thresholds)


# --------------------------------------------------------------------------
# Stage 5: evaluation
# --------------------------------------------------------------------------
def univariate_feature_auc(frame: pd.DataFrame) -> pd.DataFrame:
    """How well does each feature *alone* separate labelled anomalies (test)?

    Score = absolute distance from the feature's training-period median.
    """
    train = frame[(frame["split"] == "train") & frame["monitored"]]
    test = frame[(frame["split"] == "test") & frame["monitored"]]
    rows = []
    for f in MODEL_FEATURES + CLASSIC_ZSCORE_FEATURES:
        dist = (test[f] - train[f].median()).abs()
        rows.append({"feature": f, "roc_auc": float(roc_auc_score(test["is_anomaly"], dist))})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def evaluate(scored: ScoredData) -> dict:
    """Evaluate every detector on the held-out test period."""
    frame = scored.frame
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    y = test["is_anomaly"].to_numpy()

    comparison, by_type, fp_breakdown, tuned, suppressed, events = [], {}, {}, [], [], {}
    for name, key in MODEL_KEYS.items():
        scores, flags = test[f"score__{key}"].to_numpy(), test[f"flag__{key}"].to_numpy()
        ev_table = ev.event_metrics(test, flags)
        events[name] = ev_table
        row = {"model": name, "threshold": scored.thresholds[name]}
        row.update(ev.classification_metrics(y, flags))
        row.update(ev.ranking_metrics(y, scores))
        row["event_recall"] = float(ev_table["detected"].mean())
        row["median_delay_minutes"] = float(ev_table["delay_minutes"].median())
        comparison.append(row)

        by_type[name] = ev.recall_by_type(test, flags)
        fp_breakdown[name] = ev.false_positive_breakdown(test, flags)

        # Label-informed reference: threshold tuned on validation, applied to test.
        thr, val_f1 = ev.best_f1_threshold(val["is_anomaly"].to_numpy(), val[f"score__{key}"].to_numpy())
        t_flags = flag(scores, thr)
        t_row = {"model": name, "threshold": thr, "validation_f1": val_f1}
        t_row.update(ev.classification_metrics(y, t_flags))
        t_row["event_recall"] = float(ev.event_metrics(test, t_flags)["detected"].mean())
        tuned.append(t_row)

        # Ablation: same thresholds, but without the settling-window mask.
        s_flags = flag(test[f"rawscore__{key}"].to_numpy(), scored.thresholds[name])
        s_row = {"model": name}
        s_row.update(ev.classification_metrics(y, s_flags))
        s_row["event_recall"] = float(ev.event_metrics(test, s_flags)["detected"].mean())
        suppressed.append(s_row)

    return {
        "test": test,
        "comparison": pd.DataFrame(comparison),
        "comparison_tuned": pd.DataFrame(tuned),
        "comparison_no_settling_mask": pd.DataFrame(suppressed),
        "recall_by_type": by_type,
        "fp_breakdown": fp_breakdown,
        "events": events,
        "feature_auc": univariate_feature_auc(frame),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def build_model_bundle(scored: ScoredData, pf_model: PressureFlowModel) -> dict:
    """Everything needed to score new observations (used by the streaming system).

    ``score_reference`` holds a quantile grid of each detector's training
    scores so a new score can be expressed as a training percentile without
    shipping the training data.
    """
    frame = scored.frame
    fit_rows = (frame["split"] == "train") & frame["monitored"]
    grid = np.linspace(0.0, 1.0, 2001)
    reference = {
        name: np.quantile(frame.loc[fit_rows, f"rawscore__{key}"].to_numpy(), grid)
        for name, key in MODEL_KEYS.items()
        if name in scored.detectors
    }
    return {
        "detectors": scored.detectors,
        "thresholds": scored.thresholds,
        "score_reference": reference,
        "pressure_flow_model": pf_model,
        "model_features": MODEL_FEATURES,
        "alert_budget": cfg.ALERT_BUDGET,
        "note": "Trained on SYNTHETIC data for demonstration only.",
    }


def train_model_bundle(path=None, regenerate: bool = False) -> dict:
    """Train the detectors and save only the model bundle (no large CSV outputs)."""
    prep = prepare_data(regenerate=regenerate)
    scored = fit_and_score(prep.model_frame)
    bundle = build_model_bundle(scored, prep.pf_model)
    path = cfg.MODEL_BUNDLE_FILE if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def _dataset_summary(prep: PreparedData) -> dict:
    mf = prep.model_frame
    out = {
        "rows_raw": int(len(prep.raw)),
        "rows_clean_grid": int(len(prep.features)),
        "rows_modelled": int(len(mf)),
        "val_start": str(prep.val_start),
        "test_start": str(prep.test_start),
        "rows_in_settling_windows": int(mf["in_settling"].sum()),
        "anomalous_rows_in_settling_windows": int((mf["in_settling"] & (mf["is_anomaly"] == 1)).sum()),
        "splits": {},
    }
    for name, g in mf.groupby("split"):
        out["splits"][name] = {
            "rows": int(len(g)),
            "anomalous_rows": int(g["is_anomaly"].sum()),
            "prevalence": float(g["is_anomaly"].mean()),
            "events": int(g.loc[g["anomaly_event_id"] != "", "anomaly_event_id"].nunique()),
        }
    return out


def save_outputs(prep: PreparedData, scored: ScoredData, results: dict) -> dict:
    for d in (cfg.PROCESSED_DIR, cfg.MODELS_DIR, cfg.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    base_cols = cfg.RAW_COLUMNS + ["row_inserted", "n_filled", "usable"]
    prep.features[base_cols].to_csv(cfg.CLEAN_FILE, index=False)
    prep.features.to_csv(cfg.FEATURE_FILE, index=False)
    score_cols = [c for c in scored.frame.columns if c.startswith(("score__", "pct__", "flag__"))]
    scored.frame[cfg.KEY_COLS + ["split", "operating_state", "monitored"] + cfg.LABEL_COLS + score_cols].to_csv(
        cfg.SCORES_FILE, index=False
    )

    joblib.dump(build_model_bundle(scored, prep.pf_model), cfg.MODEL_BUNDLE_FILE)

    results["comparison"].to_csv(cfg.REPORTS_DIR / "model_comparison.csv", index=False)
    results["comparison_tuned"].to_csv(cfg.REPORTS_DIR / "model_comparison_validation_tuned.csv", index=False)
    results["comparison_no_settling_mask"].to_csv(
        cfg.REPORTS_DIR / "model_comparison_no_settling_mask.csv", index=False
    )
    results["feature_auc"].to_csv(cfg.REPORTS_DIR / "feature_univariate_auc.csv", index=False)
    by_type = pd.concat(
        [t.assign(model=name) for name, t in results["recall_by_type"].items()], ignore_index=True
    )
    by_type.to_csv(cfg.REPORTS_DIR / "recall_by_anomaly_type.csv", index=False)

    metrics = {
        "disclosure": "All data are synthetic. Metrics measure recovery of simulated anomaly "
        "patterns only and are not estimates of real-world performance.",
        "alert_budget": cfg.ALERT_BUDGET,
        "dataset": _dataset_summary(prep),
        "validation_report": prep.validation_report,
        "cleaning_log": prep.cleaning_log,
        "test_metrics_label_free_threshold": results["comparison"].to_dict(orient="records"),
        "test_metrics_validation_tuned_threshold": results["comparison_tuned"].to_dict(orient="records"),
        "test_metrics_without_settling_mask": results["comparison_no_settling_mask"].to_dict(
            orient="records"
        ),
        "false_positive_breakdown": results["fp_breakdown"],
        "recall_by_anomaly_type": by_type.to_dict(orient="records"),
        "feature_univariate_auc": results["feature_auc"].to_dict(orient="records"),
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        return str(o)

    text = json.dumps(metrics, indent=2, default=_default).replace("NaN", "null")
    (cfg.REPORTS_DIR / "metrics.json").write_text(text)
    return metrics


def run(regenerate: bool = True, make_figures: bool = True) -> tuple[PreparedData, ScoredData, dict]:
    prep = prepare_data(regenerate=regenerate)
    scored = fit_and_score(prep.model_frame)
    results = evaluate(scored)
    save_outputs(prep, scored, results)
    if make_figures:
        import matplotlib

        matplotlib.use("Agg")  # file output only; no display needed
        from .visualization import make_all_figures

        make_all_figures(prep, scored, results)
    return prep, scored, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full anomaly-detection pipeline.")
    parser.add_argument("--no-regenerate", action="store_true", help="reuse the existing raw CSV")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    prep, scored, results = run(regenerate=not args.no_regenerate, make_figures=not args.no_figures)
    pd.set_option("display.width", 200)
    cols = ["model", "precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc", "event_recall"]
    print("\nSYNTHETIC DATA - held-out test period, label-free alert-budget threshold")
    print(results["comparison"][cols].round(3).to_string(index=False))
    print("\nSame detectors, threshold tuned on validation labels (label-informed reference)")
    print(results["comparison_tuned"][["model", "precision", "recall", "f1", "event_recall"]].round(3).to_string(index=False))
    print("\nAblation: label-free alerts WITHOUT the settling-window mask")
    print(results["comparison_no_settling_mask"][["model", "precision", "recall", "f1", "event_recall"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

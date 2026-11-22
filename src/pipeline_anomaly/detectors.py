"""Anomaly detectors with one shared interface.

Every detector exposes ``fit(X)`` and ``score(X)`` where **a higher score
always means "more unusual"**. Turning scores into alerts is a separate,
explicit step (``threshold_from_budget`` + ``flag``) so the same thresholding
logic is applied to every method and comparisons stay fair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from . import config as cfg
from .features import MODEL_FEATURES, ZSCORE_FEATURES


# --------------------------------------------------------------------------
# Score handling utilities
# --------------------------------------------------------------------------
def validate_scores(scores, n_expected: int | None = None) -> np.ndarray:
    """Return scores as a clean 1-D float array or raise ``ValueError``."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Scores must be 1-D, got shape {arr.shape}")
    if n_expected is not None and len(arr) != n_expected:
        raise ValueError(f"Expected {n_expected} scores, got {len(arr)}")
    if not np.isfinite(arr).all():
        raise ValueError("Scores contain NaN or infinite values")
    return arr


def threshold_from_budget(train_scores, alert_budget: float = cfg.ALERT_BUDGET) -> float:
    """Label-free threshold: the (1 - budget) quantile of the training scores."""
    if not 0.0 < alert_budget < 1.0:
        raise ValueError("alert_budget must be strictly between 0 and 1")
    return float(np.quantile(validate_scores(train_scores), 1.0 - alert_budget))


def flag(scores, threshold: float) -> np.ndarray:
    """Binary alerts: 1 where the score is strictly above the threshold."""
    return (validate_scores(scores) > threshold).astype(int)


def to_percentile(reference_scores, scores) -> np.ndarray:
    """Map scores to [0, 1] by their rank within a reference (training) sample.

    Detectors produce scores on incomparable scales (z units, path lengths,
    density ratios, squared errors). Percentiles put them on a common scale
    before they are averaged into an ensemble.
    """
    ref = np.sort(validate_scores(reference_scores))
    return np.searchsorted(ref, validate_scores(scores), side="right") / len(ref)


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------
class BaseDetector:
    name = "base"
    feature_names: list[str] = MODEL_FEATURES

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"{self.name}: missing feature columns {missing}")
        arr = X[self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"{self.name}: features contain NaN/inf; filter on `feature_ready`")
        return arr

    def fit(self, X: pd.DataFrame) -> "BaseDetector":
        raise NotImplementedError

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RollingZScoreDetector(BaseDetector):
    """Statistical baseline: largest absolute rolling z-score across sensors.

    For each sensor the current reading is compared with the mean and standard
    deviation of the previous 6 hours. The anomaly score is the largest
    absolute z-score of the three sensors. Nothing is learned, so ``fit`` is a
    no-op; this is the classic control-chart idea applied per sensor.
    """

    name = "Rolling z-score"
    feature_names = ZSCORE_FEATURES

    def fit(self, X: pd.DataFrame) -> "RollingZScoreDetector":
        self._matrix(X)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return validate_scores(np.abs(self._matrix(X)).max(axis=1))


class _ScaledDetector(BaseDetector):
    """Shared preprocessing: robust scaling then clipping of extreme values.

    RobustScaler (median / IQR) is used because the training data itself
    contains anomalies; mean / std scaling would be distorted by them.
    Clipping stops a handful of extreme rows from dominating distance- and
    variance-based models.
    """

    def _prepare_fit(self, X: pd.DataFrame) -> np.ndarray:
        self.scaler_ = RobustScaler().fit(self._matrix(X))
        return self._prepare(X)

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.scaler_.transform(self._matrix(X)), -cfg.FEATURE_CLIP, cfg.FEATURE_CLIP)


class IsolationForestDetector(_ScaledDetector):
    """Isolation Forest: anomalies are easier to isolate with random splits.

    The model builds many random trees; observations that end up alone after
    only a few splits (short average path length) are scored as unusual.
    """

    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 300, max_samples: int = 512, seed: int = cfg.RANDOM_SEED):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",  # thresholding is done separately, see module docstring
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        self.model.fit(self._prepare_fit(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return validate_scores(-self.model.score_samples(self._prepare(X)))


class LOFDetector(_ScaledDetector):
    """Local Outlier Factor in novelty mode: compares local densities.

    A point is unusual if its neighbourhood is much sparser than the
    neighbourhoods of its nearest training points. Fitted on a random
    subsample of the training period to keep neighbour search tractable.
    """

    name = "Local Outlier Factor"

    def __init__(
        self,
        n_neighbors: int = 50,
        max_train_rows: int = cfg.LOF_MAX_TRAIN_ROWS,
        seed: int = cfg.RANDOM_SEED,
    ):
        self.max_train_rows = max_train_rows
        self.seed = seed
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1)

    def fit(self, X: pd.DataFrame) -> "LOFDetector":
        Z = self._prepare_fit(X)
        if len(Z) > self.max_train_rows:
            idx = np.random.default_rng(self.seed).choice(len(Z), self.max_train_rows, replace=False)
            Z = Z[np.sort(idx)]
        self.model.fit(Z)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return validate_scores(-self.model.score_samples(self._prepare(X)))


class PCAReconstructionDetector(_ScaledDetector):
    """PCA reconstruction error (the "Q / SPE statistic" of process monitoring).

    PCA learns the main directions in which the features normally vary
    together. An observation that cannot be rebuilt well from those directions
    breaks the usual correlation structure and receives a high score.
    """

    name = "PCA reconstruction"

    def __init__(self, explained_variance: float = 0.90, seed: int = cfg.RANDOM_SEED):
        self.model = PCA(n_components=explained_variance, svd_solver="full", random_state=seed)

    def fit(self, X: pd.DataFrame) -> "PCAReconstructionDetector":
        self.model.fit(self._prepare_fit(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._prepare(X)
        recon = self.model.inverse_transform(self.model.transform(Z))
        return validate_scores(((Z - recon) ** 2).sum(axis=1))


def default_detectors() -> list[BaseDetector]:
    return [
        RollingZScoreDetector(),
        IsolationForestDetector(),
        LOFDetector(),
        PCAReconstructionDetector(),
    ]


ENSEMBLE_NAME = "Rank-average ensemble"


def ensemble_scores(percentiles: dict[str, np.ndarray]) -> np.ndarray:
    """Average of the detectors' training-referenced percentile scores."""
    if not percentiles:
        raise ValueError("No scores supplied")
    stacked = np.vstack([validate_scores(v) for v in percentiles.values()])
    return stacked.mean(axis=0)

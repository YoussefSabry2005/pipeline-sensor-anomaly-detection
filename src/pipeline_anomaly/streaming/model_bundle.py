"""Loading (and, if needed, training) the model bundle used for streaming inference.

    python -m pipeline_anomaly.streaming.model_bundle      # train + save the bundle only
"""

from __future__ import annotations

from pathlib import Path

import joblib

from .. import config as cfg

REQUIRED_KEYS = {"detectors", "thresholds", "score_reference", "pressure_flow_model", "model_features"}


def load_bundle(path: Path | str = cfg.MODEL_BUNDLE_FILE, train_if_missing: bool = True) -> dict:
    """Load the bundle written by the batch pipeline.

    The streaming system uses the *same fitted objects* as the batch
    evaluation: detectors, thresholds and the pressure-flow model. If the file
    is missing or predates the streaming upgrade it is retrained from the
    committed synthetic dataset (about a minute).
    """
    path = Path(path)
    bundle = joblib.load(path) if path.exists() else None
    if bundle is None or not REQUIRED_KEYS <= set(bundle):
        if not train_if_missing:
            raise FileNotFoundError(f"No usable model bundle at {path}. Run `python -m pipeline_anomaly.pipeline`.")
        from ..pipeline import train_model_bundle

        bundle = train_model_bundle(path)
    return prepare_for_streaming(bundle)


def prepare_for_streaming(bundle: dict) -> dict:
    """Single-threaded inference: thread-pool start-up costs more than it saves on tiny batches."""
    for det in bundle["detectors"].values():
        model = getattr(det, "model", None)
        if model is not None and hasattr(model, "n_jobs"):
            model.n_jobs = 1
    return bundle


def main() -> None:
    from ..pipeline import train_model_bundle

    bundle = train_model_bundle(cfg.MODEL_BUNDLE_FILE)
    print(f"Saved model bundle with detectors: {', '.join(bundle['detectors'])} -> {cfg.MODEL_BUNDLE_FILE.name}")


if __name__ == "__main__":
    main()

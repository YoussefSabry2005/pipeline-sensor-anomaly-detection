"""Shared fixtures. Tests use a small, separately-seeded synthetic dataset so
they are fast and independent of the files committed under ``data/``."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline_anomaly import pipeline  # noqa: E402
from pipeline_anomaly.data_generation import generate_dataset  # noqa: E402
from pipeline_anomaly.data_loading import load_raw  # noqa: E402
from pipeline_anomaly.features import PressureFlowModel, build_features  # noqa: E402
from pipeline_anomaly.splits import assign_split  # noqa: E402
from pipeline_anomaly.validation import clean  # noqa: E402

TEST_DAYS = 30
TEST_SEED = 7


@pytest.fixture(scope="session")
def generated():
    return generate_dataset(days=TEST_DAYS, seed=TEST_SEED)


@pytest.fixture(scope="session")
def raw(generated, tmp_path_factory):
    """Raw data after a CSV round trip, exactly as the pipeline would read it."""
    path = tmp_path_factory.mktemp("raw") / "sensors.csv.gz"
    generated[0].to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return load_raw(path)


@pytest.fixture(scope="session")
def manifest(generated):
    return generated[1]


@pytest.fixture(scope="session")
def cleaned(raw):
    return clean(raw)


@pytest.fixture(scope="session")
def pf_model(cleaned):
    df = cleaned[0]
    train = df[(assign_split(df) == "train") & df["usable"] & ~df["row_inserted"]]
    return PressureFlowModel.fit(train)


@pytest.fixture(scope="session")
def features(cleaned, pf_model):
    feats = build_features(cleaned[0], pf_model)
    feats["split"] = assign_split(feats)
    return feats


@pytest.fixture(scope="session")
def model_frame(features):
    return features[features["feature_ready"]].reset_index(drop=True)


@pytest.fixture(scope="session")
def scored(model_frame):
    return pipeline.fit_and_score(model_frame)


# ---------------------------------------------------------------- streaming fixtures
@pytest.fixture(scope="session")
def bundle(scored, pf_model):
    """Model bundle built from the small test dataset (same structure as the real one)."""
    from pipeline_anomaly.streaming.model_bundle import prepare_for_streaming

    return prepare_for_streaming(pipeline.build_model_bundle(scored, pf_model))


@pytest.fixture()
def stream_settings(tmp_path):
    from pipeline_anomaly.streaming.settings import StreamSettings

    s = StreamSettings()
    s.broker_path = tmp_path / "broker.db"
    s.log_dir = tmp_path / "logs"
    s.alerts_file = tmp_path / "alerts.jsonl"
    s.stream_days, s.stream_seed, s.stream_rate, s.fault_rate = 3, 99, 0.0, 0.03
    s.stream_loop = False
    return s

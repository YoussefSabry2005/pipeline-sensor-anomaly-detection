"""Central configuration: paths, column names and modelling constants.

Everything that is a *decision* (window lengths, alert budget, split
fractions) lives here so that it is visible, documented and testable rather
than scattered through the code base.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_FILE = RAW_DIR / "synthetic_pipeline_sensors.csv.gz"
CLEAN_FILE = PROCESSED_DIR / "sensors_clean.csv"
FEATURE_FILE = PROCESSED_DIR / "sensors_features.csv"
SCORES_FILE = PROCESSED_DIR / "anomaly_scores.csv"
MODEL_BUNDLE_FILE = MODELS_DIR / "detectors.joblib"
RUNTIME_DIR = PROJECT_ROOT / "runtime"   # streaming broker file, logs, alerts (git-ignored)

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
SENSOR_COLS = ["pressure_kpa", "flow_m3h", "temperature_c"]
KEY_COLS = ["segment_id", "timestamp"]
LABEL_COLS = ["is_anomaly", "anomaly_type", "anomaly_event_id", "anomaly_sensor"]
RAW_COLUMNS = (
    ["timestamp", "segment_id", "sensor_id"]
    + SENSOR_COLS
    + ["operating_state"]
    + LABEL_COLS
)
OPERATING_STATES = ["NORMAL", "HIGH_FLOW", "LOW_FLOW", "SHUTDOWN"]
ANOMALY_TYPES = [
    "spike",
    "pressure_flow_deviation",
    "drift",
    "flatline",
    "noise_burst",
    "temperature_excursion",
]

# Plausible physical envelope of each (synthetic) instrument. Readings outside
# the envelope are treated as invalid *data*, not as process anomalies.
VALID_RANGES = {
    "pressure_kpa": (0.0, 10_000.0),
    "flow_m3h": (0.0, 2_500.0),
    "temperature_c": (-40.0, 80.0),
}

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
SAMPLE_MINUTES = 5
SAMPLES_PER_HOUR = 60 // SAMPLE_MINUTES

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
# Gaps are forward-filled (causal: uses only the past) for at most this many
# samples. Longer gaps stay missing and are excluded from modelling.
MAX_FFILL_SAMPLES = 12  # one hour

# --------------------------------------------------------------------------
# Feature engineering (all windows are trailing / causal)
# --------------------------------------------------------------------------
SHORT_WINDOW = 1 * SAMPLES_PER_HOUR   # 1 h
LONG_WINDOW = 6 * SAMPLES_PER_HOUR    # 6 h
TREND_LAG = 3 * SAMPLES_PER_HOUR      # 3 h
WARMUP_SAMPLES = LONG_WINDOW + 1      # rows per segment without a full window

# Floor on the rolling standard deviation used in z-scores. Without a floor,
# a perfectly quiet window (e.g. shutdown) makes tiny wiggles look enormous.
# Values are roughly half the nominal instrument noise in the simulator and
# play the role of an "instrument resolution" an engineer would supply.
SIGMA_FLOOR = {"pressure_kpa": 6.0, "flow_m3h": 3.0, "temperature_c": 0.08}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20          # remainder (0.20) is the held-out test period

# "Alert budget": the share of training observations an analyst is willing to
# review. The decision threshold of every detector is the (1 - budget)
# quantile of its own training scores, so thresholds need no labels.
ALERT_BUDGET = 0.03

# Settling window: for this long after a commanded operating-state change the
# sensors are *expected* to move, so those rows are not monitored (they are
# excluded from training and can never raise an alert). Anomalies that fall
# inside a settling window therefore count as missed in the evaluation.
SETTLING_MINUTES = 60

SIMULATION_DAYS = 180

LOF_MAX_TRAIN_ROWS = 20_000
FEATURE_CLIP = 10.0      # robust-scaled features are clipped to +/- this value

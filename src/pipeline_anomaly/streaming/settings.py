"""Runtime settings for the streaming simulation, read from environment variables.

Every value has a safe default, so nothing needs to be configured to run the
demo. There are no credentials or secrets in this system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import config as cfg

# Topic names
TOPIC_READINGS = "sensor.readings"      # producer -> processor
TOPIC_TRUTH = "sensor.truth"            # simulator ground truth; never read by the processor
TOPIC_SCORED = "sensor.scored"          # one record per processed observation
TOPIC_ALERTS = "sensor.alerts"          # structured alerts
TOPIC_DEADLETTER = "sensor.deadletter"  # messages that could not be processed
TOPIC_METRICS = "system.metrics"        # periodic monitoring snapshots
ALL_TOPICS = [TOPIC_READINGS, TOPIC_TRUTH, TOPIC_SCORED, TOPIC_ALERTS, TOPIC_DEADLETTER, TOPIC_METRICS]

CONSUMER_GROUP = "anomaly-processor"
SCHEMA_VERSION = 1

# Severity bands for alerts, expressed as percentiles of the TRAINING score
# distribution. These are documentation thresholds for this simulation; they
# are not operational or safety severities.
SEVERITY_BANDS = [("high", 0.997), ("medium", 0.99), ("low", 0.0)]

# Flagged observations less than this many (simulated) minutes apart on the
# same segment are grouped into one review episode.
EPISODE_GAP_MINUTES = 30


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass
class StreamSettings:
    broker_path: Path = field(default_factory=lambda: Path(_env("BROKER_PATH", cfg.RUNTIME_DIR / "broker.db")))
    log_dir: Path = field(default_factory=lambda: Path(_env("LOG_DIR", cfg.RUNTIME_DIR / "logs")))
    alerts_file: Path = field(default_factory=lambda: Path(_env("ALERTS_FILE", cfg.RUNTIME_DIR / "alerts.jsonl")))
    model_path: Path = field(default_factory=lambda: Path(_env("MODEL_PATH", cfg.MODEL_BUNDLE_FILE)))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # Simulator / producer
    stream_seed: int = field(default_factory=lambda: _env("STREAM_SEED", 2026, int))
    stream_days: int = field(default_factory=lambda: _env("STREAM_DAYS", 30, int))
    stream_start: str = field(default_factory=lambda: _env("STREAM_START", "2025-07-01"))
    # Messages per wall-clock second. Each message is one 5-minute reading of
    # one segment, so simulated time runs much faster than real time.
    stream_rate: float = field(default_factory=lambda: _env("STREAM_RATE", 40.0, float))
    fault_rate: float = field(default_factory=lambda: _env("FAULT_RATE", 0.01, float))
    stream_loop: bool = field(default_factory=lambda: _env("STREAM_LOOP", False, bool))

    # Processor
    detectors: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            d.strip() for d in _env("STREAM_DETECTORS", "Isolation Forest,Local Outlier Factor").split(",") if d.strip()
        )
    )
    max_batch: int = field(default_factory=lambda: _env("MAX_BATCH", 64, int))
    poll_interval_s: float = field(default_factory=lambda: _env("POLL_INTERVAL_S", 0.02, float))
    metrics_interval_s: float = field(default_factory=lambda: _env("METRICS_INTERVAL_S", 1.0, float))

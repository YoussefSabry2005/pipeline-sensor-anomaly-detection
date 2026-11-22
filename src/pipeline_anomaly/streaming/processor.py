"""Stream processor: consume -> validate -> features -> inference -> alerts.

    python -m pipeline_anomaly.streaming.processor

Messages are handled in **micro-batches**: features are updated one message at
a time (they are stateful and cheap), then every scorable observation in the
batch is scored with one call per detector. A scikit-learn call has a fixed
overhead of several milliseconds regardless of the number of rows, so scoring
a batch together is what makes high throughput possible; at low message rates
batches simply contain one or a few messages.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..features import MODEL_FEATURES
from .alerts import EpisodeTracker, build_alert, top_deviations
from .broker import Broker, Message, SQLiteBroker
from .logging_utils import get_logger, setup_logging
from .messages import MessageRejected, validate_message
from .metrics import StreamMetrics
from .model_bundle import load_bundle
from .online_features import OnlineFeatureEngine
from .settings import (
    CONSUMER_GROUP, TOPIC_ALERTS, TOPIC_DEADLETTER, TOPIC_METRICS, TOPIC_READINGS, TOPIC_SCORED, StreamSettings,
)

log = get_logger("processor")


def _json_safe(obj):
    """Replace NaN / inf with None and NumPy scalars with Python ones."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class StreamProcessor:
    def __init__(self, bundle: dict, settings: StreamSettings | None = None, metrics: StreamMetrics | None = None):
        self.settings = settings or StreamSettings()
        self.metrics = metrics or StreamMetrics()
        missing = [d for d in self.settings.detectors if d not in bundle["detectors"]]
        if missing:
            raise KeyError(f"Detectors not in model bundle: {missing}")
        if list(bundle["model_features"]) != MODEL_FEATURES:
            raise ValueError("Model bundle was trained with a different feature list; retrain it.")
        self.detectors = {name: bundle["detectors"][name] for name in self.settings.detectors}
        self.thresholds = {name: float(bundle["thresholds"][name]) for name in self.detectors}
        self.reference = {name: np.asarray(bundle["score_reference"][name]) for name in self.detectors}
        self.features = OnlineFeatureEngine(bundle["pressure_flow_model"])
        self.episodes = EpisodeTracker()
        # Robust centre / scale of the training features, used only to explain alerts.
        scaler = next((d.scaler_ for d in bundle["detectors"].values() if hasattr(d, "scaler_")), None)
        self._center = scaler.center_ if scaler is not None else np.zeros(len(MODEL_FEATURES))
        self._scale = scaler.scale_ if scaler is not None else np.ones(len(MODEL_FEATURES))

    # ------------------------------------------------------------------
    def _percentile(self, name: str, score: float) -> float:
        ref = self.reference[name]
        return float(np.searchsorted(ref, score, side="right") / len(ref))

    def _deadletter(self, msg: Message, reason: str, detail: str = "") -> tuple:
        self.metrics.reject(reason)
        log.warning("message_rejected", extra={"fields": {"offset": msg.offset, "reason": reason, "detail": detail}})
        record = {"reason": reason, "detail": detail, "source_offset": msg.offset, "payload": msg.raw[:500],
                  "rejected_at": time.time(), "simulated": True}
        return (TOPIC_DEADLETTER, record, msg.key)

    def process_batch(self, messages: list[Message]) -> list[tuple]:
        """Process one micro-batch and return ``(topic, record, key)`` outputs to publish."""
        m = self.metrics
        t_batch = time.perf_counter()
        outputs: list[tuple] = []
        pending: list[dict] = []

        # Stage 1 - validate and update features, one message at a time (order matters)
        for msg in messages:
            m.incr("messages_received")
            t0 = time.perf_counter()
            try:
                if msg.value is None:
                    raise MessageRejected("invalid_json")
                reading = validate_message(msg.value, self.features.known_segments)
                log.debug("observation_received", extra={"fields": {
                    "offset": msg.offset, "segment_id": reading.segment_id, "timestamp": str(reading.timestamp)}})
                result = self.features.update(reading)
            except MessageRejected as exc:
                outputs.append(self._deadletter(msg, exc.reason, exc.detail))
                continue
            except Exception as exc:  # a bad message must never stop the stream
                m.incr("processing_errors")
                log.error("processing_error", exc_info=True, extra={"fields": {"offset": msg.offset}})
                outputs.append(self._deadletter(msg, "processing_error", type(exc).__name__))
                continue

            if result.status != "ok":
                m.incr("duplicates_dropped" if result.status == "duplicate" else "late_dropped")
                log.info("observation_skipped", extra={"fields": {
                    "offset": msg.offset, "segment_id": reading.segment_id, "timestamp": str(reading.timestamp),
                    "reason": result.status}})
                continue
            m.incr("observations_processed")
            m.incr("gap_rows_inserted", result.rows_inserted)
            m.incr("values_forward_filled", result.n_filled)
            feature_ms = 1000 * (time.perf_counter() - t0)
            m.observe("feature_ms", feature_ms)
            pending.append({"msg": msg, "reading": reading, "result": result, "feature_ms": feature_ms})

        # Stage 2 - score every monitored observation of the batch together
        scorable = [p for p in pending if p["result"].monitored]
        inference_ms, scores = 0.0, {}
        if scorable:
            X = pd.DataFrame(np.vstack([p["result"].features for p in scorable]), columns=MODEL_FEATURES)
            t0 = time.perf_counter()
            try:
                scores = {name: det.score(X) for name, det in self.detectors.items()}
            except Exception:
                m.incr("processing_errors")
                log.error("inference_error", exc_info=True, extra={"fields": {"batch_size": len(scorable)}})
                scores = {}
            inference_ms = 1000 * (time.perf_counter() - t0)
            for i, p in enumerate(scorable):
                p["scores"] = {name: float(s[i]) for name, s in scores.items()} if scores else None

        # Stage 3 - build outputs
        processing_ms = 1000 * (time.perf_counter() - t_batch)
        now = time.time()
        for p in pending:
            reading, result = p["reading"], p["result"]
            latency = {
                "feature_ms": round(p["feature_ms"], 3),
                "inference_ms": round(inference_ms, 3) if "scores" in p else 0.0,
                "processing_ms": round(processing_ms, 3),
                "end_to_end_ms": round(1000 * (now - reading.produced_at), 3) if reading.produced_at else None,
                "batch_size": len(messages),
            }
            m.observe("processing_ms", processing_ms)
            if latency["end_to_end_ms"] is not None:
                m.observe("end_to_end_ms", latency["end_to_end_ms"])

            if not result.usable:
                status = "not_usable"
            elif not result.feature_ready:
                status = "warming_up"
            elif result.in_settling:
                status = "settling"
            elif p.get("scores") is None:
                status = "inference_error"
            else:
                status = "scored"
            if status in ("not_usable", "warming_up"):
                m.incr(status)
            elif status == "settling":
                m.incr("in_settling")

            record = {
                "timestamp": reading.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "segment_id": reading.segment_id,
                "sensor_id": reading.sensor_id,
                "operating_state": reading.operating_state,
                "readings": result.values,
                "status": status,
                "flagged": False,
                "anomaly_score_percentile": None,
                "severity": None,
                "alert_id": None,
                "detectors": {},
                "data_quality": {"issues": reading.issues, "values_forward_filled": result.n_filled,
                                 "gap_rows_inserted": result.rows_inserted},
                "latency_ms": latency,
                "source_offset": p["msg"].offset,
                "simulated": True,
            }

            if status == "scored":
                m.incr("observations_scored")
                m.observe("inference_ms", inference_ms)
                det = {
                    name: {"score": s, "percentile": self._percentile(name, s),
                           "threshold": self.thresholds[name], "flagged": bool(s > self.thresholds[name])}
                    for name, s in p["scores"].items()
                }
                record["detectors"] = det
                record["anomaly_score_percentile"] = max(d["percentile"] for d in det.values())
                record["flagged"] = any(d["flagged"] for d in det.values())
                if record["flagged"]:
                    m.incr("observations_flagged")
                    alert = build_alert(
                        timestamp=reading.timestamp, segment_id=reading.segment_id, sensor_id=reading.sensor_id,
                        operating_state=reading.operating_state, readings=result.values, detector_results=det,
                        deviations=top_deviations(result.features, self._center, self._scale),
                        episode=self.episodes.assign(reading.segment_id, reading.timestamp), latency_ms=latency,
                    )
                    record["severity"], record["alert_id"] = alert["severity"], alert["alert_id"]
                    outputs.append((TOPIC_ALERTS, _json_safe(alert), reading.segment_id))
                    m.incr("alerts_created")
                    log.info("alert_created", extra={"fields": {
                        "alert_id": alert["alert_id"], "segment_id": reading.segment_id,
                        "severity": alert["severity"], "anomaly_score_percentile": alert["anomaly_score_percentile"],
                        "detection_methods": alert["detection_methods"], "episode_id": alert["episode_id"]}})

            log.info("observation_scored" if status == "scored" else "observation_not_scored", extra={"fields": {
                "segment_id": reading.segment_id, "timestamp": record["timestamp"], "status": status,
                "anomaly_score_percentile": record["anomaly_score_percentile"], "flagged": record["flagged"],
                "inference_ms": latency["inference_ms"], "processing_ms": latency["processing_ms"],
                "end_to_end_ms": latency["end_to_end_ms"]}})
            outputs.append((TOPIC_SCORED, _json_safe(record), reading.segment_id))

        m.batch_done(len(messages), time.perf_counter() - t_batch)
        return outputs


def run_processor(
    broker: Broker,
    processor: StreamProcessor,
    stop_event: threading.Event | None = None,
    idle_timeout_s: float | None = None,
) -> dict:
    """Poll loop. Stops on ``stop_event`` or after ``idle_timeout_s`` without new messages."""
    settings, metrics = processor.settings, processor.metrics
    stop_event = stop_event or threading.Event()
    settings.alerts_file.parent.mkdir(parents=True, exist_ok=True)
    last_metrics, last_message, received_at_snapshot = 0.0, time.time(), -1
    log.info("processor_started", extra={"fields": {
        "detectors": list(processor.detectors), "max_batch": settings.max_batch, "simulated": True}})

    def publish_metrics(status: str) -> dict:
        snap = metrics.snapshot(status, lag=broker.lag(TOPIC_READINGS, CONSUMER_GROUP))
        broker.publish(TOPIC_METRICS, snap)
        return snap

    # One math thread: scoring tiny batches with a thread pool is slower and far
    # less predictable (threads spin while waiting) than doing it serially.
    with threadpool_limits(limits=1), open(settings.alerts_file, "a", encoding="utf-8") as alerts_out:
        while not stop_event.is_set():
            batch = broker.poll(TOPIC_READINGS, CONSUMER_GROUP, settings.max_batch)
            if batch:
                last_message = time.time()
                outputs = processor.process_batch(batch)
                t0 = time.perf_counter()
                # outputs and the offset commit are written atomically
                broker.publish_and_commit(outputs, TOPIC_READINGS, CONSUMER_GROUP, batch[-1].offset)
                metrics.observe("publish_ms", 1000 * (time.perf_counter() - t0))
                for topic, record, _ in outputs:
                    if topic == TOPIC_ALERTS:
                        alerts_out.write(json.dumps(record) + "\n")
                alerts_out.flush()
            else:
                if idle_timeout_s is not None and time.time() - last_message > idle_timeout_s:
                    break
                time.sleep(settings.poll_interval_s)
            if time.time() - last_metrics >= settings.metrics_interval_s:
                received = metrics.counts["messages_received"]
                publish_metrics("running" if received > received_at_snapshot else "idle")
                received_at_snapshot, last_metrics = received, time.time()

    snap = publish_metrics("stopped")
    log.info("processor_stopped", extra={"fields": {"counts": snap["counts"], "throughput": snap["throughput"]}})
    return snap


def main() -> None:
    settings = StreamSettings()
    parser = argparse.ArgumentParser(description="Run the streaming anomaly-detection processor.")
    parser.add_argument("--idle-timeout", type=float, default=None, help="exit after this many seconds without messages")
    args = parser.parse_args()
    setup_logging(settings.log_dir, settings.log_level, "processor")
    processor = StreamProcessor(load_bundle(settings.model_path), settings)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run_processor(SQLiteBroker(settings.broker_path), processor, stop, args.idle_timeout)


if __name__ == "__main__":
    main()

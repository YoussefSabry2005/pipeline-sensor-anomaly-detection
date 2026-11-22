"""Streaming producer: publishes simulated readings to the broker at a set rate.

    python -m pipeline_anomaly.streaming.producer --days 7 --rate 40
"""

from __future__ import annotations

import argparse
import threading
import time

import pandas as pd

from .. import config as cfg
from .broker import Broker, SQLiteBroker
from .logging_utils import get_logger, setup_logging
from .settings import TOPIC_READINGS, TOPIC_TRUTH, StreamSettings
from .simulator import FaultInjector, SensorSimulator, build_messages

log = get_logger("producer")


def run_producer(
    broker: Broker,
    settings: StreamSettings,
    stop_event: threading.Event | None = None,
    max_messages: int | None = None,
) -> dict:
    """Publish the simulated stream. Returns counts of what was sent.

    ``settings.stream_rate`` is in messages per wall-clock second; 0 means
    "as fast as possible" (used to measure processor capacity).
    """
    stop_event = stop_event or threading.Event()
    stats = {"simulated_rows": 0, "messages_published": 0, "faults": {}}
    started = time.time()
    tick = 0.05
    per_tick = max(1, int(round(settings.stream_rate * tick))) if settings.stream_rate > 0 else 500
    cycle = 0
    log.info("producer_started", extra={"fields": {
        "rate_msgs_per_s": settings.stream_rate, "days": settings.stream_days, "seed": settings.stream_seed,
        "fault_rate": settings.fault_rate, "simulated": True}})

    # Continue the simulated timeline after whatever is already in the broker, so
    # a restarted producer (or the next loop cycle) never re-sends old timestamps.
    start = pd.Timestamp(settings.stream_start)
    last = broker.tail(TOPIC_TRUTH, 1)
    if last and isinstance(last[-1].value, dict):
        start = max(start, pd.Timestamp(last[-1].value["timestamp"]) + pd.Timedelta(minutes=cfg.SAMPLE_MINUTES))

    while not stop_event.is_set():
        period = int((start - pd.Timestamp(settings.stream_start)) / pd.Timedelta(days=1))
        seed = settings.stream_seed + period          # a different seed for every simulated period
        sim = SensorSimulator(settings.stream_days, seed, str(start))
        injector = FaultInjector(settings.fault_rate, seed=seed)
        start = start + pd.Timedelta(days=settings.stream_days)
        batch: list = []
        next_tick = time.perf_counter()
        for seq, (reading, truth) in enumerate(sim):
            if stop_event.is_set() or (max_messages is not None and stats["simulated_rows"] >= max_messages):
                break
            stats["simulated_rows"] += 1
            batch.append((TOPIC_TRUTH, truth, truth["segment_id"]))
            for payload in build_messages(reading, seq, time.time(), injector):
                batch.append((TOPIC_READINGS, payload, reading["segment_id"]))
                stats["messages_published"] += 1
            if (seq + 1) % per_tick == 0:
                broker.publish_many(batch)
                batch = []
                if settings.stream_rate > 0:
                    next_tick += per_tick / settings.stream_rate
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        broker.publish_many(batch)
        for kind, n in injector.counts.items():
            stats["faults"][kind] = stats["faults"].get(kind, 0) + n
        cycle += 1
        if not settings.stream_loop or (max_messages is not None and stats["simulated_rows"] >= max_messages):
            break

    stats["duration_s"] = round(time.time() - started, 3)
    log.info("producer_finished", extra={"fields": stats})
    return stats


def main() -> None:
    settings = StreamSettings()
    parser = argparse.ArgumentParser(description="Publish simulated sensor readings to the local broker.")
    parser.add_argument("--days", type=int, default=settings.stream_days)
    parser.add_argument("--rate", type=float, default=settings.stream_rate, help="messages per second (0 = unthrottled)")
    parser.add_argument("--seed", type=int, default=settings.stream_seed)
    parser.add_argument("--fault-rate", type=float, default=settings.fault_rate)
    parser.add_argument("--loop", action="store_true", default=settings.stream_loop, help="start a new simulated period when one ends")
    args = parser.parse_args()
    settings.stream_days, settings.stream_rate, settings.stream_seed = args.days, args.rate, args.seed
    settings.fault_rate, settings.stream_loop = args.fault_rate, args.loop
    setup_logging(settings.log_dir, settings.log_level, "producer")
    run_producer(SQLiteBroker(settings.broker_path), settings)


if __name__ == "__main__":
    main()

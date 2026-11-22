"""Run the whole streaming simulation locally in one command.

    python -m pipeline_anomaly.streaming.run_demo            # finite run + measured summary
    python -m pipeline_anomaly.streaming.run_demo --live     # keep streaming for the dashboard

The finite run has two phases, each on a fresh broker file:

1. **Paced** - the producer sends at a fixed rate while the processor consumes
   concurrently. This measures latency when the processor keeps up.
2. **Backlog** - all messages are published first, then drained as fast as
   possible. This measures throughput capacity with full micro-batches.

Results are written to ``reports/streaming_run_summary.json`` / ``.md`` and a
few example alerts to ``reports/sample_alerts.json``. Every number is measured
in this process on this machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .. import config as cfg
from .broker import SQLiteBroker
from .logging_utils import get_logger, setup_logging
from .model_bundle import load_bundle
from .processor import StreamProcessor, run_processor
from .producer import run_producer
from .settings import ALL_TOPICS, TOPIC_ALERTS, StreamSettings
from .stream_eval import evaluate_stream, read_all

log = get_logger("demo")


def _fresh_broker(path: Path) -> SQLiteBroker:
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    return SQLiteBroker(path)


def _start_producer(settings: StreamSettings) -> subprocess.Popen:
    """Run the producer as a separate OS process, as it would be in a real deployment."""
    env = dict(os.environ, BROKER_PATH=str(settings.broker_path), LOG_DIR=str(settings.log_dir),
               LOG_LEVEL=settings.log_level, PYTHONPATH=str(cfg.PROJECT_ROOT / "src"))
    cmd = [sys.executable, "-m", "pipeline_anomaly.streaming.producer", "--days", str(settings.stream_days),
           "--rate", str(settings.stream_rate), "--seed", str(settings.stream_seed),
           "--fault-rate", str(settings.fault_rate)] + (["--loop"] if settings.stream_loop else [])
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def run_paced(settings: StreamSettings, bundle: dict, broker: SQLiteBroker, live: bool = False) -> dict:
    """Producer (separate process) and processor (this process) run concurrently on one broker file."""
    processor = StreamProcessor(bundle, settings)
    stop = threading.Event()
    result: dict = {}
    producer = _start_producer(settings)
    if live:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())

    def watch_producer() -> None:
        out, _ = producer.communicate()
        for line in reversed(out.splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "producer_finished":
                result["producer"] = {k: event[k] for k in ("simulated_rows", "messages_published", "faults", "duration_s")}
                break
        result["producer_done_at"] = time.time()

    watcher = threading.Thread(target=watch_producer, daemon=True)
    watcher.start()
    try:
        # the processor stops 3 s after the producer has finished and the topic is drained
        idle = None if live else 3.0
        while True:
            metrics = run_processor(broker, processor, stop, idle_timeout_s=idle)
            if live or stop.is_set() or not watcher.is_alive():
                break
        result["metrics"] = metrics
    finally:
        if producer.poll() is None:
            producer.terminate()
        watcher.join(timeout=10)
    result.pop("producer_done_at", None)
    return result


def run_backlog(settings: StreamSettings, bundle: dict, broker: SQLiteBroker) -> dict:
    """Publish everything first, then drain: throughput with full batches."""
    producer_stats = run_producer(broker, settings)
    processor = StreamProcessor(bundle, settings)
    metrics = run_processor(broker, processor, idle_timeout_s=0.5)
    return {"producer": producer_stats, "metrics": metrics}


def _fmt(v, digits=2):
    return "n/a" if v is None else f"{v:,.{digits}f}"


def write_summary(summary: dict) -> None:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORTS_DIR / "streaming_run_summary.json").write_text(json.dumps(summary, indent=2))

    paced, backlog, evaln = summary["paced_run"], summary.get("backlog_run"), summary["evaluation"]
    pm, lat, cnt = paced["metrics"], paced["metrics"]["latency_ms"], paced["metrics"]["counts"]
    lines = [
        "# Streaming run summary",
        "",
        "> Simulated data, local run. Latency and throughput were measured on the machine below and will differ",
        "> elsewhere. Detection metrics are against the simulator's labels only. Generated by",
        "> `python -m pipeline_anomaly.streaming.run_demo`.",
        "",
        f"* Machine: {summary['environment']['machine']} · Python {summary['environment']['python']}",
        f"* Stream: {summary['stream']['days']} simulated days, seed {summary['stream']['seed']} "
        f"(training seed {cfg.RANDOM_SEED}), {summary['stream']['fault_rate']:.1%} injected message faults",
        f"* Detectors: {', '.join(summary['stream']['detectors'])}",
        "",
        f"## Paced run ({_fmt(summary['stream']['rate_msgs_per_s'], 0)} messages/s offered)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Messages received | {cnt['messages_received']:,} |",
        f"| Observations processed | {cnt['observations_processed']:,} |",
        f"| Observations scored | {cnt['observations_scored']:,} |",
        f"| Observations flagged (alerts) | {cnt['observations_flagged']:,} |",
        f"| Rejected to dead-letter | {cnt['rejected_messages']:,} |",
        f"| Duplicates dropped | {cnt['duplicates_dropped']:,} |",
        f"| Gap rows inserted | {cnt['gap_rows_inserted']:,} |",
        f"| Sensor values forward-filled | {cnt['values_forward_filled']:,} |",
        f"| Processing errors | {cnt['processing_errors']:,} |",
        f"| Achieved throughput | {_fmt(pm['throughput']['overall_msgs_per_s'], 1)} messages/s |",
        f"| Mean micro-batch size | {_fmt(pm['batch_size']['mean'])} |",
        "",
        "| Latency (ms) | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|",
    ]
    labels = {"feature_ms": "Validation + features, per message", "inference_ms": "Model inference, per micro-batch",
              "processing_ms": "Total processing, per micro-batch", "publish_ms": "Write outputs + commit offset",
              "end_to_end_ms": "End to end (produced → scored)"}
    for key, label in labels.items():
        s = lat[key]
        lines.append(f"| {label} | {_fmt(s['p50'])} | {_fmt(s['p95'])} | {_fmt(s['p99'])} | {_fmt(s['max'])} |")
    if backlog:
        bm = backlog["metrics"]
        lines += [
            "",
            "## Backlog run (throughput capacity)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Messages drained | {bm['counts']['messages_received']:,} |",
            f"| Mean micro-batch size | {_fmt(bm['batch_size']['mean'])} |",
            f"| Throughput while draining | {_fmt(bm['throughput']['overall_msgs_per_s'], 0)} messages/s |",
            f"| Mean processing cost | {_fmt(bm['throughput']['mean_cost_ms_per_message'], 3)} ms per message |",
        ]
    if evaln.get("evaluated_rows"):
        lines += [
            "",
            "## Detection on the stream (simulated labels)",
            "",
            f"{evaln['evaluated_rows']:,} evaluated observations, {evaln['anomalous_rows']:,} labelled anomalous "
            f"({evaln['prevalence']:.1%}).",
            "",
            "| Alert rule | Precision | Recall | F1 | False-positive rate | Event recall |",
            "|---|---|---|---|---|---|",
        ]
        rows = {"Either detector (rule used for alerts)": evaln["combined_alert_rule"], **evaln["by_detector"]}
        for name, b in rows.items():
            lines.append(f"| {name} | {b['precision']:.2f} | {b['recall']:.2f} | {b['f1']:.2f} | "
                         f"{b['false_positive_rate']:.1%} | {_fmt(b['event_recall'])} ({b['events']} events) |")
        lines += ["", "| Simulated pattern | Events | Event recall | Point recall | Median delay (min) |", "|---|---|---|---|---|"]
        for r in evaln["recall_by_anomaly_type"]:
            lines.append(f"| {r['anomaly_type']} | {r['n_events']} | {_fmt(r['event_recall'])} | "
                         f"{_fmt(r['point_recall'])} | {_fmt(r['median_delay_minutes'], 0)} |")
        sev = evaln["alerts_by_severity"]
        lines += ["", "Alerts by severity band: " + ", ".join(f"{k} {sev.get(k, 0):,}" for k in ("high", "medium", "low")) + "."]
    (cfg.REPORTS_DIR / "streaming_run_summary.md").write_text("\n".join(lines) + "\n")


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main() -> None:
    settings = StreamSettings()
    parser = argparse.ArgumentParser(description="Run the simulated streaming system end to end.")
    parser.add_argument("--days", type=int, default=settings.stream_days, help="simulated days to stream")
    parser.add_argument("--rate", type=float, default=200.0, help="messages per second in the paced run")
    parser.add_argument("--fault-rate", type=float, default=settings.fault_rate)
    parser.add_argument("--skip-backlog", action="store_true", help="skip the throughput-capacity phase")
    parser.add_argument("--live", action="store_true", help="stream continuously until Ctrl+C (for the dashboard)")
    parser.add_argument("--quiet", action="store_true", help="log to file only")
    args = parser.parse_args()
    settings.stream_days, settings.stream_rate, settings.fault_rate = args.days, args.rate, args.fault_rate
    setup_logging(settings.log_dir, settings.log_level, "stream", console=not args.quiet and not args.live)

    bundle = load_bundle(settings.model_path)
    settings.alerts_file.unlink(missing_ok=True)

    if args.live:
        settings.stream_loop = True
        if args.rate == 200.0:
            settings.stream_rate = 40.0
        print(f"Streaming SIMULATED data at {settings.stream_rate:g} messages/s. Press Ctrl+C to stop.\n"
              f"Dashboard:  streamlit run dashboard/app.py")
        run_paced(settings, bundle, _fresh_broker(settings.broker_path), live=True)
        return

    broker = _fresh_broker(settings.broker_path)
    paced = run_paced(settings, bundle, broker)
    evaluation = _nan_to_none(evaluate_stream(broker))
    alerts = read_all(broker, TOPIC_ALERTS)
    topic_counts = {t: broker.count(t) for t in ALL_TOPICS}

    summary = {
        "disclosure": "Simulated sensor data and a local, single-machine streaming simulation. "
                      "Not connected to any real pipeline or live infrastructure.",
        "environment": {"machine": f"{platform.system()} {platform.machine()}", "python": platform.python_version()},
        "stream": {"days": settings.stream_days, "seed": settings.stream_seed, "start": settings.stream_start,
                   "rate_msgs_per_s": settings.stream_rate, "fault_rate": settings.fault_rate,
                   "detectors": list(settings.detectors), "max_batch": settings.max_batch},
        "paced_run": paced,
        "topic_message_counts": topic_counts,
        "evaluation": evaluation,
    }

    # keep a few real alerts in the repository as examples of the output format
    examples, seen = [], set()
    for a in alerts:
        if a["severity"] not in seen:
            seen.add(a["severity"])
            examples.append(a)
    (cfg.REPORTS_DIR / "sample_alerts.json").write_text(json.dumps(examples, indent=2))

    if not args.skip_backlog:
        backlog_settings = StreamSettings()
        backlog_settings.stream_days, backlog_settings.stream_rate = settings.stream_days, 0.0
        backlog_settings.fault_rate = settings.fault_rate
        backlog_settings.broker_path = settings.broker_path.with_name("broker_backlog.db")
        backlog_settings.alerts_file = settings.alerts_file.with_name("alerts_backlog.jsonl")
        backlog_settings.alerts_file.unlink(missing_ok=True)
        summary["backlog_run"] = run_backlog(backlog_settings, bundle, _fresh_broker(backlog_settings.broker_path))

    write_summary(summary)
    time.sleep(0.05)
    print((cfg.REPORTS_DIR / "streaming_run_summary.md").read_text())


if __name__ == "__main__":
    main()

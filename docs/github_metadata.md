# GitHub repository metadata

**Repository name:** `pipeline-sensor-anomaly-detection`

**Description (About field):**

> Anomaly detection on simulated pipeline sensor data, from batch study to a streaming-style system: producer, local message log, low-latency scikit-learn inference, structured JSON alerts, latency monitoring, Streamlit dashboard, Docker. 126 tests. Simulation only, not a safety system.

**Suggested topics:**

`anomaly-detection` `time-series` `machine-learning` `data-science` `streaming` `stream-processing`
`real-time-analytics` `scikit-learn` `isolation-forest` `streamlit` `docker` `data-pipeline`
`monitoring` `industrial-analytics` `sensor-data` `synthetic-data` `python`

**Suggested social preview image:** `figures/dashboard/dashboard_overview.png`

## What the repository demonstrates

| Skill | Where to look |
|---|---|
| Data science and evaluation | `notebooks/`, `reports/evaluation_report.md`, `src/pipeline_anomaly/evaluation.py` |
| Anomaly detection | `src/pipeline_anomaly/detectors.py`, `docs/methodology.md` |
| Feature engineering without leakage | `features.py`, `streaming/online_features.py`, parity tests |
| Streaming systems | `streaming/broker.py`, `producer.py`, `processor.py`, `docs/streaming_architecture.md` |
| ML inference under latency constraints | `streaming/processor.py` (micro-batching), `reports/streaming_run_summary.md` |
| Monitoring and logging | `streaming/metrics.py`, `streaming/logging_utils.py`, `dashboard/app.py` |
| Docker | `Dockerfile`, `docker-compose.yml`, `.env.example` |
| Data pipelines and testing | `pipeline.py`, `validation.py`, `tests/` (126 tests) |

## Before publishing

1. Replace `<your-repo-url>` in `README.md` and the link placeholder in `reports/linkedin_package.md`.
2. Confirm the name in `LICENSE`.
3. Run `docker compose up --build` once on a machine with a working Docker engine. The compose file validates but the
   image was not built during development; remove the "not verified" note in the README once it works for you.
4. Push: `git remote add origin <url> && git push -u origin main`.

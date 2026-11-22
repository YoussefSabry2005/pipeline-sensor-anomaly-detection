# LinkedIn publication package

Every figure quoted here comes from `reports/streaming_run_summary.json` (streaming) or `reports/metrics.json`
(batch). All data is simulated and all copy says so.

---

## 1. LinkedIn post: from notebook to streaming system

**A model in a notebook answers "does this work?" A streaming system has to answer "does it still work at 3 a.m., on message 34,000, when the data arrives broken?"**

A while ago I built an anomaly detection project on pipeline-style sensor data: pressure, flow rate and temperature. It was a batch analysis. I have now rebuilt it as a streaming-style monitoring system.

First, the honest part: real pipeline data is not public, so everything here is simulated. The sensors, the segments and the "stream" all run on my laptop. This is a portfolio project, not a leak-detection or safety system.

What I built:

🔹 Sensor simulator: four synthetic segments, normal operating patterns, gradual drift, and six kinds of injected anomalies
🔹 Streaming pipeline: producer → local message log with topics and consumer offsets → processor
🔹 Low-latency inference using the Isolation Forest and Local Outlier Factor models from the original project
🔹 Structured JSON alerts with a severity band and a plain-language explanation of what deviated
🔹 Latency and throughput monitoring, structured logs, and a live Streamlit dashboard
🔹 Docker Compose setup for the three services

What I measured on 30 simulated days (34,558 messages):

⏱ Median processing time: 7.6 ms per micro-batch. Median end to end: 23 ms.
📈 About 1,500 messages per second when draining a backlog
🧪 0 processing errors, with 1% of messages deliberately corrupted, duplicated or dropped
🎯 97% of simulated anomaly events flagged at least once, at 0.32 precision

What I learned:

1️⃣ The hard part was not the model. My batch feature code took about 40 ms per message. I rewrote it to update incrementally, then wrote tests proving the streaming features and scores match the batch ones exactly. Without that, the live model would quietly see different inputs from the ones it was trained on.

2️⃣ Bad data is the normal case. Most of the processor is validation: missing values, duplicates, gaps, unparseable messages. Each one needs a deliberate decision.

3️⃣ I chose not to use Kafka. For four simulated sensors on one machine it would be decoration. I built a small message log with the same concepts and kept it behind an interface.

4️⃣ An alert is a reason to look, not a conclusion. About two in three alerts were false alarms at the reading level, and slow drift was caught hours late. Those limits are in the README.

Code, architecture diagram and full results: [GITHUB LINK]

If you work with operational or sensor data: what is the first thing you would want a system like this to get right?

#DataScience #AnomalyDetection #MachineLearning #StreamProcessing #DataEngineering #IndustrialAnalytics #Python

---

## 2. Business problem

Monitoring teams receive more sensor data than people can inspect. Anomaly detection can rank readings so analysts
look at the most unusual first. Doing that as data arrives adds engineering questions a notebook never faces: what
happens to malformed messages, how features stay consistent with training, how long each reading takes to score, and
what an alert should contain so a person can act on it. An anomaly is a statistical signal; it is not evidence of a
leak, failure or incident.

## 3. Original model

A batch study on 207,360 simulated readings: validation and cleaning, 16 leakage-safe features (including a
physics-based pressure–flow residual), and a fair comparison of a rolling z-score baseline, Isolation Forest, Local
Outlier Factor and PCA reconstruction error. On a held-out period Isolation Forest and LOF reached about twice the
baseline's F1 (0.37 vs 0.19) and flagged 85–93 % of simulated events.

## 4. Streaming architecture

Simulator → producer → local message log (SQLite file with topics, offsets and consumer groups) → stream processor
(validate → features → inference → alerts) → topics for scored records, alerts, rejected messages and metrics →
Streamlit dashboard. Kafka and Redpanda were considered and judged excessive for a single-machine simulation.
Outputs and the consumer offset are committed in one transaction.

## 5. Sensor simulator

Four synthetic segments reporting pressure, flow rate and temperature every simulated 5 minutes, with operating
states, a pressure–flow relationship, daily and seasonal variation, autocorrelated noise and six labelled anomaly
patterns. The stream uses a different seed and later dates than the training data. 1 % of messages are deliberately
corrupted, duplicated or dropped to exercise error handling. Labels travel on a separate topic the processor never reads.

## 6. Anomaly detection in the stream

Isolation Forest and LOF from the batch project score each observation; either exceeding its label-free threshold
raises an alert. Features are computed incrementally from a small rolling buffer per segment. Tests require the
streaming features and anomaly scores to equal the batch pipeline's on the same data, including data with missing
values and gaps.

## 7. Structured alerts

JSON alerts with: alert ID, observation timestamp, synthetic segment and sensor IDs, severity band (documented
percentile thresholds), anomaly score, detectors that fired, sensor readings, a plain-language explanation of the
three largest deviations, an episode ID grouping nearby alerts, measured latency, and a disclaimer. Alert text is
tested to contain no claim of a leak, failure or incident.

## 8. Latency monitoring

Measured in-process with a monotonic clock: per-message feature time, per-batch inference time, total processing
time, write time and end-to-end time, each as p50 / p95 / p99, plus throughput and counters for processed, flagged,
rejected, duplicate and failed messages. Published about once per second and logged as JSON lines.

## 9. Dashboard

Streamlit, auto-refreshing and read-only: system status, throughput and latency, sensor trend charts with flagged
points, anomaly score against its threshold, a filterable alert table with explanations, recent observations and
rejected messages. Labelled as simulated on every view.

## 10. Docker

One image, three services (producer, processor, dashboard) sharing a volume, started with `docker compose up --build`.
Models are trained from the committed synthetic dataset during the build. No secrets.
**Note:** the compose file validates, but the image has not yet been built and run on a working Docker engine.
Do that once before saying "containerized and tested".

## 11. Actual results

Simulated data; measured locally on an Apple-silicon laptop; 30 simulated days; 200 messages/s offered.

| Measure | Result |
|---|---|
| Messages / observations processed | 34,558 / 34,443 |
| Observations scored / flagged | 32,747 / 1,577 |
| Processing time per micro-batch | p50 7.6 ms · p95 18.0 ms · p99 49.9 ms |
| Model inference per micro-batch (mean 10 messages) | p50 5.9 ms · p95 13.4 ms |
| Validation + features per message | p50 0.13 ms · p95 0.47 ms |
| End to end (produced → scored) | p50 23 ms · p95 41 ms · p99 418 ms |
| Throughput capacity (backlog drain) | 1,522 messages/s |
| Rejected / duplicate / processing errors | 73 / 42 / 0 |
| Alert rule vs simulated labels | precision 0.32 · recall 0.43 · F1 0.36 |
| Simulated events flagged at least once | 97 % of 64 |
| Tests | 126 passing |

## 12. Suggested screenshots (carousel order)

1. `figures/dashboard/dashboard_overview.png` — the live dashboard (hook)
2. The Mermaid architecture diagram rendered on GitHub (screenshot from the README)
3. `figures/06_anomaly_pattern_examples.png` — the six simulated patterns
4. A real alert as JSON (`reports/sample_alerts.json`, or the dashboard's "Latest alert as JSON" panel)
5. The latency table from `reports/streaming_run_summary.md`
6. `figures/dashboard/dashboard_full.png` cropped to the alert table and performance charts
7. `figures/10_model_comparison.png` — where the models came from
8. Terminal screenshot of `pytest` showing 126 passed

Keep the "Simulated data" banner or footer visible in every image.

## 13. LinkedIn Projects section description

**Pipeline Sensor Anomaly Detection: Streaming Simulation**

Extended a batch anomaly-detection study into a streaming-style monitoring system on simulated pipeline sensor data.
Built a sensor simulator, a producer, a local message log with topics and consumer offsets, and a processor that
validates messages, updates features incrementally, runs Isolation Forest and Local Outlier Factor inference in
micro-batches, and emits structured JSON alerts with explanations. Added latency and throughput monitoring,
structured logging, a Streamlit dashboard and Docker Compose files. On 34,558 simulated messages: median processing
time 7.6 ms, median end-to-end 23 ms, about 1,500 messages/s capacity, zero processing errors with 1 % injected
faults, and 97 % of simulated anomaly events flagged. Streaming features and scores are tested to match the batch
pipeline exactly. 126 automated tests. Simulation on synthetic data; not a real monitoring or safety system.

Skills: Python · scikit-learn · pandas · NumPy · Stream Processing · Anomaly Detection · Streamlit · Docker · SQLite · pytest

## 14. Resume bullets

* Converted a batch anomaly-detection model into a streaming-style system in Python (producer, message log,
  micro-batched scikit-learn inference, JSON alerts, Streamlit dashboard, Docker Compose); on 34.5K simulated sensor
  messages measured 7.6 ms median processing time, ~1,500 messages/s capacity and zero processing errors with 1 %
  injected faults.
* Rewrote feature engineering as an incremental engine (~40 ms → under 0.5 ms per message) and wrote parity tests
  proving streaming features and anomaly scores match the batch pipeline exactly; 126 automated tests in total.

## 15. Suggested hashtags

Primary: #DataScience #AnomalyDetection #MachineLearning #StreamProcessing #DataEngineering

Secondary: #IndustrialAnalytics #Python #ScikitLearn #Streamlit #Docker #MLOps #Energy #AlbertaEnergy #PortfolioProject

Use 5–8 per post.

## 16. GitHub link placeholder

`https://github.com/<your-username>/pipeline-sensor-anomaly-detection`

Replace `[GITHUB LINK]` in the post after pushing the repository.

---

### Wording to keep

* "simulated" or "synthetic", in the first few lines
* "streaming-style" or "streaming simulation"; "runs on my laptop"
* "simulated anomaly events" when quoting the 97 %
* "not a leak-detection or safety system"

### Wording to avoid

* "real-time pipeline monitoring", "deployed", "in production", "live infrastructure"
* "Kafka" as something you used (you evaluated it and chose not to)
* "detects leaks", "prevents failures", "improves safety"
* any cost, downtime or efficiency figure (none was measured)
* "sub-10 ms latency" without saying it is per micro-batch on a laptop with simulated data

---

## Appendix: original batch-project post

The earlier post about the batch study is still accurate and can be linked or reused:

> How do you find the unusual 3% in 200,000 sensor readings? I built a project to find out. […]
> One physics-based feature carried more signal than any other. Handling normal operating-state changes roughly
> doubled precision. Isolation Forest and LOF caught 85% and 93% of simulated events, but only LOF noticed stuck
> sensors, and every method found slow drift hours late. Point-wise F1 was about 0.37, roughly twice the simple
> baseline. An anomaly score is a reason to look, not a conclusion.

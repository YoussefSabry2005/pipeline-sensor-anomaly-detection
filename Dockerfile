# One image for all three services (producer, processor, dashboard).
# Everything inside runs on SIMULATED data; no external services or credentials are needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-stream.txt .
RUN pip install -r requirements-stream.txt

COPY src ./src
COPY data/raw ./data/raw
COPY dashboard ./dashboard

# Train the detectors on the committed synthetic dataset at build time, so the
# image is self-contained and the models match the library versions inside it.
RUN python -m pipeline_anomaly.streaming.model_bundle

# Run as an unprivileged user; /data is the shared volume (broker file, logs, alerts).
RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser:appuser /data /app
USER appuser

ENV BROKER_PATH=/data/broker.db \
    LOG_DIR=/data/logs \
    ALERTS_FILE=/data/alerts.jsonl

EXPOSE 8501
CMD ["python", "-m", "pipeline_anomaly.streaming.processor"]

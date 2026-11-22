"""Structured (JSON lines) logging for the streaming components.

Each log line is one JSON object: timestamp, level, component, event name and
event-specific fields. Usage::

    log.info("alert_created", extra={"fields": {"alert_id": ..., "severity": ...}})

Only simulated sensor values and identifiers are logged; the system holds no
credentials or personal data.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_NAME = "pipeline_stream"
# Silent unless setup_logging() is called (keeps library use and tests quiet).
logging.getLogger(ROOT_NAME).addHandler(logging.NullHandler())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": record.name.removeprefix(ROOT_NAME + "."),
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, separators=(",", ":"))


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_NAME}.{component}")


def setup_logging(log_dir: Path | str | None, level: str = "INFO", component: str = "stream",
                  console: bool = True) -> logging.Logger:
    """Configure JSON logging to stdout and, if ``log_dir`` is given, a rotating file."""
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = JsonFormatter()
    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_dir / f"{component}.jsonl", maxBytes=20_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    return root

"""A small local message log with Kafka-like semantics.

Why not Kafka or Redpanda? For a single-machine simulation with a few hundred
messages per second they add a JVM / cluster service, client libraries and
start-up orchestration without changing what is being demonstrated. This
module keeps the concepts that matter (named topics, an append-only log with
monotonically increasing offsets, consumer groups that commit offsets, replay
from any offset) in ~150 lines backed by SQLite, which is in the Python
standard library and safe for several processes sharing one file.

The ``Broker`` interface is deliberately narrow so a Kafka-backed
implementation could replace ``SQLiteBroker`` without touching the producer,
processor or dashboard. No such implementation is included or claimed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class Message:
    topic: str
    offset: int
    key: str | None
    value: Any            # decoded JSON, or None if the payload was not valid JSON
    raw: str              # payload exactly as stored
    published_at: float   # wall-clock epoch seconds


def _decode(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _encode(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), allow_nan=False)


class Broker(Protocol):
    def publish(self, topic: str, value: Any, key: str | None = None) -> int: ...
    def publish_many(self, records: Iterable[tuple[str, Any, str | None]]) -> int: ...
    def poll(self, topic: str, group: str, max_messages: int = 100) -> list[Message]: ...
    def commit(self, topic: str, group: str, offset: int) -> None: ...
    def publish_and_commit(self, records, topic: str, group: str, offset: int) -> None: ...
    def committed(self, topic: str, group: str) -> int: ...
    def read(self, topic: str, after_offset: int = 0, limit: int = 1000) -> list[Message]: ...
    def tail(self, topic: str, n: int = 100) -> list[Message]: ...
    def count(self, topic: str) -> int: ...
    def lag(self, topic: str, group: str) -> int: ...


class SQLiteBroker:
    """Durable, multi-process broker stored in a single SQLite file."""

    def __init__(self, path: Path | str, timeout: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._local = threading.local()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "offset INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, key TEXT, "
                "payload TEXT NOT NULL, published_at REAL NOT NULL)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_topic_offset ON messages(topic, offset)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS offsets ("
                "topic TEXT NOT NULL, grp TEXT NOT NULL, offset INTEGER NOT NULL, PRIMARY KEY (topic, grp))"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=self._timeout)
            conn.execute("PRAGMA journal_mode=WAL")      # readers do not block the writer
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # -- writing -----------------------------------------------------------
    def publish(self, topic: str, value: Any, key: str | None = None) -> int:
        return self.publish_many([(topic, value, key)])

    def publish_many(self, records: Iterable[tuple[str, Any, str | None]]) -> int:
        now = time.time()
        rows = [(t, k, _encode(v), now) for t, v, k in records]
        if not rows:
            return 0
        with self._conn() as c:
            cur = c.executemany("INSERT INTO messages(topic, key, payload, published_at) VALUES (?,?,?,?)", rows)
            return cur.rowcount

    def commit(self, topic: str, group: str, offset: int) -> None:
        with self._conn() as c:
            self._commit(c, topic, group, offset)

    @staticmethod
    def _commit(c, topic: str, group: str, offset: int) -> None:
        c.execute(
            "INSERT INTO offsets(topic, grp, offset) VALUES (?,?,?) "
            "ON CONFLICT(topic, grp) DO UPDATE SET offset = MAX(offset, excluded.offset)",
            (topic, group, offset),
        )

    def publish_and_commit(self, records, topic: str, group: str, offset: int) -> None:
        """Write outputs and advance the consumer offset in ONE transaction.

        Either both happen or neither does, so a crash cannot produce outputs
        for an input that will be re-read (no duplicate outputs on restart).
        """
        now = time.time()
        rows = [(t, k, _encode(v), now) for t, v, k in records]
        with self._conn() as c:
            if rows:
                c.executemany("INSERT INTO messages(topic, key, payload, published_at) VALUES (?,?,?,?)", rows)
            self._commit(c, topic, group, offset)

    # -- reading -----------------------------------------------------------
    def committed(self, topic: str, group: str) -> int:
        row = self._conn().execute("SELECT offset FROM offsets WHERE topic=? AND grp=?", (topic, group)).fetchone()
        return int(row[0]) if row else 0

    def read(self, topic: str, after_offset: int = 0, limit: int = 1000) -> list[Message]:
        rows = self._conn().execute(
            "SELECT offset, key, payload, published_at FROM messages WHERE topic=? AND offset>? ORDER BY offset LIMIT ?",
            (topic, after_offset, limit),
        ).fetchall()
        return [Message(topic, o, k, _decode(p), p, ts) for o, k, p, ts in rows]

    def poll(self, topic: str, group: str, max_messages: int = 100) -> list[Message]:
        return self.read(topic, self.committed(topic, group), max_messages)

    def tail(self, topic: str, n: int = 100) -> list[Message]:
        rows = self._conn().execute(
            "SELECT offset, key, payload, published_at FROM messages WHERE topic=? ORDER BY offset DESC LIMIT ?",
            (topic, n),
        ).fetchall()
        return [Message(topic, o, k, _decode(p), p, ts) for o, k, p, ts in reversed(rows)]

    def count(self, topic: str) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM messages WHERE topic=?", (topic,)).fetchone()[0])

    def lag(self, topic: str, group: str) -> int:
        return int(
            self._conn().execute(
                "SELECT COUNT(*) FROM messages WHERE topic=? AND offset>?", (topic, self.committed(topic, group))
            ).fetchone()[0]
        )

    def reset(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages")
            c.execute("DELETE FROM offsets")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class InMemoryBroker:
    """Same interface, kept in process memory. Used by tests and quick demos."""

    def __init__(self):
        self._log: list[Message] = []
        self._offsets: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, value: Any, key: str | None = None) -> int:
        return self.publish_many([(topic, value, key)])

    def publish_many(self, records) -> int:
        n = 0
        with self._lock:
            for topic, value, key in records:
                raw = _encode(value)
                self._log.append(Message(topic, len(self._log) + 1, key, _decode(raw), raw, time.time()))
                n += 1
        return n

    def commit(self, topic: str, group: str, offset: int) -> None:
        with self._lock:
            self._offsets[(topic, group)] = max(offset, self._offsets.get((topic, group), 0))

    def publish_and_commit(self, records, topic: str, group: str, offset: int) -> None:
        self.publish_many(records)
        self.commit(topic, group, offset)

    def committed(self, topic: str, group: str) -> int:
        return self._offsets.get((topic, group), 0)

    def read(self, topic: str, after_offset: int = 0, limit: int = 1000) -> list[Message]:
        with self._lock:
            return [m for m in self._log if m.topic == topic and m.offset > after_offset][:limit]

    def poll(self, topic: str, group: str, max_messages: int = 100) -> list[Message]:
        return self.read(topic, self.committed(topic, group), max_messages)

    def tail(self, topic: str, n: int = 100) -> list[Message]:
        with self._lock:
            return [m for m in self._log if m.topic == topic][-n:]

    def count(self, topic: str) -> int:
        with self._lock:
            return sum(1 for m in self._log if m.topic == topic)

    def lag(self, topic: str, group: str) -> int:
        return len(self.read(topic, self.committed(topic, group), 10**9))

    def reset(self) -> None:
        with self._lock:
            self._log.clear()
            self._offsets.clear()

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass

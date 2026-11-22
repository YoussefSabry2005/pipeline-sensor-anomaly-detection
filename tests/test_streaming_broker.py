import pytest

from pipeline_anomaly.streaming.broker import InMemoryBroker, SQLiteBroker


@pytest.fixture(params=["memory", "sqlite"])
def broker(request, tmp_path):
    b = InMemoryBroker() if request.param == "memory" else SQLiteBroker(tmp_path / "b.db")
    yield b
    b.close()


def test_publish_poll_commit_cycle(broker):
    for i in range(5):
        broker.publish("t", {"i": i}, key="k")
    first = broker.poll("t", "g", max_messages=3)
    assert [m.value["i"] for m in first] == [0, 1, 2]
    assert [m.offset for m in first] == sorted(m.offset for m in first)
    # without a commit the same messages are delivered again (at-least-once)
    assert [m.value["i"] for m in broker.poll("t", "g", 3)] == [0, 1, 2]
    broker.commit("t", "g", first[-1].offset)
    assert [m.value["i"] for m in broker.poll("t", "g", 10)] == [3, 4]
    assert broker.lag("t", "g") == 2


def test_topics_and_groups_are_independent(broker):
    broker.publish_many([("a", {"x": 1}, None), ("b", {"x": 2}, None), ("a", {"x": 3}, None)])
    assert broker.count("a") == 2 and broker.count("b") == 1
    msgs = broker.poll("a", "g1")
    broker.commit("a", "g1", msgs[-1].offset)
    assert broker.poll("a", "g1") == []
    assert len(broker.poll("a", "g2")) == 2          # another group still sees everything
    assert [m.value["x"] for m in broker.tail("a", 1)] == [3]


def test_commit_never_moves_backwards(broker):
    for i in range(4):
        broker.publish("t", {"i": i})
    offsets = [m.offset for m in broker.read("t")]
    broker.commit("t", "g", offsets[2])
    broker.commit("t", "g", offsets[0])
    assert broker.committed("t", "g") == offsets[2]


def test_invalid_json_payload_is_delivered_as_raw(broker):
    broker.publish("t", '{"broken": ')
    msg = broker.poll("t", "g")[0]
    assert msg.value is None and msg.raw == '{"broken": '


def test_publish_and_commit(broker):
    broker.publish("in", {"i": 1})
    msg = broker.poll("in", "g")[0]
    broker.publish_and_commit([("out", {"o": 1}, None), ("out", {"o": 2}, None)], "in", "g", msg.offset)
    assert broker.count("out") == 2
    assert broker.poll("in", "g") == []


def test_nan_is_not_valid_json(broker):
    with pytest.raises(ValueError):
        broker.publish("t", {"x": float("nan")})


def test_sqlite_broker_is_durable_and_shared(tmp_path):
    path = tmp_path / "shared.db"
    writer, reader = SQLiteBroker(path), SQLiteBroker(path)   # two connections, like two processes
    writer.publish("t", {"i": 1})
    assert reader.count("t") == 1
    reader.commit("t", "g", reader.poll("t", "g")[0].offset)
    writer.close(); reader.close()
    reopened = SQLiteBroker(path)
    assert reopened.count("t") == 1 and reopened.poll("t", "g") == []
    reopened.reset()
    assert reopened.count("t") == 0

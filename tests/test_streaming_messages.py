import math

import pandas as pd
import pytest

from pipeline_anomaly import config as cfg
from pipeline_anomaly.streaming.messages import MessageRejected, make_reading_message, validate_message

ROW = {"timestamp": pd.Timestamp("2025-07-01 00:05:00"), "segment_id": "SEG-A", "sensor_id": "STN-101",
       "pressure_kpa": 4000.0, "flow_m3h": 800.0, "temperature_c": 12.0, "operating_state": "NORMAL"}
SEGMENTS = {"SEG-A", "SEG-B"}


def msg(**changes):
    m = make_reading_message(ROW, seq=1, produced_at=123.0)
    m.update(changes)
    return m


def test_wire_format_has_no_label_fields():
    m = msg()
    assert m["simulated"] is True and m["produced_at"] == 123.0
    assert not any(k in m for k in cfg.LABEL_COLS)
    assert m["timestamp"] == "2025-07-01 00:05:00"


def test_nan_becomes_null_on_the_wire():
    m = make_reading_message({**ROW, "flow_m3h": float("nan")}, 1, 0.0)
    assert m["flow_m3h"] is None


def test_valid_message():
    r = validate_message(msg(), SEGMENTS)
    assert r.timestamp == ROW["timestamp"] and r.segment_id == "SEG-A"
    assert r.values == {"pressure_kpa": 4000.0, "flow_m3h": 800.0, "temperature_c": 12.0}
    assert r.issues == [] and r.produced_at == 123.0


@pytest.mark.parametrize("change, issue", [
    ({"pressure_kpa": None}, "pressure_kpa:missing"),
    ({"flow_m3h": "ERR"}, "flow_m3h:not_numeric"),
    ({"temperature_c": -999.0}, "temperature_c:out_of_range"),
    ({"pressure_kpa": float("inf")}, "pressure_kpa:out_of_range"),
])
def test_bad_sensor_values_are_accepted_as_missing(change, issue):
    r = validate_message(msg(**change), SEGMENTS)
    col = issue.split(":")[0]
    assert math.isnan(r.values[col]) and r.issues == [issue]


def test_absent_sensor_field_is_missing_value():
    m = msg()
    del m["flow_m3h"]
    r = validate_message(m, SEGMENTS)
    assert math.isnan(r.values["flow_m3h"]) and "flow_m3h:missing" in r.issues


@pytest.mark.parametrize("value, reason", [
    ("just a string", "not_a_json_object"),
    (None, "not_a_json_object"),
    ([1, 2], "not_a_json_object"),
])
def test_non_object_rejected(value, reason):
    with pytest.raises(MessageRejected) as exc:
        validate_message(value, SEGMENTS)
    assert exc.value.reason == reason


@pytest.mark.parametrize("change, reason", [
    ({"timestamp": "not-a-timestamp"}, "invalid_timestamp"),
    ({"timestamp": None}, "missing_required_field"),
    ({"segment_id": ""}, "missing_required_field"),
    ({"timestamp": "2025-07-01 00:07:30"}, "off_grid_timestamp"),
    ({"segment_id": "SEG-Z"}, "unknown_segment"),
    ({"operating_state": "MAINTENANCE"}, "unknown_operating_state"),
])
def test_unplaceable_messages_rejected(change, reason):
    with pytest.raises(MessageRejected) as exc:
        validate_message(msg(**change), SEGMENTS)
    assert exc.value.reason == reason


def test_other_timestamp_formats_are_parsed():
    r = validate_message(msg(timestamp="2025-07-01T00:05:00"), SEGMENTS)
    assert r.timestamp == ROW["timestamp"]

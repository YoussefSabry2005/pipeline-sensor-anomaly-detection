"""Monitoring dashboard for the simulated streaming system.

    streamlit run dashboard/app.py

Read-only: the dashboard only reads topics from the local broker file. All
data shown is simulated.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline_anomaly import config as cfg  # noqa: E402
from pipeline_anomaly.streaming.broker import SQLiteBroker  # noqa: E402
from pipeline_anomaly.streaming.settings import (  # noqa: E402
    TOPIC_ALERTS, TOPIC_DEADLETTER, TOPIC_METRICS, TOPIC_SCORED, StreamSettings,
)

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SENSOR_LABELS = {"pressure_kpa": "Pressure (kPa)", "flow_m3h": "Flow rate (m³/h)", "temperature_c": "Temperature (°C)"}
SEVERITY_ORDER = ["high", "medium", "low"]
SEVERITY_ICON = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟡 low"}
STATUS_TEXT = {
    "scored": "scored", "settling": "settling (not monitored)", "warming_up": "warming up",
    "not_usable": "missing data", "inference_error": "inference error",
}

st.set_page_config(page_title="Pipeline Sensor Monitoring (Simulation)", page_icon="📈", layout="wide")
settings = StreamSettings()


@st.cache_resource
def get_broker(path: str) -> SQLiteBroker:
    return SQLiteBroker(path)


def load(broker: SQLiteBroker, n_scored: int) -> dict:
    scored = [m.value for m in broker.tail(TOPIC_SCORED, n_scored)]
    rows = []
    for r in scored:
        row = {
            "timestamp": r["timestamp"], "segment_id": r["segment_id"], "sensor_id": r["sensor_id"],
            "operating_state": r["operating_state"], "status": r["status"], "flagged": bool(r["flagged"]),
            "score_percentile": r["anomaly_score_percentile"], "severity": r["severity"],
            "processing_ms": r["latency_ms"]["processing_ms"], "end_to_end_ms": r["latency_ms"]["end_to_end_ms"],
            **{k: r["readings"].get(k) for k in cfg.SENSOR_COLS},
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    metrics = [m.value for m in broker.tail(TOPIC_METRICS, 240)]
    return {
        "scored": df,
        "alerts": [m.value for m in broker.tail(TOPIC_ALERTS, 300)],
        "metrics": metrics,
        "deadletter": [m.value for m in broker.tail(TOPIC_DEADLETTER, 25)],
    }


def line_chart(df: pd.DataFrame, y: str, title: str, height: int = 150) -> alt.LayerChart:
    base = alt.Chart(df).encode(x=alt.X("timestamp:T", title=None))
    line = base.mark_line(color=BLUE, strokeWidth=1.6).encode(
        y=alt.Y(f"{y}:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("timestamp:T", format="%Y-%m-%d %H:%M"), alt.Tooltip(f"{y}:Q", format=",.2f"),
                 "operating_state:N"],
    )
    flagged = base.transform_filter(alt.datum.flagged).mark_circle(color=ORANGE, size=55, opacity=1).encode(
        y=f"{y}:Q",
        tooltip=[alt.Tooltip("timestamp:T", format="%Y-%m-%d %H:%M"), alt.Tooltip(f"{y}:Q", format=",.2f"),
                 alt.Tooltip("severity:N"), alt.Tooltip("score_percentile:Q", format=".3f")],
    )
    return (line + flagged).properties(height=height)


def score_chart(df: pd.DataFrame) -> alt.LayerChart:
    d = df.dropna(subset=["score_percentile"])
    base = alt.Chart(d).encode(x=alt.X("timestamp:T", title="Simulated time"))
    area = base.mark_line(color=BLUE, strokeWidth=1.2).encode(
        y=alt.Y("score_percentile:Q", title="Anomaly score (percentile)", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("timestamp:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("score_percentile:Q", format=".3f")],
    )
    points = base.transform_filter(alt.datum.flagged).mark_circle(color=ORANGE, size=45, opacity=1).encode(
        y="score_percentile:Q")
    rule = base.mark_rule(color=MUTED, strokeDash=[5, 4]).encode(y=alt.datum(1 - cfg.ALERT_BUDGET), x=alt.value(0), x2=alt.value("width"))
    return (area + points + rule).properties(height=170)


def metric_history(metrics: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        lat = m["latency_ms"]
        rows.append({
            "time": pd.to_datetime(m["updated_at_epoch"], unit="s"),
            "Throughput (messages/s)": m["throughput"]["recent_msgs_per_s"],
            "Processing p50": lat["processing_ms"]["p50"], "Processing p95": lat["processing_ms"]["p95"],
            "End-to-end p95": lat["end_to_end_ms"]["p95"],
        })
    return pd.DataFrame(rows)


def fmt(value, pattern="{:,.0f}", empty="–"):
    return empty if value is None else pattern.format(value)


def ms(value) -> str:
    if value is None:
        return "–"
    return f"{value:,.1f} ms" if value < 100 else f"{value:,.0f} ms"


# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    auto = st.toggle("Auto-refresh", value=True)
    interval = st.slider("Refresh every (seconds)", 1, 10, 2)
    history = st.select_slider("Readings to load", options=[500, 1000, 2000, 4000, 8000], value=2000)
    st.divider()
    st.caption(
        "**About this demo**\n\n"
        "A local simulation of a streaming monitoring system. A simulator publishes synthetic sensor readings to a "
        "local message log; a processor validates them, computes features, scores them with models trained in the "
        "batch project, and writes alerts.\n\n"
        "Simulated time runs much faster than wall-clock time: each message is one 5-minute reading.\n\n"
        "Alerts are analytical notifications for review. They do not indicate a leak, failure or safety incident."
    )

st.title("Pipeline Sensor Monitoring")
st.warning(
    "**Simulated data.** Every reading, segment and sensor ID on this page is generated by a simulator. "
    "This is a portfolio demonstration, not a connection to a real pipeline or a safety system.",
    icon="⚠️",
)


@st.fragment(run_every=interval if auto else None)
def body() -> None:
    if not settings.broker_path.exists():
        st.info(
            "No stream found yet. Start the simulation in another terminal:\n\n"
            "`python -m pipeline_anomaly.streaming.run_demo --live`\n\n"
            "or with Docker: `docker compose up`."
        )
        return
    data = load(get_broker(str(settings.broker_path)), history)
    scored, alerts, metrics = data["scored"], data["alerts"], data["metrics"]
    if scored.empty or not metrics:
        st.info("Waiting for the first observations…")
        return

    snap = metrics[-1]
    age = time.time() - snap["updated_at_epoch"]
    live = age < 6 and snap["status"] != "stopped"
    counts, lat, thr = snap["counts"], snap["latency_ms"], snap["throughput"]

    # ------------------------------------------------------------------ system status
    st.subheader("System status")
    scored_n = counts["observations_scored"]
    c = st.columns(6)
    c[0].metric("Stream", "🟢 Live" if live and snap["status"] == "running" else ("🟡 Idle" if live else "⚪ Stopped"),
                help=f"Last heartbeat {age:,.0f} s ago")
    c[1].metric("Observations", fmt(counts["observations_processed"]), help="Observations processed since start")
    c[2].metric("Flagged", fmt(counts["observations_flagged"]),
                help="Observations scored above the alert threshold by at least one detector")
    c[3].metric("Alert rate", fmt(100 * counts["observations_flagged"] / scored_n if scored_n else None, "{:.1f}%"),
                help="Flagged share of scored observations")
    c[4].metric("Throughput", fmt(thr["recent_msgs_per_s"], "{:,.0f}/s"), help="Messages per second, recent window")
    c[5].metric("Consumer lag", fmt(snap["consumer_lag"]), help="Messages waiting to be processed")
    c = st.columns(6)
    c[0].metric("Inference p50", ms(lat["inference_ms"]["p50"]), help="Model scoring per micro-batch")
    c[1].metric("Processing p50", ms(lat["processing_ms"]["p50"]),
                help="Wall time a message spends in the processor (its micro-batch), recent window")
    c[2].metric("Processing p95", ms(lat["processing_ms"]["p95"]))
    c[3].metric("End-to-end p95", ms(lat["end_to_end_ms"]["p95"]),
                help="From producer publish to scored output, recent window")
    c[4].metric("Rejected", fmt(counts["rejected_messages"]),
                help=f"Messages sent to the dead-letter topic. Duplicates dropped: {counts['duplicates_dropped']:,}")
    c[5].metric("Errors", fmt(counts["processing_errors"]), help="Unexpected processing errors")

    # ------------------------------------------------------------------ sensor trends
    st.subheader("Sensor trends")
    segments = sorted(scored["segment_id"].unique())
    left, right = st.columns([1, 3])
    segment = left.radio("Segment (synthetic)", segments, horizontal=True, key="segment")
    window_h = right.select_slider("Window (simulated hours)", options=[6, 12, 24, 48, 96], value=24, key="window")
    seg = scored[scored["segment_id"] == segment]
    seg = seg[seg["timestamp"] >= seg["timestamp"].max() - pd.Timedelta(hours=window_h)]
    latest = seg.iloc[-1]
    st.caption(
        f"Sensor **{latest['sensor_id']}** · latest simulated reading **{latest['timestamp']:%Y-%m-%d %H:%M}** · "
        f"operating state **{latest['operating_state']}** · status **{STATUS_TEXT.get(latest['status'], latest['status'])}** · "
        f"🟠 = flagged observation"
    )
    cols = st.columns(3)
    for col, key in zip(cols, cfg.SENSOR_COLS):
        col.metric(SENSOR_LABELS[key], fmt(latest[key], "{:,.2f}"))
        col.altair_chart(line_chart(seg, key, SENSOR_LABELS[key]), use_container_width=True)
    st.altair_chart(score_chart(seg), use_container_width=True)
    st.caption(f"Dashed line: alert threshold ({100 * (1 - cfg.ALERT_BUDGET):.0f}th percentile of training-period scores). "
               "Gaps are warm-up or settling periods, when observations are not scored.")

    # ------------------------------------------------------------------ alerts
    st.subheader("Alerts")
    if not alerts:
        st.caption("No alerts yet.")
    else:
        adf = pd.DataFrame([{
            "Simulated time": a["observation_timestamp"], "Alert ID": a["alert_id"], "Segment": a["segment_id"],
            "Sensor": a["sensor_id"], "Severity": SEVERITY_ICON[a["severity"]],
            "Score percentile": a["anomaly_score_percentile"], "Detected by": ", ".join(a["detection_methods"]),
            "Why it was flagged": "; ".join(d["description"] for d in a["top_deviations"]),
            "Episode": a["episode_id"], "Processing (ms)": a["latency_ms"]["processing_ms"],
        } for a in reversed(alerts)])
        f1, f2 = st.columns([1, 3])
        chosen = f1.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER, key="sev")
        only_segment = f2.checkbox(f"Only {segment}", value=False, key="only_seg")
        view = adf[adf["Severity"].isin([SEVERITY_ICON[s] for s in chosen])]
        if only_segment:
            view = view[view["Segment"] == segment]
        st.dataframe(
            view, use_container_width=True, hide_index=True, height=320,
            column_config={"Score percentile": st.column_config.ProgressColumn(format="%.3f", min_value=0.9, max_value=1.0),
                           "Why it was flagged": st.column_config.TextColumn(width="large")},
        )
        st.caption("Severity bands are simulation thresholds on the score percentile (low ≥ 97th, medium ≥ 99th, "
                   "high ≥ 99.7th). They are not operational severities.")
        with st.expander("Latest alert as JSON (the structured output format)"):
            st.json(alerts[-1])

    # ------------------------------------------------------------------ performance
    st.subheader("Processing performance")
    hist = metric_history(metrics)
    p1, p2 = st.columns(2)
    p1.markdown("**Latency** (recent window)")
    p2.markdown("**Throughput** (recent window)")
    lat_long = hist.melt("time", ["Processing p50", "Processing p95", "End-to-end p95"], "Series", "ms").dropna()
    p1.altair_chart(
        alt.Chart(lat_long).mark_line(strokeWidth=1.8).encode(
            x=alt.X("time:T", title="Wall-clock time"), y=alt.Y("ms:Q", title="Latency (ms)"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Processing p50", "Processing p95", "End-to-end p95"],
                                                        range=[BLUE, ORANGE, AQUA]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Series:N", alt.Tooltip("ms:Q", format=",.1f"), alt.Tooltip("time:T", format="%H:%M:%S")],
        ).properties(height=220),
        use_container_width=True,
    )
    p2.altair_chart(
        alt.Chart(hist).mark_line(color=BLUE, strokeWidth=1.8).encode(
            x=alt.X("time:T", title="Wall-clock time"),
            y=alt.Y("Throughput (messages/s):Q", title="Messages per second"),
            tooltip=[alt.Tooltip("Throughput (messages/s):Q", format=",.1f"), alt.Tooltip("time:T", format="%H:%M:%S")],
        ).properties(height=220),
        use_container_width=True,
    )

    # ------------------------------------------------------------------ recent data
    st.subheader("Recent observations")
    recent = scored.sort_values("timestamp", ascending=False).head(200).copy()
    recent["status"] = recent["status"].map(STATUS_TEXT).fillna(recent["status"])
    recent["flagged"] = recent["flagged"].map({True: "🟠 flagged", False: ""})
    st.dataframe(
        recent.rename(columns={"timestamp": "Simulated time", "segment_id": "Segment", "sensor_id": "Sensor",
                               "operating_state": "State", "status": "Status", "flagged": "Flag",
                               "score_percentile": "Score percentile", "severity": "Severity",
                               "processing_ms": "Processing (ms)", "end_to_end_ms": "End-to-end (ms)",
                               **SENSOR_LABELS}),
        use_container_width=True, hide_index=True, height=300,
    )

    with st.expander(f"Rejected messages (dead-letter topic) — {counts['rejected_messages']:,} total"):
        if data["deadletter"]:
            st.dataframe(pd.DataFrame([{
                "Reason": d["reason"], "Detail": d["detail"], "Source offset": d["source_offset"],
                "Payload (truncated)": d["payload"][:120]} for d in reversed(data["deadletter"])]),
                use_container_width=True, hide_index=True)
            st.caption("The simulator deliberately corrupts a small share of messages to exercise error handling.")
        else:
            st.caption("None so far.")
    st.caption(f"Snapshot: {snap['updated_at']} UTC · processor uptime {snap['uptime_s']:,.0f} s")


body()

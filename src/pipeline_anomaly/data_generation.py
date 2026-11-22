"""Synthetic pipeline-style sensor data generator.

IMPORTANT: every value produced here is SIMULATED. Nothing is measured from a
real pipeline, and the segment / sensor identifiers are invented.

The simulator has three layers:

1. ``simulate_normal_operation``  - physically-motivated "normal" behaviour
   (operating states, a pressure-flow relationship, thermal lag, coloured
   noise).
2. ``inject_anomalies``           - six labelled anomaly patterns.
3. ``inject_data_quality_defects`` - missing cells, missing rows, duplicate
   rows and sentinel values, so that the validation stage has real work to do.

Run ``python -m pipeline_anomaly.data_generation`` to (re)create the raw CSV.
The output is fully determined by the seed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

SENTINEL_VALUE = -999.0


@dataclass(frozen=True)
class SegmentProfile:
    """Static characteristics of one synthetic pipeline segment."""

    segment_id: str
    sensor_id: str
    static_pressure_kpa: float   # pressure at zero flow
    friction_coeff: float        # kPa per (m3/h)^2
    nominal_flow_m3h: float
    base_temperature_c: float


SEGMENTS: tuple[SegmentProfile, ...] = (
    SegmentProfile("SEG-A", "STN-101", 1500.0, 4.0e-3, 800.0, 9.0),
    SegmentProfile("SEG-B", "STN-102", 1350.0, 3.4e-3, 800.0, 8.2),
    SegmentProfile("SEG-C", "STN-103", 1200.0, 2.9e-3, 795.0, 7.5),
    SegmentProfile("SEG-D", "STN-104", 1050.0, 2.5e-3, 790.0, 7.0),
)

# Standard deviation of the instrument / process noise for each sensor.
NOISE_SD = {"pressure_kpa": 12.0, "flow_m3h": 6.0, "temperature_c": 0.15}
NOISE_PHI = {"pressure_kpa": 0.6, "flow_m3h": 0.7, "temperature_c": 0.9}

STATE_FLOW_FACTOR = {"NORMAL": 1.0, "HIGH_FLOW": 1.2, "LOW_FLOW": 0.72, "SHUTDOWN": 0.0}

# Number of events of each anomaly type injected per segment per 90 days of
# simulated time (scaled linearly with the run length).
EVENTS_PER_SEGMENT_90D = {
    "spike": 30,
    "pressure_flow_deviation": 5,
    "drift": 3,
    "flatline": 4,
    "noise_burst": 4,
    "temperature_excursion": 3,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _ar1_noise(rng: np.random.Generator, n: int, sd: float, phi: float) -> np.ndarray:
    """Stationary AR(1) noise with marginal standard deviation ``sd``."""
    white = rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + white[i]
    return out


def _first_order_lag(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential lag: models the fact that flow/heat do not change instantly."""
    out = np.empty(len(signal), dtype=float)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def simulate_operating_schedule(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sequence of operating states shared by all segments (they are in series).

    The line alternates between NORMAL running (12-48 h) and an excursion to
    HIGH_FLOW / LOW_FLOW (6-18 h) or, more rarely, a SHUTDOWN (2-8 h).
    """
    sph = cfg.SAMPLES_PER_HOUR
    states = np.empty(n, dtype=object)
    i, current = 0, "NORMAL"
    while i < n:
        if current == "NORMAL":
            duration = int(rng.integers(12, 49)) * sph
        elif current == "SHUTDOWN":
            duration = int(rng.integers(2, 9)) * sph
        else:
            duration = int(rng.integers(6, 19)) * sph
        states[i : i + duration] = current
        i += duration
        if current != "NORMAL":
            current = "NORMAL"
        else:
            current = str(rng.choice(["HIGH_FLOW", "LOW_FLOW", "SHUTDOWN"], p=[0.45, 0.45, 0.10]))
    return states


# --------------------------------------------------------------------------
# Layer 1: normal operation
# --------------------------------------------------------------------------
def simulate_normal_operation(
    profile: SegmentProfile,
    timestamps: pd.DatetimeIndex,
    states: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate anomaly-free readings for one segment."""
    n = len(timestamps)
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy()
    day_of_run = np.arange(n) / (24 * cfg.SAMPLES_PER_HOUR)

    # Flow follows the commanded state with a short lag plus a mild daily cycle.
    setpoint = np.array([STATE_FLOW_FACTOR[s] for s in states], dtype=float)
    flow_factor = _first_order_lag(setpoint, alpha=0.35)
    daily_cycle = 1.0 + 0.025 * np.sin(2 * np.pi * (hours - 9.0) / 24.0)
    flow_true = profile.nominal_flow_m3h * flow_factor * daily_cycle

    # Pressure: static head + friction loss proportional to flow squared.
    pressure_true = profile.static_pressure_kpa + profile.friction_coeff * flow_true**2

    # Temperature: slow seasonal movement + small diurnal cycle + frictional
    # heating that responds to flow with a ~2 h thermal lag.
    seasonal = 2.0 * np.sin(2 * np.pi * (day_of_run - 30.0) / 365.0)
    diurnal = 0.5 * np.sin(2 * np.pi * (hours - 15.0) / 24.0)
    heating = _first_order_lag(0.005 * flow_true, alpha=0.04)
    temperature_true = profile.base_temperature_c + seasonal + diurnal + heating

    # Flow-meter noise shrinks when there is (almost) no flow.
    flow_noise = _ar1_noise(rng, n, NOISE_SD["flow_m3h"], NOISE_PHI["flow_m3h"])
    flow_noise *= 0.1 + 0.9 * np.clip(flow_factor, 0.0, 1.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": profile.segment_id,
            "sensor_id": profile.sensor_id,
            "pressure_kpa": pressure_true
            + _ar1_noise(rng, n, NOISE_SD["pressure_kpa"], NOISE_PHI["pressure_kpa"]),
            "flow_m3h": np.clip(flow_true + flow_noise, 0.0, None),
            "temperature_c": temperature_true
            + _ar1_noise(rng, n, NOISE_SD["temperature_c"], NOISE_PHI["temperature_c"]),
            "operating_state": states,
            "is_anomaly": 0,
            "anomaly_type": "none",
            "anomaly_event_id": "",
            "anomaly_sensor": "",
        }
    )


# --------------------------------------------------------------------------
# Layer 2: labelled anomalies
# --------------------------------------------------------------------------
def _find_window(
    rng: np.random.Generator,
    n: int,
    length: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    buffer: int = 24,
    max_tries: int = 2000,
) -> int | None:
    """Pick a start index whose window is free of other events and shutdowns."""
    first = 2 * cfg.LONG_WINDOW
    if n - length - 1 <= first:      # run too short to host an event of this length
        return None
    for _ in range(max_tries):
        start = int(rng.integers(first, n - length - 1))
        lo, hi = max(0, start - buffer), min(n, start + length + buffer)
        if occupied[lo:hi].any() or forbidden[start : start + length].any():
            continue
        return start
    return None


def _smooth_window(length: int, ramp: int = 3) -> np.ndarray:
    """0 -> 1 -> 0 envelope with short linear ramps at both ends."""
    env = np.ones(length)
    ramp = min(ramp, length // 2)
    if ramp > 0:
        edge = np.linspace(0.0, 1.0, ramp + 2)[1:-1]
        env[:ramp] = edge
        env[-ramp:] = edge[::-1]
    return env


def inject_anomalies(
    df: pd.DataFrame, rng: np.random.Generator, events_per_type: dict[str, int] | None = None
) -> pd.DataFrame:
    """Overlay labelled anomaly patterns on one segment's normal data.

    Patterns (all labelled in ``anomaly_type``):

    * ``spike``                   - single-sample jump in one sensor.
    * ``pressure_flow_deviation`` - pressure falls 4-9 % while flow is flat or
      slightly up, breaking the usual pressure-flow relationship (1-3 h).
    * ``drift``                   - one sensor ramps slowly away (6-18 h).
    * ``flatline``                - one sensor repeats the same value (1-3 h).
    * ``noise_burst``             - one sensor becomes 4-7x noisier (1-3 h).
    * ``temperature_excursion``   - temperature steps up 2.5-5 degC (2-6 h).
    """
    df = df.copy().reset_index(drop=True)
    events_per_type = events_per_type or EVENTS_PER_SEGMENT_90D
    n = len(df)
    sph = cfg.SAMPLES_PER_HOUR
    segment = df["segment_id"].iat[0]

    occupied = np.zeros(n, dtype=bool)
    shutdown = (df["operating_state"] == "SHUTDOWN").to_numpy()
    # Keep events out of shutdowns (plus 30 min either side): a flat or noisy
    # sensor has no clear meaning when the line is not flowing.
    forbidden = pd.Series(shutdown).rolling(13, center=True, min_periods=1).max().to_numpy() > 0

    values = {c: df[c].to_numpy(dtype=float).copy() for c in cfg.SENSOR_COLS}
    is_anom = np.zeros(n, dtype=int)
    a_type = np.array(["none"] * n, dtype=object)
    a_event = np.array([""] * n, dtype=object)
    a_sensor = np.array([""] * n, dtype=object)
    counter = 0

    def register(start: int, length: int, kind: str, affected: str) -> None:
        nonlocal counter
        counter += 1
        sl = slice(start, start + length)
        occupied[sl] = True
        is_anom[sl] = 1
        a_type[sl] = kind
        a_event[sl] = f"{segment}-E{counter:03d}"
        a_sensor[sl] = affected

    for kind, count in events_per_type.items():
        for _ in range(count):
            if kind == "spike":
                length = 1
            elif kind == "drift":
                length = int(rng.integers(6 * sph, 18 * sph + 1))
            elif kind == "temperature_excursion":
                length = int(rng.integers(2 * sph, 6 * sph + 1))
            else:
                length = int(rng.integers(1 * sph, 3 * sph + 1))

            start = _find_window(rng, n, length, occupied, forbidden)
            if start is None:  # no room for this event (very short run or crowded timeline)
                continue
            sl = slice(start, start + length)
            sensor = str(rng.choice(cfg.SENSOR_COLS))
            sign = float(rng.choice([-1.0, 1.0]))

            if kind == "spike":
                values[sensor][sl] += sign * rng.uniform(8, 20) * NOISE_SD[sensor]
            elif kind == "pressure_flow_deviation":
                env = _smooth_window(length)
                values["pressure_kpa"][sl] *= 1.0 - rng.uniform(0.04, 0.09) * env
                values["flow_m3h"][sl] *= 1.0 + rng.uniform(0.01, 0.03) * env
            elif kind == "drift":
                final = sign * rng.uniform(10, 22) * NOISE_SD[sensor]
                values[sensor][sl] += np.linspace(0.0, final, length)
            elif kind == "flatline":
                values[sensor][sl] = values[sensor][start]
            elif kind == "noise_burst":
                values[sensor][sl] += rng.normal(0, rng.uniform(4, 7) * NOISE_SD[sensor], length)
            elif kind == "temperature_excursion":
                values["temperature_c"][sl] += rng.uniform(2.5, 5.0) * _smooth_window(length)
            else:  # pragma: no cover
                raise ValueError(f"Unknown anomaly type: {kind}")
            affected = {
                "pressure_flow_deviation": "pressure_kpa",  # primary; flow moves slightly too
                "temperature_excursion": "temperature_c",
            }.get(kind, sensor)
            register(start, length, kind, affected)

    for c in cfg.SENSOR_COLS:
        df[c] = values[c]
    df["flow_m3h"] = df["flow_m3h"].clip(lower=0.0)
    df["is_anomaly"] = is_anom
    df["anomaly_type"] = a_type
    df["anomaly_event_id"] = a_event
    df["anomaly_sensor"] = a_sensor
    return df


# --------------------------------------------------------------------------
# Layer 3: data-quality defects
# --------------------------------------------------------------------------
def inject_data_quality_defects(
    df: pd.DataFrame,
    rng: np.random.Generator,
    missing_cell_rate: float = 0.003,
    gaps_per_segment: int = 2,
    duplicate_rate: float = 0.001,
    sentinels_per_segment: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Add realistic acquisition defects and return ``(df, manifest)``.

    The manifest records exactly how many defects were injected so that the
    validation stage can be checked against known truth.
    """
    df = df.copy().reset_index(drop=True)
    manifest = {"missing_cells": 0, "sentinel_cells": 0, "dropped_rows": 0, "duplicate_rows": 0}

    # (a) isolated missing cells
    for c in cfg.SENSOR_COLS:
        mask = rng.random(len(df)) < missing_cell_rate
        df.loc[mask, c] = np.nan
        manifest["missing_cells"] += int(mask.sum())

    # (b) sentinel values written by a (simulated) faulty transmitter
    for _, idx in df.groupby("segment_id").groups.items():
        chosen = rng.choice(np.asarray(idx), size=sentinels_per_segment, replace=False)
        for row in chosen:
            col = str(rng.choice(cfg.SENSOR_COLS))
            if not np.isnan(df.at[row, col]):
                manifest["sentinel_cells"] += 1
                df.at[row, col] = SENTINEL_VALUE

    # (c) communication gaps: whole rows missing for 15-50 minutes
    drop_idx: list[int] = []
    for _, idx in df.groupby("segment_id").groups.items():
        idx = np.asarray(idx)
        for _ in range(gaps_per_segment):
            length = int(rng.integers(3, 11))
            start = int(rng.integers(100, len(idx) - length - 100))
            drop_idx.extend(idx[start : start + length].tolist())
    drop_idx = sorted(set(drop_idx))
    manifest["dropped_rows"] = len(drop_idx)
    df = df.drop(index=drop_idx)

    # (d) exact duplicate rows (e.g. a re-sent telemetry packet)
    n_dup = int(round(len(df) * duplicate_rate))
    dups = df.sample(n=n_dup, random_state=int(rng.integers(0, 2**31 - 1)))
    manifest["duplicate_rows"] = n_dup
    df = pd.concat([df, dups], ignore_index=True)
    df = df.sort_values(["segment_id", "timestamp"], kind="stable").reset_index(drop=True)
    return df, manifest


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def generate_dataset(
    days: int = cfg.SIMULATION_DAYS,
    seed: int = cfg.RANDOM_SEED,
    start: str = "2025-01-01",
    with_defects: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Generate the full multi-segment dataset. Returns ``(df, manifest)``."""
    root = np.random.SeedSequence(seed)
    schedule_seed, defect_seed, *segment_seeds = root.spawn(2 + 2 * len(SEGMENTS))

    n = days * 24 * cfg.SAMPLES_PER_HOUR
    timestamps = pd.date_range(start, periods=n, freq=f"{cfg.SAMPLE_MINUTES}min")
    states = simulate_operating_schedule(n, np.random.default_rng(schedule_seed))
    scale = days / 90.0
    events = {k: max(1, int(round(v * scale))) for k, v in EVENTS_PER_SEGMENT_90D.items()}

    frames = []
    for i, profile in enumerate(SEGMENTS):
        normal = simulate_normal_operation(
            profile, timestamps, states, np.random.default_rng(segment_seeds[2 * i])
        )
        frames.append(inject_anomalies(normal, np.random.default_rng(segment_seeds[2 * i + 1]), events))
    df = pd.concat(frames, ignore_index=True)

    manifest = {
        "synthetic": True,
        "seed": seed,
        "days": days,
        "start": start,
        "sample_minutes": cfg.SAMPLE_MINUTES,
        "segments": [p.segment_id for p in SEGMENTS],
        "rows_before_defects": int(len(df)),
        "anomalous_rows_before_defects": int(df["is_anomaly"].sum()),
        "events_per_segment": events,
    }
    if with_defects:
        df, defects = inject_data_quality_defects(df, np.random.default_rng(defect_seed))
        manifest["defects"] = defects

    df["pressure_kpa"] = df["pressure_kpa"].round(1)
    df["flow_m3h"] = df["flow_m3h"].round(2)
    df["temperature_c"] = df["temperature_c"].round(3)
    manifest["rows_written"] = int(len(df))
    return df[cfg.RAW_COLUMNS], manifest


def write_dataset(path: Path = cfg.RAW_FILE, **kwargs) -> tuple[pd.DataFrame, dict]:
    """Generate the dataset and write the CSV plus a JSON manifest beside it."""
    df, manifest = generate_dataset(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the compressed file byte-identical between runs (no git churn)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else "infer"
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S", compression=compression)
    path.with_name("generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    return df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic pipeline sensor data.")
    parser.add_argument("--days", type=int, default=cfg.SIMULATION_DAYS)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=cfg.RAW_FILE)
    args = parser.parse_args()
    df, manifest = write_dataset(args.out, days=args.days, seed=args.seed)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

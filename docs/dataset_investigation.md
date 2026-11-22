# Dataset investigation

**Goal:** find data with pipeline-style structure (several segments, pressure / flow / temperature, operating
states) and trustworthy anomaly labels, without using sensitive infrastructure data.

> **Note on verification.** The candidates below were assessed from general knowledge of these well-known
> benchmarks. They were **not downloaded or inspected as part of this project**, so their exact contents and
> licence terms should be confirmed at the source before any use.

| Candidate | What it is | Fit for this project |
|---|---|---|
| Skoltech Anomaly Benchmark (SKAB) | Labelled multivariate data from a laboratory water-circulation rig, including pressure, temperature and flow-rate channels | Closest public match and the recommended next step. A single small rig rather than multi-segment pipeline operation with operating states |
| Numenta Anomaly Benchmark (NAB) | Collection of labelled univariate streams from several domains | Univariate; no related pressure / flow / temperature channels |
| NASA SMAP / MSL telemetry | Labelled spacecraft telemetry channels | Different domain; anonymised channels with no physical relationships to exploit |
| Tennessee Eastman Process | Simulated chemical plant with fault scenarios | Also simulated, and a chemical process rather than a pipeline |
| SWaT / WADI (iTrust) | Water treatment / distribution testbeds with attack labels | Access is by request; cyber-attack labels are a different problem |
| Real pipeline SCADA data | Operator historian data | Not public: commercially and security sensitive. Deliberately not sought |

## Decision

Generate synthetic data. Reasons:

1. **Structure.** No public dataset combines segments, operating states and the three sensors of interest.
2. **Ground truth.** Injected anomalies have exact start, end, type and affected sensor, enabling event-level and
   per-pattern evaluation.
3. **Responsibility.** No real infrastructure data, locations or vulnerabilities are involved.
4. **Reproducibility.** Anyone can regenerate the exact dataset from a seed, offline.

## Cost of that decision

Synthetic data proves that the pipeline is correct and shows how methods behave on known patterns. It cannot show
that the methods work on real equipment. The simulator's simplifications and their consequences are listed in
`README.md` (Data limitations) and `docs/methodology.md`. Validating on measured data such as SKAB is the first
item under future work.

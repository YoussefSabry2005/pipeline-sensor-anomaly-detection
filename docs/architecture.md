# Project architecture

> This page covers the **batch** project. The streaming simulation built on top of it is described in
> [`streaming_architecture.md`](streaming_architecture.md).

## Data flow

```
data_generation.py ──► data/raw/synthetic_pipeline_sensors.csv.gz (+ generation_manifest.json)
                              │
data_loading.py  ── load_raw: schema check, type coercion
                              │
validation.py    ── validate (report only) ──► clean (documented fixes + log)
                              │
splits.py        ── chronological train / val / test labels
                              │
features.py      ── PressureFlowModel.fit(train only) ──► build_features (causal)
                              │
detectors.py     ── fit on monitored training rows ──► score every row ──► label-free threshold
                              │
evaluation.py    ── point-wise, ranking, event-level, per-pattern, false-positive breakdown
                              │
visualization.py ── figures/*.png          pipeline.py ── reports/*.csv, metrics.json, models/*.joblib
```

`pipeline.py` orchestrates the stages (`prepare_data` → `fit_and_score` → `evaluate` → `save_outputs` →
`make_all_figures`). The notebook calls the same functions, so the notebook, the command-line run and the tests
all exercise one code path.

## Modules

| Module | Responsibility | Key design choice |
|---|---|---|
| `config.py` | Paths, schema, window lengths, alert budget, split fractions | Every modelling decision is a named constant in one place |
| `data_generation.py` | Simulator: normal operation, anomalies, data defects | Deterministic from a seed; writes a manifest of what was injected |
| `data_loading.py` | Read raw file, enforce schema and types | Fails loudly on missing columns; never drops rows |
| `validation.py` | `validate` inspects, `clean` fixes | Inspection and modification are separate functions |
| `splits.py` | Time-based split | No random splitting of time series |
| `features.py` | Causal features, pressure–flow model, settling flag | Trailing windows only; fitted parts use training data only |
| `detectors.py` | Four detectors + score utilities | One interface; thresholding separated from scoring |
| `evaluation.py` | Metrics against synthetic labels | Event-level and per-pattern views alongside point-wise metrics |
| `visualization.py` | All figures | Each function returns a `Figure`; consistent colours per entity |
| `pipeline.py` | Orchestration and persistence | Stages are plain functions, reusable from notebook and tests |

## Directory layout

| Path | Contents | In git? |
|---|---|---|
| `data/raw/` | Synthetic dataset and manifest | Yes (2.4 MB compressed) |
| `data/processed/` | Clean data, feature table, scores | No (≈250 MB, regenerated) |
| `models/` | `detectors.joblib` (detectors, thresholds, pressure–flow model) | No (regenerated) |
| `figures/` | 16 PNG figures | Yes |
| `reports/` | Reports, metrics, comparison tables | Yes |
| `notebooks/` | Executed notebook | Yes |
| `docs/` | This documentation (the "documentation" folder) | Yes |
| `tests/` | pytest suite | Yes |

## Testing

The batch code is covered by 56 tests in seven files (the streaming code adds 70 more), run on a small
separately-seeded dataset so they are fast and independent of the committed data:

| File | Covers |
|---|---|
| `test_data_generation.py` | Determinism, schema, label consistency, manifest accuracy |
| `test_data_loading.py` | Types, round trip, missing file / column, unparseable values |
| `test_validation.py` | Injected defects are found; grid regularity; causal, limited forward-fill |
| `test_features.py` | **No look-ahead**, feature definitions, settling flag, chronological split |
| `test_detectors.py` | Detector contract, reproducibility, score validation, thresholds, percentiles |
| `test_evaluation.py` | Hand-computed metrics, event detection and delay, edge cases |
| `test_pipeline.py` | End-to-end scoring; settling rows never alert; evaluation only on test period |

# Data dictionary

All values are **synthetic**. Identifiers are invented and do not refer to real assets.

## Raw file: `data/raw/synthetic_pipeline_sensors.csv.gz`

One row per segment per 5-minute timestamp (before injected defects).

| Column | Type | Unit / values | Description |
|---|---|---|---|
| `timestamp` | datetime | 5-minute grid, no time zone | Time of the reading |
| `segment_id` | string | `SEG-A` … `SEG-D` | Synthetic pipeline segment |
| `sensor_id` | string | `STN-101` … `STN-104` | Synthetic instrument station on that segment (one station per segment reports all three measurements) |
| `pressure_kpa` | float | kPa | Simulated line pressure. May be missing or `-999` (sentinel defect) |
| `flow_m3h` | float | m³/h | Simulated volumetric flow rate. May be missing or `-999` |
| `temperature_c` | float | °C | Simulated fluid temperature. May be missing or `-999` |
| `operating_state` | string | `NORMAL`, `HIGH_FLOW`, `LOW_FLOW`, `SHUTDOWN` | Commanded operating mode, shared by all segments |
| `is_anomaly` | int | 0 / 1 | 1 if the simulator injected an anomaly at this sample |
| `anomaly_type` | string | `none`, `spike`, `pressure_flow_deviation`, `drift`, `flatline`, `noise_burst`, `temperature_excursion` | Simulated pattern |
| `anomaly_event_id` | string | e.g. `SEG-A-E012`, empty if none | Groups the samples of one event |
| `anomaly_sensor` | string | sensor column name, empty if none | Sensor the pattern was applied to (primary sensor for `pressure_flow_deviation`) |

The label columns are simulator ground truth. They are used **only** for evaluation, never as model inputs.

`data/raw/generation_manifest.json` records the seed, run length, events per segment and the exact number of
defects injected.

## Columns added by cleaning (`data/processed/sensors_clean.csv`)

| Column | Type | Description |
|---|---|---|
| `row_inserted` | bool | Row did not exist in the raw file; inserted to complete the time grid |
| `n_filled` | int | Number of sensor values in this row that were forward-filled (0–3) |
| `usable` | bool | All three sensor values are present after cleaning |

## Engineered features (`data/processed/sensors_features.csv`)

`<s>` is one of `pressure`, `flow`, `temperature`. All windows are trailing.

| Column | Description | Used by models? |
|---|---|---|
| `<s>_lag_1`, `<s>_lag_1h` | Reading 5 minutes / 1 hour earlier | No (state-dependent level) |
| `<s>_diff_1`, `<s>_diff_1h` | Change over 5 minutes / 1 hour | `diff_1` yes |
| `<s>_roll_mean_1h`, `<s>_roll_std_1h` | Mean / standard deviation of the last hour | No |
| `<s>_roll_mean_6h`, `<s>_roll_std_6h` | Mean / standard deviation of the last 6 hours | No |
| `<s>_z_6h` | Classic z-score against the previous 6 hours | No (comparison only) |
| `<s>_rz_1h` | Robust z-score: deviation from the previous hour's median, scaled by the typical step size | Yes |
| `<s>_ewma_resid` | Reading minus exponentially weighted average of the past (span 1 h) | Yes |
| `<s>_vol_ratio` | Log ratio of recent to usual sample-to-sample movement (negative = unusually quiet) | Yes |
| `<s>_trend_3h` | Change in the 1-hour mean over 3 hours | `temperature` only |
| `pf_resid_z` | Observed minus expected pressure given flow, in robust standard units | Yes |
| `pf_resid_mean_1h` | 1-hour mean of `pf_resid_z` | Yes |
| `pf_resid_trend_3h` | Change in `pf_resid_mean_1h` over 3 hours | Yes |
| `minutes_since_state_change` | Minutes since `operating_state` last changed | No (defines settling window) |
| `is_shutdown` | 1 during `SHUTDOWN` | No |
| `hour_sin`, `hour_cos`, `day_of_week` | Calendar features | No (anomalies are not time-of-day dependent here) |
| `feature_ready` | Real, usable row with every model feature available | Row filter |
| `in_settling` | Within 60 minutes after a state change | Row filter |
| `monitored` | `feature_ready` and not `in_settling` | Row filter |
| `split` | `train`, `val` or `test` | Row filter |

## Scores (`data/processed/anomaly_scores.csv`)

`<m>` is one of `zscore`, `iforest`, `lof`, `pca`, `ensemble`.

| Column | Description |
|---|---|
| `score__<m>` | Anomaly score (higher = more unusual); settling rows set to the training minimum |
| `pct__<m>` | Score as a percentile of the training scores (0–1) |
| `flag__<m>` | 1 if the score exceeds the detector's alert-budget threshold |

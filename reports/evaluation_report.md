# Evaluation report

> **Scope.** All data is synthetic. These results measure recovery of the anomaly patterns injected by this
> project's simulator on a held-out period. They are **not** estimates of performance on a real pipeline and must
> not be quoted as such. Source of every number: `reports/metrics.json`, produced by
> `python -m pipeline_anomaly.pipeline` (seed 42).

## 1. Setup

| Item | Value |
|---|---|
| Rows modelled | 207,143 (of 207,360 on the clean grid; the rest are warm-up or inserted rows) |
| Train | 2025-01-01 to 2025-04-18 · 124,233 rows · 228 events · 4.0 % anomalous |
| Validation | 2025-04-19 to 2025-05-24 · 41,438 rows · 83 events · 2.6 % anomalous |
| **Test** | 2025-05-25 to 2025-06-29 · 41,472 rows · 81 events · 1,292 anomalous samples (3.1 %) |
| Features | 16 causal features (`features.MODEL_FEATURES`) |
| Threshold | 97th percentile of each detector's training scores (3 % alert budget, no labels used) |
| Settling rule | Rows within 60 min of a state change cannot alert (4.9 % of rows) |

Test events by pattern: spike 52, pressure–flow deviation 7, noise burst 7, flatline 6, temperature excursion 5,
drift 4.

## 2. Main results (test period, label-free threshold)

| Detector | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | Event recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.197 | 0.192 | 0.195 | 0.025 | 0.603 | 0.141 | 0.840 | 248 | 1,009 | 1,044 | 39,171 |
| Isolation Forest | 0.371 | 0.365 | 0.368 | 0.020 | 0.845 | 0.258 | 0.852 | 471 | 799 | 821 | 39,381 |
| Local Outlier Factor | 0.362 | 0.357 | 0.359 | 0.020 | 0.703 | 0.236 | 0.926 | 461 | 812 | 831 | 39,368 |
| PCA reconstruction | 0.298 | 0.275 | 0.286 | 0.021 | 0.698 | 0.153 | 0.519 | 355 | 836 | 937 | 39,344 |
| Rank-average ensemble | 0.241 | 0.243 | 0.242 | 0.025 | 0.748 | 0.150 | 0.778 | 314 | 989 | 978 | 39,191 |

No-skill PR-AUC baseline = prevalence = 0.031. Figures: `figures/10_model_comparison.png`,
`11_pr_roc_curves.png`, `12_confusion_matrices.png`.

**Reading.** Isolation Forest and LOF are the best at the operating threshold and about twice the baseline's F1.
Isolation Forest ranks best overall (ROC-AUC 0.85). LOF surfaces the most events (93 %). PCA catches only about half
of the events. The ensemble is worse than its two best members.

## 3. Recall by simulated pattern

Event recall (share of events with ≥ 1 alert):

| Pattern (test events) | Z-score | Isolation Forest | LOF | PCA | Ensemble |
|---|---|---|---|---|---|
| Spike (52) | 0.88 | 0.88 | 0.88 | 0.37 | 0.75 |
| Pressure–flow deviation (7) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Drift (4) | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| Flatline (6) | 0.00 | 0.00 | 1.00 | 0.00 | 0.17 |
| Noise burst (7) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Temperature excursion (5) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Point-wise recall (share of anomalous samples flagged):

| Pattern (test samples) | Z-score | Isolation Forest | LOF | PCA | Ensemble |
|---|---|---|---|---|---|
| Spike (52) | 0.88 | 0.88 | 0.88 | 0.37 | 0.75 |
| Pressure–flow deviation (167) | 0.43 | 0.73 | 0.31 | 0.36 | 0.41 |
| Drift (548) | 0.02 | 0.27 | 0.10 | 0.13 | 0.04 |
| Flatline (148) | 0.00 | 0.00 | 0.57 | 0.00 | 0.01 |
| Noise burst (161) | 0.37 | 0.46 | 0.83 | 0.60 | 0.66 |
| Temperature excursion (216) | 0.29 | 0.37 | 0.43 | 0.51 | 0.36 |

Median delay from event start to first alert: 0–10 minutes for spikes, noise bursts, temperature excursions and
pressure–flow deviations; **175–365 minutes for drift**; 47.5 minutes for flatline (LOF).

With 4–7 events per pattern (spikes excepted) these cells have wide uncertainty; one event changes a cell by
14–25 percentage points.

## 4. Threshold sensitivity (label-informed reference)

Threshold chosen to maximise F1 on the validation period, then applied to the test period:

| Detector | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|
| Rolling z-score | 0.429 | 0.138 | 0.209 | 0.790 |
| Isolation Forest | 0.369 | 0.433 | 0.398 | 0.852 |
| Local Outlier Factor | 0.414 | 0.314 | 0.357 | 0.926 |
| PCA reconstruction | 0.314 | 0.254 | 0.281 | 0.481 |
| Rank-average ensemble | 0.334 | 0.192 | 0.244 | 0.691 |

Using labels to set the threshold changes F1 by at most 0.03. The limit on performance is how well the scores
rank anomalies, not where the line is drawn.

## 5. Ablation: without the settling-window rule

| Detector | Precision | Recall | F1 | Event recall |
|---|---|---|---|---|
| Rolling z-score | 0.094 | 0.248 | 0.136 | 0.938 |
| Isolation Forest | 0.158 | 0.420 | 0.230 | 0.938 |
| Local Outlier Factor | 0.167 | 0.411 | 0.238 | 1.000 |
| PCA reconstruction | 0.136 | 0.327 | 0.192 | 0.605 |
| Rank-average ensemble | 0.114 | 0.298 | 0.165 | 0.864 |

The rule roughly doubles precision and costs 7–10 points of event recall: it hides the events that begin just
after a state change. Whether that trade is right depends on the review capacity of the team using the alerts.

## 6. False-positive analysis (test period, samples)

| Detector | Steady operation | Within 1 h after a labelled event | During shutdown | Total |
|---|---|---|---|---|
| Rolling z-score | 658 | 141 | 210 | 1,009 |
| Isolation Forest | 481 | 144 | 174 | 799 |
| Local Outlier Factor | 396 | 350 | 66 | 812 |
| PCA reconstruction | 513 | 208 | 115 | 836 |
| Rank-average ensemble | 650 | 202 | 137 | 989 |

* **Steady operation** (49–66 %): ordinary noise tails. A fixed alert budget always fills itself.
* **After an event** (14–43 %): the rolling windows still contain the anomaly. An analyst would see these as the
  same episode; grouping alerts into episodes would remove most of them.
* **Shutdown** (8–22 %): temperature continues to fall for hours, longer than the 60-minute settling window, and
  flow noise collapses, which resembles a stuck sensor.

## 7. Feature analysis

Single-feature ROC-AUC on the test period (distance from the training median):

| Feature | ROC-AUC |
|---|---|
| `pf_resid_z` | 0.814 |
| `pf_resid_mean_1h` | 0.777 |
| `pf_resid_trend_3h` | 0.727 |
| `pressure_vol_ratio` | 0.605 |
| all other features | 0.51 – 0.56 |

The three physics-based residual features dominate. Per-sensor features are individually weak because each one
only responds to anomalies on its own sensor; they contribute in combination. Full table:
`reports/feature_univariate_auc.csv`.

## 8. Potential missed anomalies and false positives

**Likely to be missed**

* The early part of any slow drift, and slow temperature drift in particular (no physical reference for temperature).
* Stuck sensors, unless a density-based detector or a dedicated rule is used.
* Anything beginning within 60 minutes of a state change.
* Anything in the first ~6 hours of data or immediately after a data gap longer than one hour.
* Patterns unlike the six simulated here. This evaluation says nothing about them.

**Likely false positives**

* Random noise excursions in steady operation.
* The hour after a genuine event.
* Long thermal transients after shutdowns and restarts.

## 9. What these results do and do not support

Supported: the pipeline is correct and leakage-free; multivariate detectors with a physics-based feature
outperform a per-sensor baseline on these simulated patterns; operating-state handling matters more than model
choice; detectors are complementary.

Not supported: any claim about real pipelines, about leak or failure detection, about operational or financial
benefit, or about performance on anomaly types that were not simulated.

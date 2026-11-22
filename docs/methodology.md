# Methodology and technical decisions

Each section states the decision, the alternative considered, and the reason.

## 1. Data generation

* **Three layers**: normal operation → labelled anomalies → data-quality defects. Keeping them separate means
  process anomalies (to be detected) and data defects (to be cleaned) are never confused.
* **Normal operation.** One operating-state schedule is shared by all segments because they are in series.
  Flow follows the state through a first-order lag with a ±2.5 % daily cycle. Pressure = static head + friction
  coefficient × flow². Temperature = segment base + seasonal term + daily cycle + frictional heating with a slow
  lag. Noise is AR(1) rather than white, because real sensor noise is autocorrelated and white noise would make
  detection unrealistically easy.
* **Anomalies** are placed at random, never overlapping, never during shutdown (a flat or noisy reading has no
  clear meaning on a line that is not flowing), but they *can* overlap state changes.
* **Defects**: 0.3 % missing cells, two communication gaps per segment, 0.1 % duplicate rows, ten `-999`
  sentinel values per segment. Counts are written to a manifest so validation can be checked against truth.

## 2. Validation and cleaning

| Decision | Alternative | Reason |
|---|---|---|
| Out-of-range values → missing | Keep and let detectors flag them | They are telemetry artefacts; scoring them would inflate detection results with trivial catches |
| Forward-fill, max 1 hour | Linear interpolation | Interpolation uses the next reading: look-ahead a live monitor would not have |
| Insert rows for missing timestamps | Leave gaps | Rolling windows are defined in samples and assume a regular grid |
| Flag rather than delete | Drop bad rows | Keeps the grid regular and the decisions auditable |
| Inserted rows excluded from evaluation | Score them | No real observation exists to judge |

## 3. Split

Chronological 60 / 20 / 20. A random split would place samples of the same event, and overlapping rolling
windows, on both sides. The validation period is used only for the label-informed reference threshold.

## 4. Features and leakage

* Windows are trailing. Baseline statistics use `shift(1)` so the current reading is compared with the past only.
* The pressure–flow curve is the only fitted feature component and is fitted on training rows only, with a robust
  (MAD) residual scale so anomalies in the training period do not inflate it.
* `tests/test_features.py::test_no_look_ahead` rewrites the second half of a series and asserts that every
  feature in the first half is unchanged.
* **Raw levels are excluded from the model matrix.** Levels identify the operating state; an unsupervised model
  given levels learns that shutdown is "rare" and flags it.
* **Classic vs robust z-score.** A 6-hour mean / standard deviation baseline is inflated for hours after every
  state change, which first triggers alerts and then hides real ones. The robust version uses a 1-hour median for
  the level and the rolling *median* of absolute sample-to-sample changes for the scale; a median ignores the few
  large steps of a transition. Both are in the feature table; only the robust one is used.
* **Sigma floors** (`config.SIGMA_FLOOR`) stop near-constant windows from producing enormous z-scores. They play
  the role of an instrument resolution supplied by an engineer.

## 5. Operating-state handling: the settling window

For 60 minutes after a commanded state change, rows are *not monitored*: excluded from training and unable to
alert. This is a simplified version of state-based alarm suppression. It is a trade, measured in the evaluation:

* benefit: Isolation Forest precision 0.16 → 0.37 on the test period;
* cost: recall 0.42 → 0.36, and a blind spot covering 4.9 % of rows (328 labelled anomalous samples fall inside it
  across the whole dataset).

Anomalies inside settling windows remain in the evaluation and count as missed.

## 6. Detectors

| Detector | Parameters | Reasoning |
|---|---|---|
| Rolling z-score | max of three robust z-scores | Simplest defensible baseline; anything more complex must beat it |
| Isolation Forest | 300 trees, 512 samples per tree, `contamination="auto"` | More trees stabilise scores; sub-sampling is what makes isolation work; contamination is irrelevant because thresholding is done separately |
| Local Outlier Factor | 50 neighbours, novelty mode, 20,000-row training subsample | Larger neighbourhoods are less sensitive to small clusters of training anomalies; subsample keeps neighbour search fast |
| PCA reconstruction | components explaining 90 % of variance | Standard choice for the residual (Q / SPE) statistic |
| Ensemble | mean of training-referenced percentiles | Percentiles make incomparable score scales commensurable |

Preprocessing for the learned models: `RobustScaler` then clipping to ±10. Hyper-parameters were set from
standard practice and **not** searched, because the only labels are synthetic and tuning to them would be
tuning to the simulator.

Not included, deliberately: One-Class SVM (poor scaling to 124k training rows and adds little beyond LOF here);
deep sequence models (unjustified before simple baselines are understood; listed as future work).

## 7. Thresholding

Primary: 97th percentile of each detector's own training scores (3 % alert budget). This needs no labels, gives
every detector the same alert volume on training data, and corresponds to a question a team can answer: "how
much can we review?". Secondary, for reference only: the F1-maximising threshold on the validation period.

## 8. Evaluation design

* **Point-wise metrics** are strict: a 12-hour drift counts as 144 separate samples to find.
* **Event recall** and **detection delay** reflect use: was the episode surfaced at all, and how fast?
* **PR-AUC** is preferred over ROC-AUC at 3 % prevalence; ROC-AUC is reported for completeness.
* **Per-pattern recall** shows which behaviours each method can and cannot see.
* **False-positive breakdown** separates steady-operation noise, post-event recovery and shutdown periods.
* **Ablations**: label-informed thresholds; no settling window.

## 9. Honesty notes

* All labels are synthetic; see the README disclosure.
* After a first full run (90 days, classic rolling features, no settling rule) showed state changes dominating
  the alerts, the features and the settling rule were redesigned and the simulation extended to 180 days so the
  test period would contain more events. This was a design change, not threshold tuning, but it means the test
  period informed one design decision.
* Per-pattern results rest on 4–7 events for five of the six patterns.

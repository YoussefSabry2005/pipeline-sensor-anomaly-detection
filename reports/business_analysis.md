# Business analysis: where anomaly detection could help, and where it should not be used

> This project uses synthetic data. The discussion below describes how this *kind* of analysis could support
> analysts; it does not report benefits observed in any real operation.

## The problem it addresses

A monitoring team receives far more sensor data than people can inspect. Four segments at 5-minute sampling
already produce about 1,150 rows a day; real systems have thousands of tags at higher rates. Most readings are
unremarkable. The useful question is: *which few readings deserve a person's attention first?*

Anomaly detection answers that with a ranked list. It does not answer *why* a reading is unusual or whether it
matters. That judgement stays with people who know the system.

## Ways it could support analysts

**1. Sensor data review.**
Instead of paging through trends, an analyst opens the periods with the highest scores. In this study, a 3 %
review budget surfaced 85–93 % of simulated events at least once (Isolation Forest, LOF).

**2. Prioritising observations for further investigation.**
Scores rank; they do not just flag. Sorting by score and by how long a score stays high puts sustained,
strongly unusual periods ahead of single noisy samples. The alert budget is an explicit, adjustable statement of
how much the team can review.

**3. Monitoring analytics.**
Per-pattern results show which behaviours each method sees. A team could run a density-based detector for
stuck-sensor behaviour and an isolation-based one for relationship breaks, rather than relying on one model.

**4. Operational reporting.**
Counts of alerts by segment, sensor, shift or week show where unusual behaviour concentrates and whether it is
increasing. This is descriptive reporting, useful for planning reviews and maintenance discussions.

**5. Data quality investigation.**
Several simulated patterns (flatline, noise burst, spikes) are what instrument and telemetry problems look like.
The validation stage separately catches duplicates, gaps and sentinel values. Finding these protects every
downstream use of the data, including reporting and modelling.

## What the study says about making it usable

* **Operating context is essential.** Without the settling-window rule, precision roughly halved because normal
  state changes were flagged. A system that ignores operating state will bury analysts in alerts.
* **Encode engineering knowledge.** The single most informative feature was the deviation from the expected
  pressure–flow relationship. Collaboration with people who understand the hydraulics is worth more than a
  more complex algorithm.
* **Expect a review workload.** At the chosen budget about two of every three alerted samples were not labelled
  anomalous. Grouping alerts into episodes and suppressing post-event echoes would reduce this.
* **Know the blind spots.** Slow drift was found hours late; stuck sensors were missed by three of four methods.
  A team should know what its tooling cannot see.
* **No single model wins everywhere,** and naive averaging made things worse.

## What this must not be used for

* **Not a safety system.** It has not been validated on real data, against real incidents, or to any standard.
* **Not leak detection.** A pressure–flow deviation in this simulator is a pattern, not a leak. Real leak
  detection relies on engineered, validated and regulated methods.
* **No automatic operational actions.** Scores should inform a person's review queue, never open or close a
  valve, stop a pump or raise an operational alarm.
* **Not a replacement for existing alarms or procedures.** At most it is an additional lens for analysts.

## What would be needed before any real use

1. Real historical data with reviewed event records, under appropriate data governance.
2. Evaluation with operations and integrity specialists on what counts as a useful alert.
3. Measurement of analyst workload and missed-event rates over a long shadow period.
4. Alert grouping, explanations and feedback capture.
5. Monitoring of the detectors themselves (data drift, score drift) and a retraining policy.

## Summary

Anomaly detection is a way to direct limited human attention. This project shows the mechanics, the importance
of operating context and domain features, and the kinds of mistakes such a system makes. It does not show, and
does not claim, any real-world operational benefit.

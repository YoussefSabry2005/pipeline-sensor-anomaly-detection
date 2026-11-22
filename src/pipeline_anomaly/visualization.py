"""Figures for the report, README and notebook.

Every function returns a Matplotlib ``Figure``; ``make_all_figures`` saves the
full set to ``figures/``. All figures are drawn from SYNTHETIC data and say so
in a footer.

Style choices: one y-axis per panel (never dual axes), a fixed colour per
entity (model / segment / state) across all figures, thin marks, light grid,
and a legend whenever a panel shows more than one series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as cfg

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
LABEL_BAND = "#c3c2b7"  # neutral shading for simulator-labelled anomaly periods

MODEL_COLORS = {
    "Rolling z-score": BLUE,
    "Isolation Forest": ORANGE,
    "Local Outlier Factor": AQUA,
    "PCA reconstruction": YELLOW,
    "Rank-average ensemble": MAGENTA,
}
MODEL_KEYS = {
    "Rolling z-score": "zscore",
    "Isolation Forest": "iforest",
    "Local Outlier Factor": "lof",
    "PCA reconstruction": "pca",
    "Rank-average ensemble": "ensemble",
}
SEGMENT_COLORS = {"SEG-A": BLUE, "SEG-B": ORANGE, "SEG-C": AQUA, "SEG-D": YELLOW}
STATE_COLORS = {"NORMAL": BLUE, "HIGH_FLOW": ORANGE, "LOW_FLOW": AQUA, "SHUTDOWN": YELLOW}
SENSOR_LABELS = {
    "pressure_kpa": "Pressure (kPa)",
    "flow_m3h": "Flow rate (m³/h)",
    "temperature_c": "Temperature (°C)",
}
TYPE_LABELS = {
    "spike": "Spike",
    "pressure_flow_deviation": "Pressure–flow deviation",
    "drift": "Drift",
    "flatline": "Flatline",
    "noise_burst": "Noise burst",
    "temperature_excursion": "Temperature excursion",
}
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#86b6ef", "#2a78d6", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
FOOTER = "Synthetic data — simulated for demonstration; not real pipeline measurements."


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "figure.dpi": 110,
            "savefig.dpi": 170,
        }
    )


def _finish(fig, title: str, subtitle: str | None = None, legend_above: bool = False):
    """Title block with fixed spacing in inches, footer disclosure, tight layout."""
    h = fig.get_figheight()
    n_lines = subtitle.count("\n") + 1 if subtitle else 0
    fig.text(0.012, 1 - 0.10 / h, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 1 - 0.42 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK2,
                 linespacing=1.35)
    fig.text(0.012, 0.06 / h, FOOTER, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    header = 0.50 + 0.21 * n_lines + (0.30 if legend_above else 0.0)
    fig.tight_layout(rect=(0, 0.30 / h, 1, 1 - header / h))
    return fig


def _shade_events(ax, g: pd.DataFrame) -> None:
    """Grey bands over periods the simulator labelled as anomalous."""
    lab = g["is_anomaly"].to_numpy().astype(bool)
    ts = g["timestamp"].to_numpy()
    step = np.timedelta64(cfg.SAMPLE_MINUTES, "m")
    edges = np.flatnonzero(np.diff(np.r_[0, lab.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        ax.axvspan(ts[s] - step / 2, ts[e - 1] + step / 2, color=LABEL_BAND, alpha=0.45, lw=0)


def _date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


# --------------------------------------------------------------------------
# EDA figures
# --------------------------------------------------------------------------
def fig_sensor_overview(feats: pd.DataFrame, segment: str = "SEG-A", days: int = 14):
    g = feats[feats["segment_id"] == segment]
    start = g["timestamp"].min() + pd.Timedelta(days=7)
    g = g[(g["timestamp"] >= start) & (g["timestamp"] < start + pd.Timedelta(days=days))]
    fig, axes = plt.subplots(
        4, 1, figsize=(11, 7.6), sharex=True, gridspec_kw={"height_ratios": [0.28, 1, 1, 1]}
    )
    # Operating-state strip
    ax = axes[0]
    codes = g["operating_state"].map({s: i for i, s in enumerate(cfg.OPERATING_STATES)}).to_numpy()
    cmap = mpl.colors.ListedColormap([STATE_COLORS[s] for s in cfg.OPERATING_STATES])
    t0, t1 = mpl.dates.date2num(g["timestamp"].iloc[0]), mpl.dates.date2num(g["timestamp"].iloc[-1])
    ax.imshow(codes[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3, extent=(t0, t1, 0, 1))
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Operating state", fontsize=9.5, fontweight="normal", color=INK2)
    ax.legend(
        handles=[Patch(color=STATE_COLORS[s], label=s.replace("_", " ").title()) for s in cfg.OPERATING_STATES],
        ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0), handlelength=1.2,
    )
    for ax, col in zip(axes[1:], cfg.SENSOR_COLS):
        ax.plot(g["timestamp"], g[col], color=BLUE, lw=1.0)
        _shade_events(ax, g)
        ax.set_ylabel(SENSOR_LABELS[col])
    axes[1].legend(
        handles=[Patch(color=LABEL_BAND, alpha=0.6, label="Simulator-labelled anomaly period")],
        loc="upper right",
    )
    _date_axis(axes[-1])
    return _finish(
        fig,
        f"Two weeks of sensor readings — segment {segment}",
        "Pressure and flow step together when the operating state changes; temperature follows with a thermal lag.",
        legend_above=True,
    )


def fig_distributions_by_state(feats: pd.DataFrame):
    d = feats[feats["usable"] & (feats["is_anomaly"] == 0)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, col in zip(axes, cfg.SENSOR_COLS):
        sns.boxplot(
            data=d, x="operating_state", y=col, order=cfg.OPERATING_STATES, hue="operating_state",
            hue_order=cfg.OPERATING_STATES, palette=STATE_COLORS, legend=False, showfliers=False,
            width=0.55, linewidth=1.0, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(SENSOR_LABELS[col])
        ax.set_xticks(range(4))
        ax.set_xticklabels([s.replace("_", "\n").title() for s in cfg.OPERATING_STATES])
        ax.grid(axis="x", visible=False)
    return _finish(
        fig,
        "Sensor distributions depend on the operating state",
        "Normal (unlabelled) rows, all segments. A single global threshold per sensor would confuse states with anomalies.",
    )


def fig_pressure_flow(feats: pd.DataFrame, pf_model):
    d = feats[feats["usable"] & (feats["is_anomaly"] == 0)]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for seg, color in SEGMENT_COLORS.items():
        g = d[d["segment_id"] == seg].sample(3000, random_state=0)
        ax.scatter(g["flow_m3h"], g["pressure_kpa"], s=5, color=color, alpha=0.25, lw=0)
        x = np.linspace(0, d["flow_m3h"].max(), 200)
        ax.plot(x, np.polyval(pf_model.coefs[seg], x), color=color, lw=2, label=f"{seg} fitted curve")
    ax.set_xlabel(SENSOR_LABELS["flow_m3h"])
    ax.set_ylabel(SENSOR_LABELS["pressure_kpa"])
    ax.legend(loc="upper left")
    return _finish(
        fig,
        "Pressure rises with the square of flow",
        "Points: sampled normal readings. Lines: quadratic fit per segment (training period only).\n"
        "The distance from the line becomes a feature that is comparable across operating states.",
    )


def fig_correlations(feats: pd.DataFrame):
    d = feats[feats["usable"] & (feats["is_anomaly"] == 0)]
    names = [SENSOR_LABELS[c].split(" (")[0] for c in cfg.SENSOR_COLS]
    panels = {
        "All operating states": d,
        "Within NORMAL state only": d[(d["operating_state"] == "NORMAL") & ~d["in_settling"]],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, (title, sub) in zip(axes, panels.items()):
        corr = sub[cfg.SENSOR_COLS].corr()
        sns.heatmap(
            corr, vmin=-1, vmax=1, cmap=DIV_CMAP, annot=True, fmt=".2f", square=True, cbar=False,
            xticklabels=names, yticklabels=names, linewidths=2, linecolor=SURFACE,
            annot_kws={"color": INK, "fontsize": 10}, ax=ax,
        )
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(length=0)
    return _finish(
        fig,
        "Sensor correlations are driven by state changes",
        "Pearson correlation of normal rows. Across states pressure and flow move together; inside a single state the link is much weaker.",
    )


def fig_temporal_trends(feats: pd.DataFrame):
    d = feats[feats["usable"]].copy()
    d["date"] = d["timestamp"].dt.floor("D")
    daily = d.groupby(["segment_id", "date"])[cfg.SENSOR_COLS].mean().reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    for ax, col in zip(axes, cfg.SENSOR_COLS):
        for seg, color in SEGMENT_COLORS.items():
            g = daily[daily["segment_id"] == seg]
            ax.plot(g["date"], g[col], color=color, lw=1.4, label=seg)
        ax.set_ylabel(SENSOR_LABELS[col])
    axes[0].legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))
    _date_axis(axes[-1])
    return _finish(
        fig,
        "Daily mean readings over the full simulated period",
        "Segments share the same flow regime; temperature also carries a slow seasonal rise.",
        legend_above=True,
    )


def _example_event(feats: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the event of this type whose surroundings contain the fewest state changes."""
    best = None
    events = feats[feats["anomaly_type"] == kind]
    for event_id, ev in list(events.groupby("anomaly_event_id", sort=True))[:25]:
        span = ev["timestamp"].max() - ev["timestamp"].min()
        pad = max(pd.Timedelta(hours=2), span * 0.6)
        g = feats[
            (feats["segment_id"] == ev["segment_id"].iat[0])
            & (feats["timestamp"] >= ev["timestamp"].min() - pad)
            & (feats["timestamp"] <= ev["timestamp"].max() + pad)
        ]
        changes = int((g["operating_state"] != g["operating_state"].shift()).sum()) - 1
        if best is None or changes < best[0]:
            best = (changes, ev, g)
        if changes == 0:
            break
    return best[1], best[2]


def fig_anomaly_examples(feats: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, kind in zip(axes.ravel(), cfg.ANOMALY_TYPES):
        ev, g = _example_event(feats, kind)
        col = ev["anomaly_sensor"].iat[0]
        ax.plot(g["timestamp"], g[col], color=BLUE, lw=1.2)
        _shade_events(ax, g)
        if kind == "spike":
            ax.scatter(ev["timestamp"], ev[col], s=42, color=ORANGE, zorder=3, edgecolor=SURFACE, lw=1.5)
        ax.set_title(TYPE_LABELS[kind])
        ax.set_ylabel(SENSOR_LABELS[col])
        ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator(minticks=3, maxticks=5))
        _date_axis(ax)
    return _finish(
        fig,
        "The six simulated anomaly patterns",
        "One example of each. Grey band = period labelled by the simulator (the spike is a single sample, marked in orange).",
    )


def fig_label_distribution(feats: pd.DataFrame):
    lab = feats[feats["is_anomaly"] == 1]
    rows = lab.groupby("anomaly_type").size().reindex(cfg.ANOMALY_TYPES)
    events = lab.groupby("anomaly_type")["anomaly_event_id"].nunique().reindex(cfg.ANOMALY_TYPES)
    names = [TYPE_LABELS[t] for t in cfg.ANOMALY_TYPES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, series, title in zip(axes, [events, rows], ["Number of events", "Number of anomalous samples"]):
        bars = ax.barh(names, series.to_numpy(), color=BLUE, height=0.6)
        ax.bar_label(bars, labels=[f"{v:,}" for v in series], padding=4, fontsize=9, color=INK2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, series.max() * 1.15)
        ax.invert_yaxis()
    share = 100 * feats["is_anomaly"].mean()
    return _finish(
        fig,
        "Distribution of simulated anomalies",
        f"{share:.1f}% of all rows are labelled. Spikes are frequent but one sample long; drifts are rare but long, "
        "so they dominate point-wise metrics.",
    )


# --------------------------------------------------------------------------
# Feature figures
# --------------------------------------------------------------------------
def fig_feature_auc(feature_auc: pd.DataFrame):
    d = feature_auc.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    bars = ax.barh(d["feature"], d["roc_auc"] - 0.5, left=0.5, color=BLUE, height=0.62)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in d["roc_auc"]], padding=4, fontsize=8.5, color=INK2)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.set_xlim(0.45, max(0.9, d["roc_auc"].max() + 0.06))
    ax.set_xlabel("ROC-AUC of the single feature (0.5 = no better than chance)")
    ax.grid(axis="y", visible=False)
    return _finish(
        fig,
        "How well does each feature separate labelled anomalies on its own?",
        "Test period. Score = distance from the feature's training median. The physics-based pressure–flow\n"
        "residual carries the most signal; no single feature is sufficient.",
    )


def fig_feature_distributions(frame: pd.DataFrame, features: list[str]):
    test = frame[(frame["split"] == "test") & frame["monitored"]]
    fig, axes = plt.subplots(1, len(features), figsize=(3.1 * len(features), 3.9))
    for ax, f in zip(np.atleast_1d(axes), features):
        lo, hi = test[f].quantile([0.001, 0.999])
        bins = np.linspace(lo, hi, 50)
        for lab, color, name in [(0, BLUE, "Normal"), (1, ORANGE, "Labelled anomaly")]:
            ax.hist(test.loc[test["is_anomaly"] == lab, f].clip(lo, hi), bins=bins, density=True,
                    histtype="step", lw=1.8, color=color, label=name)
        ax.set_title(f, fontsize=9.5)
        ax.set_yticks([])
        ax.grid(False)
    np.atleast_1d(axes)[0].legend(loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("Density")
    return _finish(
        fig,
        "Feature distributions: normal versus labelled-anomaly rows",
        "Test period. Anomalies appear as heavier tails, but the bulk of the two distributions overlaps.",
    )


# --------------------------------------------------------------------------
# Model figures
# --------------------------------------------------------------------------
def fig_model_comparison(comparison: pd.DataFrame):
    metrics = [("precision", "Precision"), ("recall", "Recall\n(point-wise)"), ("f1", "F1"), ("event_recall", "Event recall")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    n, width = len(comparison), 0.15
    for i, (_, row) in enumerate(comparison.iterrows()):
        x = np.arange(len(metrics)) + (i - (n - 1) / 2) * (width + 0.012)
        vals = [row[m] for m, _ in metrics]
        bars = ax.bar(x, vals, width=width, color=MODEL_COLORS[row["model"]], label=row["model"])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics], fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.1), handlelength=1.2, columnspacing=1.4)
    return _finish(
        fig,
        "Detector comparison on the held-out test period",
        "Label-free threshold (3% alert budget set on training scores). Point-wise scores are modest; most simulated events are still caught at least once.",
        legend_above=True,
    )


def fig_pr_roc(test: pd.DataFrame):
    y = test["is_anomaly"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, key in MODEL_KEYS.items():
        s = test[f"score__{key}"].to_numpy()
        p, r, _ = precision_recall_curve(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(r, p, color=MODEL_COLORS[name], label=name)
        axes[1].plot(fpr, tpr, color=MODEL_COLORS[name], label=name)
    base = axes[0].axhline(y.mean(), color=MUTED, lw=1, ls="--", label="No-skill baseline (prevalence)")
    axes[0].legend(handles=[base], loc="upper right")
    axes[1].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–recall", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(loc="lower right")
    return _finish(
        fig,
        "Threshold-free view of the detectors",
        "Test period, point-wise, against simulated labels. With ~3% prevalence the precision–recall curve is the more informative panel.",
    )


def fig_confusion_matrices(comparison: pd.DataFrame):
    fig, axes = plt.subplots(1, len(comparison), figsize=(13, 3.5))
    for ax, (_, r) in zip(axes, comparison.iterrows()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{int(v):,}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if share[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"], rotation=90, va="center")
        ax.set_title(r["model"], fontsize=9.5)
        ax.grid(False)
        ax.tick_params(length=0)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Simulated label")
    return _finish(
        fig,
        "Confusion matrices (test period, label-free threshold)",
        "Counts of samples; percentages are row shares (share of normal / of anomalous samples).",
    )


def fig_recall_by_type(recall_by_type: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (col, title) in zip(
        axes,
        [("event_recall", "Event recall — share of events with at least one alert"),
         ("point_recall", "Point-wise recall — share of anomalous samples flagged")],
    ):
        mat = pd.DataFrame({m: t.set_index("anomaly_type")[col] for m, t in recall_by_type.items()})
        mat = mat.reindex(cfg.ANOMALY_TYPES)
        sns.heatmap(mat, vmin=0, vmax=1, cmap=SEQ_CMAP, annot=True, fmt=".2f", cbar=False,
                    linewidths=2, linecolor=SURFACE, ax=ax, annot_kws={"fontsize": 9})
        ax.set_title(title, fontsize=10)
        ax.set_yticklabels([TYPE_LABELS[t] for t in cfg.ANOMALY_TYPES], rotation=0)
        ax.set_xticklabels([m.replace(" ", "\n", 1) for m in mat.columns], rotation=0, fontsize=8.5)
        ax.set_ylabel("")
        ax.grid(False)
        ax.tick_params(length=0)
    n = next(iter(recall_by_type.values())).set_index("anomaly_type")["n_events"]
    counts = ", ".join(f"{TYPE_LABELS[t].lower()} {int(n[t])}" for t in cfg.ANOMALY_TYPES)
    return _finish(
        fig,
        "Which simulated patterns does each detector catch?",
        f"Test period. Events per type: {counts}. Small counts mean wide uncertainty.",
    )


def fig_fp_breakdown(fp_breakdown: dict[str, dict]):
    cats = [
        ("steady_operation", "Steady operation", BLUE),
        ("within_1h_after_event", "Within 1 h after a labelled event", ORANGE),
        ("during_shutdown", "During shutdown", AQUA),
    ]
    models = list(fp_breakdown)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    left = np.zeros(len(models))
    for key, label, color in cats:
        vals = np.array([fp_breakdown[m][key] for m in models], dtype=float)
        ax.barh(models, vals, left=left, color=color, label=label, height=0.58, edgecolor=SURFACE, lw=2)
        left += vals
    for i, m in enumerate(models):
        ax.text(left[i] + left.max() * 0.01, i, f"{int(left[i]):,}", va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("False-positive samples in the test period")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16), handlelength=1.2)
    return _finish(
        fig,
        "Where do the false alarms come from?",
        "Alerts on rows the simulator did not label. Some follow a real event (rolling windows still recovering); others reflect ordinary noise.",
        legend_above=True,
    )


def fig_score_distributions(test: pd.DataFrame, thresholds_pct: float):
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.5), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, (name, key) in zip(axes, MODEL_KEYS.items()):
        for lab, color, label in [(0, BLUE, "Normal"), (1, ORANGE, "Labelled anomaly")]:
            ax.hist(test.loc[test["is_anomaly"] == lab, f"pct__{key}"], bins=bins, density=True,
                    histtype="step", lw=1.8, color=color, label=label)
        ax.axvline(thresholds_pct, color=MUTED, lw=1, ls="--")
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("Score percentile")
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Density (log scale)")
    axes[0].legend(loc="upper left")
    return _finish(
        fig,
        "Anomaly-score distributions",
        "Scores as percentiles of the training scores. Dashed line = alert threshold. The bar at 0 is rows inside settling windows, which can never alert.",
    )


def _busiest_window(test: pd.DataFrame, segment: str, days: int) -> pd.Timestamp:
    g = test[(test["segment_id"] == segment) & (test["anomaly_event_id"] != "")]
    starts = g.groupby("anomaly_event_id").agg(t=("timestamp", "min"), kind=("anomaly_type", "first"))
    best, best_score = test["timestamp"].min(), -1
    for t0 in pd.date_range(test["timestamp"].min(), test["timestamp"].max() - pd.Timedelta(days=days), freq="12h"):
        inside = starts[(starts["t"] >= t0) & (starts["t"] < t0 + pd.Timedelta(days=days))]
        score = inside["kind"].nunique() * 10 + len(inside)
        if score > best_score:
            best, best_score = t0, score
    return best


def fig_anomaly_overlay(test: pd.DataFrame, thresholds: dict, model: str = "Isolation Forest",
                        segment: str = "SEG-B", days: int = 6):
    key = MODEL_KEYS[model]
    t0 = _busiest_window(test, segment, days)
    g = test[(test["segment_id"] == segment) & (test["timestamp"] >= t0)
             & (test["timestamp"] < t0 + pd.Timedelta(days=days))]
    hits = g[g[f"flag__{key}"] == 1]
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for ax, col in zip(axes[:3], cfg.SENSOR_COLS):
        ax.plot(g["timestamp"], g[col], color=BLUE, lw=1.0)
        _shade_events(ax, g)
        ax.scatter(hits["timestamp"], hits[col], s=13, color=ORANGE, zorder=3, lw=0)
        ax.set_ylabel(SENSOR_LABELS[col])
    axes[0].legend(
        handles=[
            Patch(color=LABEL_BAND, alpha=0.6, label="Simulator-labelled anomaly period"),
            mpl.lines.Line2D([], [], marker="o", ls="", color=ORANGE, markersize=6, label=f"Flagged by {model}"),
        ],
        ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0),
    )
    ax = axes[3]
    ax.plot(g["timestamp"], g[f"pct__{key}"], color=BLUE, lw=0.9)
    _shade_events(ax, g)
    ax.axhline(1 - cfg.ALERT_BUDGET, color=MUTED, lw=1, ls="--")
    ax.text(1.005, 1 - cfg.ALERT_BUDGET, "alert\nthreshold", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED, va="center", ha="left")
    ax.set_ylabel("Score percentile")
    ax.set_ylim(0, 1.03)
    _date_axis(ax)
    return _finish(
        fig,
        f"Alerts overlaid on sensor trends — {segment}, test period",
        "Flagged points are candidates for review, not confirmed events. Note alerts outside grey bands (false alarms) "
        "and grey bands with few alerts (missed).",
        legend_above=True,
    )


# --------------------------------------------------------------------------
# Batch export
# --------------------------------------------------------------------------
def make_all_figures(prep, scored, results, out_dir: Path = cfg.FIGURES_DIR) -> list[Path]:
    set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats, test = prep.features, results["test"]
    top_features = results["feature_auc"]["feature"].head(4).tolist()
    figures = {
        "01_sensor_overview": fig_sensor_overview(feats),
        "02_distributions_by_state": fig_distributions_by_state(feats),
        "03_pressure_flow_relationship": fig_pressure_flow(feats, prep.pf_model),
        "04_sensor_correlations": fig_correlations(feats),
        "05_temporal_trends": fig_temporal_trends(feats),
        "06_anomaly_pattern_examples": fig_anomaly_examples(feats),
        "07_anomaly_label_distribution": fig_label_distribution(feats),
        "08_feature_univariate_auc": fig_feature_auc(results["feature_auc"]),
        "09_feature_distributions": fig_feature_distributions(scored.frame, top_features),
        "10_model_comparison": fig_model_comparison(results["comparison"]),
        "11_pr_roc_curves": fig_pr_roc(test),
        "12_confusion_matrices": fig_confusion_matrices(results["comparison"]),
        "13_recall_by_anomaly_type": fig_recall_by_type(results["recall_by_type"]),
        "14_anomaly_overlay": fig_anomaly_overlay(test, scored.thresholds),
        "15_score_distributions": fig_score_distributions(test, 1 - cfg.ALERT_BUDGET),
        "16_false_positive_breakdown": fig_fp_breakdown(results["fp_breakdown"]),
    }
    paths = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths

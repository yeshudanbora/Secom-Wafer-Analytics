"""
Root-cause analysis workflow for SECOM FDC alarms.

Phase 3 produces the raw T^2 and SPE statistics with per-sensor
contribution decompositions. This module wraps that math in the
*engineer-facing workflow* a fab actually uses on shift:

    1. Alarm catalog    -- every alarmed wafer with its top contributors,
                           sortable / filterable like an Inficon excursion
                           report.

    2. Pareto plot      -- publication-quality bar + cumulative-line chart
                           for a single wafer's contributors. This is the
                           chart projected in a disposition meeting.

    3. Sensor leaderboard
                        -- across all alarms, which sensors most often
                           appear as top contributors? Distinguishes
                           "usual suspect" sensors that consistently
                           drive excursions from sensors that merely drift.

    4. Drill-down       -- given an alarmed wafer and a top-contributor
                           sensor, return the data needed to overlay that
                           wafer onto the sensor's univariate I-MR chart.
                           Connects the multivariate alarm to its
                           univariate root cause.

The math (MacGregor decomposition for T^2, residual squared for SPE) lives
in src/fdc.py and is unit-tested there. This module is pure workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.fdc import (
    PCAModel, FDCScores,
    t2_contributions, spe_contributions,
)


# ---------------------------------------------------------------------------
# Alarm catalog
# ---------------------------------------------------------------------------

@dataclass
class AlarmRecord:
    """One row of the alarm catalog."""
    wafer_idx: int
    timestamp: pd.Timestamp
    t2: float
    spe: float
    t2_alarm: bool
    spe_alarm: bool
    alarm_type: str          # "T2-only" / "SPE-only" / "both"
    is_fail: bool
    top_t2_sensors: list[str]
    top_t2_values: list[float]
    top_spe_sensors: list[str]
    top_spe_values: list[float]


def build_alarm_catalog(
    model: PCAModel,
    monitoring: pd.DataFrame,
    scores: FDCScores,
    n_top: int = 3,
) -> pd.DataFrame:
    """Catalog every alarmed wafer with its top-N contributors.

    Parameters
    ----------
    model       : fitted PCAModel from src.fdc
    monitoring  : the held-out wafer dataframe (must contain timestamp,
                  is_fail, and all sensor columns referenced by `model`)
    scores      : FDCScores returned by score(model, monitoring)
    n_top       : number of top contributors to record per wafer

    Returns
    -------
    DataFrame with one row per alarm, sorted by max(T2/limit, SPE/limit)
    descending so the most severe excursions are at the top.
    """
    alarmed_idx = np.where(scores.any_alarm())[0]
    rows: list[dict] = []
    for i in alarmed_idx:
        wafer = monitoring.iloc[i]
        ct2 = t2_contributions(model, wafer)
        cspe = spe_contributions(model, wafer)
        t2_flag = bool(scores.t2_alarm()[i])
        spe_flag = bool(scores.spe_alarm()[i])
        if t2_flag and spe_flag:
            atype = "both"
        elif t2_flag:
            atype = "T2-only"
        else:
            atype = "SPE-only"

        rows.append({
            "wafer_idx": int(i),
            "timestamp": wafer["timestamp"],
            "t2": float(scores.t2[i]),
            "spe": float(scores.spe[i]),
            "t2_norm": float(scores.t2[i] / model.t2_limit),
            "spe_norm": float(scores.spe[i] / model.spe_limit)
                        if model.spe_limit > 0 else float("nan"),
            "alarm_type": atype,
            "is_fail": bool(wafer["is_fail"]),
            "top_t2_sensors": ct2.head(n_top).index.tolist(),
            "top_t2_values": ct2.head(n_top).values.tolist(),
            "top_spe_sensors": cspe.head(n_top).index.tolist(),
            "top_spe_values": cspe.head(n_top).values.tolist(),
        })

    cat = pd.DataFrame(rows)
    if len(cat):
        # Severity = max(T2_norm, SPE_norm), so a wafer ringing 8x its T^2
        # limit comes ahead of one ringing 2x its SPE limit.
        cat["severity"] = cat[["t2_norm", "spe_norm"]].max(axis=1)
        cat = cat.sort_values("severity", ascending=False).reset_index(drop=True)
    return cat


# ---------------------------------------------------------------------------
# Pareto contribution plot data
# ---------------------------------------------------------------------------

def pareto_contribution_data(
    model: PCAModel,
    wafer: pd.Series,
    statistic: str = "t2",
    n: int = 10,
) -> pd.DataFrame:
    """Build the dataframe that drives a Pareto contribution chart.

    Parameters
    ----------
    statistic : 't2' or 'spe'
    n         : number of bars to keep (top-N contributors)

    Returns
    -------
    DataFrame with columns:
        sensor, contribution, contribution_abs, percent_of_total,
        cumulative_percent
    Sorted by contribution_abs descending.
    """
    if statistic == "t2":
        contribs = t2_contributions(model, wafer)
        total = contribs.abs().sum() if (contribs.abs().sum() > 0) else 1.0
    elif statistic == "spe":
        contribs = spe_contributions(model, wafer)
        total = contribs.sum() if contribs.sum() > 0 else 1.0
    else:
        raise ValueError(f"statistic must be 't2' or 'spe', got {statistic!r}")

    df = pd.DataFrame({
        "sensor": contribs.index,
        "contribution": contribs.values,
        "contribution_abs": np.abs(contribs.values),
    })
    df = df.sort_values("contribution_abs", ascending=False).reset_index(drop=True)
    df["percent_of_total"] = 100.0 * df["contribution_abs"] / total
    df["cumulative_percent"] = df["percent_of_total"].cumsum()
    return df.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sensor leaderboard: usual suspects across all alarms
# ---------------------------------------------------------------------------

def sensor_alarm_leaderboard(
    catalog: pd.DataFrame,
    n_top: int = 3,
) -> pd.DataFrame:
    """Aggregate alarm catalog to find sensors that *consistently* contribute.

    A sensor that drifts steadily will show up as a top contributor in
    many alarms, but a sensor with one isolated excursion will appear
    once. The leaderboard surfaces the former.

    Returns
    -------
    DataFrame with columns:
        sensor, t2_appearances, spe_appearances, total_appearances,
        appearance_rate (vs total alarms), fail_correlated_appearances
    """
    n_alarms = len(catalog)
    if n_alarms == 0:
        return pd.DataFrame(columns=[
            "sensor", "t2_appearances", "spe_appearances",
            "total_appearances", "appearance_rate",
            "fail_correlated_appearances",
        ])

    counts: dict[str, dict[str, int]] = {}

    for _, row in catalog.iterrows():
        for s in row["top_t2_sensors"][:n_top]:
            counts.setdefault(s, {"t2": 0, "spe": 0, "fail": 0})
            counts[s]["t2"] += 1
            if row["is_fail"]:
                counts[s]["fail"] += 1
        for s in row["top_spe_sensors"][:n_top]:
            counts.setdefault(s, {"t2": 0, "spe": 0, "fail": 0})
            counts[s]["spe"] += 1
            # don't double-count fail appearances; counted on T2 side only
            # if sensor appears in both, fail counted on T2 side already

    rows = []
    for sensor, c in counts.items():
        total = c["t2"] + c["spe"]
        rows.append({
            "sensor": sensor,
            "t2_appearances": c["t2"],
            "spe_appearances": c["spe"],
            "total_appearances": total,
            "appearance_rate": total / n_alarms,
            "fail_correlated_appearances": c["fail"],
        })
    return (pd.DataFrame(rows)
              .sort_values("total_appearances", ascending=False)
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Univariate drill-down for one sensor at one alarm
# ---------------------------------------------------------------------------

@dataclass
class DrillDownView:
    """Data needed to render a univariate I-MR chart with one wafer
    highlighted as the alarm point."""
    sensor: str
    monitoring_values: np.ndarray
    timestamps: np.ndarray
    highlight_idx: int
    mu: float
    ucl: float
    lcl: float
    zone_a_upper: float
    zone_a_lower: float
    zone_b_upper: float
    zone_b_lower: float
    alarm_t2: float
    alarm_spe: float
    contribution_t2: float
    contribution_spe: float


def drill_down(
    model: PCAModel,
    monitoring: pd.DataFrame,
    scores: FDCScores,
    wafer_idx: int,
    sensor: str,
    spc_limits,
) -> DrillDownView:
    """Build the data for a per-sensor drill-down on a specific alarm.

    Parameters
    ----------
    spc_limits : ControlLimits dict {sensor: ControlLimits} from
                 src.spc.run_univariate_spc, used so the drill-down chart
                 shows the *same* limits as the standalone I-MR chart in
                 Phase 2.
    """
    if sensor not in spc_limits:
        raise ValueError(
            f"No SPC limits found for {sensor}. The sensor must be in the "
            f"critical-sensor list passed to run_univariate_spc."
        )
    lim = spc_limits[sensor]
    ct2 = t2_contributions(model, monitoring.iloc[wafer_idx])
    cspe = spe_contributions(model, monitoring.iloc[wafer_idx])

    return DrillDownView(
        sensor=sensor,
        monitoring_values=monitoring[sensor].to_numpy(dtype=float),
        timestamps=monitoring["timestamp"].to_numpy(),
        highlight_idx=wafer_idx,
        mu=lim.mu,
        ucl=lim.ucl_i,
        lcl=lim.lcl_i,
        zone_a_upper=lim.zone_a_upper,
        zone_a_lower=lim.zone_a_lower,
        zone_b_upper=lim.zone_b_upper,
        zone_b_lower=lim.zone_b_lower,
        alarm_t2=float(scores.t2[wafer_idx]),
        alarm_spe=float(scores.spe[wafer_idx]),
        contribution_t2=float(ct2.get(sensor, 0.0)),
        contribution_spe=float(cspe.get(sensor, 0.0)),
    )


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib)
# ---------------------------------------------------------------------------

def plot_pareto(
    pareto_df: pd.DataFrame,
    ax=None,
    title: Optional[str] = None,
):
    """Render a Pareto chart from pareto_contribution_data() output."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    bar_colors = ["#c0392b" if v >= 0 else "#2980b9"
                  for v in pareto_df["contribution"]]
    ax.bar(pareto_df["sensor"], pareto_df["contribution_abs"],
           color=bar_colors, edgecolor="black")
    ax.set_ylabel("|contribution|", color="black")
    ax.set_xlabel("Sensor")
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(pareto_df["sensor"], pareto_df["cumulative_percent"],
             color="black", marker="o", linewidth=1.4)
    ax2.set_ylabel("Cumulative % of |total|", color="black")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    if title:
        ax.set_title(title)
    return ax


def plot_drilldown(view: DrillDownView, ax=None):
    """Render the univariate I-MR chart with the alarmed wafer highlighted."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))

    x = np.arange(len(view.monitoring_values))
    ax.plot(x, view.monitoring_values, color="steelblue",
            linewidth=0.8, marker=".", markersize=2.5, alpha=0.7)

    ax.axhline(view.mu, color="black", linewidth=1.1, label=f"μ = {view.mu:.3g}")
    ax.axhline(view.ucl, color="crimson", linestyle="--",
               label=f"UCL = {view.ucl:.3g}")
    ax.axhline(view.lcl, color="crimson", linestyle="--",
               label=f"LCL = {view.lcl:.3g}")
    ax.axhspan(view.zone_a_lower, view.zone_a_upper,
               color="orange", alpha=0.07)

    # Highlight the alarmed wafer
    ax.scatter([view.highlight_idx],
               [view.monitoring_values[view.highlight_idx]],
               color="red", s=110, zorder=10, edgecolor="black",
               linewidth=1.2,
               label=f"Alarm wafer (idx {view.highlight_idx})")

    ax.set_title(
        f"{view.sensor}  —  drill-down for FDC alarm "
        f"(T²={view.alarm_t2:.1f}, SPE={view.alarm_spe:.1f}; "
        f"this sensor's contributions: T²={view.contribution_t2:+.2f}, "
        f"SPE={view.contribution_spe:.2f})"
    )
    ax.set_xlabel("Wafer index (monitoring set)")
    ax.set_ylabel(view.sensor)
    ax.legend(loc="upper right", fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from src.data_loader import select_critical_sensors
    from src.spc import split_baseline_monitoring, run_univariate_spc
    from src.fdc import fit_pca, score

    cleaned = pd.read_parquet("data/processed/secom_clean.parquet")
    critical = select_critical_sensors(cleaned, n=25, method="fail_corr")

    baseline, monitoring = split_baseline_monitoring(cleaned, 0.70)
    model = fit_pca(baseline, critical, var_threshold=0.90, alpha=0.01)
    scores = score(model, monitoring)
    _, limits_by, _ = run_univariate_spc(cleaned, critical)

    # Build the catalog
    catalog = build_alarm_catalog(model, monitoring, scores, n_top=3)
    out = Path("data/processed")
    catalog_view = catalog.copy()
    # Stringify list columns for CSV portability
    for col in ["top_t2_sensors", "top_t2_values",
                "top_spe_sensors", "top_spe_values"]:
        catalog_view[col] = catalog_view[col].apply(
            lambda v: ";".join(str(x) for x in v))
    catalog_view.to_csv(out / "alarm_catalog.csv", index=False)

    # Build the leaderboard
    leaderboard = sensor_alarm_leaderboard(catalog, n_top=3)
    leaderboard.to_csv(out / "sensor_alarm_leaderboard.csv", index=False)

    print("\n=== Phase 4: root-cause workflow ===")
    print(f"\nAlarm catalog ({len(catalog)} alarms total):")
    print(catalog[["wafer_idx", "timestamp", "t2_norm", "spe_norm",
                   "alarm_type", "is_fail",
                   "top_t2_sensors", "top_spe_sensors"]].head(10)
          .to_string(index=False))

    print(f"\nSensor leaderboard (top sensors across all alarms):")
    print(leaderboard.head(10).to_string(index=False,
        float_format=lambda v: f"{v:.3f}"))

    # Demo Pareto data for the worst alarm
    if len(catalog):
        worst = catalog.iloc[0]
        worst_idx = int(worst["wafer_idx"])
        print(f"\nMost severe alarm: monitoring idx {worst_idx}, "
              f"T² norm = {worst['t2_norm']:.2f}, "
              f"SPE norm = {worst['spe_norm']:.2f}, "
              f"is_fail = {worst['is_fail']}")
        pareto = pareto_contribution_data(
            model, monitoring.iloc[worst_idx], statistic="t2", n=10)
        print(f"\nT² Pareto data for that alarm:")
        print(pareto.to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))

        # Drill-down demo
        dd = drill_down(model, monitoring, scores, worst_idx,
                        worst["top_t2_sensors"][0], limits_by)
        print(f"\nDrill-down on top T² contributor "
              f"({worst['top_t2_sensors'][0]}):")
        print(f"  Sensor value at alarm   : "
              f"{dd.monitoring_values[dd.highlight_idx]:.4g}")
        print(f"  Baseline μ / UCL / LCL  : "
              f"{dd.mu:.4g} / {dd.ucl:.4g} / {dd.lcl:.4g}")
        print(f"  Contribution to T²      : {dd.contribution_t2:+.3f}")
        print(f"  Contribution to SPE     : {dd.contribution_spe:.3f}")

    print(f"\nWrote {out / 'alarm_catalog.csv'} ({len(catalog)} rows)")
    print(f"Wrote {out / 'sensor_alarm_leaderboard.csv'} "
          f"({len(leaderboard)} rows)")

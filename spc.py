"""
Univariate Statistical Process Control (SPC) for SECOM.

This module implements the SPC primitives a fab process engineer uses on
shift to decide whether a tool is running in-control:

    * I-MR (Individuals & Moving Range) control charts -- the right chart
      for wafer-by-wafer data, where each wafer is a single measurement
      and there are no rational subgroups.
    * Nelson rules 1-8 -- the canonical out-of-control pattern detectors.
      Western Electric rules are a strict subset; Nelson rules 1-4 cover
      WE 1-4 with equivalent behaviour.
    * Cp / Cpk capability indices -- with spec limits set to
      baseline_mean +/- 3*sigma_within (the natural process tolerance,
      since the SECOM dataset has no engineering specs of its own).

Design decisions documented inline:

    BASELINE WINDOW
        We use the first 70% of pass-only wafers (chronologically ordered)
        as the Phase-1 baseline used to fit control limits and spec limits.
        The remaining 30% of pass wafers plus all 104 failures form the
        Phase-2 monitoring set on which the rules are evaluated. This
        prevents the control limits from "seeing" the excursions we want
        to detect -- a real-fab analogue would be the first month of
        production after qualification.

    SIGMA ESTIMATE
        sigma_within = MR_bar / d2,  d2 = 1.128 for n=2 moving ranges.
        This is the within-run estimate appropriate for I-MR charts; it
        is intentionally smaller than the overall sample std because the
        latter contains between-run shifts the chart is supposed to flag.

    SPEC LIMITS & CAPABILITY
        We use the industry-standard split between *overall* and *within*
        variation when computing capability:

            sigma_overall = std(baseline values)        # long-run, drift-included
            sigma_within  = MR_bar / d2                 # short-run, MR-based

        USL / LSL = mu_baseline +/- 3*sigma_overall
            -- the natural tolerance the *customer* sees, since SECOM has
               no engineering specs of its own.

        Cp  = (USL - LSL) / (6 * sigma_within)
        Cpk = min(USL - mu, mu - lsl) / (3 * sigma_within)

        This produces a meaningful Cp != 1 whenever the baseline contains
        run-to-run drift (sigma_overall > sigma_within), which is the
        diagnostic of interest. Cpk diverges from Cp as the long-run
        process mean shifts away from the spec midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hartley constants for moving-range (n = 2):
D2_N2 = 1.128       # E[range] / sigma for n=2
D3_N2 = 0.0         # lower MR limit coefficient for n=2
D4_N2 = 3.267       # upper MR limit coefficient for n=2


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ControlLimits:
    """Control & spec limits for a single sensor, fit on the baseline.

    All zone boundaries are stored so Nelson rules 5/6 can be evaluated
    later without re-deriving them.
    """
    sensor: str
    mu: float                 # baseline mean (centerline of I chart)
    sigma_within: float       # MR_bar / d2  (short-run, used for Cp/Cpk)
    sigma_overall: float      # std(baseline) (long-run, used for spec limits)
    ucl_i: float              # mu + 3*sigma_within
    lcl_i: float              # mu - 3*sigma
    zone_a_upper: float       # mu + 2*sigma
    zone_a_lower: float       # mu - 2*sigma
    zone_b_upper: float       # mu + 1*sigma
    zone_b_lower: float       # mu - 1*sigma
    mr_bar: float             # mean of moving ranges on baseline
    ucl_mr: float             # D4 * MR_bar
    lcl_mr: float             # D3 * MR_bar (= 0 for n=2)
    usl: float                # spec upper (mu + 3*sigma)
    lsl: float                # spec lower (mu - 3*sigma)
    cp: float
    cpk: float
    n_baseline: int


@dataclass
class RuleViolations:
    """Boolean masks (one per Nelson rule) over a chart's points."""
    rule_1: np.ndarray  # one point > 3 sigma from centerline
    rule_2: np.ndarray  # 9 in a row on same side of centerline
    rule_3: np.ndarray  # 6 in a row trending up or down
    rule_4: np.ndarray  # 14 in a row alternating up/down
    rule_5: np.ndarray  # 2 of 3 consecutive in zone A or beyond, same side
    rule_6: np.ndarray  # 4 of 5 consecutive in zone B or beyond, same side
    rule_7: np.ndarray  # 15 in a row inside zone C (both sides) - stratification
    rule_8: np.ndarray  # 8 in a row outside zone C, either side - mixture

    def any_violation(self) -> np.ndarray:
        return (self.rule_1 | self.rule_2 | self.rule_3 | self.rule_4
                | self.rule_5 | self.rule_6 | self.rule_7 | self.rule_8)

    def counts(self) -> dict[str, int]:
        return {
            f"rule_{i}": int(getattr(self, f"rule_{i}").sum())
            for i in range(1, 9)
        }


# ---------------------------------------------------------------------------
# Baseline split
# ---------------------------------------------------------------------------

def split_baseline_monitoring(
    cleaned: pd.DataFrame,
    baseline_frac: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned frame into a Phase-1 baseline and Phase-2 monitoring set.

    Phase-1 baseline = first `baseline_frac` of pass-only wafers,
                       in chronological order.
    Phase-2 monitoring = everything else (the held-out pass wafers plus
                         all the fail wafers).

    The split is on a sorted-by-timestamp copy; the original frame is
    untouched.
    """
    df = cleaned.sort_values("timestamp").reset_index(drop=True)
    pass_mask = ~df["is_fail"]
    pass_idx = np.where(pass_mask)[0]
    n_baseline = int(len(pass_idx) * baseline_frac)
    baseline_idx = set(pass_idx[:n_baseline])
    is_baseline = df.index.isin(baseline_idx)

    baseline = df.loc[is_baseline].reset_index(drop=True)
    monitoring = df.loc[~is_baseline].reset_index(drop=True)
    return baseline, monitoring


# ---------------------------------------------------------------------------
# Limit fitting & capability
# ---------------------------------------------------------------------------

def fit_control_limits(
    baseline_values: np.ndarray,
    sensor_name: str,
) -> ControlLimits:
    """Compute I-MR limits, zone boundaries, and Cp/Cpk for one sensor."""
    x = np.asarray(baseline_values, dtype=float)
    if x.size < 2:
        raise ValueError(f"Need at least 2 baseline points for {sensor_name}")

    mu = float(np.mean(x))
    mr = np.abs(np.diff(x))
    mr_bar = float(np.mean(mr))
    sigma_within = mr_bar / D2_N2
    sigma_overall = float(np.std(x, ddof=1))

    if sigma_within == 0.0:
        # Degenerate baseline; flag with NaN limits so downstream code skips.
        sigma_within = float("nan")

    # Control limits use the within-run sigma (standard I-chart practice).
    ucl_i = mu + 3 * sigma_within
    lcl_i = mu - 3 * sigma_within

    # Spec limits use the overall sigma -- the natural tolerance the
    # customer sees, including any baseline drift.
    usl = mu + 3 * sigma_overall
    lsl = mu - 3 * sigma_overall

    # Capability indices are computed against sigma_within so they reflect
    # what the tool *could* hold if drift were eliminated.
    if np.isfinite(sigma_within) and sigma_within > 0:
        cp = (usl - lsl) / (6 * sigma_within)
        cpk = min(usl - mu, mu - lsl) / (3 * sigma_within)
    else:
        cp = float("nan")
        cpk = float("nan")

    return ControlLimits(
        sensor=sensor_name,
        mu=mu,
        sigma_within=sigma_within,
        sigma_overall=sigma_overall,
        ucl_i=ucl_i,
        lcl_i=lcl_i,
        zone_a_upper=mu + 2 * sigma_within,
        zone_a_lower=mu - 2 * sigma_within,
        zone_b_upper=mu + 1 * sigma_within,
        zone_b_lower=mu - 1 * sigma_within,
        mr_bar=mr_bar,
        ucl_mr=D4_N2 * mr_bar,
        lcl_mr=D3_N2 * mr_bar,
        usl=usl,
        lsl=lsl,
        cp=cp,
        cpk=cpk,
        n_baseline=len(x),
    )


# ---------------------------------------------------------------------------
# Nelson rules
# ---------------------------------------------------------------------------

def nelson_rules(
    values: np.ndarray,
    limits: ControlLimits,
) -> RuleViolations:
    """Apply Nelson rules 1-8 to a series of individual measurements.

    Each returned mask is True at index i if rule k flags point i. We
    follow the conventional definition where the violation is recorded
    at the *terminal* point of the qualifying pattern (i.e. for rule 2
    "9 in a row on same side", indices 0..7 cannot have flagged it yet,
    but index 8 onward can).

    References:
        Nelson, L.S. (1984). The Shewhart Control Chart -- Tests for
            Special Causes. Journal of Quality Technology, 16(4).
        Montgomery, D.C. (2019). Introduction to Statistical Quality
            Control, 8th ed., Wiley.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    mu = limits.mu
    sigma = limits.sigma_within

    # Side relative to centerline
    above = x > mu
    below = x < mu

    # ---- Rule 1: one point beyond +/-3 sigma ----------------------------
    rule_1 = (x > limits.ucl_i) | (x < limits.lcl_i)

    # ---- Rule 2: 9 consecutive points on same side of centerline --------
    rule_2 = _run_length_flag(above, 9) | _run_length_flag(below, 9)

    # ---- Rule 3: 6 consecutive points strictly increasing or decreasing -
    diffs = np.diff(x)
    increasing = diffs > 0
    decreasing = diffs < 0
    # A run of 6 strictly-increasing values requires 5 consecutive
    # positive diffs, terminating at point i.
    rule_3 = np.zeros(n, dtype=bool)
    if n >= 6:
        rule_3 |= np.r_[np.zeros(5, dtype=bool), _consec_true(increasing, 5)]
        rule_3 |= np.r_[np.zeros(5, dtype=bool), _consec_true(decreasing, 5)]

    # ---- Rule 4: 14 alternating up/down ---------------------------------
    rule_4 = np.zeros(n, dtype=bool)
    if n >= 14:
        sign = np.sign(diffs)
        # alternation: sign[i] != 0 and sign[i] != sign[i-1] for 13 in a row
        alt = np.zeros(len(diffs), dtype=bool)
        alt[1:] = (sign[1:] != 0) & (sign[1:] == -sign[:-1])
        rule_4 = np.r_[np.zeros(13, dtype=bool), _consec_true(alt, 13)]

    # ---- Rule 5: 2 of 3 consecutive in zone A or beyond, same side ------
    in_zone_a_or_beyond_upper = x > limits.zone_a_upper       # > mu + 2 sigma
    in_zone_a_or_beyond_lower = x < limits.zone_a_lower       # < mu - 2 sigma
    rule_5 = _k_of_n_window(in_zone_a_or_beyond_upper, k=2, n_window=3) | \
             _k_of_n_window(in_zone_a_or_beyond_lower, k=2, n_window=3)

    # ---- Rule 6: 4 of 5 consecutive in zone B or beyond, same side ------
    in_zone_b_or_beyond_upper = x > limits.zone_b_upper       # > mu + 1 sigma
    in_zone_b_or_beyond_lower = x < limits.zone_b_lower       # < mu - 1 sigma
    rule_6 = _k_of_n_window(in_zone_b_or_beyond_upper, k=4, n_window=5) | \
             _k_of_n_window(in_zone_b_or_beyond_lower, k=4, n_window=5)

    # ---- Rule 7: 15 in a row inside +/- 1 sigma (stratification) --------
    in_zone_c = (x >= limits.zone_b_lower) & (x <= limits.zone_b_upper)
    rule_7 = _run_length_flag(in_zone_c, 15)

    # ---- Rule 8: 8 in a row outside +/- 1 sigma, either side (mixture) --
    outside_zone_c = (x > limits.zone_b_upper) | (x < limits.zone_b_lower)
    rule_8 = _run_length_flag(outside_zone_c, 8)

    return RuleViolations(
        rule_1=rule_1, rule_2=rule_2, rule_3=rule_3, rule_4=rule_4,
        rule_5=rule_5, rule_6=rule_6, rule_7=rule_7, rule_8=rule_8,
    )


# ---------------------------------------------------------------------------
# Boolean-pattern helpers
# ---------------------------------------------------------------------------

def _run_length_flag(mask: np.ndarray, run_length: int) -> np.ndarray:
    """Flag every index that terminates a run of >= `run_length` Trues.

    Example: mask = [T, T, T, F], run_length=3 -> [F, F, T, F]
    """
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    if run_length <= 0 or n < run_length:
        return out
    # rolling sum of last `run_length` mask values; flag where == run_length
    csum = np.cumsum(mask.astype(int))
    window_sum = csum.copy()
    window_sum[run_length:] = csum[run_length:] - csum[:-run_length]
    out[run_length - 1:] = window_sum[run_length - 1:] == run_length
    return out


def _consec_true(mask: np.ndarray, k: int) -> np.ndarray:
    """Indices where there are k consecutive Trues ending here.

    Returned array has length len(mask) - k + 1 (aligned to terminal index).
    """
    n = len(mask)
    if k <= 0 or n < k:
        return np.zeros(max(0, n - k + 1), dtype=bool)
    csum = np.cumsum(mask.astype(int))
    window = csum.copy()
    window[k:] = csum[k:] - csum[:-k]
    return window[k - 1:] == k


def _k_of_n_window(mask: np.ndarray, k: int, n_window: int) -> np.ndarray:
    """Flag terminal index of any window of length n_window with >= k Trues.

    Used for Nelson rules 5 and 6.
    """
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    if n_window <= 0 or n < n_window:
        return out
    csum = np.cumsum(mask.astype(int))
    window_sum = csum.copy()
    window_sum[n_window:] = csum[n_window:] - csum[:-n_window]
    out[n_window - 1:] = window_sum[n_window - 1:] >= k
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_univariate_spc(
    cleaned: pd.DataFrame,
    sensor_cols: list[str],
    baseline_frac: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, ControlLimits], dict[str, RuleViolations]]:
    """Fit limits on baseline, score Nelson rules on monitoring, summarise.

    Parameters
    ----------
    cleaned       : full cleaned dataframe
    sensor_cols   : list of sensor names to monitor (e.g. critical 25)
    baseline_frac : fraction of pass-only wafers used as Phase-1 baseline

    Returns
    -------
    summary    : DataFrame, one row per sensor with mu, sigma, Cp, Cpk,
                 violation counts, and total flagged-points
    limits_by  : dict sensor -> ControlLimits
    rules_by   : dict sensor -> RuleViolations (computed on monitoring set)
    """
    baseline, monitoring = split_baseline_monitoring(cleaned, baseline_frac)

    limits_by: dict[str, ControlLimits] = {}
    rules_by: dict[str, RuleViolations] = {}
    rows = []
    for s in sensor_cols:
        try:
            lim = fit_control_limits(baseline[s].to_numpy(), s)
        except ValueError:
            continue
        if not np.isfinite(lim.sigma_within):
            continue
        rules = nelson_rules(monitoring[s].to_numpy(), lim)
        limits_by[s] = lim
        rules_by[s] = rules

        counts = rules.counts()
        flagged = int(rules.any_violation().sum())

        # Cpk_monitoring uses the same fixed USL/LSL fit on baseline, but
        # substitutes the monitoring-set mean for mu. This reveals drift:
        # if the long-run mean has shifted off-center, Cpk_monitoring drops
        # below Cpk (which by construction equals Cp on the baseline).
        mon_mean = float(monitoring[s].mean())
        if lim.sigma_within > 0:
            cpk_mon = min(lim.usl - mon_mean,
                          mon_mean - lim.lsl) / (3 * lim.sigma_within)
        else:
            cpk_mon = float("nan")

        rows.append({
            "sensor": s,
            "mu": lim.mu,
            "mu_monitoring": mon_mean,
            "mean_shift_sigma": (mon_mean - lim.mu) / lim.sigma_within
                                 if lim.sigma_within > 0 else float("nan"),
            "sigma_within": lim.sigma_within,
            "sigma_overall": lim.sigma_overall,
            "drift_ratio": (lim.sigma_overall / lim.sigma_within
                            if lim.sigma_within > 0 else float("nan")),
            "Cp": lim.cp,
            "Cpk_baseline": lim.cpk,
            "Cpk_monitoring": cpk_mon,
            "flagged_points": flagged,
            "n_monitoring": len(monitoring),
            "flag_rate": flagged / len(monitoring) if len(monitoring) else 0.0,
            **counts,
        })

    summary = pd.DataFrame(rows).sort_values("Cpk_monitoring").reset_index(drop=True)
    return summary, limits_by, rules_by


if __name__ == "__main__":
    # Smoke run end-to-end on the cleaned parquet from Phase 1.
    from pathlib import Path
    from src.data_loader import select_critical_sensors

    cleaned = pd.read_parquet("data/processed/secom_clean.parquet")
    critical = select_critical_sensors(cleaned, n=25)
    summary, limits_by, rules_by = run_univariate_spc(cleaned, critical)

    out = Path("data/processed")
    summary.to_csv(out / "spc_summary.csv", index=False)

    print("\nUnivariate SPC summary (sorted by Cpk ascending -- worst first):\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out / 'spc_summary.csv'}")

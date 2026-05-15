"""
EWMA run-to-run (R2R) controller demo for SECOM.

Univariate SPC detects drift *after* it has accumulated. Run-to-run
control prevents drift from accumulating at all by issuing a recipe
correction after every wafer based on the smoothed deviation from
target. This is the standard industrial form of EWMA R2R control
(Ingolfsson & Sachs 1993; Box & Jenkins 1976), used in production fabs
on critical photo and etch parameters such as overlay, CD, and
deposition thickness.

Algorithm:

    EWMA_t  = lambda * y_t + (1 - lambda) * EWMA_{t-1}    # filtered y
    e_t     = EWMA_t - target                              # filtered error
    u_{t+1} = u_t - K * e_t                                # next correction
    y'_{t+1} = y_{t+1,raw} + u_{t+1}                       # corrected reading

Tuning rules of thumb (Del Castillo, "Statistical Process Adjustment for
Quality Control" 2002):

    lambda = 0.2 - 0.4    -- smooths noise, tracks drift on photo tools
    K      = 0.5 - 1.0    -- K=1 is full one-step compensation,
                             K=0.5 is the conservative default that
                             avoids ringing on tools with deadtime.

This module:

    1. Simulates the closed-loop trajectory on a chosen monitoring sensor.
    2. Re-evaluates Nelson rules on the controlled trajectory using the
       *same* baseline-fit limits as Phase 2, so "excursions prevented"
       is a fair apples-to-apples count.
    3. Sweeps (lambda, K) on a grid and reports excursion-prevention
       headline numbers.

References:
    Ingolfsson, A. & Sachs, E. (1993). Stability and Sensitivity of
        an EWMA Controller. Journal of Quality Technology, 25(4).
    Del Castillo, E. (2002). Statistical Process Adjustment for
        Quality Control. Wiley.
    Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis:
        Forecasting and Control, Holden-Day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.spc import (
    ControlLimits, fit_control_limits, nelson_rules, RuleViolations,
)


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class R2RSimulation:
    """Result of one closed-loop EWMA controller run.

    Stores both the raw and controlled trajectories so downstream
    plotting code can render them on the same axis.
    """
    sensor: str
    lambda_: float
    K: float
    target: float
    raw: np.ndarray              # actual measured sensor values (no controller)
    ewma: np.ndarray             # EWMA-filtered values
    correction: np.ndarray       # cumulative recipe correction u_t at each step
    controlled: np.ndarray       # y_raw + correction (what the chart would see)
    raw_violations: RuleViolations
    controlled_violations: RuleViolations
    raw_violation_count: int
    controlled_violation_count: int
    excursions_prevented: int
    raw_cpk_monitoring: float
    controlled_cpk_monitoring: float
    limits: ControlLimits


def simulate_ewma_r2r(
    monitoring_values: np.ndarray,
    limits: ControlLimits,
    lambda_: float = 0.30,
    K: float = 0.5,
    target: Optional[float] = None,
) -> R2RSimulation:
    """Simulate an EWMA R2R controller running on a monitoring trajectory.

    Parameters
    ----------
    monitoring_values : raw sensor measurements during Phase-2 monitoring
    limits            : ControlLimits fit on the baseline (defines target,
                        spec limits, control limits, sigma_within)
    lambda_           : EWMA smoothing factor (0 < lambda <= 1)
    K                 : controller gain (typical 0.5-1.0)
    target            : desired sensor value. Defaults to the baseline
                        mean (the sensor's natural in-control center).

    Returns
    -------
    R2RSimulation with both trajectories and the excursion-prevention
    count using the *same* baseline-fit Nelson rules as Phase 2.
    """
    if not (0 < lambda_ <= 1):
        raise ValueError(f"lambda must be in (0, 1], got {lambda_}")
    y_raw = np.asarray(monitoring_values, dtype=float)
    n = len(y_raw)
    if target is None:
        target = limits.mu

    ewma = np.zeros(n)
    correction = np.zeros(n)        # cumulative correction u_t
    controlled = np.zeros(n)

    # Initial state: assume zero correction in effect, EWMA seeded at target
    # (i.e. controller believes the process is on-target until it sees data).
    ewma[0] = target
    correction[0] = 0.0
    controlled[0] = y_raw[0] + correction[0]

    for t in range(1, n):
        # Apply correction computed from previous step BEFORE measurement.
        # The correction was computed at step t-1 based on EWMA_{t-1}.
        controlled[t] = y_raw[t] + correction[t - 1]

        # Update EWMA on the *controlled* (apparent) value -- this is what
        # the controller actually observes on the chart.
        ewma[t] = lambda_ * controlled[t] + (1 - lambda_) * ewma[t - 1]

        # Compute next-step correction
        e = ewma[t] - target
        correction[t] = correction[t - 1] - K * e

    # Score violations using the same baseline-fit limits as Phase 2.
    raw_v = nelson_rules(y_raw, limits)
    ctl_v = nelson_rules(controlled, limits)

    raw_vc = int(raw_v.any_violation().sum())
    ctl_vc = int(ctl_v.any_violation().sum())

    # Cpk on the monitoring set against the same fixed USL/LSL
    if limits.sigma_within > 0:
        raw_cpk = min(limits.usl - y_raw.mean(),
                      y_raw.mean() - limits.lsl) / (3 * limits.sigma_within)
        ctl_cpk = min(limits.usl - controlled.mean(),
                      controlled.mean() - limits.lsl) / (3 * limits.sigma_within)
    else:
        raw_cpk = ctl_cpk = float("nan")

    return R2RSimulation(
        sensor=limits.sensor,
        lambda_=lambda_,
        K=K,
        target=target,
        raw=y_raw,
        ewma=ewma,
        correction=correction,
        controlled=controlled,
        raw_violations=raw_v,
        controlled_violations=ctl_v,
        raw_violation_count=raw_vc,
        controlled_violation_count=ctl_vc,
        excursions_prevented=raw_vc - ctl_vc,
        raw_cpk_monitoring=float(raw_cpk),
        controlled_cpk_monitoring=float(ctl_cpk),
        limits=limits,
    )


# ---------------------------------------------------------------------------
# Gain sweep
# ---------------------------------------------------------------------------

def sweep_gains(
    monitoring_values: np.ndarray,
    limits: ControlLimits,
    lambdas: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50, 0.70),
    Ks: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> pd.DataFrame:
    """Sweep (lambda, K) on a grid and return a comparison table.

    Each row reports raw violations, controlled violations, excursions
    prevented, and Cpk on the controlled trajectory. The "best" cell
    (most excursions prevented without over-correcting) is the controller
    setting we'd recommend.
    """
    rows = []
    for lam in lambdas:
        for K in Ks:
            sim = simulate_ewma_r2r(monitoring_values, limits,
                                     lambda_=lam, K=K)
            rows.append({
                "lambda": lam,
                "K": K,
                "raw_violations": sim.raw_violation_count,
                "controlled_violations": sim.controlled_violation_count,
                "excursions_prevented": sim.excursions_prevented,
                "prevention_pct": (sim.excursions_prevented
                                   / sim.raw_violation_count * 100
                                   if sim.raw_violation_count else 0.0),
                "raw_Cpk_mon": sim.raw_cpk_monitoring,
                "controlled_Cpk_mon": sim.controlled_cpk_monitoring,
                "Cpk_lift": (sim.controlled_cpk_monitoring
                             - sim.raw_cpk_monitoring),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from src.data_loader import select_critical_sensors
    from src.spc import split_baseline_monitoring

    cleaned = pd.read_parquet("data/processed/secom_clean.parquet")
    critical = select_critical_sensors(cleaned, n=25, method="fail_corr")
    baseline, monitoring = split_baseline_monitoring(cleaned, 0.70)

    # Pick the sensor with the largest mean shift between baseline and
    # monitoring -- this is where R2R control has the most to recover.
    shifts = {}
    limits_cache: dict[str, ControlLimits] = {}
    for s in critical:
        lim = fit_control_limits(baseline[s].to_numpy(), s)
        if lim.sigma_within > 0:
            shift = (monitoring[s].mean() - lim.mu) / lim.sigma_within
            shifts[s] = abs(shift)
            limits_cache[s] = lim
    target_sensor = max(shifts, key=shifts.get)
    lim = limits_cache[target_sensor]
    print(f"\n=== Phase 5: EWMA R2R controller demo ===")
    print(f"\nTarget sensor selected by largest mean shift:")
    print(f"  Sensor              : {target_sensor}")
    print(f"  Baseline mu         : {lim.mu:.4g}")
    print(f"  Monitoring mu       : "
          f"{monitoring[target_sensor].mean():.4g}")
    print(f"  Mean shift (sigma)  : {shifts[target_sensor]:+.3f}")
    print(f"  Sigma_within        : {lim.sigma_within:.4g}")

    # Single-run simulation at recommended defaults
    sim = simulate_ewma_r2r(
        monitoring[target_sensor].to_numpy(),
        lim, lambda_=0.30, K=0.5,
    )
    print(f"\nSimulation @ lambda=0.30, K=0.5:")
    print(f"  Raw violations            : {sim.raw_violation_count}")
    print(f"  Controlled violations     : {sim.controlled_violation_count}")
    print(f"  Excursions prevented      : {sim.excursions_prevented}")
    if sim.raw_violation_count:
        prev_pct = (sim.excursions_prevented
                    / sim.raw_violation_count * 100)
        print(f"  Prevention rate           : {prev_pct:.1f}%")
    print(f"  Raw Cpk_monitoring        : {sim.raw_cpk_monitoring:.3f}")
    print(f"  Controlled Cpk_monitoring : "
          f"{sim.controlled_cpk_monitoring:.3f}")
    print(f"  Cpk lift                  : "
          f"{sim.controlled_cpk_monitoring - sim.raw_cpk_monitoring:+.3f}")

    # Gain sweep
    print(f"\nGain sweep (lambda x K):")
    sweep = sweep_gains(monitoring[target_sensor].to_numpy(), lim)
    pivot = sweep.pivot(index="lambda", columns="K",
                        values="excursions_prevented")
    print("Excursions prevented by gain combination:")
    print(pivot.to_string())

    print("\nFull sweep table:")
    print(sweep.round(3).to_string(index=False))

    # Persist
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "r2r_gain_sweep.csv", index=False)
    pd.DataFrame({
        "raw": sim.raw,
        "ewma": sim.ewma,
        "correction": sim.correction,
        "controlled": sim.controlled,
    }).to_csv(out / "r2r_simulation.csv", index=False)
    print(f"\nWrote {out / 'r2r_gain_sweep.csv'}")
    print(f"Wrote {out / 'r2r_simulation.csv'}")

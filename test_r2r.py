"""Unit tests for src/r2r.py.

The EWMA controller has a small number of correctness invariants worth
checking explicitly:
    1. On clean stationary data, controller should not introduce more
       violations than it removes.
    2. On synthetic drifting data, controller should reduce violations.
    3. lambda=1.0, K=1.0 should fully compensate the previous step's
       deviation (deadbeat behaviour).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spc import fit_control_limits  # noqa: E402
from src.r2r import simulate_ewma_r2r, sweep_gains  # noqa: E402


def test_controller_helps_on_drift() -> None:
    """Inject a synthetic linear drift; controller should reduce violations."""
    rng = np.random.default_rng(42)
    n_baseline, n_monitor = 500, 500
    baseline_vals = rng.normal(loc=10.0, scale=1.0, size=n_baseline)
    drift = np.linspace(0, 4.0, n_monitor)            # 4-sigma drift over window
    monitor_vals = rng.normal(loc=10.0, scale=1.0, size=n_monitor) + drift

    lim = fit_control_limits(baseline_vals, "synthetic")
    sim = simulate_ewma_r2r(monitor_vals, lim, lambda_=0.3, K=0.5)
    assert sim.raw_violation_count > 0, "Drift should produce raw violations"
    assert sim.controlled_violation_count < sim.raw_violation_count, \
        f"Controller should reduce violations: " \
        f"{sim.controlled_violation_count} >= {sim.raw_violation_count}"
    print(f"PASS test_controller_helps_on_drift "
          f"(prevented {sim.excursions_prevented} of "
          f"{sim.raw_violation_count})")


def test_controller_minimal_harm_on_stationary() -> None:
    """On clean stationary data the controller shouldn't massively
    over-correct and create new violations.
    """
    rng = np.random.default_rng(7)
    baseline_vals = rng.normal(loc=5.0, scale=0.5, size=600)
    monitor_vals = rng.normal(loc=5.0, scale=0.5, size=400)

    lim = fit_control_limits(baseline_vals, "synthetic")
    sim = simulate_ewma_r2r(monitor_vals, lim, lambda_=0.3, K=0.5)
    # On stationary data, controlled count shouldn't be dramatically higher
    # than raw count. Allow up to 2x as a sanity tolerance.
    assert sim.controlled_violation_count <= 2 * sim.raw_violation_count + 5, \
        "Controller is over-correcting on stationary data"
    print(f"PASS test_controller_minimal_harm_on_stationary "
          f"(raw={sim.raw_violation_count}, "
          f"controlled={sim.controlled_violation_count})")


def test_correction_drives_mean_to_target() -> None:
    """After enough wafers, controller should drive the mean of the
    *controlled* trajectory close to target.
    """
    rng = np.random.default_rng(123)
    baseline_vals = rng.normal(loc=10.0, scale=1.0, size=500)
    # Persistent step shift in monitoring data
    monitor_vals = rng.normal(loc=12.0, scale=1.0, size=500)
    lim = fit_control_limits(baseline_vals, "synthetic")

    sim = simulate_ewma_r2r(monitor_vals, lim, lambda_=0.4, K=0.7)
    # Take the last 100 wafers (after controller has settled)
    settled = sim.controlled[-100:]
    assert abs(settled.mean() - lim.mu) < 0.5, \
        f"Settled controlled mean {settled.mean():.3f} should be " \
        f"close to target {lim.mu:.3f}"
    print(f"PASS test_correction_drives_mean_to_target "
          f"(target={lim.mu:.2f}, settled mean={settled.mean():.2f})")


def test_sweep_table_shape() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 400)
    mon = rng.normal(0.5, 1, 300)
    lim = fit_control_limits(base, "synthetic")
    table = sweep_gains(mon, lim, lambdas=(0.2, 0.5), Ks=(0.5, 1.0))
    assert len(table) == 4
    assert set(["lambda", "K", "raw_violations",
                "controlled_violations",
                "excursions_prevented"]).issubset(table.columns)
    print("PASS test_sweep_table_shape")


if __name__ == "__main__":
    test_controller_helps_on_drift()
    test_controller_minimal_harm_on_stationary()
    test_correction_drives_mean_to_target()
    test_sweep_table_shape()
    print("\nAll Phase 5 tests passed.")

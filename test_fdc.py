"""Unit tests for src/fdc.py.

The FDC math is easy to get wrong in subtle ways. These tests verify
key invariants on synthetic data where ground truth is known.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdc import (  # noqa: E402
    fit_pca, score, t2_contributions, spe_contributions,
)


def _make_synthetic(n: int = 600, p: int = 8, K_true: int = 3, seed: int = 0):
    """Build a (n x p) frame that lives close to a K_true-dim subspace."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, K_true))
    A = rng.normal(size=(K_true, p))
    X = Z @ A + 0.05 * rng.normal(size=(n, p))   # tiny residual noise
    cols = [f"sensor_{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["is_fail"] = False
    return df, cols


def test_pca_recovers_intrinsic_dim() -> None:
    df, cols = _make_synthetic(n=600, p=8, K_true=3)
    model = fit_pca(df, cols, var_threshold=0.90, alpha=0.01)
    assert model.K == 3, f"Expected K=3, got K={model.K}"
    var_kept = model.explained_variance_ratio[:model.K].sum()
    assert var_kept > 0.90
    print(f"PASS test_pca_recovers_intrinsic_dim (K={model.K}, var={var_kept:.3f})")


def test_baseline_alarm_rate_near_alpha() -> None:
    df, cols = _make_synthetic(n=2000, p=10, K_true=4, seed=1)
    model = fit_pca(df, cols, var_threshold=0.90, alpha=0.01)
    s = score(model, df)
    rate = s.any_alarm().mean()
    assert rate < 0.05, f"In-sample alarm rate too high: {rate:.3f}"
    print(f"PASS test_baseline_alarm_rate_near_alpha ({rate:.3%})")


def test_outlier_triggers_alarm() -> None:
    df, cols = _make_synthetic(n=1000, p=8, K_true=3, seed=2)
    model = fit_pca(df, cols, var_threshold=0.90, alpha=0.01)
    extreme = df.iloc[0].copy()
    for c in cols:
        extreme[c] = 50.0
    s = score(model, pd.DataFrame([extreme]))
    assert bool(s.any_alarm()[0]), "Extreme wafer must alarm"
    print(f"PASS test_outlier_triggers_alarm "
          f"(T^2={s.t2[0]:.1f}, SPE={s.spe[0]:.1f})")


def test_t2_contributions_sum_close_to_t2() -> None:
    df, cols = _make_synthetic(n=500, p=6, K_true=2, seed=3)
    model = fit_pca(df, cols, var_threshold=0.90, alpha=0.01)
    s = score(model, df)
    idx = int(np.argmax(s.t2))
    contribs = t2_contributions(model, df.iloc[idx])
    assert abs(contribs.sum() - s.t2[idx]) < 1e-6, \
        f"contribs sum {contribs.sum()} != T^2 {s.t2[idx]}"
    print("PASS test_t2_contributions_sum_close_to_t2")


def test_spe_contributions_sum_equals_spe() -> None:
    df, cols = _make_synthetic(n=500, p=6, K_true=2, seed=4)
    model = fit_pca(df, cols, var_threshold=0.90, alpha=0.01)
    s = score(model, df)
    idx = int(np.argmax(s.spe))
    contribs = spe_contributions(model, df.iloc[idx])
    assert abs(contribs.sum() - s.spe[idx]) < 1e-9, \
        f"contribs sum {contribs.sum()} != SPE {s.spe[idx]}"
    print("PASS test_spe_contributions_sum_equals_spe")


def test_limits_are_finite_positive() -> None:
    df, cols = _make_synthetic(n=400, p=10, K_true=3, seed=5)
    model = fit_pca(df, cols, var_threshold=0.90, alpha=0.01)
    assert np.isfinite(model.t2_limit) and model.t2_limit > 0
    assert np.isfinite(model.spe_limit) and model.spe_limit >= 0
    print(f"PASS test_limits_are_finite_positive "
          f"(T^2_lim={model.t2_limit:.2f}, SPE_lim={model.spe_limit:.2f})")


if __name__ == "__main__":
    test_pca_recovers_intrinsic_dim()
    test_baseline_alarm_rate_near_alpha()
    test_outlier_triggers_alarm()
    test_t2_contributions_sum_close_to_t2()
    test_spe_contributions_sum_equals_spe()
    test_limits_are_finite_positive()
    print("\nAll Phase 3 tests passed.")

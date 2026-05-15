"""Unit tests for src/rootcause.py."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdc import fit_pca, score  # noqa: E402
from src.rootcause import (  # noqa: E402
    build_alarm_catalog, pareto_contribution_data,
    sensor_alarm_leaderboard,
)


def _make_synthetic(n: int = 600, p: int = 8, K_true: int = 3,
                    seed: int = 0, n_outliers: int = 5):
    """Make a frame with `n_outliers` injected anomalies at the end."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, K_true))
    A = rng.normal(size=(K_true, p))
    X = Z @ A + 0.05 * rng.normal(size=(n, p))
    # Inject outliers in the last n_outliers rows
    X[-n_outliers:] += 30.0 * rng.normal(size=(n_outliers, p))
    cols = [f"sensor_{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="h")
    df["is_fail"] = False
    df.loc[df.index[-n_outliers:], "is_fail"] = True
    return df, cols


def test_catalog_records_each_alarm() -> None:
    df, cols = _make_synthetic(n=400, p=10, n_outliers=8, seed=1)
    # Train on the clean head
    train = df.iloc[:300].reset_index(drop=True)
    test = df.iloc[300:].reset_index(drop=True)
    model = fit_pca(train, cols, var_threshold=0.90, alpha=0.01)
    s = score(model, test)
    cat = build_alarm_catalog(model, test, s, n_top=3)
    assert len(cat) >= 5, "Should catch most injected outliers"
    # All recorded indices should be alarmed
    assert all(s.any_alarm()[i] for i in cat["wafer_idx"])
    # Catalog should have severity column and be sorted descending
    assert "severity" in cat.columns
    assert (cat["severity"].diff().dropna() <= 0).all()
    print(f"PASS test_catalog_records_each_alarm ({len(cat)} alarms cataloged)")


def test_pareto_contributions_sum_correct() -> None:
    df, cols = _make_synthetic(n=400, p=8, n_outliers=5, seed=2)
    model = fit_pca(df.iloc[:300].reset_index(drop=True), cols)
    pareto = pareto_contribution_data(
        model, df.iloc[-1], statistic="t2", n=8)
    # Cumulative should reach 100% on top-N (where N = total sensors)
    assert abs(pareto["cumulative_percent"].iloc[-1] - 100.0) < 1e-6
    print("PASS test_pareto_contributions_sum_correct")


def test_pareto_spe_nonneg() -> None:
    df, cols = _make_synthetic(n=400, p=6, n_outliers=5, seed=3)
    model = fit_pca(df.iloc[:300].reset_index(drop=True), cols)
    pareto = pareto_contribution_data(
        model, df.iloc[-1], statistic="spe", n=6)
    assert (pareto["contribution"] >= 0).all(), "SPE contributions must be nonneg"
    print("PASS test_pareto_spe_nonneg")


def test_leaderboard_appearance_counts() -> None:
    df, cols = _make_synthetic(n=400, p=10, n_outliers=20, seed=4)
    model = fit_pca(df.iloc[:300].reset_index(drop=True), cols)
    s = score(model, df.iloc[300:].reset_index(drop=True))
    cat = build_alarm_catalog(
        model, df.iloc[300:].reset_index(drop=True), s, n_top=3)
    lb = sensor_alarm_leaderboard(cat, n_top=3)
    # Total appearances should equal sum of T2 + SPE per row
    assert (lb["total_appearances"]
            == lb["t2_appearances"] + lb["spe_appearances"]).all()
    # No more than n_alarms appearances per sensor on either statistic
    n = len(cat)
    assert (lb["t2_appearances"] <= n).all()
    assert (lb["spe_appearances"] <= n).all()
    print(f"PASS test_leaderboard_appearance_counts (top sensor: "
          f"{lb.iloc[0]['sensor']} with {lb.iloc[0]['total_appearances']} "
          f"appearances)")


if __name__ == "__main__":
    test_catalog_records_each_alarm()
    test_pareto_contributions_sum_correct()
    test_pareto_spe_nonneg()
    test_leaderboard_appearance_counts()
    print("\nAll Phase 4 tests passed.")

"""Unit tests for src/spc.py Nelson rule implementations.

Each test crafts a minimal signal that should trigger exactly one rule,
then verifies the implementation flags it at the expected terminal index.
This is the kind of test a reviewer will look at first when auditing the
SPC code -- the rules are easy to get subtly wrong.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spc import (  # noqa: E402
    ControlLimits,
    nelson_rules,
    fit_control_limits,
    _run_length_flag,
    _consec_true,
    _k_of_n_window,
)


def _make_limits(mu: float = 0.0, sigma: float = 1.0) -> ControlLimits:
    """Construct a ControlLimits matching what fit_control_limits would
    produce, but using arbitrary mu / sigma_within for direct testing."""
    return ControlLimits(
        sensor="test",
        mu=mu,
        sigma_within=sigma,
        sigma_overall=sigma,
        ucl_i=mu + 3 * sigma,
        lcl_i=mu - 3 * sigma,
        zone_a_upper=mu + 2 * sigma,
        zone_a_lower=mu - 2 * sigma,
        zone_b_upper=mu + 1 * sigma,
        zone_b_lower=mu - 1 * sigma,
        mr_bar=sigma * 1.128,
        ucl_mr=sigma * 1.128 * 3.267,
        lcl_mr=0.0,
        usl=mu + 3 * sigma,
        lsl=mu - 3 * sigma,
        cp=1.0,
        cpk=1.0,
        n_baseline=100,
    )


def test_helper_run_length_flag() -> None:
    mask = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    out = _run_length_flag(mask, 3)
    # terminal indices of every >= 3-run: idx 2, idx 6, idx 7
    assert out.tolist() == [False, False, True, False, False, False, True, True]
    print("PASS test_helper_run_length_flag")


def test_helper_k_of_n_window() -> None:
    mask = np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=bool)
    out = _k_of_n_window(mask, k=2, n_window=3)
    # window ending at idx 2: [1,0,1] -> 2 trues -> flag
    # window ending at idx 6: [0,1,1] -> 2 trues -> flag
    # window ending at idx 7: [1,1,0] -> 2 trues -> flag
    expected = [False, False, True, False, False, False, True, True]
    assert out.tolist() == expected, f"got {out.tolist()}"
    print("PASS test_helper_k_of_n_window")


def test_rule_1_beyond_3sigma() -> None:
    lims = _make_limits()
    x = np.array([0.0, 0.0, 4.0, 0.0, -3.5])  # idx 2 and 4 outside +/- 3
    r = nelson_rules(x, lims)
    assert r.rule_1.tolist() == [False, False, True, False, True]
    print("PASS test_rule_1_beyond_3sigma")


def test_rule_2_nine_same_side() -> None:
    lims = _make_limits()
    # 8 points above centerline -> NOT flagged; 9th point flagged
    x = np.array([0.5] * 8 + [0.5] + [0.0])
    r = nelson_rules(x, lims)
    assert r.rule_2[8] == True, "9th consecutive same-side point should flag"
    assert r.rule_2[7] == False, "8 in a row alone should not flag"
    assert r.rule_2[9] == False, "after returning to centerline should not flag"
    print("PASS test_rule_2_nine_same_side")


def test_rule_3_six_trending() -> None:
    lims = _make_limits()
    # strictly increasing for 6 points: indices 0..5
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4])
    r = nelson_rules(x, lims)
    assert r.rule_3[5] == True, "6th increasing point should flag"
    assert r.rule_3[4] == False, "5 increasing alone should not flag"
    print("PASS test_rule_3_six_trending")


def test_rule_5_two_of_three_zone_a() -> None:
    lims = _make_limits()
    # values near +2.5 sigma (zone A or beyond)
    x = np.array([0.0, 2.5, 0.0, 2.5])
    r = nelson_rules(x, lims)
    # window [0..2]: 1 zone-A point -> no flag
    # window [1..3]: 2 zone-A points -> flag at idx 3
    assert r.rule_5[3] == True
    assert r.rule_5[2] == False
    print("PASS test_rule_5_two_of_three_zone_a")


def test_rule_7_fifteen_in_zone_c() -> None:
    lims = _make_limits()
    # 15 points all within +/- 1 sigma -> stratification flag at idx 14
    x = np.array([0.3] * 15 + [2.0])
    r = nelson_rules(x, lims)
    assert r.rule_7[14] == True
    assert r.rule_7[13] == False
    print("PASS test_rule_7_fifteen_in_zone_c")


def test_rule_8_eight_outside_zone_c() -> None:
    lims = _make_limits()
    # alternating outside +/- 1 sigma: 8 points should trigger
    x = np.array([1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5])
    r = nelson_rules(x, lims)
    assert r.rule_8[7] == True
    assert r.rule_8[6] == False
    print("PASS test_rule_8_eight_outside_zone_c")


def test_fit_control_limits_basic() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=2.0, size=500)
    lim = fit_control_limits(x, "synthetic")
    assert abs(lim.mu - 10.0) < 0.3
    # sigma_within and sigma_overall should both be ~2.0 for clean iid data
    assert 1.7 < lim.sigma_within < 2.3
    assert 1.7 < lim.sigma_overall < 2.3
    print("PASS test_fit_control_limits_basic")


if __name__ == "__main__":
    test_helper_run_length_flag()
    test_helper_k_of_n_window()
    test_rule_1_beyond_3sigma()
    test_rule_2_nine_same_side()
    test_rule_3_six_trending()
    test_rule_5_two_of_three_zone_a()
    test_rule_7_fifteen_in_zone_c()
    test_rule_8_eight_outside_zone_c()
    test_fit_control_limits_basic()
    print("\nAll Phase 2 tests passed.")

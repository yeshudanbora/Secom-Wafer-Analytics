"""Sanity tests for src/data_loader.py.

These are not exhaustive unit tests -- they verify that the cleaning
pipeline behaves as documented on the actual SECOM file in data/raw/.
Run with:  python -m tests.test_data_loader
"""
import sys
from pathlib import Path

# add project root to path so we can import src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw, clean, select_critical_sensors  # noqa: E402


def test_shapes() -> None:
    raw_dir = ROOT / "data" / "raw"
    df = load_raw(raw_dir / "secom.data", raw_dir / "secom_labels.data")
    assert df.shape == (1567, 593), f"Expected (1567, 593), got {df.shape}"
    assert df["label"].isin([-1, 1]).all()
    assert df["is_fail"].sum() == 104, "Expected 104 fail wafers"
    print("PASS test_shapes")


def test_cleaning_drops_expected_buckets() -> None:
    raw_dir = ROOT / "data" / "raw"
    df = load_raw(raw_dir / "secom.data", raw_dir / "secom_labels.data")
    cleaned, report = clean(df)
    assert report.n_sensors_final < report.n_sensors_raw
    assert report.n_sensors_final > 100, "Cleaning should not be too aggressive"
    sensor_cols = [c for c in cleaned.columns if c.startswith("sensor_")]
    assert cleaned[sensor_cols].isna().sum().sum() == 0, "All NaNs should be imputed"
    print("PASS test_cleaning_drops_expected_buckets")


def test_critical_selection_size() -> None:
    raw_dir = ROOT / "data" / "raw"
    df = load_raw(raw_dir / "secom.data", raw_dir / "secom_labels.data")
    cleaned, _ = clean(df)
    crit = select_critical_sensors(cleaned, n=25)
    assert len(crit) == 25
    assert len(set(crit)) == 25, "Critical list must be unique"
    print("PASS test_critical_selection_size")


if __name__ == "__main__":
    test_shapes()
    test_cleaning_drops_expected_buckets()
    test_critical_selection_size()
    print("\nAll Phase 1 tests passed.")

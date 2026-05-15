"""Unit tests for the SQL layer (db_schema, db_ingest_psycopg2, db_query).

These tests require a running PostgreSQL instance reachable at the
default course settings (or whatever env vars override them). If Postgres
is not available, all tests are SKIPPED rather than failing — this keeps
CI green on machines without a database.

Run with:
    python tests/test_db.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _postgres_available() -> bool:
    """Quick connection test. Returns False if Postgres isn't reachable."""
    try:
        import psycopg2
        from src.db_config import psycopg2_dsn
        conn = psycopg2.connect(connect_timeout=2, **psycopg2_dsn())
        conn.close()
        return True
    except Exception:
        return False


def test_schema_applies_cleanly() -> None:
    from src.db_schema import apply_schema, verify_schema
    apply_schema()
    counts = verify_schema()
    expected = {"wafers", "measurements", "spc_limits", "alarms", "r2r_runs"}
    assert set(counts.keys()) == expected, \
        f"Expected tables {expected}, got {set(counts.keys())}"
    print(f"PASS test_schema_applies_cleanly (tables: {list(counts)})")


def test_ingest_loads_expected_counts() -> None:
    from src.db_ingest_psycopg2 import ingest_psycopg2
    counts = ingest_psycopg2()
    assert counts["wafers"] == 1567, f"Expected 1567 wafers, got {counts['wafers']}"
    assert counts["measurements"] == 1567 * 590, (
        f"Expected {1567 * 590} measurements (1567 wafers × 590 sensors), "
        f"got {counts['measurements']}"
    )
    print(f"PASS test_ingest_loads_expected_counts "
          f"({counts['wafers']} wafers, {counts['measurements']:,} measurements)")


def test_readback_matches_file_pipeline() -> None:
    """The Postgres-sourced frame should match the file-sourced frame
    exactly on shape, pass/fail counts, and cleaned-frame outputs."""
    from src.data_loader import load_raw, clean
    from src.db_query import load_wafers_pandas

    file_df = load_raw(
        ROOT / "data" / "raw" / "secom.data",
        ROOT / "data" / "raw" / "secom_labels.data",
    )
    db_df = load_wafers_pandas()

    assert file_df.shape == db_df.shape, \
        f"Shapes differ: file={file_df.shape}, db={db_df.shape}"
    assert file_df["is_fail"].sum() == db_df["is_fail"].sum(), \
        "Fail count mismatch between file and DB"

    # Run cleaning on both and compare summary counts
    _, file_rep = clean(file_df)
    _, db_rep = clean(db_df)
    assert file_rep.n_sensors_final == db_rep.n_sensors_final, (
        f"Cleaning produces different sensor counts: "
        f"file={file_rep.n_sensors_final}, db={db_rep.n_sensors_final}"
    )
    print(f"PASS test_readback_matches_file_pipeline "
          f"(both pipelines → {db_rep.n_sensors_final} sensors)")


def test_fail_rate_sql_query() -> None:
    from src.db_query import fail_rate
    rate = fail_rate()
    # 104 / 1567 = 0.06636
    assert 0.06 < rate < 0.07, f"Fail rate looks wrong: {rate}"
    print(f"PASS test_fail_rate_sql_query ({rate:.4f})")


if __name__ == "__main__":
    if not _postgres_available():
        print("SKIPPED: PostgreSQL not reachable at "
              "localhost:5432 with course-default credentials. "
              "Set DB_HOST/DB_USER/DB_PASSWORD env vars to override.")
        sys.exit(0)

    test_schema_applies_cleanly()
    test_ingest_loads_expected_counts()
    test_readback_matches_file_pipeline()
    test_fail_rate_sql_query()
    print("\nAll Phase 6.5 (SQL layer) tests passed.")

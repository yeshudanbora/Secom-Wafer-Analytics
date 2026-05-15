"""
DDL runner: execute sql/01_schema.sql against PostgreSQL.

Follows Lecture 3's guidance: DDL operations live in plain .sql files and
are run independently of any data-engineering pipeline. We use psycopg2
(not SparkSQL) because Spark's JDBC is designed for DML, not DDL.

Usage:
    python -m src.db_schema
"""

from __future__ import annotations

from pathlib import Path

import psycopg2

from src.db_config import psycopg2_dsn


SCHEMA_FILE = Path(__file__).resolve().parents[1] / "sql" / "01_schema.sql"


def apply_schema() -> None:
    """Read the DDL file and execute it.

    Runs the whole file as one statement block; psycopg2 handles multi-
    statement SQL when autocommit is on.
    """
    sql_text = SCHEMA_FILE.read_text()
    conn = psycopg2.connect(**psycopg2_dsn())
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql_text)
    finally:
        conn.close()


def verify_schema() -> dict[str, int]:
    """Confirm all five secom.* tables exist; return their row counts."""
    expected = ["wafers", "measurements", "spc_limits", "alarms", "r2r_runs"]
    counts: dict[str, int] = {}
    conn = psycopg2.connect(**psycopg2_dsn())
    try:
        with conn.cursor() as cur:
            for tbl in expected:
                cur.execute(f"SELECT COUNT(*) FROM secom.{tbl}")
                counts[tbl] = cur.fetchone()[0]
    finally:
        conn.close()
    return counts


if __name__ == "__main__":
    print(f"Applying schema from {SCHEMA_FILE}")
    apply_schema()
    counts = verify_schema()
    print("Schema applied. Current row counts:")
    for t, n in counts.items():
        print(f"  secom.{t:<14} {n:>8} rows")

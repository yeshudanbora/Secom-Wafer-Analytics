"""
Pure-psycopg2 ingestion (fallback for environments without SparkSQL).

This is the same end result as src.db_ingest, but goes:

    raw CSV  ─►  pandas DataFrame  ─►  psycopg2 COPY  ─►  PostgreSQL

instead of routing through Spark. Useful in two situations:

    1. Your laptop doesn't have the postgres JDBC driver installed yet,
       and you want to get the database loaded while you set up Spark.
    2. You want a "no Spark" reproduction path for grading or CI.

In production at a fab, the SparkSQL pipeline in src.db_ingest is the
correct shape because real telemetry volumes are >> SECOM. This fallback
is a convenience, not a substitute.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import psycopg2

from src.data_loader import load_raw
from src.db_config import psycopg2_dsn
from src.db_schema import verify_schema


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def truncate_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE secom.alarms RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE secom.measurements RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE secom.wafers RESTART IDENTITY CASCADE;")


def copy_wafers(conn, df: pd.DataFrame) -> dict[int, int]:
    """COPY rows into secom.wafers and return (wafer_index → wafer_id) map."""
    wafer_rows = df[["timestamp", "label", "is_fail"]].reset_index(drop=True)

    buf = io.StringIO()
    for i, row in wafer_rows.iterrows():
        buf.write(
            f"{i}\t{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\t"
            f"{int(row['label'])}\t{'t' if row['is_fail'] else 'f'}\n"
        )
    buf.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY secom.wafers (wafer_index, ts, label, is_fail) "
            "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')",
            buf,
        )
        cur.execute(
            "SELECT wafer_index, wafer_id FROM secom.wafers ORDER BY wafer_index"
        )
        return {wi: wid for wi, wid in cur.fetchall()}


def copy_measurements(conn, df: pd.DataFrame, id_map: dict[int, int]) -> None:
    """Long-pivot the wide sensor frame and COPY into secom.measurements."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    print(f"      streaming {len(df) * len(sensor_cols):,} measurement rows...")

    # Stream to a single COPY for speed
    buf = io.StringIO()
    for wafer_index, row in df.iterrows():
        wid = id_map[wafer_index]
        for s in sensor_cols:
            v = row[s]
            v_str = "\\N" if pd.isna(v) else f"{v}"
            buf.write(f"{wid}\t{s}\t{v_str}\n")

    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY secom.measurements (wafer_id, sensor_name, value) "
            "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')",
            buf,
        )


def ingest_psycopg2(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    print("[1/4] Reading raw files into pandas...")
    df = load_raw(raw_dir / "secom.data", raw_dir / "secom_labels.data")
    print(f"      shape: {df.shape}")

    print("[2/4] Truncating destination tables...")
    conn = psycopg2.connect(**psycopg2_dsn())
    try:
        truncate_tables(conn)
        conn.commit()

        print("[3/4] COPYing secom.wafers...")
        id_map = copy_wafers(conn, df)
        conn.commit()
        print(f"      inserted {len(id_map)} wafers")

        print("[4/4] COPYing secom.measurements...")
        copy_measurements(conn, df, id_map)
        conn.commit()
    finally:
        conn.close()

    return verify_schema()


if __name__ == "__main__":
    counts = ingest_psycopg2()
    print("\nIngestion complete. Row counts:")
    for t, n in counts.items():
        print(f"  secom.{t:<14} {n:>10,}")

"""
Read SECOM data out of PostgreSQL.

Two paths are provided:

    * `load_wafers_pandas()` — direct query that returns a pandas DataFrame
      in the same shape as src/data_loader.py's cleaned frame (wide format:
      timestamp, label, is_fail, sensor_0...sensor_589). This is what the
      existing Phase 1–5 analysis modules need.

    * `load_wafers_spark(spark)` — SparkSQL read returning a Spark
      DataFrame. Useful when you want to demonstrate the SparkSQL JDBC
      read-back pattern (also Lecture 3), or when the dataset is large
      enough that pandas would be slow.

The SQL layer is the system of record; the file-based parquet outputs
written by the Phase 1 cleaner are a fast read-optimized cache for
downstream analysis. This dual-source pattern (warehouse + cache) is the
standard fab MDE design.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text

from src.db_config import db_properties, psycopg2_dsn


def _engine():
    """SQLAlchemy engine — used for pandas read_sql_query to silence its
    SQLAlchemy-only future-warning."""
    p = db_properties()
    return create_engine(
        f"postgresql+psycopg2://{p['username']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['database']}"
    )


# ---------------------------------------------------------------------------
# Pandas read-back (wide format)
# ---------------------------------------------------------------------------

WIDE_SQL = """
WITH base AS (
    SELECT
        w.wafer_id,
        w.wafer_index,
        w.ts        AS timestamp,
        w.label,
        w.is_fail,
        m.sensor_name,
        m.value
    FROM secom.wafers w
    JOIN secom.measurements m USING (wafer_id)
)
SELECT * FROM base
ORDER BY wafer_index, sensor_name;
"""


def load_wafers_pandas() -> pd.DataFrame:
    """Read SECOM data from Postgres and pivot to wide format.

    Returns a DataFrame with columns:
        timestamp, label, is_fail, sensor_0, sensor_1, ..., sensor_589

    matching the shape produced by `src.data_loader.load_raw()`. The Phase
    1 cleaning pipeline can be applied to this frame as-is.
    """
    engine = _engine()
    long_df = pd.read_sql_query(text(WIDE_SQL), engine)

    # Pivot long -> wide. Keep one row per wafer.
    wide = long_df.pivot_table(
        index=["wafer_index", "timestamp", "label", "is_fail"],
        columns="sensor_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Re-order sensor columns numerically (sensor_0 .. sensor_589)
    meta_cols = ["wafer_index", "timestamp", "label", "is_fail"]
    sensor_cols = sorted(
        [c for c in wide.columns if c.startswith("sensor_")],
        key=lambda s: int(s.split("_")[1]),
    )
    wide = wide[meta_cols + sensor_cols]

    # Sort chronologically and drop wafer_index from the analysis frame
    wide = wide.sort_values("timestamp").reset_index(drop=True)
    return wide.drop(columns=["wafer_index"])


# ---------------------------------------------------------------------------
# SparkSQL read-back
# ---------------------------------------------------------------------------

def load_wafers_spark(spark, wide: bool = False):
    """Read SECOM data into a Spark DataFrame.

    Parameters
    ----------
    wide : if True, pivot to wide format (one row per wafer). If False,
           return the long format as stored in Postgres.

    Useful for big-data demonstration: in a real fab you'd never want to
    pivot 600M sensor readings into pandas memory, so the long-format
    read is the production-shaped path.
    """
    p = db_properties()
    long_df = (
        spark.read.format("jdbc")
             .option("url",     p["url"])
             .option("dbtable", """
                (SELECT w.wafer_index, w.ts, w.label, w.is_fail,
                        m.sensor_name, m.value
                 FROM secom.wafers w
                 JOIN secom.measurements m USING (wafer_id)) AS joined
             """)
             .option("user",    p["username"])
             .option("password", p["password"])
             .option("driver",  p["driver"])
             .load()
    )
    if not wide:
        return long_df

    pivoted = (
        long_df.groupBy("wafer_index", "ts", "label", "is_fail")
               .pivot("sensor_name")
               .agg({"value": "first"})
    )
    return pivoted.orderBy("ts")


# ---------------------------------------------------------------------------
# Convenience queries
# ---------------------------------------------------------------------------

def fail_rate() -> float:
    """Quick pure-SQL query — what fraction of wafers failed?"""
    conn = psycopg2.connect(**psycopg2_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                  SUM(CASE WHEN is_fail THEN 1 ELSE 0 END)::float
                  / COUNT(*)::float AS fail_rate
                FROM secom.wafers;
            """)
            return float(cur.fetchone()[0])
    finally:
        conn.close()


def sensor_summary_sql() -> pd.DataFrame:
    """Per-sensor mean / std / NaN-rate computed in PostgreSQL.

    Useful as a sanity check: numbers should agree with what
    src.data_loader.clean() computes on the same data.
    """
    sql = """
    SELECT sensor_name,
           AVG(value)          AS mean,
           STDDEV_SAMP(value)  AS sd,
           1.0 - COUNT(value)::float / COUNT(*)::float AS missing_rate,
           COUNT(*) AS n
    FROM secom.measurements
    GROUP BY sensor_name
    ORDER BY sensor_name;
    """
    return pd.read_sql_query(text(sql), _engine())


if __name__ == "__main__":
    print(f"Fail rate (computed via SQL): {fail_rate():.4f}")
    summary = sensor_summary_sql()
    print(f"\nSensor summary ({len(summary)} sensors):")
    print(summary.head(10).to_string(index=False))

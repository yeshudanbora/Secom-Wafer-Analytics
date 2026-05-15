"""
Ingest raw SECOM files into PostgreSQL via SparkSQL.

This module mirrors the Lecture 3 pattern (`db_properties` dict, JDBC write
through `df.write.format("jdbc")`) and demonstrates the standard fab MDE
ingestion shape:

    raw CSV  ─►  SparkSQL DataFrame  ─►  JDBC  ─►  PostgreSQL

Two writes happen in order:

    1. `secom.wafers`        : one row per wafer with timestamp + label
    2. `secom.measurements`  : long-format (wafer_id, sensor_name, value)

The wafer table is written first so we can resolve the `wafer_id` foreign
key when writing measurements. We pull the inserted wafer_ids back from
Postgres after step 1 (a small "lookup" read) and join them onto the
measurements DataFrame before the second write.

The cleaning pipeline in `src/data_loader.py` is the *modeling* layer; this
file is the *data engineering* layer. Both can coexist — for fast local
analysis the file-based pipeline is fine, while the database is the
canonical system of record at production scale.

Usage:
    python -m src.db_ingest
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import psycopg2

# Spark is heavy; only import it inside functions so importing this module
# in environments without Spark (e.g. when only the psycopg2 fallback is
# being used) doesn't fail.
from src.db_config import db_properties, psycopg2_dsn


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# ---------------------------------------------------------------------------
# Spark session helper
# ---------------------------------------------------------------------------

def make_spark_session(app_name: str = "secom-ingest"):
    """Build a local SparkSession with the PostgreSQL JDBC driver wired in.

    Mirrors the Lecture 3 setup. If `findspark` is installed (typical on
    Windows), we initialize it. We also point Spark at the postgres JDBC
    JAR via the `spark.jars.packages` mechanism, which auto-downloads the
    driver into the Ivy cache the first time it runs — this avoids the
    manual "drop the .jar into $SPARK_HOME/jars" step from Lecture 3.
    """
    try:
        import findspark
        findspark.init()
    except ImportError:
        pass  # not needed on macOS/Linux when SPARK_HOME is set

    from pyspark.sql import SparkSession

    # Allow override via env var if the user already has the JAR locally.
    jdbc_pkg = os.environ.get(
        "POSTGRES_JDBC_PACKAGE",
        "org.postgresql:postgresql:42.7.3",
    )

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", jdbc_pkg)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")  # tiny dataset
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Raw → SparkSQL DataFrames
# ---------------------------------------------------------------------------

def load_raw_into_spark(spark, raw_dir: Path = RAW_DIR):
    """Read secom.data and secom_labels.data into SparkSQL DataFrames.

    The label file's quoted timestamps confuse Spark's whitespace tokenizer,
    so we parse labels in pure Python (same fix as src/data_loader.py) and
    then `spark.createDataFrame()` from the parsed Python lists.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, IntegerType, BooleanType, TimestampType,
    )

    # ----- Labels --------------------------------------------------------
    label_path = raw_dir / "secom_labels.data"
    rows: list[tuple] = []
    import datetime as _dt
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        lbl_str, _, rest = line.partition(" ")
        ts = _dt.datetime.strptime(rest.strip().strip('"'),
                                   "%d/%m/%Y %H:%M:%S")
        lbl = int(lbl_str)
        rows.append((len(rows), ts, lbl, lbl == 1))   # wafer_index, ts, label, is_fail

    labels_schema = StructType([
        StructField("wafer_index", IntegerType(), False),
        StructField("ts",          TimestampType(), False),
        StructField("label",       IntegerType(), False),
        StructField("is_fail",     BooleanType(), False),
    ])
    wafers_df = spark.createDataFrame(rows, schema=labels_schema)

    # ----- Sensor measurements -------------------------------------------
    # secom.data is whitespace-separated, 590 cols, NaN encoded as the
    # literal string "NaN". Easiest path: read with a permissive schema,
    # then long-pivot.
    sensor_path = raw_dir / "secom.data"
    # Spark's CSV reader handles repeated whitespace if we set sep=" "
    # and inferSchema; but NaN tokens need explicit handling. We use the
    # standard nanValue option.
    sensor_df_wide = (
        spark.read
             .option("header", "false")
             .option("sep", " ")
             .option("nullValue", "NaN")
             .option("inferSchema", "true")
             .csv(str(sensor_path))
    )
    # Sometimes pyspark's whitespace handling leaves a trailing empty
    # column; drop any all-null columns it produced.
    n_cols_expected = 590
    cols = sensor_df_wide.columns
    if len(cols) > n_cols_expected:
        # Drop trailing empties (these are columns where every value is null)
        # by counting non-null rows.
        keep = []
        for c in cols:
            non_null = sensor_df_wide.filter(F.col(c).isNotNull()).limit(1).count()
            if non_null > 0:
                keep.append(c)
            if len(keep) == n_cols_expected:
                break
        sensor_df_wide = sensor_df_wide.select(*keep)

    # Rename columns to sensor_0 .. sensor_589
    new_names = [f"sensor_{i}" for i in range(len(sensor_df_wide.columns))]
    sensor_df_wide = sensor_df_wide.toDF(*new_names)

    # Add a row-number wafer_index so we can join to labels
    from pyspark.sql.window import Window
    sensor_df_wide = sensor_df_wide.withColumn(
        "wafer_index",
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )

    return wafers_df, sensor_df_wide


# ---------------------------------------------------------------------------
# Pivot wide → long
# ---------------------------------------------------------------------------

def melt_to_long(sensor_df_wide):
    """Long-pivot the wide sensor frame to (wafer_index, sensor_name, value).

    PySpark doesn't have a built-in `.melt()` (in older versions); we use
    `stack()` via selectExpr to do the unpivot in one pass.
    """
    from pyspark.sql import functions as F

    sensor_cols = [c for c in sensor_df_wide.columns if c.startswith("sensor_")]
    n = len(sensor_cols)
    # Build a stack(N, 'sensor_0', sensor_0, 'sensor_1', sensor_1, ...)
    stack_args = ", ".join([f"'{c}', `{c}`" for c in sensor_cols])
    stack_expr = f"stack({n}, {stack_args}) as (sensor_name, value)"

    long_df = sensor_df_wide.selectExpr("wafer_index", stack_expr)
    # Cast value to double to be explicit
    long_df = long_df.withColumn("value", F.col("value").cast("double"))
    return long_df


# ---------------------------------------------------------------------------
# JDBC writers (Lecture 3 pattern)
# ---------------------------------------------------------------------------

def write_wafers(wafers_df, mode: str = "append") -> None:
    """Insert into secom.wafers via JDBC.

    We use mode='append' because the schema has already been created by
    src.db_schema and we don't want JDBC to drop and recreate the table.
    The wafers table uses SERIAL wafer_id, which Postgres assigns server-
    side; we let it.
    """
    p = db_properties()
    (
        wafers_df.select("wafer_index", "ts", "label", "is_fail")
        .write
        .format("jdbc")
        .mode(mode)
        .option("url",     p["url"])
        .option("dbtable", "secom.wafers")
        .option("user",    p["username"])
        .option("password", p["password"])
        .option("driver",  p["driver"])
        .save()
    )


def fetch_wafer_id_lookup(spark):
    """Read the just-inserted (wafer_index, wafer_id) mapping back out."""
    p = db_properties()
    return (
        spark.read
             .format("jdbc")
             .option("url",     p["url"])
             .option("dbtable", "(SELECT wafer_index, wafer_id FROM secom.wafers) AS wafer_map")
             .option("user",    p["username"])
             .option("password", p["password"])
             .option("driver",  p["driver"])
             .load()
    )


def write_measurements(long_df, wafer_lookup, mode: str = "append") -> None:
    """Join long_df with the wafer_id lookup and bulk-insert into secom.measurements."""
    p = db_properties()
    joined = (
        long_df.join(wafer_lookup, on="wafer_index", how="inner")
               .select("wafer_id", "sensor_name", "value")
    )
    (
        joined.write
              .format("jdbc")
              .mode(mode)
              .option("url",     p["url"])
              .option("dbtable", "secom.measurements")
              .option("user",    p["username"])
              .option("password", p["password"])
              .option("driver",  p["driver"])
              # Larger batch size = faster on large inserts
              .option("batchsize", "10000")
              .save()
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def truncate_tables() -> None:
    """Wipe wafers + measurements so re-ingesting doesn't double-insert."""
    conn = psycopg2.connect(**psycopg2_dsn())
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE secom.alarms RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE secom.measurements RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE secom.wafers RESTART IDENTITY CASCADE;")
    finally:
        conn.close()


def ingest(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Full ingestion pipeline. Returns row counts after the run."""
    spark = make_spark_session()
    try:
        print("[1/4] Reading raw files into Spark...")
        wafers_df, sensor_df_wide = load_raw_into_spark(spark, raw_dir)
        n_wafers = wafers_df.count()
        n_sensors = len([c for c in sensor_df_wide.columns
                         if c.startswith("sensor_")])
        print(f"      wafers={n_wafers}, sensors={n_sensors}")

        print("[2/4] Truncating destination tables...")
        truncate_tables()

        print("[3/4] Writing secom.wafers...")
        write_wafers(wafers_df)

        print("[4/4] Pivoting and writing secom.measurements...")
        long_df = melt_to_long(sensor_df_wide)
        wafer_lookup = fetch_wafer_id_lookup(spark)
        write_measurements(long_df, wafer_lookup)
    finally:
        spark.stop()

    # Final counts
    from src.db_schema import verify_schema
    counts = verify_schema()
    return counts


if __name__ == "__main__":
    counts = ingest()
    print("\nIngestion complete. Row counts:")
    for t, n in counts.items():
        print(f"  secom.{t:<14} {n:>10,}")

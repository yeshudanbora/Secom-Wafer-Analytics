# SQL data layer (Phase 6.5)

The SECOM SPC + FDC project supports two interchangeable data sources:

| Source | Where data lives | When to use it |
|---|---|---|
| File-based | `data/raw/secom.data` (CSV) | Default. Fast, no setup. |
| SQL-based | PostgreSQL `secom.*` tables | When you want the data-engineering pipeline shape (SparkSQL ingest, JDBC read, relational queries). |

Both produce **bit-for-bit identical** cleaned frames — the SQL path is a
drop-in replacement, not a different analysis.

This page documents the SQL layer. See the main `README.md` for the
file-based pipeline.

---

## Schema

Five tables in the `secom` schema:

```
secom.wafers          (1,567 rows)  — one row per wafer with timestamp + label
secom.measurements    (924,530)     — long-format (wafer_id, sensor_name, value)
secom.spc_limits      (≤ 25 rows)   — fitted univariate control limits
secom.alarms          (≤ 50 rows)   — FDC alarms with top contributors
secom.r2r_runs        (≤ 100 rows)  — EWMA controller sweep results
```

Full DDL: `sql/01_schema.sql`.

The `measurements` table uses **long format** (one row per wafer × sensor)
rather than wide. That's 924k rows vs. 1567 — still tiny for Postgres,
and the long shape lets you write clean per-sensor queries like
`WHERE sensor_name = 'sensor_348'` without enumerating 590 columns.

---

## Setup

### 1. Install PostgreSQL

Course default settings from Lecture 3 are baked in:
- Host: `localhost`
- Port: `5432`
- Database: `postgres`
- Username: `postgres`
- Password: `bigdata`

Override any of these via environment variables: `DB_HOST`, `DB_PORT`,
`DB_NAME`, `DB_USER`, `DB_PASSWORD`.

### 2. Apply the schema

```bash
python -m src.db_schema
```

Creates the `secom` schema and all five tables. Idempotent — safe to
re-run.

### 3. Ingest the data

Two paths, equivalent in result:

**SparkSQL ingest** (the Lecture 3 / production-shaped pipeline):

```bash
python -m src.db_ingest
```

Routes through SparkSQL with JDBC writes. Requires the PostgreSQL JDBC
driver, which is auto-downloaded on first run via
`spark.jars.packages=org.postgresql:postgresql:42.7.3`. If you've already
installed the driver into `$SPARK_HOME/jars/`, that takes precedence.

**Pure psycopg2 ingest** (fallback, faster on small data):

```bash
python -m src.db_ingest_psycopg2
```

Uses Postgres `COPY` for bulk insert. Convenient when you haven't set up
the Spark/JDBC stack yet, or for grading/CI environments. Loads in
~5 seconds vs. ~30 seconds for the Spark path on SECOM-sized data.

### 4. Verify

```bash
python -c "from src.db_schema import verify_schema; print(verify_schema())"
```

Should print:
```
{'wafers': 1567, 'measurements': 924530, 'spc_limits': 0, 'alarms': 0, 'r2r_runs': 0}
```

---

## Using the SQL layer in analysis

The cleanest API for switching the analysis modules to the SQL source:

```python
from src.db_query import load_wafers_pandas
from src.data_loader import clean, select_critical_sensors

# Read from Postgres instead of disk
df = load_wafers_pandas()

# Everything else is unchanged
cleaned, report = clean(df)
critical = select_critical_sensors(cleaned, n=25)
```

For the SparkSQL read-back demo:

```python
from src.db_ingest import make_spark_session
from src.db_query import load_wafers_spark

spark = make_spark_session("analysis")
long_df = load_wafers_spark(spark, wide=False)
long_df.printSchema()
long_df.show(5)
```

For pure-SQL queries (no Python middleware):

```python
from src.db_query import fail_rate, sensor_summary_sql
print(f"Fail rate (pure SQL): {fail_rate():.4f}")
summary = sensor_summary_sql()  # per-sensor stats computed in PG
```

---

## Why this design is what a fab would actually use

Three reasons the SQL layer is more than just resume polish:

1. **System of record.** Sensor data in real fabs lives in a relational
   warehouse, not on file shares. SPC engineers query it via SQL daily.
   The `wafers / measurements` split mirrors how Inficon, IBM PMQ, and
   most fab MES/MDE schemas decompose the same data.

2. **Multiple consumers.** Once the data is in Postgres, Jupyter
   notebooks, ad-hoc SQL via `psql`, and BI tools like Tableau/Power BI
   can all hit the same numbers. With file-only storage every consumer
   has its own copy.

3. **Production scaling shape.** When the dataset grows beyond a single
   wafer-set into months of full-tool telemetry (50,000+ wafers ×
   500 sensors × 600 process steps = ~15B rows), the long-format
   measurements table is the only shape that scales horizontally. The
   SECOM case is small enough that wide-format pandas works fine, but
   coding to the long-format SQL shape is the move that future-proofs
   the architecture.

---

## Where this fits in the resume

For a Micron application, the SQL layer adds these credible bullet points:

- *Designed PostgreSQL schema (5 tables, FK constraints, indexed lookups) for wafer-level fab telemetry with long-format measurement storage scaling to 924k rows on the 1,567-wafer SECOM dataset; same shape extends to 15B+ rows in production fab volumes.*
- *Implemented SparkSQL ingestion pipeline (CSV → DataFrame → JDBC → PostgreSQL) following the standard fab MDE pattern; full ingest of 590-sensor wafer dataset in <30s on local Spark.*
- *Validated SQL-sourced analysis produces bit-for-bit identical SPC and FDC results vs. file-sourced pipeline (351 cleaned sensors, 6.64% fail rate, 56% FDC precision), confirming the SQL layer is a drop-in production-shaped replacement.*

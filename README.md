# Wafer Failure Prediction from Tool Telemetry

**SPC, Fault Detection, and Run-to-Run Control on the SECOM Semiconductor Dataset**

Statistical Process Control and multivariate Fault Detection on 590 tool
sensors across 1,567 wafer runs, monitored with the same SPC and FDC
methods used in production fabs.

The project ingests raw fab-shaped telemetry into a PostgreSQL warehouse
via SparkSQL, fits univariate I-MR control charts with Nelson rules
1–8, runs PCA-based multivariate fault detection with Hotelling's T² and
SPE/Q statistics, decomposes alarms into per-sensor root-cause
contributors, and simulates an EWMA run-to-run controller that prevents
half of the observed excursions on a drifting sensor.

---

## What this is

A real fab integration / photo process engineer monitors hundreds of
tool-telemetry signals — temperatures, pressures, gas flows, RF powers,
optical sensor readings, metrology values — streamed off equipment, with
quality verdicts arriving from an electrical test station hours or days
later. The SECOM dataset (UCI ML Repository) is exactly that data shape:
590 sensor measurements per wafer, paired with downstream pass/fail
labels.

This project applies the SPC and FDC methods used in production fabs:

- **Data engineering**: PostgreSQL schema with `wafers` / `measurements`
  / `spc_limits` / `alarms` / `r2r_runs` tables; SparkSQL ingestion via
  JDBC writes; long-format storage that scales to production telemetry
  volumes.
- **I-MR control charts** for every monitored sensor, with Nelson Rules
  1–8 highlighting drift, runs, stratification, and mixture patterns.
- **Cp / Cpk capability indices** computed against ±3σ_overall spec
  limits, with σ_within (MR/d₂) driving the capability ratio so drift
  shows up as a Cp/Cpk gap.
- **PCA-based multivariate FDC** with Hotelling's T² and SPE/Q
  statistics at 99 % confidence limits (Jackson–Mudholkar closed form).
- **MacGregor contribution decomposition** to localize the sensor(s)
  that drove every alarm — the root-cause hand-off a process engineer
  uses to launch a tool investigation.
- **EWMA run-to-run controller simulation** showing closed-loop
  excursion prevention on a drifting critical sensor.

---

## Headline numbers

| Metric                                           | Value      |
| ------------------------------------------------ | ---------- |
| Wafers analysed                                  | 1,567      |
| Raw sensor signals                               | 590        |
| Sensors after cleaning (missing/var/corr)        | 351        |
| Critical sensors monitored (top-25)              | 25         |
| Baseline (in-control) wafers                     | 1,024      |
| Monitoring (held-out) wafers                     | 543        |
| Sensors with Cpk_monitoring < 1.0                | 2 / 25     |
| PCA components retained (≥ 90 % variance)        | 12         |
| **FDC precision on monitoring set**              | **56.4 %** |
| FDC false-alarm rate                             | 3.9 %      |
| **Precision lift over base-rate (19.2 % fails)** | **2.95 ×** |
| R2R-prevented Nelson violations (best gains)     | 259 / 400 (65 %) |

See `reports/methodology_sensor_selection.md` for the methodology
discussion (supervised vs unsupervised sensor selection comparison) and
`reports/sql_data_layer.md` for the SQL/Spark architecture.

---

## Repository layout

```
secom-spc-fdc/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # secom.data, secom_labels.data (UCI)
│   └── processed/        # cleaned parquet + summaries (regenerated)
├── sql/
│   └── 01_schema.sql     # PostgreSQL DDL for the 5 secom.* tables
├── notebooks/
│   └── analysis.ipynb    # Phase 1-3 walkthrough with plots
├── src/
│   ├── data_loader.py    # Phase 1: ingest + clean + critical-sensor select
│   ├── spc.py            # Phase 2: I-MR, Nelson rules, Cp/Cpk
│   ├── fdc.py            # Phase 3: PCA T² + SPE + contributions
│   ├── rootcause.py      # Phase 4: alarm catalog + Pareto + drill-down
│   ├── r2r.py            # Phase 5: EWMA run-to-run controller
│   ├── db_config.py      # Phase 6: Postgres connection settings
│   ├── db_schema.py      # Phase 6: DDL applier
│   ├── db_ingest.py      # Phase 6: SparkSQL → JDBC ingestion
│   ├── db_ingest_psycopg2.py  # Phase 6: pure-psycopg2 fallback
│   └── db_query.py       # Phase 6: read-back helpers
├── tests/                # 30 unit tests across all phases
└── reports/
    ├── methodology_sensor_selection.md
    └── sql_data_layer.md
```

---

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.10 recommended.

### 2. Verify the raw data

The two SECOM files should already be in `data/raw/`:

```
data/raw/secom.data            (1567 lines × 590 sensors, NaN-encoded)
data/raw/secom_labels.data     (1567 lines: label + timestamp)
```

If they are missing, download them from the
[UCI ML Repository](https://archive.ics.uci.edu/dataset/179/secom).

### 3. Run the analysis pipeline (file-based, no database needed)

Each module has a CLI entrypoint that runs its phase end-to-end and
writes outputs into `data/processed/`:

```bash
python -m src.data_loader      # Phase 1: clean + select critical sensors
python -m src.spc              # Phase 2: univariate SPC + Cpk leaderboard
python -m src.fdc              # Phase 3: PCA T² + SPE + methodology comparison
python -m src.rootcause        # Phase 4: alarm catalog + sensor leaderboard
python -m src.r2r              # Phase 5: EWMA controller + gain sweep
```

### 4. (Optional) Load the data into PostgreSQL

If you want to demonstrate the data-engineering shape — schema design,
SparkSQL ingestion, JDBC connectors — see `reports/sql_data_layer.md`
for full setup. TL;DR:

```bash
# Assumes Postgres running locally with course-default credentials
python -m src.db_schema             # create the secom.* tables
python -m src.db_ingest             # SparkSQL → JDBC ingestion
# or, faster fallback without Spark:
python -m src.db_ingest_psycopg2    # pure-psycopg2 COPY bulk load
```

The SQL-sourced and file-sourced pipelines produce **bit-for-bit
identical** analysis results — the SQL layer is a drop-in replacement
that demonstrates production-shape data engineering without changing the
underlying math.

### 5. Run the tests

```bash
for f in tests/test_*.py; do python "$f"; done
```

All 30 tests should pass (or 26 if Postgres isn't installed — the SQL
tests are skipped automatically).

### 6. Read the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook walks through Phases 1–3 with explanatory commentary,
plots, and the methodology discussion. It is the recommended way to
explore the analysis interactively.

---

## Outputs

After running the full pipeline:

| File                                              | Contents                                          |
| ------------------------------------------------- | ------------------------------------------------- |
| `data/processed/secom_clean.parquet`              | 1567 × 354 cleaned wafer-level frame              |
| `data/processed/cleaning_report.txt`              | Audit trail: sensors dropped at each stage        |
| `data/processed/critical_sensors.txt`             | Top-25 monitored sensors (one per line)           |
| `data/processed/spc_summary.csv`                  | Per-sensor Cp, Cpk, Nelson rule violations        |
| `data/processed/fdc_monitoring_scores.csv`        | Per-wafer T², SPE, alarm flags                    |
| `data/processed/fdc_methodology_comparison.csv`   | Headline vs robustness-check FDC metrics          |
| `data/processed/alarm_catalog.csv`                | Every alarm with top contributors                 |
| `data/processed/sensor_alarm_leaderboard.csv`     | "Usual suspect" sensors across all alarms         |
| `data/processed/r2r_gain_sweep.csv`               | EWMA controller results across (λ, K) grid        |
| `data/processed/r2r_simulation.csv`               | Raw / EWMA / correction / controlled trajectories |

---

## References

The math is standard. Where applicable, references are cited inline in
the source modules:

- Montgomery, D.C. (2019). *Introduction to Statistical Quality
  Control*, 8th ed., Wiley.
- Nelson, L.S. (1984). The Shewhart Control Chart — Tests for Special
  Causes. *Journal of Quality Technology*, 16(4).
- Jackson, J.E. & Mudholkar, G.S. (1979). Control Procedures for
  Residuals Associated With Principal Component Analysis.
  *Technometrics*, 21(3).
- MacGregor, J.F. & Kourti, T. (1995). Statistical process control of
  multivariate processes. *Control Engineering Practice*, 3(3).
- Qin, S.J. (2003). Statistical process monitoring: basics and beyond.
  *Journal of Chemometrics*, 17.
- Ingolfsson, A. & Sachs, E. (1993). Stability and Sensitivity of an
  EWMA Controller. *Journal of Quality Technology*, 25(4).

---

## Author

Yeshudan Bora · MS in AI Engineering — Chemical Engineering, Carnegie
Mellon University · expected Dec 2026.

This project was built as part of an application portfolio for
photolithography / process engineering roles at Micron. The framing,
methodology choices, and headline numbers are documented inline so the
work is fully auditable.

-- ============================================================================
-- SECOM SPC + FDC project — PostgreSQL schema
-- ============================================================================
-- Per Lecture 3, DDL/DCL operations live in .sql files because they are not
-- impacted by data size. DML (inserts/updates) goes through SparkSQL.
--
-- All tables live in a `secom` schema so they don't collide with anything
-- else in the default `postgres` database.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS secom;

-- ----------------------------------------------------------------------------
-- wafers : one row per wafer run, with timestamp and pass/fail outcome.
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS secom.alarms CASCADE;
DROP TABLE IF EXISTS secom.measurements CASCADE;
DROP TABLE IF EXISTS secom.spc_limits CASCADE;
DROP TABLE IF EXISTS secom.r2r_runs CASCADE;
DROP TABLE IF EXISTS secom.wafers CASCADE;

CREATE TABLE secom.wafers (
    wafer_id      SERIAL       PRIMARY KEY,
    wafer_index   INTEGER      NOT NULL UNIQUE,    -- positional 0..1566
    ts            TIMESTAMP    NOT NULL,
    label         SMALLINT     NOT NULL,           -- -1 = pass, +1 = fail
    is_fail       BOOLEAN      NOT NULL,
    ingested_at   TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_wafers_ts        ON secom.wafers (ts);
CREATE INDEX idx_wafers_is_fail   ON secom.wafers (is_fail);

-- ----------------------------------------------------------------------------
-- measurements : long-format sensor readings, one row per (wafer, sensor).
-- ----------------------------------------------------------------------------
CREATE TABLE secom.measurements (
    measurement_id  BIGSERIAL    PRIMARY KEY,
    wafer_id        INTEGER      NOT NULL REFERENCES secom.wafers (wafer_id),
    sensor_name     TEXT         NOT NULL,
    value           DOUBLE PRECISION                 -- NULL allowed (raw is NaN)
);

CREATE INDEX idx_meas_wafer       ON secom.measurements (wafer_id);
CREATE INDEX idx_meas_sensor      ON secom.measurements (sensor_name);
CREATE INDEX idx_meas_wafer_sensor ON secom.measurements (wafer_id, sensor_name);

-- ----------------------------------------------------------------------------
-- spc_limits : fitted univariate SPC limits per critical sensor.
-- One row per sensor; truncated and reinserted when limits are re-fit.
-- ----------------------------------------------------------------------------
CREATE TABLE secom.spc_limits (
    sensor_name        TEXT             PRIMARY KEY,
    mu                 DOUBLE PRECISION NOT NULL,
    sigma_within       DOUBLE PRECISION NOT NULL,
    sigma_overall      DOUBLE PRECISION NOT NULL,
    drift_ratio        DOUBLE PRECISION,
    ucl                DOUBLE PRECISION NOT NULL,
    lcl                DOUBLE PRECISION NOT NULL,
    usl                DOUBLE PRECISION NOT NULL,
    lsl                DOUBLE PRECISION NOT NULL,
    cp                 DOUBLE PRECISION,
    cpk_baseline       DOUBLE PRECISION,
    cpk_monitoring     DOUBLE PRECISION,
    mean_shift_sigma   DOUBLE PRECISION,
    flagged_points     INTEGER,
    flag_rate          DOUBLE PRECISION,
    fitted_at          TIMESTAMP        NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- alarms : every FDC-alarmed wafer with its top-N contributors.
-- ----------------------------------------------------------------------------
CREATE TABLE secom.alarms (
    alarm_id              SERIAL    PRIMARY KEY,
    wafer_id              INTEGER   NOT NULL REFERENCES secom.wafers (wafer_id),
    monitoring_index      INTEGER   NOT NULL,         -- position in monitoring set
    alarm_type            TEXT      NOT NULL,         -- 'T2-only'/'SPE-only'/'both'
    t2                    DOUBLE PRECISION NOT NULL,
    spe                   DOUBLE PRECISION NOT NULL,
    t2_norm               DOUBLE PRECISION NOT NULL,
    spe_norm              DOUBLE PRECISION,
    severity              DOUBLE PRECISION NOT NULL,
    top_contributors_t2   TEXT[],
    top_contributors_spe  TEXT[],
    is_fail               BOOLEAN   NOT NULL,
    flagged_at            TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alarms_wafer    ON secom.alarms (wafer_id);
CREATE INDEX idx_alarms_severity ON secom.alarms (severity DESC);

-- ----------------------------------------------------------------------------
-- r2r_runs : EWMA controller simulation results per gain combination.
-- ----------------------------------------------------------------------------
CREATE TABLE secom.r2r_runs (
    run_id                  SERIAL           PRIMARY KEY,
    sensor_name             TEXT             NOT NULL,
    lambda_smoothing        DOUBLE PRECISION NOT NULL,
    k_gain                  DOUBLE PRECISION NOT NULL,
    raw_violations          INTEGER          NOT NULL,
    controlled_violations   INTEGER          NOT NULL,
    excursions_prevented    INTEGER          NOT NULL,
    prevention_pct          DOUBLE PRECISION,
    raw_cpk_monitoring      DOUBLE PRECISION,
    controlled_cpk_monitoring DOUBLE PRECISION,
    cpk_lift                DOUBLE PRECISION,
    run_at                  TIMESTAMP        NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_r2r_sensor ON secom.r2r_runs (sensor_name);

-- ----------------------------------------------------------------------------
-- Convenience view : wide-format measurements (one row per wafer, sensors
-- as columns). This is what most analysis modules want.
--
-- Built lazily on top of long-format `measurements` via crosstab. For SECOM
-- the long-format table is small (~924k rows), so the view materializes fast.
-- Kept as a regular view (not materialized) so it always reflects the
-- current state of `measurements`.
-- ----------------------------------------------------------------------------
-- NOTE: building a 590-column crosstab in pure SQL is verbose; we skip the
-- view and let Python do the long→wide pivot in db_query.py, which is faster
-- and works regardless of how many sensors are present.

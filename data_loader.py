"""
SECOM data ingestion and cleaning pipeline.

The SECOM dataset (UCI ML Repository) contains 590 sensor measurements per
wafer run with a downstream pass/fail label. In a real fab this is the data
shape an integration / photo process engineer monitors -- temperatures,
pressures, gas flows, RF powers, optical sensor readings, metrology values --
streamed off equipment, with the quality verdict arriving from an electrical
test station hours or days later.

This module:
    1. Parses the raw whitespace-separated sensor file and the label file.
    2. Joins them on row index and attaches the wafer timestamp.
    3. Drops dead / redundant sensors (>50% missing, zero variance, near-duplicate).
    4. Median-imputes remaining missing values.
    5. Writes a cleaned parquet for downstream SPC / FDC modules.

Authoring note: the cleaning thresholds are intentionally conservative and
documented inline so a hiring manager can audit each decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CleaningConfig:
    """Tunable thresholds for the cleaning pipeline.

    Each threshold maps to a real engineering decision:
        max_missing_frac  : sensors with too much downtime are unreliable
        min_variance      : constant signals carry no SPC information
        max_corr          : sensors above this correlation are redundant
                            (one of each pair is dropped to keep the matrix
                            invertible for PCA in fdc.py)
    """
    max_missing_frac: float = 0.50
    min_variance: float = 1e-10
    max_corr: float = 0.99


@dataclass
class CleaningReport:
    """Audit trail of what the pipeline removed and why.

    Stored alongside the processed parquet so the cleaning is reproducible
    and reviewable -- an engineer should never have to ask "why did sensor
    412 disappear?" without an answer in writing.
    """
    n_rows: int
    n_sensors_raw: int
    n_sensors_after_missing: int
    n_sensors_after_variance: int
    n_sensors_after_corr: int
    n_sensors_final: int
    pass_count: int
    fail_count: int
    fail_rate: float
    dropped_high_missing: list
    dropped_zero_variance: list
    dropped_high_corr: list

    def summary(self) -> str:
        return (
            f"SECOM cleaning report\n"
            f"---------------------\n"
            f"Rows (wafer runs)          : {self.n_rows}\n"
            f"Sensors raw                : {self.n_sensors_raw}\n"
            f"  after missing-rate filter: {self.n_sensors_after_missing} "
            f"(-{self.n_sensors_raw - self.n_sensors_after_missing})\n"
            f"  after variance filter    : {self.n_sensors_after_variance} "
            f"(-{self.n_sensors_after_missing - self.n_sensors_after_variance})\n"
            f"  after correlation filter : {self.n_sensors_after_corr} "
            f"(-{self.n_sensors_after_variance - self.n_sensors_after_corr})\n"
            f"Sensors final              : {self.n_sensors_final}\n"
            f"\n"
            f"Pass wafers                : {self.pass_count}\n"
            f"Fail wafers                : {self.fail_count}\n"
            f"Fail rate                  : {self.fail_rate:.2%}\n"
        )


# ---------------------------------------------------------------------------
# Raw IO
# ---------------------------------------------------------------------------

def load_raw(
    sensor_path: str | Path,
    label_path: str | Path,
) -> pd.DataFrame:
    """Read raw SECOM files and return a single dataframe.

    The sensor file is whitespace-separated with NaN encoded as the literal
    string 'NaN'. The label file has two whitespace-separated columns:
    label (-1 pass / +1 fail) and a quoted timestamp.

    Returns
    -------
    DataFrame with columns:
        timestamp        : pd.Timestamp
        label            : int   (-1 pass, +1 fail)
        is_fail          : bool  (label == 1)
        sensor_0 .. sensor_589 : float
    """
    sensor_path = Path(sensor_path)
    label_path = Path(label_path)

    sensors = pd.read_csv(
        sensor_path,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )
    sensors.columns = [f"sensor_{i}" for i in range(sensors.shape[1])]

    # The label file has the form:   -1 "19/07/2008 11:55:00"
    # Pandas' whitespace tokenizer treats the inner space inside the quoted
    # timestamp as a delimiter, so we parse it line-by-line instead.
    raw_lines = Path(label_path).read_text().splitlines()
    label_vals: list[int] = []
    ts_vals: list[str] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        lbl_str, _, rest = line.partition(" ")
        label_vals.append(int(lbl_str))
        ts_vals.append(rest.strip().strip('"'))
    labels = pd.DataFrame({"label": label_vals, "timestamp": ts_vals})
    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )
    labels["is_fail"] = labels["label"] == 1

    if len(sensors) != len(labels):
        raise ValueError(
            f"Row count mismatch: sensors={len(sensors)} labels={len(labels)}"
        )

    df = pd.concat([labels, sensors], axis=1)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def clean(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline.

    Stage 1 -- drop sensors that are mostly NaN (broken/offline equipment).
    Stage 2 -- drop sensors with zero / near-zero variance (no info content).
    Stage 3 -- greedy drop of one sensor in each highly correlated pair.
    Stage 4 -- median-impute residual NaNs in the surviving sensors.

    Returns
    -------
    cleaned : DataFrame
    report  : CleaningReport
    """
    cfg = config or CleaningConfig()
    sensor_cols = _sensor_cols(df)
    n_raw = len(sensor_cols)

    # --- Stage 1: missing-rate filter -------------------------------------
    miss_frac = df[sensor_cols].isna().mean()
    dropped_missing = miss_frac[miss_frac > cfg.max_missing_frac].index.tolist()
    cols_after_missing = [c for c in sensor_cols if c not in dropped_missing]

    # --- Stage 2: variance filter -----------------------------------------
    variances = df[cols_after_missing].var(ddof=1)
    dropped_var = variances[variances < cfg.min_variance].index.tolist()
    cols_after_var = [c for c in cols_after_missing if c not in dropped_var]

    # --- Stage 3: correlation filter --------------------------------------
    # Compute |corr| on median-imputed view (just for the corr calc; final
    # imputation happens after this stage on the surviving columns).
    tmp = df[cols_after_var].fillna(df[cols_after_var].median())
    corr = tmp.corr().abs()
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    dropped_corr = []
    for i, col_i in enumerate(corr.columns):
        if col_i in dropped_corr:
            continue
        # find any partner with corr above threshold; drop the partner
        partners = corr.columns[(corr.values[i] > cfg.max_corr) & upper[i]]
        for partner in partners:
            if partner not in dropped_corr:
                dropped_corr.append(partner)
    cols_final = [c for c in cols_after_var if c not in dropped_corr]

    # --- Stage 4: median imputation on survivors --------------------------
    cleaned = df.copy()
    cleaned = cleaned[["timestamp", "label", "is_fail"] + cols_final]
    cleaned[cols_final] = cleaned[cols_final].fillna(cleaned[cols_final].median())

    report = CleaningReport(
        n_rows=len(cleaned),
        n_sensors_raw=n_raw,
        n_sensors_after_missing=len(cols_after_missing),
        n_sensors_after_variance=len(cols_after_var),
        n_sensors_after_corr=len(cols_final),
        n_sensors_final=len(cols_final),
        pass_count=int((~cleaned["is_fail"]).sum()),
        fail_count=int(cleaned["is_fail"].sum()),
        fail_rate=float(cleaned["is_fail"].mean()),
        dropped_high_missing=dropped_missing,
        dropped_zero_variance=dropped_var,
        dropped_high_corr=dropped_corr,
    )
    return cleaned, report


# ---------------------------------------------------------------------------
# Critical-sensor selection
# ---------------------------------------------------------------------------

def select_critical_sensors(
    cleaned: pd.DataFrame,
    n: int = 25,
    method: str = "fail_corr",
    fit_on: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Pick the top-N sensors most worth monitoring.

    In a real fab the "critical" list is set by process engineers based on
    physics + historical excursions. Here we approximate it from the data:

        method='fail_corr' -- absolute Pearson correlation with the fail
                              label. SUPERVISED -- to avoid leakage, pass
                              the in-control baseline subset as `fit_on`
                              so selection only sees data that would have
                              been available before the monitoring period.
        method='variance'  -- highest log-variance sensors (most dynamic
                              range). Unsupervised and leakage-free, but
                              tends to pick high-energy noise channels.

    Parameters
    ----------
    cleaned   : full cleaned dataframe (used to enumerate sensor cols
                and -- if fit_on is None -- to compute the ranking)
    n         : number of sensors to keep
    method    : 'fail_corr' or 'variance'
    fit_on    : if provided, compute the ranking on this subset only
                (typical: pass the Phase-1 baseline). When None we fall
                back to ranking on the full cleaned frame.

    Returns
    -------
    list of sensor column names, length <= n
    """
    sensor_cols = _sensor_cols(cleaned)
    src = fit_on if fit_on is not None else cleaned

    if method == "fail_corr":
        target = src["is_fail"].astype(float)
        # If the baseline contains zero failures (the typical case for the
        # first 70% pass-only split), |corr| with a constant target is
        # undefined. Detect this and fail loudly so the caller picks a
        # different fit_on or method.
        if target.std(ddof=1) == 0:
            raise ValueError(
                "select_critical_sensors(method='fail_corr') needs a "
                "fit_on subset that contains both pass and fail wafers. "
                "Either pass the full cleaned frame, or use "
                "method='variance' for an unsupervised, leakage-free split."
            )
        scores = src[sensor_cols].apply(
            lambda s: abs(np.corrcoef(s, target)[0, 1])
            if s.std(ddof=1) > 0 else 0.0
        )
    elif method == "variance":
        scores = np.log1p(src[sensor_cols].var(ddof=1))
    else:
        raise ValueError(f"Unknown method: {method}")

    return scores.sort_values(ascending=False).head(n).index.tolist()


def select_critical_sensors_no_leak(
    cleaned: pd.DataFrame,
    n: int = 25,
    method: str = "fail_corr",
    selection_frac: float = 0.40,
) -> tuple[list[str], int]:
    """Leakage-free critical-sensor selection via chronological lookahead.

    This is the no-leakage version of select_critical_sensors. It mimics
    how a real fab decides which sensors to monitor:

        1. Sort wafers by timestamp.
        2. Use the first `selection_frac` of wafers (which contain both
           passes and fails) as the *selection window*. Sensor ranking
           is fit on this window only.
        3. Downstream code can now use the remaining (1 - selection_frac)
           of the timeline for SPC baseline + monitoring without ever
           having shown selection any of those wafers.

    Returns
    -------
    sensors        : list of selected sensor names
    n_selection    : number of wafers used in the selection window
                     (caller can re-split the remainder for SPC)
    """
    df = cleaned.sort_values("timestamp").reset_index(drop=True)
    n_sel = int(len(df) * selection_frac)
    selection_window = df.iloc[:n_sel]

    sensors = select_critical_sensors(
        cleaned=df,
        n=n,
        method=method,
        fit_on=selection_window,
    )
    return sensors, n_sel


# ---------------------------------------------------------------------------
# Convenience entrypoint
# ---------------------------------------------------------------------------

def build_processed(
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/processed",
    config: Optional[CleaningConfig] = None,
) -> tuple[pd.DataFrame, CleaningReport, list[str]]:
    """End-to-end: load -> clean -> save -> select critical sensors.

    Writes:
        <out_dir>/secom_clean.parquet
        <out_dir>/cleaning_report.txt
        <out_dir>/critical_sensors.txt
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_dir / "secom.data", raw_dir / "secom_labels.data")
    cleaned, report = clean(df, config)
    critical = select_critical_sensors(cleaned, n=25, method="fail_corr")

    cleaned.to_parquet(out_dir / "secom_clean.parquet", index=False)
    (out_dir / "cleaning_report.txt").write_text(report.summary())
    (out_dir / "critical_sensors.txt").write_text("\n".join(critical))

    return cleaned, report, critical


if __name__ == "__main__":
    cleaned, report, critical = build_processed()
    print(report.summary())
    print(f"Top 25 critical sensors (by |corr| with fail label):")
    for i, c in enumerate(critical, 1):
        print(f"  {i:2d}. {c}")

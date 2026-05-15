"""
Multivariate Fault Detection & Classification (FDC) for SECOM.

Where Phase 2's univariate SPC monitors each sensor in isolation, this
module monitors the *joint* sensor signature using Principal Component
Analysis. This is the math that backs every commercial FDC platform
(Inficon FabGuard, Applied Materials E3, BISTel SmartEES, etc.):

    1. Standardize the baseline data (z-score per sensor) so each sensor
       contributes equally regardless of physical units.
    2. Fit PCA on the standardized in-control wafers. Keep K components
       capturing 90% cumulative variance (typical fab default).
    3. For every new wafer, compute two statistics:

           T^2 = sum_{k=1..K} (t_k^2 / lambda_k)
                 -- Hotelling's T-squared, distance from the origin
                    *inside* the K-dim model subspace, normalized by
                    each PC's variance. Catches wafers where the
                    retained sensors vary in unusual but in-model ways.

           Q (SPE) = || x - x_hat ||^2
                 -- Squared Prediction Error, distance from the model
                    subspace. Catches wafers whose correlation structure
                    breaks down -- sensors that normally move together
                    suddenly don't.

    4. Set 99% control limits using the standard closed-form
       distributions:
           T^2_limit  : scaled F-distribution
           SPE_limit  : Jackson-Mudholkar approximation (Box 1954)

    5. Decompose every excursion into per-sensor contributions so a
       process engineer knows which sensor(s) drove the alarm
       (the "root-cause" piece of the JD).

References:
    Jackson, J.E. & Mudholkar, G.S. (1979). Control Procedures for
        Residuals Associated With Principal Component Analysis.
        Technometrics, 21(3).
    MacGregor, J.F. & Kourti, T. (1995). Statistical process control
        of multivariate processes. Control Engineering Practice, 3(3).
    Qin, S.J. (2003). Statistical process monitoring: basics and beyond.
        Journal of Chemometrics, 17.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PCAModel:
    """Fitted PCA model + control limits.

    Stores everything needed to score new wafers and decompose their
    statistics. Designed to be pickled / re-loaded by downstream code
    without re-fitting.
    """
    sensor_cols: list[str]
    mu: np.ndarray            # per-sensor mean from baseline (length p)
    sigma: np.ndarray         # per-sensor std  from baseline (length p)
    P: np.ndarray             # loading matrix, shape (p, K)
    eigvals: np.ndarray       # PC variances (length K)
    explained_variance_ratio: np.ndarray  # full p-length, useful for plots
    K: int                    # number of retained components
    N_baseline: int
    alpha: float              # significance level (0.01 = 99% conf)
    t2_limit: float
    spe_limit: float
    # cached residual-eigenvalue statistics for SPE limit (Jackson-Mudholkar)
    theta1: float
    theta2: float
    theta3: float


@dataclass
class FDCScores:
    """Per-wafer T^2 and SPE statistics with limit overlays.

    Indexed positionally to match the input dataframe row order.
    """
    t2: np.ndarray
    spe: np.ndarray
    t2_limit: float
    spe_limit: float

    def t2_alarm(self) -> np.ndarray:
        return self.t2 > self.t2_limit

    def spe_alarm(self) -> np.ndarray:
        return self.spe > self.spe_limit

    def any_alarm(self) -> np.ndarray:
        return self.t2_alarm() | self.spe_alarm()


# ---------------------------------------------------------------------------
# PCA fit
# ---------------------------------------------------------------------------

def fit_pca(
    baseline: pd.DataFrame,
    sensor_cols: list[str],
    var_threshold: float = 0.90,
    alpha: float = 0.01,
) -> PCAModel:
    """Fit PCA on standardized baseline data and compute control limits.

    Parameters
    ----------
    baseline      : Phase-1 in-control wafers (DataFrame)
    sensor_cols   : sensors used as PCA inputs (e.g. critical 25)
    var_threshold : cumulative variance fraction to retain (0.90 default)
    alpha         : significance level for control limits (0.01 = 99%)

    Returns
    -------
    PCAModel with fitted loadings, eigenvalues, and limits.
    """
    X_raw = baseline[sensor_cols].to_numpy(dtype=float)
    N, p = X_raw.shape
    if N < p + 2:
        raise ValueError(
            f"Baseline too small for PCA: N={N}, p={p}. Need N > p + 1."
        )

    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0, ddof=1)
    sigma = np.where(sigma > 0, sigma, 1.0)  # guard against zero-var
    X = (X_raw - mu) / sigma

    # SVD: X = U S V^T, columns of V are the loadings (P).
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    eigvals_full = (s ** 2) / (N - 1)
    explained = eigvals_full / eigvals_full.sum()
    cum = np.cumsum(explained)
    K = int(np.searchsorted(cum, var_threshold) + 1)
    K = max(K, 1)

    P = Vt[:K].T              # (p, K)
    eigvals = eigvals_full[:K]

    # ---- T^2 limit (scaled F distribution) ------------------------------
    # T^2_a = K(N^2 - 1) / (N(N - K)) * F_{a, K, N-K}
    f_crit = stats.f.ppf(1 - alpha, K, N - K)
    t2_limit = K * (N ** 2 - 1) / (N * (N - K)) * f_crit

    # ---- SPE limit (Jackson-Mudholkar / Box approximation) --------------
    residual_eigs = eigvals_full[K:]
    theta1 = float(residual_eigs.sum())
    theta2 = float((residual_eigs ** 2).sum())
    theta3 = float((residual_eigs ** 3).sum())
    if theta1 <= 0 or theta2 <= 0:
        # Degenerate: model captures all variance. SPE limit -> 0.
        spe_limit = 0.0
    else:
        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2)
        c_alpha = stats.norm.ppf(1 - alpha)
        # Standard JM closed form
        spe_limit = theta1 * (
            c_alpha * np.sqrt(2 * theta2 * h0 ** 2) / theta1
            + 1
            + theta2 * h0 * (h0 - 1) / (theta1 ** 2)
        ) ** (1 / h0)

    return PCAModel(
        sensor_cols=list(sensor_cols),
        mu=mu,
        sigma=sigma,
        P=P,
        eigvals=eigvals,
        explained_variance_ratio=explained,
        K=K,
        N_baseline=N,
        alpha=alpha,
        t2_limit=float(t2_limit),
        spe_limit=float(spe_limit),
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(model: PCAModel, frame: pd.DataFrame) -> FDCScores:
    """Compute T^2 and SPE for every row in `frame`."""
    X_raw = frame[model.sensor_cols].to_numpy(dtype=float)
    X = (X_raw - model.mu) / model.sigma

    # Project into PC space
    T = X @ model.P                  # (N, K) scores
    # T^2 per row: sum_k (T_ik^2 / lambda_k)
    t2 = np.sum((T ** 2) / model.eigvals, axis=1)

    # Reconstruction in original (standardized) space
    X_hat = T @ model.P.T
    residual = X - X_hat
    spe = np.sum(residual ** 2, axis=1)

    return FDCScores(
        t2=t2, spe=spe,
        t2_limit=model.t2_limit, spe_limit=model.spe_limit,
    )


# ---------------------------------------------------------------------------
# Contribution analysis (root-cause)
# ---------------------------------------------------------------------------

def t2_contributions(
    model: PCAModel,
    wafer_row: pd.Series | np.ndarray,
) -> pd.Series:
    """Per-sensor contribution to a single wafer's T^2 statistic.

    Uses the standard MacGregor decomposition:
        T^2 = sum_k (t_k^2 / lambda_k)
        and  t_k = sum_j P_jk * x_j
        so   contrib_j to t_k = P_jk * x_j * t_k / lambda_k
        and  contrib_j to T^2 = sum_k (P_jk * x_j * t_k / lambda_k)

    Returns a Series indexed by sensor name.
    """
    if isinstance(wafer_row, pd.Series):
        x_raw = wafer_row[model.sensor_cols].to_numpy(dtype=float)
    else:
        x_raw = np.asarray(wafer_row, dtype=float)
    x = (x_raw - model.mu) / model.sigma

    t = x @ model.P                              # (K,)
    contribs = (model.P * x[:, None]) @ (t / model.eigvals)
    return pd.Series(contribs, index=model.sensor_cols).sort_values(
        key=lambda s: s.abs(), ascending=False
    )


def spe_contributions(
    model: PCAModel,
    wafer_row: pd.Series | np.ndarray,
) -> pd.Series:
    """Per-sensor contribution to a single wafer's SPE.

    SPE = sum_j (x_j - x_hat_j)^2
    so contribution of sensor j is just the squared residual on j.
    Always non-negative.
    """
    if isinstance(wafer_row, pd.Series):
        x_raw = wafer_row[model.sensor_cols].to_numpy(dtype=float)
    else:
        x_raw = np.asarray(wafer_row, dtype=float)
    x = (x_raw - model.mu) / model.sigma
    x_hat = (x @ model.P) @ model.P.T
    return pd.Series(
        (x - x_hat) ** 2, index=model.sensor_cols
    ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Detection-vs-fail-label evaluation
# ---------------------------------------------------------------------------

def evaluate_against_labels(
    scores: FDCScores,
    is_fail: np.ndarray,
) -> dict:
    """Confusion-matrix-style metrics treating fail wafers as positives."""
    is_fail = np.asarray(is_fail, dtype=bool)
    alarm = scores.any_alarm()
    tp = int((alarm & is_fail).sum())
    fp = int((alarm & ~is_fail).sum())
    fn = int((~alarm & is_fail).sum())
    tn = int((~alarm & ~is_fail).sum())

    pos = tp + fn
    neg = fp + tn
    tpr = tp / pos if pos else 0.0
    fpr = fp / neg if neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * tpr / (precision + tpr)
          if (precision + tpr) > 0 else 0.0)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "alarms": tp + fp,
        "fails": pos,
        "passes": neg,
        "TPR (recall)": tpr,
        "FPR (false alarm)": fpr,
        "precision": precision,
        "F1": f1,
        "t2_only_alarms": int(scores.t2_alarm().sum()),
        "spe_only_alarms": int(scores.spe_alarm().sum()),
        "both_alarms": int((scores.t2_alarm() & scores.spe_alarm()).sum()),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_fdc(
    cleaned: pd.DataFrame,
    sensor_cols: list[str],
    baseline_frac: float = 0.70,
    var_threshold: float = 0.90,
    alpha: float = 0.01,
) -> tuple[PCAModel, FDCScores, FDCScores, dict]:
    """End-to-end FDC: split, fit, score, evaluate."""
    from src.spc import split_baseline_monitoring
    baseline, monitoring = split_baseline_monitoring(cleaned, baseline_frac)
    model = fit_pca(baseline, sensor_cols, var_threshold, alpha)
    s_base = score(model, baseline)
    s_mon = score(model, monitoring)
    metrics = evaluate_against_labels(s_mon, monitoring["is_fail"].to_numpy())
    return model, s_base, s_mon, metrics


if __name__ == "__main__":
    from pathlib import Path
    from src.data_loader import select_critical_sensors

    cleaned = pd.read_parquet("data/processed/secom_clean.parquet")
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    # ---- Run A: supervised selection (|corr| with fail label) -----------
    # Standard SECOM-literature practice (e.g. UCI baseline, Salem et al.
    # 2018). Uses outcome data for *selection*, but the PCA model itself
    # is fit on pass-only baseline -- so it is not train-test leakage in
    # the traditional sense. The variance-based run below is the
    # leakage-free robustness check.
    crit_corr = select_critical_sensors(cleaned, n=25, method="fail_corr")
    model_a, s_base_a, s_mon_a, m_a = run_fdc(cleaned, crit_corr)

    # ---- Run B: unsupervised selection (top-25 variance) ----------------
    crit_var = select_critical_sensors(cleaned, n=25, method="variance")
    model_b, s_base_b, s_mon_b, m_b = run_fdc(cleaned, crit_var)

    # Persist Run A outputs (the headline run) for downstream consumers
    # (e.g. the root-cause module in src.rootcause)
    pd.DataFrame({
        "t2": s_mon_a.t2,
        "spe": s_mon_a.spe,
        "t2_alarm": s_mon_a.t2_alarm(),
        "spe_alarm": s_mon_a.spe_alarm(),
        "any_alarm": s_mon_a.any_alarm(),
    }).to_csv(out / "fdc_monitoring_scores.csv", index=False)

    def _fmt(metrics: dict, model) -> dict:
        return {
            "K": model.K,
            "var_explained": float(
                model.explained_variance_ratio[:model.K].sum()),
            "T2_limit": model.t2_limit,
            "SPE_limit": model.spe_limit,
            "TP": metrics["TP"], "FP": metrics["FP"],
            "FN": metrics["FN"], "TN": metrics["TN"],
            "TPR": metrics["TPR (recall)"],
            "FPR": metrics["FPR (false alarm)"],
            "Precision": metrics["precision"],
            "F1": metrics["F1"],
        }

    table = pd.DataFrame({
        "fail_corr (headline)": _fmt(m_a, model_a),
        "variance (robustness)": _fmt(m_b, model_b),
    }).T
    table.to_csv(out / "fdc_methodology_comparison.csv")

    print("\nFDC scorecard -- supervised vs unsupervised sensor selection")
    print("-" * 65)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out / 'fdc_methodology_comparison.csv'}")

    # Worst alarm root-cause demo on the headline run
    from src.spc import split_baseline_monitoring
    _, monitoring = split_baseline_monitoring(cleaned, 0.70)
    worst_idx = int(np.argmax(s_mon_a.t2))
    worst_wafer = monitoring.iloc[worst_idx]
    contribs_t2 = t2_contributions(model_a, worst_wafer)
    contribs_spe = spe_contributions(model_a, worst_wafer)
    print(f"\nWorst T^2 wafer (Run A): index {worst_idx}, "
          f"timestamp {worst_wafer['timestamp']}")
    print(f"  T^2 = {s_mon_a.t2[worst_idx]:.2f} "
          f"(limit {model_a.t2_limit:.2f}), "
          f"is_fail = {bool(worst_wafer['is_fail'])}")
    print("  Top 5 T^2 contributors :")
    for s, v in contribs_t2.head(5).items():
        print(f"    {s:<14} {v:+.3f}")
    print("  Top 5 SPE contributors :")
    for s, v in contribs_spe.head(5).items():
        print(f"    {s:<14} {v:.3f}")

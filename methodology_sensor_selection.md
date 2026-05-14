# Methodology note: sensor selection

The FDC model in this project monitors 25 of the 351 cleaned sensors. The
choice of *which 25* is itself a methodology decision worth documenting,
because it affects every downstream metric.

## Two reasonable strategies

| Strategy | What it ranks on | Leakage status |
|---|---|---|
| `fail_corr` | Absolute Pearson correlation between each sensor and the binary fail label, computed on the full cleaned set | Supervised: uses outcome data for *selection*. The model itself is then fit on a pass-only baseline, so this is not train-test leakage in the conventional ML sense — but it is also not fully blind. |
| `variance` | Log-variance per sensor, no labels involved | Fully unsupervised, leakage-free. |

We use **`fail_corr` as the headline run**, with `variance` reported as a
robustness check. The reasoning:

1. **Industry parallel.** A real fab process engineer chooses critical
   sensors using *historical excursion data* — i.e. outcomes — to decide
   what to monitor. A sensor that has never correlated with yield loss
   doesn't go on the SPC chart. Pretending to ignore historical outcomes
   would be more "rigorous" but less realistic.

2. **Literature precedent.** Most SECOM publications (UCI baseline,
   Salem et al. 2018, Pham et al. 2018) apply feature selection on the
   full dataset before cross-validation. The 2018 BDCC paper explicitly
   notes this is the dominant pattern in published work on this dataset.

3. **What's actually being held out.** The PCA model itself never sees a
   fail wafer during fitting — the baseline is pass-only by construction.
   So while the *sensor list* is selected with outcome awareness, the
   *anomaly model* is fully blind to the failures it is later asked to
   detect.

## Robustness check

The unsupervised `variance` run is reported alongside in
`data/processed/fdc_methodology_comparison.csv`:

|                       | K  | Var explained | TPR   | FPR   | Precision | F1    |
|----------------------|---:|--------------:|------:|------:|----------:|------:|
| fail_corr (headline)  | 12 | 0.903         | 0.212 | 0.039 | 0.564     | 0.308 |
| variance (robustness) | 18 | 0.920         | 0.144 | 0.109 | 0.238     | 0.180 |

The two selections share only **2 of 25 sensors** in common, and the
supervised run dominates on every metric. This is the expected result:
fail-correlated sensors carry the most information about product
outcomes, so monitoring them gives the FDC system more signal to work
with. The variance run picks high-energy but mostly uninformative
channels and pays for it in both recall and precision.

## What this is not

- It is not a claim that 21% recall is the best achievable on SECOM.
  Published classifiers using the full feature set, class rebalancing
  (SMOTE, ROSE), and supervised models routinely report higher F1 than
  PCA-based FDC. PCA-based FDC trades raw classification accuracy for
  *interpretability* (which sensor caused this alarm?) and *online
  applicability* (control limits are fixed before a new wafer arrives).

- It is not equivalent to fitting on the test set. The PCA loadings,
  eigenvalues, control limits, and baseline statistics are all derived
  from pass-only baseline data only.

- It is not a substitute for proper time-rolling validation, which would
  refit the model periodically as new data arrives. Phase 5's APC
  controller demo shows the closed-loop version of that idea.

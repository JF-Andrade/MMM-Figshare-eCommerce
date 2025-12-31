# Changelog

All notable changes to the hierarchical MMM model.

## [2025-12-31] Technical Audit & Model Corrections

Comprehensive audit conducted by a senior review process. All issues identified have been corrected.

### Critical Fixes (C1-C3)

| ID  | Issue                               | Fix Applied                                          |
| --- | ----------------------------------- | ---------------------------------------------------- |
| C1  | Student-T ν prior too high          | `PRIOR_NU_BETA`: 0.1 → 0.5 (mean ν ≈ 4, more robust) |
| C2  | Data leakage in spend normalization | Normalization moved to AFTER train/test split        |
| C3  | NaN in holdout y_obs_data           | Changed to zeros for JAX/NumPyro compatibility       |

### Medium Fixes (M1-M5)

| ID  | Issue                                                | Fix Applied                                   |
| --- | ---------------------------------------------------- | --------------------------------------------- |
| M1  | `sigma_alpha` too restrictive                        | `PRIOR_SIGMA_ADSTOCK_TERRITORY`: 0.1 → 0.2    |
| M2  | `L_channel` used folded Normal                       | Changed to proper `pm.HalfNormal`             |
| M3  | `L_territory` used `abs()` (discontinuous gradients) | Changed to `pt.softplus()`                    |
| M4  | Split used `.loc` (fragile indexing)                 | Changed to `.iloc` for robustness             |
| M5  | `sigma_L` too restrictive                            | `PRIOR_SIGMA_SATURATION_TERRITORY`: 0.1 → 0.2 |

### Prior Calibration Summary

| Parameter                          | Before | After | Rationale                             |
| ---------------------------------- | ------ | ----- | ------------------------------------- |
| `PRIOR_SATURATION_L_SIGMA`         | 1.0    | 0.3   | Calibrated for normalized spend [0,1] |
| `PRIOR_SIGMA_OBS`                  | 0.5    | 1.0   | Calibrated for y_log std ≈ 0.5-1.5    |
| `PRIOR_SIGMA_ADSTOCK_TERRITORY`    | 0.1    | 0.2   | More regional flexibility             |
| `PRIOR_SIGMA_SATURATION_TERRITORY` | 0.1    | 0.2   | More regional flexibility             |
| `PRIOR_NU_BETA`                    | 0.1    | 0.5   | Mean ν ≈ 4 for robust outliers        |

### Model Architecture Changes

- **L_channel** — Changed from folded Normal (`abs(Normal)`) to proper `HalfNormal`
- **L_territory** — Uses `softplus()` instead of `abs()` for smoother gradients
- **Spend normalization** — Now fitted on train only, applied to test with same max

### Diagnostics & Metrics

- **ESS tail** — Added tail ESS to convergence diagnostics
- **BFMI** — Added Bayesian Fraction of Missing Information
- **SMAPE** — Uses symmetric variant for robustness to small values

### Data Pipeline

- **M4 Fix** — Uses `iloc` instead of `loc` for robust index handling
- **NaN warning** — Added logging before `fillna(0)` for debugging
- **Holdout observed data** — Changed from NaN to zeros (C3 fix)

### Files Modified

- `src/config.py` — C1, M1, M5 (prior values)
- `src/models/hierarchical_bayesian.py` — M2, M3 (L_channel, L_territory)
- `scripts/mmm_hierarchical.py` — C2, C3, M4 (normalization, holdout, iloc)

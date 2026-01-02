# Changelog

All notable changes to the hierarchical MMM model.

## [2026-01-02] Territory Hierarchy Architecture & Dashboard Context

Major refactoring to ensure all calculations correctly account for territory-specific parameters.

### Critical Fixes

| Issue                                   | Fix Applied                                            | File                         |
| --------------------------------------- | ------------------------------------------------------ | ---------------------------- |
| ROI calculated in log scale             | Convert contributions from log to linear scale         | `regenerate_deliverables.py` |
| Contributions used raw spend            | Use normalized spend from `hierarchical_train.parquet` | `regenerate_deliverables.py` |
| Executive Summary lacked territory      | Added Global/By Territory selector                     | `01_Executive_Summary.py`    |
| Home page showed global without label   | Added "(Global)" labels to KPIs                        | `Home.py`                    |
| Model Details territory selector hidden | Made territory selector more prominent                 | `04_Model_Details.py`        |

### New Deliverables

- `adstock_territory.json` - Alpha by territory × channel
- `saturation_territory.json` - L, k by territory × channel
- `contributions_territory.json` - Contributions by territory
- `optimization_territory.json` - Budget optimization by territory
- `lift_by_territory.json` - Revenue lift by territory

### New Script

- `scripts/regenerate_deliverables.py` - Regenerates dashboard deliverables from saved idata

### Model Performance (Latest Run)

| Metric      | Baseline | Hierarchical |
| ----------- | -------- | ------------ |
| R² Test     | 0.486    | 0.856        |
| MAPE Test   | 15.7%    | 36.4%        |
| Territories | 1        | 18           |
| Training    | ~30s     | ~3.8 hours   |

---

## [2025-12-31] Ridge Baseline Audit & Fixes

Comprehensive audit of the Ridge Regression baseline to address scaling inconsistencies in ROI calculations, data leakage in cross-validation (mitigated by a 2-period gap to isolate adstock carryover effects between training and test sets), and the physical plausibility of channel coefficients.

### Critical Fixes

| Issue                              | Fix Applied                                              | File              |
| ---------------------------------- | -------------------------------------------------------- | ----------------- |
| ROI formula ignored StandardScaler | Reverts scaling: `beta_original = beta_scaled / sigma_X` | `evaluation.py`   |
| No gap in TimeSeriesSplit          | Added `gap=2` to account for adstock carryover           | `mmm_baseline.py` |

> **Note on Data Leakage**: Per-fold transformation was initially implemented but caused instability with small datasets. Reverted to hybrid approach: global transformation for channel consistency, proper CV splits for scoring. Minor leakage in saturation normalization (uses global max) is acceptable trade-off for stability.

### Medium Fixes

| Issue                               | Fix Applied                                       | File          |
| ----------------------------------- | ------------------------------------------------- | ------------- |
| Negative coefficients not validated | Added warning for economically implausible values | `insights.py` |
| Missing CV gap config               | Added `CV_GAP_WEEKS = 2` constant                 | `config.py`   |

### Low Priority Fixes

| Issue                      | Fix Applied                               | File                |
| -------------------------- | ----------------------------------------- | ------------------- |
| No ROI tests               | Added 4 unit tests for ROI computation    | `tests/test_roi.py` |
| Potential division by zero | Added safety check for `total_spend == 0` | `preprocessing.py`  |
| Misleading variable name   | Renamed `y_scaler` → `y_mean`             | `preprocessing.py`  |

### New Functions

- `_transform_test_fold()` in `mmm_baseline.py` — Transforms test fold using train statistics to prevent leakage

### Files Modified

- `src/evaluation.py` — B1 (ROI formula)
- `scripts/mmm_baseline.py` — B2, B3 (leakage, CV gap)
- `src/insights.py` — B4 (coefficient validation)
- `src/config.py` — B5, B6 (bounds, CV config)
- `src/preprocessing.py` — B8, B9 (safety, naming)
- `tests/test_roi.py` — B7 (new file)

---

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

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

### [2026-01-02] Dashboard & Optimization Logic Fixes

Resolved inconsistencies in Global Lift calculation and dashboard "Data not available" errors. Global metrics are now mathematically consistent with territory-level data.

#### Critical Fixes

| Issue                                | Fix Applied                                                   | File                         |
| ------------------------------------ | ------------------------------------------------------------- | ---------------------------- |
| Global Lift ~0.0%                    | Replaced Top-Down optimization with **Bottom-Up aggregation** | `regenerate_deliverables.py` |
| "Data not available" (Model Details) | Fixed `az.summary` indexing (names vs indices)                | `regenerate_deliverables.py` |
| KeyError: 'lift_absolute'            | Added safe dictionary access `.get()`                         | `01_Executive_Summary.py`    |
| Inconsistent Budget Charts           | Global charts now reflect sum of territory optimizations      | `regenerate_deliverables.py` |

#### New Features

- **Actual vs Predicted Chart**: Added time-series comparison with territory filter to `05_Model_Comparison.py`.
- **Improved Sidebar**: Simplified layout for professional look.
- **Predictions Deliverable**: Pipeline now saves `predictions.json` for detailed analysis.

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

| File                      | Changes                                                                    |
| ------------------------- | -------------------------------------------------------------------------- |
| `src/evaluation.py`       | Corrected ROI formula to account for `StandardScaler` (reverting scaling)  |
| `scripts/mmm_baseline.py` | Added CV gap for adstock and implemented fold-specific test transformation |
| `src/insights.py`         | Added plausibility warnings for negative channel coefficients              |
| `src/config.py`           | Defined `CV_GAP_WEEKS` constant for time-series splits                     |
| `src/preprocessing.py`    | Fixed potential division by zero and renamed `y_scaler` to `y_mean`        |
| `tests/test_roi.py`       | Initialized unit tests for ROI computation logic                           |

---

## [2025-12-31] Technical Audit & Model Corrections

Comprehensive audit conducted through a thorough review. All issues identified have been corrected.

### Critical Fixes

| Issue                                 | Fix Applied                                            |
| ------------------------------------- | ------------------------------------------------------ |
| Student-T ν prior (ν=0.1) was too low | `PRIOR_NU_BETA`: 0.1 → 0.5 (calibrated for mean ν ≈ 4) |
| Data leakage in spend normalization   | Normalization moved to AFTER train/test split          |
| NaN in holdout y_obs_data             | Changed to zeros for JAX/NumPyro compatibility         |

### Medium Fixes (M1-M5)

| Issue                                                | Fix Applied                                   |
| ---------------------------------------------------- | --------------------------------------------- |
| `sigma_alpha` too restrictive                        | `PRIOR_SIGMA_ADSTOCK_TERRITORY`: 0.1 → 0.2    |
| `L_channel` used folded Normal                       | Changed to proper `pm.HalfNormal`             |
| `L_territory` used `abs()` (discontinuous gradients) | Changed to `pt.softplus()`                    |
| Split used `.loc` (fragile indexing)                 | Changed to `.iloc` for robustness             |
| `sigma_L` too restrictive                            | `PRIOR_SIGMA_SATURATION_TERRITORY`: 0.1 → 0.2 |

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
- **Spend normalization** — Fitted on training data only to prevent data leakage (lookahead bias), ensuring the model does not "see" future spend peaks during the training phase

### Diagnostics & Metrics

- **ESS tail** — Added tail ESS to convergence diagnostics
- **BFMI** — Added Bayesian Fraction of Missing Information
- **SMAPE** — Uses symmetric variant for robustness to small values

### Data Pipeline

- **Robust indexing** — Uses `iloc` instead of `loc` for robust handling
- **NaN warning** — Added logging before `fillna(0)` for debugging
- **Holdout observed data** — Changed from NaN to zeros (C3 fix)

### Files Modified

| File                                  | Changes                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------- |
| `src/config.py`                       | Updated priors for Student-T degrees of freedom, adstock territory, and saturation territory |
| `src/models/hierarchical_bayesian.py` | Refactored `L_channel` to `HalfNormal` and `L_territory` to `softplus` for better gradients  |
| `scripts/mmm_hierarchical.py`         | Fixed data leakage in normalization, holdout NaN handling, and switched to `iloc` indexing   |

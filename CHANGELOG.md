# Changelog

All notable changes to the MMM project.

## [2026-04-06] Pipeline Stabilization & Git Cleanup

Critical fixes for runtime failures during holdout evaluation and budget optimization.

### Stability & Robustness

- **Adstock Dimensions**: Fixed a `ValueError` in `geometric_adstock_pytensor` by explicitly slicing lag tensors to `n_obs`, ensuring compatibility with short holdout sequences (where `n_obs < l_max`).
- **Budget Optimizer**: Implemented a "Zero-Spend Defense" in `optimize_hierarchical_budget` to filter out channels with no historical spend, preventing numerical instability and solver crashes.
- **Metadata Extraction**: Enhanced `generate_all_deliverables` to robustly handle `max_spend` for zero-volume channels, ensuring metadata integrity for the dashboard.

### Repository Management

- **Git Cleanup**: Removed the `specs/` folder from the remote repository history and updated `.gitignore` to prevent future tracking of private specification files.
- **Testing**: Added `tests/test_adstock_dimensions.py` to validate adstock logic for variable-length sequences.

## [2026-04-06] Hierarchical MMM Cleanup

Major technical debt reduction by pruning orphaned legacy code and streamlining math helpers.

### Pruning & System Cleanup

Comprehensive cleanup of the codebase, removing technical debt and stabilizing the hierarchical pipeline for production release.

- **Pruning**: Removed 9 legacy `log_*` functions in `src/insights.py`, deleted orphaned `src/schemas.py` and `src/utils/pymc_marketing_helpers.py`, and removed redundant `create_pipeline` in `src/pipeline.py`.
- **Regression Fix**: Restored missing `create_hierarchy_indices` export in `src/preprocessing.py`, resolving a critical `ImportError` in the hierarchical model.
- **Documentation**: Updated `README.md`, `reading_roadmap.md`, and `CHANGELOG.md` to reflect the current, lean project structure.
- **Linting**: Applied global `ruff` rules to eliminate unused imports and variables across all core modules.
- **Verification**: Confirmed 100% test pass rate (13/13) and verified functional stability of the Streamlit dashboard.

## [2026-04-04] Math Correction: Log-to-Linear & Jensen’s Inequality

Major mathematical overhaul to resolve flaws in how channel contributions and budget optimization were computed in linear space.

### 📐 Mathematical Innovation
- **Counterfactual Marginals**: Moved from naive scaling to exact marginal contributions: `E[exp(μ_full)] - E[exp(μ_full - β_c·Sat_c)]`. This respects the multiplicative nature of the log-linear model and addresses **Jensen's Inequality** bias.
- **Non-Additive Model**: Explicitly acknowledged that contributions in a multiplicative model do not sum to total revenue; the difference is now correctly categorized as a **Base/Synergy** effect.
- **True Objective Optimizer**: The budget optimizer now uses the full multiplicative objective function `Σ_t exp(base_t + Σ_c β_eff_c · Sat_c(x_c))` across every observation, leading to more accurate allocation.

### 🛠️ Refactorings
- **Math Standardization**: Moved internal consistency helpers into the test suite (`tests/test_contribution_math.py`).
- **Clean Deliverables**: Removed `scale_factor` from all JSON outputs; all metrics are now natively in linear dollars.

## [2026-01-21] Zero Lift Fix & Optimization Robustness

Critical fix for the "0% Revenue Lift" issue in budget optimization, achieved by adjusting saturation priors and improving optimizer stability.

### Critical Fixes

| Issue                  | Fix Applied                                                               | File              |
| ---------------------- | ------------------------------------------------------------------------- | ----------------- |
| Zero Projected Lift    | Increased `PRIOR_SATURATION_L_SIGMA` (0.3 -> 0.7) to fix early saturation | `src/config.py`   |
| Optimizer Instability  | Added robustness to ignore channels with negative contributions           | `src/insights.py` |
| SLSQP Failure (Mode 8) | Modified optimizer to accept partial results that improve objective       | `src/insights.py` |

## [2026-01-19] Scale Consistency & Pipeline Decoupling

Critical fix for scale mismatch between training and deliverables, and architectural decoupling of model training from dashboard artifacts.

### Scale Consistency (Lift Fix)

| Change                   | Description                                                             | File                  |
| ------------------------ | ----------------------------------------------------------------------- | --------------------- |
| `spend_max_by_channel`   | Now saved during training to ensure consistent normalization            | `mmm_hierarchical.py` |
| Consistent Normalization | Deliverables now use training-stage max-spend instead of global raw max | `deliverables.py`     |
| Lift 0% Resolution       | Fixed scale mismatch in budget optimizer that caused 0% projected lift  | `insights.py`         |

### Architectural Decoupling

| Change                 | Description                                                               | File                              |
| ---------------------- | ------------------------------------------------------------------------- | --------------------------------- |
| Training Autonomy      | Removed internal call to `generate_all_deliverables` from training        | `mmm_hierarchical.py`             |
| Pipeline Orchestration | Deliverables are now exclusively generated by the pipeline stage          | `pipeline.py` / `run_pipeline.py` |
| Clean Artifacts        | Training now only saves core artifacts (`.nc`, `.pkl`) for downstream use | `mmm_hierarchical.py`             |

## [2026-01-18] Pipeline Refactoring & Deliverables Fix

Major refactoring to separate deliverables generation from model training, enabling quick iteration without re-running MCMC.

### Pipeline Architecture

| Change                            | Description                                                        | File                          |
| --------------------------------- | ------------------------------------------------------------------ | ----------------------------- |
| New `GENERATE_DELIVERABLES` stage | Separate stage for deliverables generation                         | `src/pipeline.py`             |
| `--deliverables-only` flag        | Run only deliverables generation                                   | `scripts/run_pipeline.py`     |
| Artifact saving                   | Saves `model_data.pkl`, `regions.pkl`, `idata.nc` for regeneration | `scripts/mmm_hierarchical.py` |

**New Module:**

- `src/deliverables.py` - Contains `generate_all_deliverables()` function with all dashboard deliverables logic

### Critical Fix: Marginal ROAS

| Issue                        | Fix Applied                                           | File              |
| ---------------------------- | ----------------------------------------------------- | ----------------- |
| Incorrect normalized spend   | Changed from `multiplier` to `x_current × multiplier` | `src/insights.py` |
| Duplicate `k` multiplication | Removed extra `k` factor from formula                 | `src/insights.py` |
| Missing scale conversion     | Added `1/max_spend` factor for raw spend scale        | `src/insights.py` |

### Documentation Updates

| Change                           | Description                                                                       | File                                |
| -------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------- |
| Known Limitations (README)       | Added notes on log-to-linear conversion, optimizer heuristic, What-If limitations | `README.md`                         |
| Known Limitations (Math)         | Added detailed LaTeX explanations                                                 | `docs/model_mathematics.md`         |
| What-If Simulator warning        | Added info box about adstock limitations                                          | `app/pages/03_What_If_Simulator.py` |
| Removed "Potential Improvements" | Section removed from README                                                       | `README.md`                         |

### New Tests

- `tests/test_marginal_roas.py` - Unit tests for marginal ROAS calculation (diminishing returns, normalization, non-negativity)

---

## [2026-01-05] Dashboard Reorganization & New Features

Major dashboard restructuring: consolidated 10 pages into 5 focused pages with clear user journey.

### Dashboard Reorganization

| Before                           | After                         | Change       |
| -------------------------------- | ----------------------------- | ------------ |
| 10 pages with redundant content  | 5 focused pages               | Consolidated |
| Territory selector on each page  | Global selector in sidebar    | Centralized  |
| Mixed business/technical content | Audience-appropriate grouping | Reorganized  |

**New Page Structure:**

- `Home.py` - Navigation hub with section cards
- `01_Performance_Analysis.py` - Channel performance, regional insights, alerts
- `02_Budget_Optimization.py` - Optimal allocation recommendations
- `03_What_If_Simulator.py` - Interactive budget simulation
- `04_Technical_Details.py` - Parameters, comparison, diagnostics (tabbed)
- `05_Historical_Tracking.py` - ROI trends and benchmarks over time

### New Features

| Feature               | Description                        | File                        |
| --------------------- | ---------------------------------- | --------------------------- |
| Saturation Alerts     | Highlight channels >80% saturated  | `alerts.py`                 |
| ROI Anomaly Detection | Flag territories with outlier ROI  | `alerts.py`                 |
| Excel Export          | Multi-sheet report download        | `export.py`                 |
| What-If Simulator     | Hill function budget simulation    | `03_What_If_Simulator.py`   |
| ROI Trends            | Cross-run ROI time series          | `05_Historical_Tracking.py` |
| Historical Benchmarks | Compare current vs 6-month average | `05_Historical_Tracking.py` |

### Files Deleted

- `01_Executive_Summary.py` (merged into Performance Analysis)
- `03_Regional_Analysis.py` (merged into Performance Analysis)
- `04_Model_Details.py` (merged into Technical Details)
- `05_Model_Comparison.py` (merged into Technical Details)
- `06_Channel_Efficiency.py` (merged into Performance Analysis)
- `08_ROI_Trends.py` (merged into Historical Tracking)
- `09_Historical_Benchmarks.py` (merged into Historical Tracking)

---

## [2026-01-04] Territory Data Scale Fix & Dashboard Enhancements

Critical fix for territory-level contributions and optimization data.

### Critical Fixes

| Issue                                | Fix Applied                                            | File                  |
| ------------------------------------ | ------------------------------------------------------ | --------------------- |
| Territory contributions in log scale | Applied `scale_factor` for log→linear conversion       | `mmm_hierarchical.py` |
| Territory spend normalized (0-1)     | Replaced with raw spend from `{CHANNEL}_SPEND` columns | `mmm_hierarchical.py` |
| Territory ROI incorrect              | Recalculated using scaled contributions and raw spend  | `mmm_hierarchical.py` |

### New Features

| Feature                | Description                                        | File                  |
| ---------------------- | -------------------------------------------------- | --------------------- |
| Marginal ROAS Analysis | Compute incremental ROAS at different spend levels | `src/insights.py`     |
| Max Spend Reference    | Added `max_spend` column to saturation parameters  | `mmm_hierarchical.py` |

### Dashboard Improvements

| Change                  | Description                                              | File                  |
| ----------------------- | -------------------------------------------------------- | --------------------- |
| Global Territory Filter | Single selector applies to all sections in Model Details | `04_Model_Details.py` |
| Predictions Aggregation | Fixed timestamp error when "All Territories" selected    | `charts.py`           |
| Saturation L-points     | Added vertical lines at half-saturation points           | `charts.py`           |

---

## [2026-01-04] Structural Refactoring

Separate concerns between low-level math and pipeline orchestration.

### Refactoring

| Change                   | Description                                                    | File                     |
| ------------------------ | -------------------------------------------------------------- | ------------------------ |
| Create Math Layer        | Extracted `adstock`, `saturation`, `scaling` to pure functions | `src/transformations.py` |
| Orchestration Layer      | `preprocessing.py` now orchestrates data flow and validation   | `src/preprocessing.py`   |
| Remove Validation Module | Absorbed `validation.py` into `preprocessing.py` for cohesion  | `src/validation.py`      |

---

## [2026-01-03] Preprocessing & Validation Refactoring

Major code quality improvements in preprocessing, validation, and baseline model.

### Refactoring

| Change                            | Description                                                           | File               |
| --------------------------------- | --------------------------------------------------------------------- | ------------------ |
| Split `compute_temporal_features` | Created `add_seasonality_features` and `add_event_features` for SRP   | `preprocessing.py` |
| Consolidated feature loops        | Merged 3 redundant loops into 1 in `prepare_baseline_features`        | `preprocessing.py` |
| Centralized validation logic      | Moved `transform_test_fold` from `mmm_baseline.py` to `validation.py` | `validation.py`    |
| Module-level imports              | Moved lazy imports to top-level (PEP 8)                               | `validation.py`    |

### Baseline Performance (Post-Fix)

- **R² Train**: 0.728
- **R² Test**: 0.227
- **MAPE**: 19.4%

---

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

### Added

- **Metric Upgrade**: Integrated **CAC** (Customer Acquisition Cost) and **iROAS** (Incremental Return on Ad Spend) into the core pipeline and dashboard.
- **New Dashboard Page**: `06_Channel_Efficiency.py` for detailed efficiency analysis (CAC x iROAS matrix).
- **Configuration**: Centralized `DEFAULT_CURRENCY` in `src/config.py`.

### Changed

- **Terminology**: Standardized dashboard labels from "ROI" to "**iROAS**" to accurately reflect incremental return logic.
- **Docs**: Updated `README.md` and `docs/` to reflect metric definitions and data loading logic.
- `LICENSE` file (MIT).

### Files Modified

| File                                  | Changes                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------- |
| `src/config.py`                       | Updated priors for Student-T degrees of freedom, adstock territory, and saturation territory |
| `src/models/hierarchical_bayesian.py` | Refactored `L_channel` to `HalfNormal` and `L_territory` to `softplus` for better gradients  |
| `scripts/mmm_hierarchical.py`         | Fixed data leakage in normalization, holdout NaN handling, and switched to `iloc` indexing   |

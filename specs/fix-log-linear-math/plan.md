# Technical Plan: Fix Log-to-Linear Mathematical Flaws

## Spec Reference
Implements: `specs/fix-log-linear-math/spec.md`

## Architecture Overview

The fix propagates bottom-up through four layers: (1) the model's contribution computation produces linear-dollar marginals directly from the exponentiated posterior, (2) the deliverables layer stops applying `scale_factor`, (3) the insights layer (MROAS + optimizer) uses posterior betas and a multiplicative objective, and (4) the simulator consumes pre-saved model internals. Each layer depends only on the one below it.

## Component Breakdown

### Component 1: Linear Contribution Engine
- **Responsibility:** Compute counterfactual marginal contributions in linear dollars using `exp(μ_full) - exp(μ_full - β_eff_c · sat_c)`
- **Location:** `src/models/hierarchical_bayesian.py`
- **Functions modified:** `compute_channel_contributions()`, `compute_channel_contributions_by_territory()`
- **New helper:** `_compute_base_effects()` — extracts intercept + features + seasonality per observation from idata
- **Accepts:** `idata`, `X_spend`, `territory_idx`, `channel_names`, + new: `X_features`, `X_season`
- **Returns:** DataFrame with `contribution` (linear $), `contribution_log` (diagnostic), `roi`, `beta_mean`, `synergy_base` row
- **AC Coverage:** AC-1, AC-2, AC-6, AC-E1, AC-E3

### Component 2: Deliverables Adapter
- **Responsibility:** Remove `scale_factor` application, pass through linear-dollar contributions, save model internals
- **Location:** `src/deliverables.py`
- **Functions modified:** `_compute_contributions()`, `_compute_regional_metrics()`, `generate_all_deliverables()`
- **Accepts:** Same inputs as current, but passes additional model data to Component 1
- **Returns:** Same deliverables dict structure, but with mathematically correct values + new `model_internals` key
- **AC Coverage:** AC-3, AC-7, AC-10

### Component 3: Posterior-Based Marginal ROAS
- **Responsibility:** Compute MROAS using posterior `beta_channel` directly, not reverse-engineered from contributions
- **Location:** `src/insights.py :: compute_marginal_roas()`
- **Accepts:** `contributions_df`, `saturation_params`, `n_obs`, + new: `beta_channel` (array of posterior mean betas)
- **Returns:** Same `list[dict]` structure with corrected `marginal_roas` values
- **AC Coverage:** AC-4

### Component 4: Multiplicative Budget Optimizer
- **Responsibility:** Replace additive objective `Σ scale·Sat(x)` with multiplicative `Σ_t exp(base_t + Σ_c β·Sat_c(x_c))`
- **Location:** `src/insights.py :: optimize_hierarchical_budget()`, `optimize_budget_by_territory()`
- **Accepts:** New: `base_array` (n_obs,), `beta_eff_matrix` (n_obs, n_channels) instead of using contrib_df to derive `scale`
- **Returns:** Same dict structure with corrected optimal allocations and lift
- **AC Coverage:** AC-5, AC-E2

### Component 5: Multiplicative What-If Simulator
- **Responsibility:** Use saved model internals for multiplicative revenue projection
- **Location:** `app/pages/03_What_If_Simulator.py :: simulate_budget()`
- **Accepts:** `spend_dict`, `saturation_params`, `model_internals` (loaded from deliverable)
- **Returns:** Same dict structure with corrected projections
- **AC Coverage:** AC-8

### Component 6: Dead Code Deprecation
- **Responsibility:** Mark flawed scaling functions as deprecated
- **Location:** `src/insights.py`
- **Functions:** `compute_scaling_factor`, `log_global_contributions`, `log_roi_with_hdi`, `log_regional_metrics`, `compute_marginal_roas_custom`, `compute_marginal_roas_by_territory`
- **AC Coverage:** AC-9

### Component 7: Test Suite
- **Responsibility:** Validate mathematical correctness with synthetic known-coefficient tests
- **Location:** `tests/test_contribution_math.py` (new), `tests/test_marginal_roas.py`, `tests/test_optimization.py`
- **AC Coverage:** AC-10, AC-1 (synthetic), AC-2 (synthetic)

## Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Contribution method | Counterfactual marginal | Exact in multiplicative model, O(C) passes |
| Non-additivity | Raw marginals + synergy row | Mathematically honest, no distortion |
| μ reconstruction | Internal from idata | Keeps function API simple |
| Model internals storage | JSON deliverable | Streamlit app loads JSON already |
| Optimizer objective | `np.sum(np.exp(mu_array))` | Vectorized NumPy, no Python loop over obs |

## AC Coverage Map

| AC | Component(s) | Verification |
|----|-------------|--------------|
| AC-1 | Comp 1: Linear Contribution Engine | test_contribution_math.py: synthetic model |
| AC-2 | Comp 1: Linear Contribution Engine | test_contribution_math.py: sum < total |
| AC-3 | Comp 2: Deliverables Adapter | grep: no `scale_factor *` in contribution chain |
| AC-4 | Comp 3: Posterior-Based MROAS | test_marginal_roas.py: beta from posterior |
| AC-5 | Comp 4: Multiplicative Optimizer | test_optimization.py: objective uses exp |
| AC-6 | Comp 1: Linear Contribution Engine | test_contribution_math.py: territory variant |
| AC-7 | Comp 2: Deliverables Adapter | manual: check model_internals.json exists |
| AC-8 | Comp 5: What-If Simulator | manual: Streamlit slider test |
| AC-9 | Comp 6: Dead Code Deprecation | grep: DeprecationWarning in each function |
| AC-10 | Comp 7: Test Suite | pytest tests/ green |
| AC-E1 | Comp 1 | test_contribution_math.py: zero-spend case |
| AC-E2 | Comp 4 | test_optimization.py: clamped normalized spend |
| AC-E3 | Comp 1 + Comp 4 | test_optimization.py: negative beta case |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `_compute_base_effects` needs feature/season data not in m_data | Medium | High | Verify m_data keys first; add to function signature if missing |
| Contribution values change dramatically, breaking dashboard visual expectations | High | Medium | Log both old and new values for comparison; document magnitude change |
| SLSQP convergence issues with multiplicative objective (non-convex) | Medium | Medium | Keep existing convergence fallback logic (accept if objective improved) |
| JSON serialization of large `base_array` / `beta_eff_matrix` | Low | Low | Use float32 precision, compress if >1MB |

## Out of Scope (Technical)

- No modification to `build_hierarchical_model()`, `fit_model()`, or `predict()`
- No changes to adstock/saturation primitive functions (`geometric_adstock_numpy`, `hill_saturation_numpy`)
- No changes to `src/preprocessing.py` or `src/transformations.py`
- No new Streamlit pages

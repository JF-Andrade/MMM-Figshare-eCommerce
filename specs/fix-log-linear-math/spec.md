# Fix Log-to-Linear Mathematical Flaws

Status: Approved
Version: 1.0
Last updated: 2026-04-04

## Overview

The MMM codebase computes channel contributions, marginal ROAS, and budget optimization using a naive linear scaling factor that violates Jensen's Inequality. All post-model metrics reported in the Streamlit dashboard are mathematically incorrect. This feature replaces the flawed scaling approach with counterfactual marginal contributions computed directly over the exponentiated posterior.

## User Stories

### Primary
As a marketing analyst, I want channel contributions reported in correct linear dollars so that budget allocation decisions are based on mathematically valid metrics.

### Secondary
As a data scientist reviewing this portfolio project, I want the mathematical methodology to be defensible and consistent with the model's DGP (Data Generating Process) so that the work demonstrates rigorous Bayesian modeling practice.

## Boundaries

**Always do:**
- Respect Jensen's Inequality: `E[exp(X)] ≥ exp(E[X])`
- Compute contributions as counterfactual marginals: `exp(μ_full) - exp(μ_full - β_c·Sat_c)`
- Use posterior mean parameters when a point estimate is required
- Validate new outputs against known synthetic inputs before integrating
- Preserve existing function signatures where possible; extend, don't break

**Ask first (do not proceed unilaterally):**
- Adding new dependencies
- Changing the PyMC model specification itself
- Modifying the Streamlit app layout or navigation structure
- Removing test files

**Never do:**
- Apply a naive `scale_factor` to convert log-space to linear-space values
- Optimize an additive objective `Σ k·Sat(x)` when the true model is multiplicative
- Reverse-engineer β from corrupted contributions (`beta = contribution / saturation`)
- Modify the PyMC model definition in `build_hierarchical_model()`
- Break backward compatibility of deliverables JSON schema without explicit handling

## Acceptance Criteria

### AC-1: Counterfactual Channel Contributions [MUST]
Given a fitted hierarchical Bayesian model with known posterior parameters
When `compute_channel_contributions()` is called
Then each channel's contribution is computed as `Σ_t [exp(μ_full_t) - exp(μ_full_t - β_eff_c_t · sat_c_t)]` — the exact marginal contribution in linear dollars

### AC-2: Contributions Are Non-Additive [MUST]
Given the multiplicative structure of the model in linear space
When all channel contributions are summed
Then `sum(contributions) ≤ total_predicted_revenue` — with the difference attributed to a synergy/base row

### AC-3: No Scale Factor [MUST]
Given the updated contribution pipeline
When deliverables are generated
Then no `scale_factor` multiplication is applied anywhere in the contribution or ROI calculation chain

### AC-4: Marginal ROAS Uses Posterior Beta [MUST]
Given posterior mean `beta_channel` values from `idata`
When `compute_marginal_roas()` computes β for the Hill derivative
Then β is the posterior mean `beta_channel[c]`, not reverse-engineered from `contribution / Sat(x)`

### AC-5: Multiplicative Optimizer Objective [MUST]
Given the budget optimizer
When the SLSQP objective function is evaluated at spend vector x
Then it computes `Σ_t exp(base_t + Σ_c β_eff_c · Sat_c(x_c / max_spend_c))` — not `Σ_c scale_c · Sat_c(x_c)`

### AC-6: Territory Contributions [MUST]
Given territory-level model parameters
When `compute_channel_contributions_by_territory()` is called
Then contributions per territory are computed using the same counterfactual logic as AC-1, scoped to each territory's observations

### AC-7: Model Internals Saved as Deliverable [MUST]
Given a fitted model
When deliverables are generated
Then `base_array` (non-channel effects per observation) and `beta_eff_matrix` (effective betas per obs × channel) are saved to `deliverables/model_internals.json`

### AC-8: What-If Simulator Uses Multiplicative Model [SHOULD]
Given the saved model internals
When the user adjusts spend sliders in the What-If Simulator page
Then projected contributions are computed using `Σ_t exp(base_t + Σ_c β_eff_c · Sat_c(x_new_c))` — not `beta * new_response`

### AC-9: Dead Scaling Functions Deprecated [SHOULD]
Given `compute_scaling_factor`, `log_global_contributions`, `log_roi_with_hdi`, `log_regional_metrics`, `compute_marginal_roas_custom`, `compute_marginal_roas_by_territory` in `insights.py`
When the refactoring is complete
Then these functions are marked deprecated with docstring warnings

### AC-10: Existing Tests Pass [MUST]
Given the test suite under `tests/`
When `pytest tests/` is run after all changes
Then all existing tests pass (with updated assertions where needed for new contribution semantics)

### AC-E1: Zero-Spend Channel Handling [MUST]
Given a channel with zero total spend
When contributions are computed
Then contribution for that channel is 0.0 (not NaN or negative)

### AC-E2: Saturation Extrapolation Guard [MUST]
Given a new spend value that exceeds `max_spend` (the training maximum)
When the optimizer or simulator evaluates Hill saturation
Then the normalized spend is clamped to `2.0` to prevent wild extrapolation

### AC-E3: Negative Contribution Guard [MUST]
Given a channel where posterior beta_territory offset produces a negative effective beta
When contributions are computed
Then the contribution is computed correctly (may be negative) and the optimizer fixes that channel at current spend rather than attempting to optimize it

### AC-W1: Shapley Attribution [WONT]
This feature will NOT include Shapley-value attribution. Reason: computationally expensive (2^C evaluations) and not needed for a portfolio project with 5 channels. Raw marginal contributions with a synergy row are sufficient.

### AC-W2: Full Posterior Sampling for Contributions [WONT]
This feature will NOT sample over the full posterior for contribution uncertainty intervals. Reason: posterior mean parameters provide a computationally efficient point estimate. The existing `compute_roi_with_hdi()` already provides ROI intervals via sampling; contribution intervals can be added later.

## Out of Scope

- Modifying the PyMC model specification (`build_hierarchical_model`)
- Changing the adstock or saturation transformation functions
- Altering the preprocessing pipeline or data ingestion
- Adding new Streamlit pages or modifying navigation
- Performance optimization of MCMC sampling
- Adding new marketing channels or territories

## Non-Functional Requirements

- Performance: `compute_channel_contributions()` completes in < 5 seconds for 200 observations × 5 channels
- Backward compatibility: deliverables JSON keys (`contributions`, `roi`, `optimization`, etc.) retain their names; internal values change
- Numerical stability: all divisions protected by EPSILON; all exp() calls bounded

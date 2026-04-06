# Task List: Fix Log-to-Linear Mathematical Flaws

## Plan Reference
Implements: `specs/fix-log-linear-math/plan.md`

## Tasks

### Foundation (Component 1: Linear Contribution Engine)

- [x] **TASK-001** [S] Add `_compute_base_effects()` helper to `hierarchical_bayesian.py`
  - Creates: new function in `src/models/hierarchical_bayesian.py`
  - Extracts intercept + feature_effect + season_effect per observation from idata + m_data
  - Returns: `base_array` (n_obs,), `beta_eff_matrix` (n_obs, n_channels)
  - Depends on: none

- [x] **TASK-002** [M] Write tests for counterfactual contribution math
  - Creates: `tests/test_contribution_math.py`
  - Tests: AC-1 (synthetic known model), AC-2 (sum < total), AC-E1 (zero spend), AC-E3 (negative beta)
  - Depends on: TASK-001

- [x] **TASK-003** [M] Modify `compute_channel_contributions()` for counterfactual marginals
  - Modifies: `src/models/hierarchical_bayesian.py`
  - AC Coverage: AC-1, AC-2, AC-E1, AC-E3
  - Depends on: TASK-002

- [x] **TASK-004** [M] Modify `compute_channel_contributions_by_territory()` for counterfactual marginals
  - Modifies: `src/models/hierarchical_bayesian.py`
  - AC Coverage: AC-6
  - Depends on: TASK-003

### Deliverables Adapter (Component 2)

- [x] **TASK-005** [M] Modify `_compute_contributions()` — remove scale_factor
  - Modifies: `src/deliverables.py`
  - AC Coverage: AC-3
  - Depends on: TASK-003

- [x] **TASK-006** [S] Modify `_compute_regional_metrics()` — remove scale_factor
  - Modifies: `src/deliverables.py`
  - AC Coverage: AC-3
  - Depends on: TASK-003

- [x] **TASK-007** [M] Modify `generate_all_deliverables()` — remove scale_factor, save model_internals
  - Modifies: `src/deliverables.py`
  - AC Coverage: AC-3, AC-7
  - Depends on: TASK-005, TASK-006

### Marginal ROAS (Component 3)

- [x] **TASK-008** [S] Write tests for posterior-beta MROAS
  - Modifies: `tests/test_marginal_roas.py`
  - Tests: AC-4 (beta from posterior, not reverse-engineered)
  - Depends on: TASK-005

- [x] **TASK-009** [M] Modify `compute_marginal_roas()` — use posterior beta
  - Modifies: `src/insights.py`
  - AC Coverage: AC-4
  - Depends on: TASK-008

### Budget Optimizer (Component 4)

- [x] **TASK-010** [S] Write tests for multiplicative optimizer objective
  - Modifies: `tests/test_optimization.py`
  - Tests: AC-5, AC-E2 (clamped spend), AC-E3 (fixed negative-beta channel)
  - Depends on: TASK-007

- [x] **TASK-011** [L] Modify `optimize_hierarchical_budget()` — multiplicative objective
  - Modifies: `src/insights.py`
  - AC Coverage: AC-5, AC-E2
  - Depends on: TASK-010

- [x] **TASK-012** [M] Modify `optimize_budget_by_territory()` — multiplicative objective
  - Modifies: `src/insights.py`
  - AC Coverage: AC-5, AC-E2
  - Depends on: TASK-011

### What-If Simulator (Component 5)

- [x] **TASK-013** [M] Modify `simulate_budget()` — multiplicative model
  - Modifies: `app/pages/03_What_If_Simulator.py`
  - AC Coverage: AC-8
  - Depends on: TASK-007

### Cleanup (Component 6)

- [x] **TASK-014** [S] [P] Deprecate dead scaling functions in `insights.py`
  - Modifies: `src/insights.py`
  - AC Coverage: AC-9
  - Depends on: TASK-009

### Validation (Component 7)

- [x] **TASK-015** [M] Run full test suite, fix any failures
  - Runs: `pytest tests/`
  - AC Coverage: AC-10
  - Depends on: TASK-001 through TASK-014

- [x] **TASK-016** [S] Run `gitnexus_detect_changes()` — confirm blast radius matches plan
  - Depends on: TASK-015

## Legend
- `[S]` Small — under 1 hour
- `[M]` Medium — 1–3 hours
- `[L]` Large — 3–6 hours (consider splitting)
- `[P]` Parallelizable — can run concurrently with other `[P]` tasks at same level

## Dependency Graph

```bash
TASK-001 ─────────────────────────┐
    │                             │
TASK-002 (tests)                  │
    │                             │
TASK-003 ───┬─────────────────────┤
    │       │                     │
TASK-004    │                     │
    │       │                     │
TASK-005  TASK-006                │
    │       │                     │
    └───┬───┘                     │
        │                         │
    TASK-007 ──────┬──────┬───────┘
        │          │      │
    TASK-008   TASK-010  TASK-013
        │          │
    TASK-009   TASK-011
        │          │
    TASK-014   TASK-012
        │          │
        └──────┬───┘
               │
           TASK-015
               │
           TASK-016
```

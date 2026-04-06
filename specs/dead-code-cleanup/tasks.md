# Task List: Dead Code Cleanup

## Plan Reference

Implements: `specs/dead-code-cleanup/plan.md`

## Tasks

### Component 1: General Import & Variable Hygiene

- [x] **TASK-001** [S] [P] Clean up unused imports and variables in `src/`.
  - [x] Executed `ruff` fixes
  - [x] Verified via `gitnexus_detect_changes()`

- [x] **TASK-002** [M] Remove legacy MLFlow logging functions from `src/insights.py`.
  - [x] Pruned 9 orphaned wrappers
  - [x] Verified zero upstream callers

- [x] **TASK-003** [M] Prune unused Pydantic model fields in `src/schemas.py`.
  - [x] Deleted entirely as file was orphaned

- [x] **TASK-004** [S] Remove `create_pipeline` from `src/pipeline.py`.
  - [x] Removed factory function
  - [x] Verified direct instantiation in `scripts/run_pipeline.py`

- [x] **TASK-005** [S] [P] Remove orphaned helpers in `src/utils/pymc_marketing_helpers.py`.
  - [x] Deleted file and legacy tests

- [x] **TASK-006** [S] Clean up unused imports in `tests/`.
  - [x] Executed `ruff` fixes on `tests/`

- [x] **TASK-007** [M] Standardize math helpers location.
  - [x] Moved `_compute_linear_contributions` to `tests/test_contribution_math.py` (inlined)
  - [x] Removed from `src/models/hierarchical_bayesian.py`

- [x] **TASK-008** [L] Final verification and push.
  - [x] Full `pytest` run (13/13 passed)
  - [x] Streamlit app verified via browser tool
  - [x] `npx gitnexus analyze` refresh

## Legend
- `[S]` Small — under 1 hour
- `[M]` Medium — 1–3 hours
- `[L]` Large — 3–6 hours (consider splitting)
- `[P]` Parallelizable — can run concurrently with other `[P]` tasks at same level

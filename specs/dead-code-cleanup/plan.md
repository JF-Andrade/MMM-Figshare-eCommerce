# Technical Plan: Dead Code Cleanup

## Spec Reference

Implements: `specs/dead-code-cleanup/spec.md`

## Architecture Overview

The cleanup follows a "Verify-then-Prune" approach using static analysis (`Ruff`, `Vulture`) and graph analysis (`GitNexus`). We will categorize removals into low-risk (imports), medium-risk (variables/private methods), and high-risk (public methods/cross-module files).

## Component Breakdown

### [Component 1] General Import & Variable Hygiene

- **Responsibility:** Eliminate `F401` and `F841` warnings across all modules.
- **Location:** `src/`, `scripts/`, `tests/`
- **AC Coverage:** AC-1

### [Component 2] Domain-Specific Pruning: `src/insights.py`

- **Responsibility:** Remove legacy MLFlow logging helpers that were superseded by the deliverables system.
- **Location:** `src/insights.py`
- **AC Coverage:** AC-2

### [Component 3] Schema Pruning: `src/schemas.py`

- **Responsibility:** Remove unused Pydantic model attributes that clutter the JSON serialization of results.
- **Location:** `src/schemas.py`
- **AC Coverage:** AC-3

### [Component 4] Orphaned Helpers Cleanup

- **Responsibility:** Prune `extract_adstock_params` and `extract_saturation_params` that are not referenced in the hierarchical model flow.
- **Location:** `src/utils/pymc_marketing_helpers.py`, `src/preprocessing.py`, `src/transformations.py`
- **AC Coverage:** AC-2

## Technology Choices

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Linting | Ruff | Fastest and standard in this project's constitution. |
| Dead Code | Vulture | Reliable for finding unused attributes and functions. |
| Graph Analysis | GitNexus | Validates reachability across complex module imports. |
| Impact Analysis | GitNexus MCP | REQUIRED for checking blast radius before deleting symbols. |

## AC Coverage Map

| AC | Component(s) | Verification Strategy |
| :--- | :--- | :--- |
| AC-1 | Component 1 | `ruff check .` |
| AC-2 | Component 2, 4 | `vulture` + `gitnexus_impact` |
| AC-3 | Component 3 | `vulture` + `gitnexus_context` |
| AC-E1 | All | `pytest tests/` + `gitnexus_detect_changes` |

## Risks

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Deleting a "reflection-based" or "dynamic" call | Low | Medium | Tests cover core flows; GitNexus graph identifies indirect refs. |
| Removing a test dependency incorrectly | Medium | Low | `pytest` will catch it immediately. |
| Deleting a field required by the frontend/app | Medium | High | Manual verification of the Streamlit dashboard after pruning `schemas.py`. |

## Out of Scope (Technical)

- Modifying `app/` components unless they rely on a deleted schema property (then they will be updated or pruned).
- Modifying `.yml` or `.toml` configs unless a dependency needs removal.

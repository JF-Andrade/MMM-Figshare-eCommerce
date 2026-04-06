# Dead Code Cleanup: Specification

Status: Draft
Version: 1.0
Last updated: 2026-04-05

## Overview

Systematically identify and remove unreferenced functions, classes, variables, and redundant imports to improve codebase maintainability and reduce technical debt. All removals must be verified against the existing test suite.

## User Stories

### Primary

As a developer, I want a lean and well-documented codebase so that I can understand and extend it without being distracted by orphaned logic or unused helpers.

## Boundaries

**Always do:**

- Verify all removals using `ruff` (unused imports/vars) and `vulture` (unreferenced symbols).
- MUST run `gitnexus_impact({target: "symbolName", direction: "upstream"})` before deleting any non-private function, class, or method.
- MUST run `gitnexus_detect_changes()` after each batch of deletions to ensure zero unintended impact.
- Cross-reference with the GitNexus call graph to catch "orphaned" but technically "reachable" code.
- Run the full `pytest` suite after every minor batch of removals.
- Keep mathematical and research functions if they are explicitly mentioned in `docs/` or required for `tests/`.

**Ask first (do not proceed unilaterally):**

- Removing top-level script files (e.g., `github_version.py`).
- Removing functions that are currently unused but look like "near-future" features or part of a public API contract (though this repo seems private).

**Never do:**

- Remove logic that causes `pytest` to fail.
- Remove documentation-only files (`.md`) unless they are explicitly marked as "TO DELETE".
- Modify the mathematical core (PyMC model logic) unless it's strictly an unused helper.

## Acceptance Criteria

### AC-1: Zero Redundant Imports [MUST]

Given the entire codebase
When `ruff check . --select F401` is run
Then no unused import warnings should remain.

### AC-2: Zero Unreferenced Private Helpers [MUST]

Given internal modules (`src/insights.py`, `src/pipeline.py`, etc.)
When `vulture` is run with a 60% confidence threshold (excluding known entry points)
Then all verified "dead" functions and variables should be removed.

### AC-3: Pruned Schemas [MUST]

Given `src/schemas.py`
When a field is identified as unused in both production logic and tests
Then that field must be removed from the Pydantic model.

### AC-4: Orphaned Scripts Audit [SHOULD]

Given root-level scripts
When a script is not referenced in `README.md`, `constitution.md`, or CI/CD
Then it should be proposed for deletion.

### AC-E1: Test Suite Protection [MUST]

Given any cleanup action
When the action is completed
Then `pytest tests/` must still return 100% success.

## Out of Scope

- Refactoring working logic for performance (only removal of dead code).
- Updating dependencies (unless version conflict arises).
- Formatting/Linting outside of `F401` and `F841`.

## Open Questions

- [NEEDS CLARIFICATION] Is `github_version.py` used by any external GitHub Action?
- [NEEDS CLARIFICATION] Should test-only helpers (`_compute_linear_contributions`) stay in `src/`?

---

## AI Assumptions Surface

1. **Hierarchy Entry Points**: I assume `app/Home.py`, `src/pipeline.py`, and `scripts/run_pipeline.py` are the only production entry points. Anything unreachable from these is "dead".
2. **Test Instrumentation**: I assume that if a function is *only* called by a test, it's NOT dead but "test instrumentation", and should be kept or moved to `tests/`.
3. **Pydantic Schemas**: I assume all fields in `src/schemas.py` are intended for use. If Vulture flags them as "unused attribute", I will check if they are expected in an API response before deleting.
4. **Environment**: I assume `ruff` and `vulture` are the primary tools for validation.

# Project Constitution

Version: 1.0.0
Last updated: 2026-04-04

## Architecture Principles

- **Model-first:** The PyMC Bayesian posterior is the single source of truth for all marketing metrics. No downstream calculation may contradict the model's mathematical structure.
- **Log-space discipline:** The model is additive in log-space and multiplicative in linear space. All post-modeling transformations must respect this duality.
- **Separation of concerns:** Model fitting (`src/models/`), post-model analysis (`src/insights.py`), deliverable generation (`src/deliverables.py`), and dashboard presentation (`app/`) are distinct layers. Each layer consumes the outputs of the previous; none may bypass the chain.
- **Pipeline reproducibility:** All deliverables are generated from a single entry point (`scripts/run_pipeline.py` or `scripts/generate_deliverables.py`) using MLflow-tracked artifacts.

## Technology Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.11+ | Type hints required on all public functions |
| Probabilistic framework | PyMC 5.x | Hierarchical Bayesian MMM |
| Inference | NUTS (via PyMC or NumPyro backend) | Student-T likelihood |
| Numerical | NumPy, SciPy | Vectorized operations preferred over Python loops |
| Data | Pandas | DataFrames for tabular results |
| Experiment tracking | MLflow | All deliverables logged as artifacts |
| Dashboard | Streamlit | Multi-page app under `app/` |
| Visualization | Plotly, Matplotlib | Plotly for interactive, Matplotlib for static |
| Testing | pytest | Tests under `tests/` |
| Code quality | ruff | Formatting and linting |

## Mathematical Constraints

- **Jensen's Inequality must be respected.** For any convex function f: E[f(X)] ≥ f(E[X]). Never approximate E[exp(X)] as C·E[X].
- **Counterfactual marginal contributions.** Channel contribution in linear dollars is the difference in expected revenue with vs. without that channel, computed over the exponentiated posterior.
- **No naive log-to-linear scaling.** Never compute a global `scale_factor = total_revenue / (mean_log * n_obs)` and multiply log-space components by it.
- **Additive in log = multiplicative in linear.** Any optimization objective or simulation must model revenue as `exp(base + Σ β·Sat(x))`, not `Σ k·Sat(x)`.
- **Posterior mean parameters for point estimates.** When a single set of parameters is needed (not full posterior), use the posterior mean averaged over chains and draws.

## Naming Conventions

- Files: `snake_case.py`
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE` (in `src/config.py`)
- Test files: `test_<module>.py`

## Banned Patterns

- No `scale_factor` multiplication of log-space contributions to produce linear-space values
- No `contribution * scaling_factor` anywhere in the codebase
- No `print()` for debugging in committed code (use proper logging or structured output)
- No hardcoded epsilon values scattered through code — use `EPSILON` from `src/config.py`

## File Structure Rules

```
src/                  # Core library
  models/             # PyMC model definition, fitting, prediction
  config.py           # All hyperparameters and constants
  transformations.py  # Adstock, saturation, log transforms
  preprocessing.py    # Data preparation
  insights.py         # Post-model analysis (MROAS, optimization, metrics)
  deliverables.py     # Deliverable generation orchestration
  pipeline.py         # Pipeline step execution
  schemas.py          # Pydantic schemas
app/                  # Streamlit dashboard
  pages/              # Multi-page app pages
  components/         # Reusable UI components
scripts/              # Entry-point scripts
tests/                # pytest test suite
docs/                 # Documentation and proofs
specs/                # SDD specifications
```

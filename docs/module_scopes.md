# MMM Project Module Scopes

## src/ Modules

| Module                            | Scope / Purpose                                           |
| --------------------------------- | --------------------------------------------------------- |
| `config.py`                       | Constants, hyperparameters, PipelineConfig dataclass      |
| `data_loader.py`                  | Load raw data, currency filtering, data validation        |
| `preprocessing.py`                | **Orchestrator**: Pipelines, splits, validation           |
| `transformations.py`              | **Core Math**: Adstock, Saturation, Scaling (Low-level)   |
| `evaluation.py`                   | Metrics calculation, convergence diagnostics, ROI         |
| `insights.py`                     | Parameter extraction, visualizations, budget optimization |
| `deliverables.py`                 | Dashboard deliverables generation (standalone execution)  |
| `schemas.py`                      | Pydantic models for deliverables                          |
| `comparison.py`                   | Compare baseline vs hierarchical model                    |
| `pipeline.py`                     | Orchestration, retry logic, state management              |
| `models/hierarchical_bayesian.py` | Custom PyMC model with learned adstock/saturation         |

## scripts/

| Script                | Scope / Purpose                              |
| --------------------- | -------------------------------------------- |
| `run_pipeline.py`     | CLI entry point, orchestrates full pipeline  |
| `mmm_baseline.py`     | Baseline model training + MLflow logging     |
| `mmm_hierarchical.py` | Hierarchical model training + MLflow logging |

**Rule:** Scripts should only contain:

- Entry point function (`run_*`)
- CLI argument parsing
- MLflow tracking/logging calls
- Calls to `src/` modules (no business logic)

# Project Technical Context

> Architecting and Optimizing Bayesian Marketing Mix Models.
> Last Technical Audit: 2026-03-28.

---

## 1. [PROJECT_STRUCTURE]

```text
MMM-Figshare-eCommerce/
├── /data
│   ├── raw/              # Original Figshare download
│   └── processed/        # Cleaned Parquet datasets
├── /app                  # Streamlit Dashboard (6 pages)
│   ├── Home.py
│   ├── components/
│   └── pages/
├── /src                  # Core Library (src-layout)
│   ├── config.py         # Centralized hyperparameters
│   ├── models/
│   │   └── hierarchical_bayesian.py
│   ├── preprocessing.py
│   ├── transformations.py
│   ├── insights.py       # Budget optimization & ROI
│   └── schemas.py        # Pydantic validation
├── /scripts              # CLI Entrypoints
│   ├── run_pipeline.py
│   ├── mmm_baseline.py
│   └── mmm_hierarchical.py
├── /tests
├── /models               # Saved traces (.nc)
├── /mlruns               # MLflow experiment tracking
└── /notebooks            # EDA & prototyping

* Pattern: Custom src-layout + Streamlit Dashboard
```

---

## 2. [DATA_STACK]

| Category          | Technologies                                         |
| :---------------- | :--------------------------------------------------- |
| **Core**          | Pandas, NumPy, PyArrow, FastParquet                  |
| **Bayesian/ML**   | PyMC ≥5.10, PyMC-Marketing ≥0.4, ArviZ, Scikit-learn |
| **Sampling**      | NumPyro, JAX (XLA Compilation)                       |
| **Tracking/Ops**  | MLflow ≥2.10                                         |
| **Visualization** | Matplotlib, Seaborn, Plotly                          |
| **Dashboard**     | Streamlit                                            |
| **Time Series**   | Statsmodels, Holidays                                |
| **Environment**   | Conda/Mamba (MKL-optimized BLAS)                     |

---

## 3. [OBJECTIVE]

- **Business Goal:** Quantify the true ROI of digital advertising spend across Google, Meta, and TikTok platforms, and optimize multi-region budget allocation for ~100 eCommerce brands.
- **ML Task:** Hierarchical Bayesian Regression for Marketing Mix Modeling (MMM) on time-series panel data. Decomposes revenue into additive channel contributions using learned Adstock (carryover) and Hill Saturation (diminishing returns) transformations.
- **Key Metrics:**
  - **Predictive:** R², SMAPE (Symmetric MAPE), CRPS (probabilistic).
  - **Business:** Channel ROI, Contribution Share, Marginal ROAS, Projected Revenue Lift.
  - **Diagnostics:** R-hat < 1.01, ESS > 400, 0 divergences.

---

## 4. [STANDARDS]

- **Linter/Formatter:** Ruff (line-length: 100, target: py310). Selected rules: `E`, `F`, `I`, `W`. Ignores `E501`.
- **Type Checking:** Pydantic schemas (`src/schemas.py`) for runtime validation of I/O data structures.
- **Testing:** Pytest (in `dev` dependencies). Tests cover data loading, preprocessing, model creation, insights extraction, and ROI computation.
- **Docstrings:** Google-style (inferred from codebase conventions).
- **Configuration:** All hyperparameters (priors, MCMC settings, feature lists) are centralized in `src/config.py`. **Never hardcode values in modules.**
- **Data Validation:** Strict temporal holdout (last 8 weeks). Panel-aware time-series splitting.

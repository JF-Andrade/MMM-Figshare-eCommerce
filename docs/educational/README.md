# Marketing Mix Modeling: Educational Material

Comprehensive graduate-level educational material covering theory, methodology, and implementation of Bayesian Marketing Mix Models using PyMC-Marketing.

---

## Target Audience

- **Data Scientists** seeking to add marketing analytics to their toolkit
- **Marketing Analysts** wanting rigorous measurement methodology
- **Graduate Students** in quantitative marketing or statistics

**Expected Background**: Python programming, basic statistics, linear regression fundamentals.

---

## Table of Contents

| # | Section | Learning Outcome | Time |
|---|---------|------------------|------|
| 0 | [Executive Summary](00_executive_summary.md) | Understand material scope and learning path | 15 min |
| 1 | [Introduction to MMM](01_introduction.md) | Compare MMM vs attribution; identify use cases | 45 min |
| 2 | [Mathematical Foundations](02_mathematical_foundations.md) | Apply Bayes' theorem; understand MCMC sampling | 2 hr |
| 3 | [Adstock Transformations](03_adstock.md) | Implement carryover effects in time series | 45 min |
| 4 | [Saturation Functions](04_saturation.md) | Model diminishing returns with logistic curves | 45 min |
| 5 | [PyMC-Marketing Implementation](05_pymc_marketing.md) | Build a working MMM using PyMC-Marketing | 1.5 hr |
| 6 | [MCMC Diagnostics](06_mcmc_diagnostics.md) | Diagnose convergence; configure GPU acceleration | 1 hr |
| 7 | [Hierarchical Models](07_hierarchical_models.md) | Implement partial pooling across regions | 1 hr |
| 8 | [Model Validation](08_validation.md) | Validate with holdout and posterior checks | 45 min |
| 9 | [ROI Computation](09_roi_computation.md) | Compute channel ROI with uncertainty | 45 min |
| 10 | [Budget Optimization](10_budget_optimization.md) | Generate optimal allocation recommendations | 45 min |
| 11 | [Advanced Topics](11_advanced_topics.md) | Explore extensions: interactions, causality | 1 hr |

**Total Time: ~11 hours**

---

## How to Use This Material

**Sequential Learning (Recommended)**:
Follow sections 0-8 in order for complete understanding.

**Practitioner Fast-Track**:
Sections 0 → 1 → 5 → 6 → 9 for implementation focus.

**Business Stakeholder Summary**:
Section 0 (Executive Summary) + Section 9 (ROI) only.

---

## Prerequisites

| Topic | Self-Assessment | Resources |
|-------|-----------------|-----------|
| Python | Can write functions and classes | [Python Tutorial](https://docs.python.org/3/tutorial/) |
| Pandas | Can filter, group, and merge DataFrames | [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html) |
| Statistics | Understand distributions and hypothesis testing | [Think Stats (Free)](https://greenteapress.com/thinkstats/) |
| Regression | Can interpret linear regression coefficients | [Khan Academy: Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data) |

---

## Connection to Project

This material accompanies the [MMM-Figshare-eCommerce](../../README.md) project:

- **Scripts**: `scripts/mmm_baseline.py` (Ridge Regression), `scripts/mmm_hierarchical.py` (Bayesian)
- **Dashboard**: `streamlit run app/dashboard.py` (6 interactive pages)
- **Modules**: `src/` contains reusable functions for preprocessing, modeling, and optimization
- **Execution**: Local CPU or Google Colab (GPU recommended for hierarchical model)

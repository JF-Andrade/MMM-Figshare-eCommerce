# Marketing Mix Modeling: eCommerce Digital Channels

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMC-Marketing](https://img.shields.io/badge/PyMC--Marketing-0.8+-green.svg)](https://www.pymc-marketing.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bayesian Marketing Mix Model to **quantify the ROI** of digital advertising channels and **optimize budget allocation** across Google, Meta, and TikTok platforms by territory.

> **Note:**
> See [CHANGELOG.md](CHANGELOG.md) for recent updates.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Motivation

Digital marketing teams allocate millions in advertising spend across multiple platforms, but often lack rigorous methods to measure true ROI and optimize budget allocation. Traditional last-click attribution fails to capture:

- **Carryover effects** (adstock): Ads continue to influence conversions days/weeks after exposure
- **Saturation effects**: Diminishing returns as spend increases
- **Cross-channel interactions**: How channels work together

This project applies Bayesian Media Mix Modeling to solve these challenges, providing:

1. Statistically rigorous ROI estimates with uncertainty quantification
2. Optimal budget allocation recommendations
3. Multi-region analysis with hierarchical modeling

---

## Key Results

### Aggregated Channel ROI (All Territories)

> **Source:** Latest Hierarchical Run (Representative results from 2023 evaluation data)

| Channel            | ROI  | Contribution Share |
| ------------------ | ---- | ------------------ |
| GOOGLE_SHOPPING    | 4.42 | 62.7%              |
| GOOGLE_PMAX        | 1.76 | 17.2%              |
| META_FACEBOOK      | 1.47 | 17.0%              |
| GOOGLE_DISPLAY     | 0.33 | 0.8%               |
| GOOGLE_PAID_SEARCH | 0.29 | 3.3%               |

### Model Performance

| Model                     | R² Train | R² Test | MAPE Test |
| ------------------------- | -------- | ------- | --------- |
| **Ridge Baseline**        | 0.75     | 0.49    | 15.7%     |
| **Hierarchical Bayesian** | 0.67     | 0.86    | 36.4%     |

### Interactive Dashboard

The primary way to consume model results is through the **Streamlit Dashboard**:

```bash
streamlit run app/Home.py
```

**Dashboard Pages (5 focused views):**

1. **Performance Analysis**: Channel ROI, contribution charts, saturation alerts, regional heatmap
2. **Budget Optimization**: Optimal allocation recommendations, reallocation table
3. **What-If Simulator**: Interactive budget sliders with real-time contribution projection
4. **Technical Details**: Adstock/saturation parameters, model comparison, diagnostics (tabbed)
5. **Historical Tracking**: ROI trends over time, benchmarks vs 6-month average

### Deliverables (MLflow)

All results are stored in MLflow artifacts. Model training and deliverables generation are now decoupled, allowing for quick iteration on dashboard metrics without re-training:

- `roi.json` / `roi_hierarchical.csv` - ROI aggregated metrics
- `revenue_lift.json` - Projected revenue growth from optimal reallocation
- `optimization.json` - Budget allocation recommendations
- `marginal_roas.json` - Diminishing returns analysis (Marginal ROAS curves)
- `predictions.json` - Detailed actual vs predicted time series

> **Tip:** Use `--deliverables-only` to regenerate artifacts using a previously trained model.

---

## Normalization & Scale Integrity

To ensure accurate cross-region and cross-currency comparisons, the model uses **currency-aware normalization**:

1. Spend is normalized internally (0-1) based on the maximum weekly spend per channel/currency during training.
2. The learned parameters (`L`, `k`) are tied to this normalized scale.
3. The optimization engine uses the training-stage `max_spend` to maintain consistent scale when calculating Marginal ROAS and Projected Lift.

---

## Dataset

**Source:** [Multi-Region MMM Dataset (Figshare)](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841)

> "This dataset contains anonymised, multi-region marketing and purchase data for close to 100 eCommerce brands to support the advancement and benchmarking of Marketing Mix Modelling techniques. This dataset is suitable for modelling customer acquisition costs (CAC) and return on advertising spend (ROAS) for online-only purchases and digital marketing channels (currently limited to Google, Meta, and Tiktok)."

| Attribute   | Value                        |
| ----------- | ---------------------------- |
| Rows        | 132,259                      |
| Columns     | 104                          |
| Time Period | 2019-2023                    |
| Granularity | Daily (aggregated to weekly) |
| Regions     | 19 territories               |
| Brands      | ~100 eCommerce brands        |

### Marketing Channels (9)

| Platform | Channels                                    |
| -------- | ------------------------------------------- |
| Google   | Paid Search, Shopping, PMax, Display, Video |
| Meta     | Facebook, Instagram, Other                  |
| TikTok   | TikTok Ads                                  |

### Target Variable

`ALL_PURCHASES_ORIGINAL_PRICE` - Total revenue before discounts (avoids data leakage from GROSS_DISCOUNT)

### Data Preprocessing

- **Aggregation**: Daily to weekly (Monday-Sunday) to align with marketing cycles.
- **Channel Selection**: Variance filter (removes >80% zeros) and low-volume aggregation (<1% share moved to `OTHER_SPEND`).
- **ROI / ROAS**: Incremental Return on Ad Spend (Contribution / Spend).
- **CAC (Heuristic)**: Customer Acquisition Cost modeled via revenue contribution share (Attributed New Customers = Total New Customers \* Channel Revenue Share).
- **Control Variables**: Trend, Holidays, Seasonality (Fourier terms).
- **Priors**: Informative priors based on platform defaults and business logic.
- **Exogenous Demand**: Non-paid traffic sources (Direct, Organic, Email) included as base demand drivers.
- **Transformations**: Geometric Adstock (carryover) and Hill Saturation (diminishing returns).

### Features (33 Active)

All features are centralized in `src/config.py`.

| Category    | Count | Features                                                                                                                                                                                                  |
| ----------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SPEND**   | 9     | `GOOGLE_PAID_SEARCH_SPEND`, `GOOGLE_SHOPPING_SPEND`, `GOOGLE_PMAX_SPEND`, `GOOGLE_DISPLAY_SPEND`, `GOOGLE_VIDEO_SPEND`, `META_FACEBOOK_SPEND`, `META_INSTAGRAM_SPEND`, `META_OTHER_SPEND`, `TIKTOK_SPEND` |
| **CONTROL** | 4     | `trend`, `is_holiday`, `is_q4`, `is_black_friday`                                                                                                                                                         |
| **TRAFFIC** | 6     | `DIRECT_CLICKS`, `BRANDED_SEARCH_CLICKS`, `ORGANIC_SEARCH_CLICKS`, `EMAIL_CLICKS`, `REFERRAL_CLICKS`, `ALL_OTHER_CLICKS`                                                                                  |
| **SEASON**  | 8     | `WEEK_SIN/COS (1/2)`, `MONTH_SIN/COS (1/2)` (High-order Fourier terms)                                                                                                                                    |

> **Model Usage:** SPEND features go through Bayesian adstock/saturation. All other features use Horseshoe regularization to handle high dimensionality.

### Validation

- **Ridge Baseline (Expanding Window CV)**: Uses 5 expanding window folds for hyperparameter tuning.
  - Ensures that selected adstock/saturation parameters are robust across different time periods and seasonalities, minimizing overfitting to a specific timeframe.
- **Hierarchical Bayesian (Out-of-Time Holdout)**: Uses the last N weeks as a strict holdout set.
  - Mimics a real-world production forecasting scenario where future data is completely unseen. Also computationally necessary as full Bayesian cross-validation is prohibitively expensive for large hierarchical models.
- **Data Integrity**: Pipeline includes strict type-checking and automated calculation verification.

### Region Usage

| Model              | Regions      | Selection Criteria                                                     |
| ------------------ | ------------ | ---------------------------------------------------------------------- |
| **Ridge Baseline** | UK (Default) | Single-region focus for rapid iteration (configurable via `--region`). |
| **Hierarchical**   | All Valid    | Must have ≥52 weeks of data (automatically filtered).                  |

> **Note:** Use `--max-regions N` to limit hierarchical model to the top N regions by revenue for faster training.

---

## Methodology

### Architecture Overview

#### Pipeline Flow

```mermaid
flowchart LR
    subgraph Data["Data Layer"]
        RAW[(Raw Data<br/>Daily)]
        WEEKLY[(Weekly<br/>Aggregated)]
    end

    subgraph Preprocess["Preprocessing"]
        AGG[Aggregation]
        FEAT[Feature<br/>Engineering]
        NORM[Currency<br/>Normalization]
    end

    subgraph Transform["Transformations"]
        ADS[Geometric<br/>Adstock]
        SAT[Hill<br/>Saturation]
    end

    subgraph Model["Bayesian Model"]
        HIER[Hierarchical<br/>Pooling]
        MCMC[NUTS<br/>Sampler]
    end

    subgraph Output["Deliverables"]
        ROI[Channel ROI]
        OPT[Budget<br/>Optimization]
        DASH[Dashboard]
    end

    RAW --> AGG --> WEEKLY
    WEEKLY --> FEAT --> NORM
    NORM --> ADS --> SAT
    SAT --> HIER --> MCMC
    MCMC --> ROI & OPT --> DASH
```

#### Hierarchical Model Structure

```mermaid
flowchart TB
    subgraph Hyperpriors["Hyperpriors"]
        SA[σ_α]
        SL[σ_L]
        SB[σ_β]
    end

    subgraph Global["Global Parameters"]
        ALPHA[α_channel<br/>Beta 2,2]
        L[L_channel<br/>HalfNormal]
        K[k_channel<br/>Gamma 2,1]
        BETA[β_channel<br/>HalfNormal]
    end

    subgraph Territory["Territory Parameters"]
        ALPHA_T[α_territory]
        L_T[L_territory]
        BETA_T[β_territory]
    end

    subgraph Transforms["Data Transforms"]
        ADS_T[Adstock<br/>x + α·x_prev]
        SAT_T[Saturation<br/>x^k / L^k+x^k]
    end

    subgraph Likelihood["Likelihood"]
        MU[μ = intercept +<br/>Σ β·saturation]
        Y[y ~ StudentT<br/>ν, μ, σ]
    end

    SA --> ALPHA_T
    SL --> L_T
    SB --> BETA_T

    ALPHA --> ALPHA_T
    L --> L_T
    BETA --> BETA_T
    K --> SAT_T

    ALPHA_T --> ADS_T
    ADS_T --> SAT_T
    L_T --> SAT_T
    BETA_T --> MU
    SAT_T --> MU
    MU --> Y
```

### Bayesian Media Mix Model

The model uses a **Hierarchical Bayesian** approach to decompose revenue into additive components, using a robust Student-T likelihood to handle outliers.

#### 1. Revenue Model (Likelihood)

$$ Revenue_t \sim \text{Student-T}(\nu, \mu_t, \sigma) $$

- **$\nu$ (Nu)**: Degrees of freedom. Controls the "heaviness" of the tails. Lower values (<30) make the model more robust to outliers (e.g., Black Friday spikes).
- **$\sigma$ (Sigma)**: Observation noise. The baseline variance of revenue not explained by the model.
- **$\mu_t$ (Mu)**: Expected revenue at time $t$, composed of:
  `Intercept + Seasonality + Trend + Σ(Media Effects)`

#### 2. Media Transformations

Raw spend is transformed to capture consumer behavior dynamics:

1. **Geometric Adstock (Carryover)**
   $$ Adstock(x*t) = x_t + \alpha \cdot x*{t-1} $$
   - **$\alpha$ (Alpha)**: Decay rate $[0, 1]$. Represents the % of ad effect retained in the following week.
     - High $\alpha$ (>0.5) $\rightarrow$ Brand building (TV, Video).
     - Low $\alpha$ (<0.3) $\rightarrow$ Direct response (PMax, Search).

2. **Hill Saturation (Diminishing Returns)**
   $$ Saturation(x) = \frac{x^K}{x^K + L^K} $$
   - **$K$ (Shape)**: Steepness of the S-curve.
     - $K > 1$: S-shape (threshold effect).
     - $K \le 1$: C-shape (diminishing returns start immediately).
   - **$L$ (Scale)**: Half-saturation point. The spend level where the channel reaches 50% of its maximum potential impact.

#### 3. Hierarchical Pooling

The model learns "Global" effects while allowing "Local" variations for each Territory:

$$ \beta*{region} \sim \text{Normal}(\beta*{global}, \sigma\_{heterogeneity}) $$

- **$\beta_{global}$**: The average effectiveness of a channel across all markets.
- **$\sigma_{heterogeneity}$**: How much a specific territory is allowed to deviate from the global average.
  - Small $\sigma$: Territories behave similarly (strong pooling).
  - Large $\sigma$: Territories are independent (weak pooling).

### Prior Distributions (Bayesian Hierarchical Model)

All priors are externalized to `src/config.py` for easy tuning.

| Component      | Parameter     | Distribution     | Config                             | Description                                                                                                                     |
| -------------- | ------------- | ---------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Adstock**    | α (alpha)     | Beta(2, 2)       | `PRIOR_ADSTOCK_ALPHA/BETA`         | Decay rate controlling how quickly ad effects fade. Values near 0 = rapid decay, near 1 = slow decay. Beta(2,2) centers at 0.5. |
| **Adstock**    | σ_α           | HalfNormal(0.2)  | `PRIOR_SIGMA_ADSTOCK_TERRITORY`    | Variation in decay rates across territories.                                                                                    |
| **Saturation** | L             | HalfNormal(0.3)  | `PRIOR_SATURATION_L_SIGMA`         | Half-saturation point (calibrated for normalized spend in [0,1]).                                                               |
| **Saturation** | k             | Gamma(2, 1)      | `PRIOR_SATURATION_K_ALPHA/BETA`    | Hill curve steepness. k=1 is linear, k>2 creates sharp S-curve.                                                                 |
| **Saturation** | σ_L           | HalfNormal(0.2)  | `PRIOR_SIGMA_SATURATION_TERRITORY` | Regional variation in saturation points.                                                                                        |
| **Hierarchy**  | σ_territory   | HalfNormal(0.5)  | `PRIOR_SIGMA_TERRITORY`            | Baseline revenue variation between territories.                                                                                 |
| **Channels**   | β_channel     | HalfNormal(0.5)  | `PRIOR_BETA_CHANNEL_SIGMA`         | Global channel effect magnitude. Positive-only ensures spend increases revenue.                                                 |
| **Channels**   | σ_β_territory | HalfNormal(0.05) | `PRIOR_SIGMA_BETA_TERRITORY`       | Regional variation in channel effectiveness.                                                                                    |
| **Horseshoe**  | τ             | HalfStudentT(3)  | Computed: `τ0 = m0/(D-m0)/√n`      | Global shrinkage (Piironen & Vehtari, 2017).                                                                                    |
| **Horseshoe**  | λ             | HalfStudentT(3)  | `PRIOR_HORSESHOE_LAMBDA_BETA`      | Local shrinkage per feature. Enables sparse feature selection.                                                                  |
| **Likelihood** | σ_obs         | HalfNormal(1.0)  | `PRIOR_SIGMA_OBS`                  | Observation noise scale (calibrated for y_log std ≈ 0.5-1.5).                                                                   |
| **Likelihood** | ν             | Gamma(2, 0.5)    | `PRIOR_NU_ALPHA/BETA`              | Student-T degrees of freedom. Mean ν ≈ 4 for robust outlier handling.                                                           |

### Control & Seasonality Variables

| Component       | Variable          | Distribution   | Config                     | Description                                                    |
| :-------------- | :---------------- | :------------- | :------------------------- | :------------------------------------------------------------- |
| **Trend**       | `trend`           | Horseshoe      | `PRIOR_HORSESHOE_...`      | Normalized linear trend (0 to 1) capturing long-term growth.   |
| **Events**      | `is_holiday`      | Horseshoe      | `PRIOR_HORSESHOE_...`      | Binary holiday indicator for general public holidays.          |
| **Events**      | `is_q4`           | Horseshoe      | `PRIOR_HORSESHOE_...`      | Binary indicator for Q4 (Oct-Dec) high season.                 |
| **Events**      | `is_black_friday` | Horseshoe      | `PRIOR_HORSESHOE_...`      | Binary indicator for Black Friday week (major revenue spikes). |
| **Seasonality** | `WEEK_SIN/COS`    | Normal(0, 0.5) | `PRIOR_GAMMA_SEASON_SIGMA` | Fourier terms (1st/2nd order) capturing weekly cycles.         |
| **Seasonality** | `MONTH_SIN/COS`   | Normal(0, 0.5) | `PRIOR_GAMMA_SEASON_SIGMA` | Fourier terms (1st/2nd order) capturing monthly/yearly cycles. |

### MCMC Sampling

| Component    | Parameter     | Value         | Config               | Description                                                                                                                                                                            |
| :----------- | :------------ | :------------ | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sampling** | Engine        | NumPyro (JAX) | `MCMC_SAMPLER`       | High-performance JAX-based sampler (significantly faster than standard PyMC). [Source](https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html) |
| **Sampling** | Chains        | 4             | `MCMC_CHAINS`        | Number of independent Markov chains running in parallel.                                                                                                                               |
| **Sampling** | Draws         | 4,000         | `MCMC_DRAWS`         | Number of posterior samples per chain (after tuning).                                                                                                                                  |
| **Sampling** | Tune          | 500           | `MCMC_TUNE`          | Warm-up steps to adapt step size and mass matrix.                                                                                                                                      |
| **Sampling** | Target Accept | 0.85          | `MCMC_TARGET_ACCEPT` | Target acceptance probability for NUTS. Higher values reduce divergences.                                                                                                              |
| **Sampling** | Max Treedepth | 12            | `MCMC_MAX_TREEDEPTH` | Maximum depth of the NUTS trajectory tree to prevent infinite loops.                                                                                                                   |

### Validation Strategy

- **Temporal Holdout**: Last 8 weeks held out for validation (strictly out-of-time to mimic production).
- **Diagnostics**:
  - **Convergence**: R-hat < 1.01, ESS > 400.
  - **Divergences**: 0 divergences required for reliable posterior.
  - **BFMI**: > 0.3 (ensure energy distribution matches).
- **Metrics**:
  - **Deterministic**: $R^2$, SMAPE (Symmetric MAPE for robustness), RMSE.
  - **Probabilistic**: CRPS (Continuous Ranked Probability Score) and Coverage (HDI).
- **Splitting**:
  - **Ridge Baseline**: `TimeSeriesSplit` (5 folds) for hyperparameter tuning.
  - **Hierarchical**: **Panel Time-Based Holdout** (Last 8 weeks of each territory).

### Known Limitations

> [!NOTE]
> **Contribution Calculation.** Model contributions are computed in log-space and
> converted to monetary scale via a linear scaling factor. This first-order
> approximation may introduce 5-15% error in absolute values. Channel rankings
> and relative comparisons remain reliable.

> [!NOTE]  
> **Budget Optimizer.** The optimization routine estimates channel effectiveness (β)
> from observed contribution and saturation response. This heuristic does not re-run
> the full posterior predictive. Maximum precision requires model re-fitting with
> modified spend scenarios.

> [!NOTE]
> **What-If Simulator.** The simulator models returns via Hill saturation curves.
> Carryover (adstock) effects from modified budget allocations are not simulated;
> they are implicitly captured in the estimated β. This approach is best suited
> for short-term scenarios (1-4 weeks).

---

## Project Structure

```text
MMM-Figshare-eCommerce/
├── data/
│   ├── raw/                    # Original Figshare download
│   └── processed/              # Cleaned datasets
│       └── mmm_data.parquet    # Main dataset (31 MB)
│
├── environment.yml             # Conda environment configuration
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing demo
│   ├── 03_inspection.ipynb     # Data health checks & inspection
│   └── 05_optimization.ipynb   # Budget optimization workflow
│
├── scripts/
│   ├── run_pipeline.py         # Pipeline orchestrator CLI
│   ├── mmm_baseline.py         # Ridge Regression baseline
│   └── mmm_hierarchical.py     # Bayesian hierarchical model
│
├── src/
│   ├── config.py               # Centralized configuration (all hyperparameters)
│   ├── data_loader.py          # Data loading and validation
│   ├── preprocessing.py        # Pipeline orchestration & data prep
│   ├── transformations.py      # Core math & feature engineering helpers
│   ├── pipeline.py             # Pipeline orchestration logic
│   ├── baseline_evaluation.py  # Ridge model metrics & ROI computation
│   ├── insights.py             # Optimization, parameter extraction
│   ├── comparison.py           # Model comparison utilities
│   ├── utils/                  # Utility modules
│   │   └── __init__.py
│   └── models/                 # Model package
│       ├── __init__.py         # Package exports
│       └── hierarchical_bayesian.py  # Bayesian MMM with learned transforms
│
├── app/
│   ├── Home.py                 # Streamlit main entry point
│   ├── mlflow_loader.py        # MLflow data loading (adapter pattern)
│   ├── shared.py               # Shared UI components and configs
│   ├── components/             # Reusable UI widgets
│   └── pages/                  # Dashboard pages
│       ├── 01_Executive_Summary.py
│       ├── 02_Budget_Optimization.py
│       ├── 03_Regional_Analysis.py
│       ├── 04_Model_Details.py
│       ├── 05_Model_Comparison.py
│       └── 06_Channel_Efficiency.py  # CAC, ROAS metrics
│
├── tests/
│   ├── test_data_loader.py     # Data loading tests
│   ├── test_preprocessing.py   # Preprocessing tests
│   ├── test_model.py           # Model creation tests
│   ├── test_optimization.py    # Optimization tests
│   ├── test_horseshoe_tau.py   # Horseshoe prior formula validation
│   └── test_roi.py             # ROI computation tests
│
├── models/                     # Saved models and traces
├── mlruns/                     # MLflow experiment tracking
│
├── pyproject.toml              # Project dependencies
└── README.md
```

---

## Installation

### Local Development (Conda + MKL)

```powershell
# Clone repository
git clone https://github.com/JF-Andrade/MMM-Figshare-eCommerce.git
cd MMM-Figshare-eCommerce

# Create Conda environment with MKL (requires mamba)
# Note: You can use `conda env create` if mamba is not installed (slower)
mamba env create -f environment.yml

# Activate environment with BLAS optimization
. .\activate_mmm.ps1
```

> **Note**: The `activate_mmm.ps1` script activates the Conda environment and configures the `PYTENSOR_FLAGS` variable to link Intel MKL. This is necessary because PyTensor on Windows does not automatically detect BLAS libraries, and without this configuration, matrix operations are severely degraded.

---

## Usage

### Pipeline Execution (Recommended)

```bash
# Run full pipeline
python scripts/run_pipeline.py

# Baseline only (Ridge Regression)
python scripts/run_pipeline.py --baseline-only

# Hierarchical only (Bayesian)
python scripts/run_pipeline.py --hierarchical-only

# Preview execution plan
python scripts/run_pipeline.py --dry-run
```

### Direct Script Execution

```bash
# Baseline model (Ridge)
python scripts/mmm_baseline.py

# Hierarchical model (Bayesian)
# Optional: limit regions for faster testing
python scripts/mmm_hierarchical.py --max-regions 5
```

### Output Files

Results are saved to `models/` and tracked in MLflow (`mlruns/`).

| File                         | Description                                      |
| :--------------------------- | :----------------------------------------------- |
| `ridge_baseline_results.png` | Baseline model actual vs predicted visualization |
| `ridge_coefficients.csv`     | Ridge regression coefficients                    |
| `ridge_roi.csv`              | ROI estimates from baseline                      |
| `mmm_hierarchical_trace.nc`  | Full MCMC trace (ArviZ NetCDF)                   |
| `saturation_curves.png`      | Learned saturation curves visualization          |

---

## Tech Stack

| Category         | Technologies                               |
| :--------------- | :----------------------------------------- |
| **Modeling**     | PyMC-Marketing, PyMC, ArviZ                |
| **Sampling**     | NumPyro (JAX Backend for speed)            |
| **Tracking**     | MLflow (Experiments & Registry)            |
| **Dashboard**    | Streamlit, Plotly, Altair                  |
| **Environment**  | Conda/Mamba, Intel MKL (BLAS optimization) |
| **Acceleration** | JAX (XLA Compilation)                      |

---

## Author

**Jordao Fernandes de Andrade**

- Email: <jordaoandrade@gmail.com>
- LinkedIn: [linkedin.com/in/jordaofernandes](https://www.linkedin.com/in/jordaofernandes/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset provided by [Figshare](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841)
- PyMC-Marketing team for the excellent MMM framework

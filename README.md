# Marketing Mix Modeling: eCommerce Digital Channels

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMC-Marketing](https://img.shields.io/badge/PyMC--Marketing-0.8+-green.svg)](https://www.pymc-marketing.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bayesian Marketing Mix Model to quantify the ROI of digital advertising channels and optimize budget allocation across Google, Meta, and TikTok platforms.

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
- [Future Work](#future-work)
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

### Channel ROI Estimates (GBP, UK Territory)

| Channel         | ROI    | Contribution Share |
| --------------- | ------ | ------------------ |
| GOOGLE_SHOPPING | 0.0126 | 64.0%              |
| GOOGLE_PMAX     | 0.0092 | 17.3%              |
| META_FACEBOOK   | 0.0039 | 15.3%              |
| OTHER_SPEND     | 0.0032 | 2.0%               |
| META_INSTAGRAM  | 0.0009 | 1.0%               |

### Model Performance

| Model                     | R² Train | R² Test | MAPE Test |
| ------------------------- | -------- | ------- | --------- |
| **Ridge Baseline**        | 0.60     | 0.13    | 14.7%     |
| **Hierarchical Bayesian** | 0.85\*   | 0.42\*  | 21.3%\*   |

> \* Preliminary results from the custom nested hierarchical model with 18 territories.

### Visualizations

Results and visualizations are saved to `models/`:

- `mmm_baseline_results.png` - Model fit and channel contributions
- `channel_contributions.png` - Channel contributions over time (original scale)
- `channel_share_hdi.png` - Channel share with 94% HDI
- `waterfall_decomposition.png` - Revenue decomposition
- `contribution_curves.png` - Saturation curves
- `roi_baseline.csv` - ROI rankings

---

## Dataset

**Source:** [Multi-Region MMM Dataset (Figshare)](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841)

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

1. Aggregation from daily to weekly (253 weeks)
2. Channel variance filter: removes channels with >80% zeros (TIKTOK, GOOGLE_VIDEO)
3. Feature engineering: CTR, CPC, rolling stats, spend share
4. Holiday indicator and trend features
5. Low-spend channels aggregated into OTHER_SPEND (< 5% of total spend)

### Features (59 Total)

All features are centralized in `src/config.py`.

| Category    | Count | Features                                                                                                                                                                                                  |
| ----------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SPEND**   | 9     | `GOOGLE_PAID_SEARCH_SPEND`, `GOOGLE_SHOPPING_SPEND`, `GOOGLE_PMAX_SPEND`, `GOOGLE_DISPLAY_SPEND`, `GOOGLE_VIDEO_SPEND`, `META_FACEBOOK_SPEND`, `META_INSTAGRAM_SPEND`, `META_OTHER_SPEND`, `TIKTOK_SPEND` |
| **CONTROL** | 6     | `trend`, `is_holiday`, `month`, `week_of_year`, `quarter`, `is_q4`                                                                                                                                        |
| **CTR**     | 7     | `GOOGLE_PAID_SEARCH_CTR`, `GOOGLE_SHOPPING_CTR`, `GOOGLE_PMAX_CTR`, `GOOGLE_DISPLAY_CTR`, `GOOGLE_VIDEO_CTR`, `META_FACEBOOK_CTR`, `META_INSTAGRAM_CTR`                                                   |
| **CPC**     | 9     | `GOOGLE_PAID_SEARCH_CPC`, `GOOGLE_SHOPPING_CPC`, `GOOGLE_PMAX_CPC`, `GOOGLE_DISPLAY_CPC`, `GOOGLE_VIDEO_CPC`, `META_FACEBOOK_CPC`, `META_INSTAGRAM_CPC`, `META_OTHER_CPC`, `TIKTOK_CPC`                   |
| **SHARE**   | 9     | `{CHANNEL}_SHARE` for each spend channel                                                                                                                                                                  |
| **TRAFFIC** | 6     | `DIRECT_CLICKS`, `BRANDED_SEARCH_CLICKS`, `ORGANIC_SEARCH_CLICKS`, `EMAIL_CLICKS`, `REFERRAL_CLICKS`, `ALL_OTHER_CLICKS`                                                                                  |
| **SEASON**  | 4     | `sin_1`, `cos_1`, `sin_2`, `cos_2` (Fourier terms)                                                                                                                                                        |

> **Model Usage:** SPEND features go through Bayesian adstock/saturation. All other features use Horseshoe regularization.

### Validation

- Expanding window cross-validation (3 folds)
- Pydantic schema validation for all deliverables

### Region Usage

| Model              | Regions   | Selection Criteria                                              |
| ------------------ | --------- | --------------------------------------------------------------- |
| **Ridge Baseline** | UK only   | Grid Search over Adstock/Saturation + TimeSeriesSplit (5 folds) |
| **Hierarchical**   | All valid | All currencies, ≥52 weeks of data                               |

> Use `--max-regions N` flag to limit hierarchical model to top N regions by revenue.

---

## Methodology

### Bayesian Media Mix Model

The model decomposes revenue into:

```text
Revenue = Base + Σ(Channel Effects) + Controls + Seasonality + Noise
```

Where each channel effect is transformed by:

1. **Adstock (Carryover)**: GeometricAdstock with learned decay rate

   - Captures how ads continue to influence conversions over time
   - `l_max = 12` weeks maximum lag

2. **Saturation (Diminishing Returns)**: LogisticSaturation
   - Models decreasing marginal returns at higher spend levels

### Prior Distributions (Bayesian Hierarchical Model)

All priors are externalized to `src/config.py` for easy tuning.

| Component      | Parameter     | Distribution     | Config                             | Description                                                                                                                     |
| -------------- | ------------- | ---------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Adstock**    | α (alpha)     | Beta(2, 2)       | `PRIOR_ADSTOCK_ALPHA/BETA`         | Decay rate controlling how quickly ad effects fade. Values near 0 = rapid decay, near 1 = slow decay. Beta(2,2) centers at 0.5. |
| **Adstock**    | σ_α           | HalfNormal(0.1)  | `PRIOR_SIGMA_ADSTOCK_TERRITORY`    | Variation in decay rates across territories.                                                                                    |
| **Saturation** | L             | HalfNormal(1.0)  | `PRIOR_SATURATION_L_SIGMA`         | Half-saturation point: spend level where 50% of max effect is reached. Higher L = more spend needed.                            |
| **Saturation** | k             | Gamma(2, 1)      | `PRIOR_SATURATION_K_ALPHA/BETA`    | Hill curve steepness. k=1 is linear, k>2 creates sharp S-curve.                                                                 |
| **Saturation** | σ_L           | HalfNormal(0.1)  | `PRIOR_SIGMA_SATURATION_TERRITORY` | Regional variation in saturation points.                                                                                        |
| **Hierarchy**  | σ_currency    | HalfNormal(0.5)  | `PRIOR_SIGMA_CURRENCY`             | Baseline revenue variation between currencies (GBP, USD, EUR).                                                                  |
| **Hierarchy**  | σ_territory   | HalfNormal(0.3)  | `PRIOR_SIGMA_TERRITORY`            | Territory variation nested within currency.                                                                                     |
| **Channels**   | β_channel     | HalfNormal(0.5)  | `PRIOR_BETA_CHANNEL_SIGMA`         | Global channel effect magnitude. Positive-only ensures spend increases revenue.                                                 |
| **Channels**   | σ_β_territory | HalfNormal(0.05) | `PRIOR_SIGMA_BETA_TERRITORY`       | Regional variation in channel effectiveness.                                                                                    |
| **Horseshoe**  | τ             | HalfCauchy(1)    | `PRIOR_HORSESHOE_TAU_BETA`         | Global shrinkage for 43 auxiliary features.                                                                                     |
| **Horseshoe**  | λ             | HalfCauchy(1)    | `PRIOR_HORSESHOE_LAMBDA_BETA`      | Local shrinkage per feature. Enables sparse feature selection.                                                                  |
| **Likelihood** | σ_obs         | HalfNormal(0.5)  | `PRIOR_SIGMA_OBS`                  | Observation noise scale (log-revenue).                                                                                          |
| **Likelihood** | ν             | Gamma(2, 0.1)    | `PRIOR_NU_ALPHA/BETA`              | Student-T degrees of freedom. Low ν (~3-10) = heavy tails for outlier robustness.                                               |

> [!TIP] > **Student-T Likelihood** (`USE_STUDENT_T=True`): Automatically down-weights extreme observations like Black Friday spikes.

### Control Variables

| Variable       | Description                                |
| -------------- | ------------------------------------------ |
| `trend`        | Normalized linear trend (0 to 1)           |
| `is_holiday`   | Binary holiday indicator                   |
| `month`        | Normalized month (1/12 to 12/12)           |
| `week_of_year` | Normalized week of year (1/52 to 52/52)    |
| `quarter`      | Normalized quarter (0.25 to 1.0)           |
| `is_q4`        | Binary Q4 indicator (holiday sales season) |

### MCMC Sampling

| Parameter     | Value                             |
| ------------- | --------------------------------- |
| Sampler       | PyMC (CPU with **Numba Backend**) |
| Chains        | 4                                 |
| Draws         | 2,000                             |
| Tune          | 1,500                             |
| Target Accept | 0.99                              |
| Max Treedepth | 15                                |

### Validation Strategy

- **Temporal Holdout**: Last 12 weeks held out for validation
- **Convergence Diagnostics**: R-hat < 1.01 (optimized via high `target_accept` and `max_treedepth`), ESS > 400, Zero divergences.
- **Metrics**: R², MAE, MAPE
- **Cross-Validation**:
  - **Ridge Baseline**: `TimeSeriesSplit` (5 folds) preventing data leakage.
  - **Hierarchical**: Expanding window CV (Module available in `src/validation.py`).

---

## Project Structure

```text
MMM-Figshare-eCommerce/
├── data/
│   ├── raw/                    # Original Figshare download
│   └── processed/              # Cleaned datasets
│       └── mmm_data.parquet    # Main dataset (31 MB)
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing demo
│   ├── 03_mmm_baseline.ipynb   # Baseline model development
│   ├── 04_hierarchical.ipynb   # Hierarchical model development
│   └── 05_optimization.ipynb   # Budget optimization workflow
│
├── scripts/
│   ├── run_pipeline.py        # Pipeline orchestrator CLI
│   ├── mmm_baseline.py        # Ridge Regression baseline
│   └── mmm_hierarchical.py    # Bayesian hierarchical model
│
├── src/
│   ├── config.py              # Centralized configuration (all hyperparameters)
│   ├── data_loader.py         # Data loading and validation
│   ├── preprocessing.py       # Adstock, saturation, feature engineering
│   ├── models/                # Model package
│   │   ├── __init__.py        # Package exports
│   │   └── hierarchical_bayesian.py  # Bayesian MMM with learned transforms
│   ├── evaluation.py          # Convergence, metrics, ROI computation
│   ├── insights.py            # Budget optimization, parameter extraction
│   ├── comparison.py          # Model comparison utilities
│   ├── schemas.py             # Pydantic data schemas
│   └── validation.py          # Expanding window CV module
│
├── app/
│   ├── dashboard.py           # Streamlit main entry point
│   ├── mlflow_loader.py       # MLflow data loading
│   ├── components/            # Reusable UI components
│   └── pages/                 # Dashboard pages
│       ├── 01_Executive_Summary.py
│       ├── 02_Budget_Optimization.py
│       ├── 03_Regional_Analysis.py
│       ├── 04_Model_Details.py
│       └── 05_Model_Comparison.py
│
├── tests/
│   ├── test_data_loader.py    # Data loading tests
│   ├── test_preprocessing.py  # Preprocessing tests
│   ├── test_model.py          # Model creation tests
│   ├── test_model_insights.py # Insights extraction tests
│   └── test_optimization.py   # Optimization tests
│
├── models/                     # Saved models and traces
├── mlruns/                     # MLflow experiment tracking
│
├── pyproject.toml             # Project dependencies
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
mamba env create -f environment.yml

# Activate environment with BLAS optimization
. .\activate_mmm.ps1
```

> **Nota**: O script `activate_mmm.ps1` ativa o ambiente Conda e configura a variável `PYTENSOR_FLAGS` para linkar o Intel MKL. Isso é necessário porque o PyTensor no Windows não detecta automaticamente as bibliotecas BLAS, e sem essa configuração as operações matriciais ficam severamente degradadas.

### Google Colab (Recommended for GPU)

See [docs/execution_guide.md](docs/execution_guide.md) for detailed instructions.

---

## Usage

### Pipeline Execution (Recommended)

```bash
# Run full pipeline
python scripts/run_pipeline.py

# Baseline only (2-4 hours)
python scripts/run_pipeline.py --baseline-only

# Hierarchical only (4-8 hours)
python scripts/run_pipeline.py --hierarchical-only

# Preview what would run
python scripts/run_pipeline.py --dry-run
```

### Direct Script Execution

```bash
# Baseline model (single holdout, ~15 min)
python scripts/mmm_baseline.py

# Baseline with expanding window CV (~45 min)
python scripts/mmm_baseline.py --cv

# Hierarchical model only
python scripts/mmm_hierarchical.py
```

See [docs/execution_guide.md](docs/execution_guide.md) for full CLI options.

### Output Files

After execution, results are saved to `models/`:

| File                       | Description               |
| -------------------------- | ------------------------- |
| `mmm_baseline_trace.nc`    | MCMC trace (ArviZ format) |
| `roi_baseline.csv`         | ROI by channel            |
| `metrics_baseline.json`    | Performance metrics       |
| `mmm_baseline_results.png` | Visualization             |

---

## Tech Stack

| Category            | Technologies                                 |
| ------------------- | -------------------------------------------- |
| Modeling            | PyMC-Marketing, PyMC, ArviZ                  |
| Experiment Tracking | MLflow                                       |
| Mode                | Numba (LLVM compilation)                     |
| Sampling            | PyMC (optimized via custom `fit_model`)      |
| Backend             | **Numba Mode** (activated via `.pytensorrc`) |
| Environment         | UV, Intel MKL                                |

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
- Google Colab for free GPU access

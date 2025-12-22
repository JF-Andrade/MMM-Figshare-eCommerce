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

### Validation

- Expanding window cross-validation (3 folds)
- Pydantic schema validation for all deliverables

### Region Usage

| Model              | Regions   | Selection Criteria                |
| ------------------ | --------- | --------------------------------- |
| **Ridge Baseline** | UK only   | Filtered to GBP currency          |
| **Hierarchical**   | All valid | All currencies, ≥52 weeks of data |

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

### Prior Distributions (Explicit)

| Parameter    | Distribution  | Meaning                                                                                              |
| ------------ | ------------- | ---------------------------------------------------------------------------------------------------- |
| **alpha**    | Beta(1, 1)    | Adstock decay rate. Equivalent to Uniform(0,1), no prior bias. Values close to 1 = longer carryover. |
| **lam (λ)**  | Gamma(3, 1)   | Saturation curve steepness. Higher values = more abrupt saturation.                                  |
| **beta (β)** | HalfNormal(2) | Channel effect magnitude. Larger = more revenue impact per spend.                                    |

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
| Target Accept | 0.98                              |
| Max Treedepth | 15                                |

### Validation Strategy

- **Temporal Holdout**: Last 12 weeks held out for validation
- **Convergence Diagnostics**: R-hat < 1.01 (optimized via high `target_accept` and `max_treedepth`), ESS > 400, Zero divergences.
- **Metrics**: R², MAE, MAPE
- **Expanding Window CV**: Module available in `src/validation.py`

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
│   ├── config.py              # Configuration (MLflow settings)
│   ├── data_loader.py         # Data loading and validation
│   ├── preprocessing.py       # Adstock, saturation, calendar features
│   ├── feature_engineering.py # Feature derivation functions
│   ├── model.py               # MMM model creation and fitting
│   ├── evaluation.py          # Convergence, metrics, ROI computation
│   ├── optimization.py        # Budget allocation optimizer
│   ├── model_insights.py      # Adstock/saturation parameter extraction
│   ├── comparison.py          # Model comparison utilities
│   ├── value_estimation.py    # Project value and ROI calculation
│   ├── project_pricing.py     # Market pricing estimates
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
│       ├── 05_Model_Comparison.py
│       └── 06_Project_Value.py
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

## Future Work

- [x] ~~Cross-validation with multiple temporal splits~~ (Expanding Window CV)
- [x] ~~Explicit prior distributions~~ (Prior class implemented)
- [x] ~~Modular src/ architecture~~ (data_loader, model, evaluation, optimization)
- [x] ~~Test suite~~ (pytest tests for all modules)
- [x] ~~Budget optimization module~~ (src/optimization.py)
- [x] ~~Ridge Regression baseline~~ (frequentist comparison)
- [x] ~~Interactive Streamlit dashboard~~ (6 pages implemented)
- [x] ~~Model comparison page~~ (baseline vs hierarchical)
- [x] ~~Project value estimation~~ (ROI calculation)
- [ ] Tune priors based on domain knowledge
- [ ] Add interaction effects between channels
- [ ] Improve model R² with additional features

---

## Author

**Jordao Fernandes de Andrade**

- Email: <jordaoandrade@gmail.com>
- LinkedIn: [linkedin.com/in/jordaoandrade](https://linkedin.com/in/jordaoandrade)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset provided by [Figshare](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841)
- PyMC-Marketing team for the excellent MMM framework
- Google Colab for free GPU access

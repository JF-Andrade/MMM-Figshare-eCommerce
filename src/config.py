# MMM Project Configuration
# Centralized configuration - ALL hyperparameters defined here

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# 1. GLOBAL CONSTANTS
# =============================================================================

SEED = 1991

# =============================================================================
# 2. INFRASTRUCTURE & PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_FILENAME = "conjura_mmm_data.csv"
PROCESSED_FILENAME = "mmm_data.parquet"

# =============================================================================
# 3. DATA SCHEMA
# =============================================================================

# Target Variable (Revenue)
TARGET_COL = "ALL_PURCHASES_ORIGINAL_PRICE"
REVENUE_COL = TARGET_COL  # Alias for clarity
RAW_DATE_COL = "DATE_DAY"   # Raw daily date column
RAW_REGION_COL = "TERRITORY_NAME" # Raw region column (e.g. territory)
DATE_COL = "week"           # Aggregated weekly date column
GEO_COL = "geo"             # Aggregated geometric column
MIN_WEEKS_PER_REGION = 52
DEFAULT_CURRENCY = "GBP"
RAW_CURRENCY_COL = "CURRENCY_CODE"

# Acquisition Metrics (for CAC/ROAS)
ACQUISITION_COLS = [
    "FIRST_PURCHASES",         # New Customers (for CAC)
    "ALL_PURCHASES"            # Total Transactions
]

# Spend Channels
SPEND_COLS = [
    "GOOGLE_PAID_SEARCH_SPEND",
    "GOOGLE_SHOPPING_SPEND",
    "GOOGLE_PMAX_SPEND",
    "GOOGLE_DISPLAY_SPEND",
    "GOOGLE_VIDEO_SPEND",
    "META_FACEBOOK_SPEND",
    "META_INSTAGRAM_SPEND",
    "META_OTHER_SPEND",
    "TIKTOK_SPEND",
]

# Channel Prefixes (for preprocessing)
CHANNELS = [
    "GOOGLE_PAID_SEARCH",
    "GOOGLE_SHOPPING",
    "GOOGLE_PMAX",
    "GOOGLE_DISPLAY",
    "GOOGLE_VIDEO",
    "META_FACEBOOK",
    "META_INSTAGRAM",
    "META_OTHER",
    "TIKTOK",
]

METRICS = ["SPEND", "CLICKS", "IMPRESSIONS"]

# Regions with holiday calendars
REGION_HOLIDAY_MAP = {
    "US": "US",
    "UK": "GB",
    "AU": "AU",
    "NL": "NL",
    "ES": "ES",
    "HK": "HK",
    "IE": "IE",
    "CA": "CA",
    "NZ": "NZ",
    "DE": "DE",
    "AT": "AT",
    "JP": "JP",
    "FR": "FR",
    "IT": "IT",
    "SE": "SE",
    "DK": "DK",
    "NO": "NO",
    "CH": "CH",
}

# Control Variables
CONTROL_COLS = ["trend", "is_holiday", "is_q4", "is_black_friday"]

# Seasonality Terms (Cyclic) - 1st and 2nd order Fourier harmonics
SEASON_COLS = [
    "WEEK_SIN", "WEEK_COS", "MONTH_SIN", "MONTH_COS",  # 1st order
    "WEEK_SIN_2", "WEEK_COS_2", "MONTH_SIN_2", "MONTH_COS_2",  # 2nd order
]

# Non-paid Traffic (Exogenous Demand)
TRAFFIC_COLS = [
    "DIRECT_CLICKS",
    "BRANDED_SEARCH_CLICKS",
    "ORGANIC_SEARCH_CLICKS",
    "EMAIL_CLICKS",
    "REFERRAL_CLICKS",
    "ALL_OTHER_CLICKS",
]

# Final Feature Set for Modeling
ALL_FEATURES = SPEND_COLS + TRAFFIC_COLS + CONTROL_COLS + SEASON_COLS

# Regional defaults (baseline model)
DEFAULT_CURRENCY = "GBP"
TARGET_TERRITORY = "UK"

# =============================================================================
# 4. PREPROCESSING & QUALITY
# =============================================================================

# Data Quality Filters
MIN_SPEND_THRESHOLD = 0.05
MIN_NONZERO_RATIO = 0.20
DEFAULT_IMPUTE_VALUE = 0.0     # Default value for missing data imputation
DEFAULT_LOG_OFFSET = 1.0       # Offset for log transformation (log(x + offset))

# Adstock & Saturation Defaults (for baseline functions)
DEFAULT_ADSTOCK_DECAY = 0.5          # Geometric decay rate
DEFAULT_HALF_SATURATION_PCT = 0.5    # Percentile for Hill function
DEFAULT_ADSTOCK_LMAX = 8             # Max lag for convolution-based adstock
DEFAULT_SATURATION_SLOPE = 2       # Hill function steepness (1=smooth, 2+=sharper S-curve)

# Feature Engineering Settings
YEARLY_SEASONALITY = 2  # Number of Fourier terms

# =============================================================================
# 5. VALIDATION STRATEGY
# =============================================================================

HOLDOUT_WEEKS = 8

# =============================================================================
# 6. BAYESIAN MODEL CONFIGURATION (Hierarchical)
# =============================================================================

# --- MCMC Settings ---
MCMC_CHAINS = 4
MCMC_DRAWS = 4000
MCMC_TUNE = 500
MCMC_TARGET_ACCEPT = 0.85
MCMC_MAX_TREEDEPTH = 12
MCMC_SAMPLER = "numpyro"          # Options: "pymc", "numpyro" (requires JAX)

# --- Priors: Adstock (Geometric) ---
L_MAX = 10                     # Maximum lag weeks
PRIOR_ADSTOCK_ALPHA = 2        # Beta(alpha, beta) for decay rate
PRIOR_ADSTOCK_BETA = 2         # Beta(alpha, beta) for decay rate
PRIOR_SIGMA_ADSTOCK_TERRITORY = 0.2  # Regional variation in adstock

# --- Priors: Saturation (Hill Function) ---
PRIOR_SATURATION_L_SIGMA = 0.3       # HalfNormal sigma for L (half-saturation). Calibrated for X_spend_norm in [0, 1]
PRIOR_SATURATION_K_ALPHA = 2         # Gamma(alpha, beta) for k (steepness). K > 1: S shape. K < 1: Inverted S shape.
PRIOR_SATURATION_K_BETA = 1          # Gamma(alpha, beta) for k (steepness). K > 1: S shape. K < 1: Inverted S shape.
PRIOR_SIGMA_SATURATION_TERRITORY = 0.2 # Regional variation in L

# --- Priors: Hierarchical Intercepts ---
PRIOR_SIGMA_TERRITORY = 0.5  # Variation between territories

# --- Priors: Channel Effects ---
PRIOR_BETA_CHANNEL_SIGMA = 0.5     # HalfNormal sigma for channel betas
PRIOR_SIGMA_BETA_TERRITORY = 0.05  # Regional variation in channel effects

# --- Priors: Feature Effects (Regularized Horseshoe - Piironen & Vehtari, 2017) --- Read in: https://arxiv.org/abs/1707.01694
# Note: tau0 is computed dynamically based on m0, D, and n in the model
PRIOR_HORSESHOE_M0 = 5          # Expected ~5 relevant features out of ~10
PRIOR_HORSESHOE_LAMBDA_BETA = 1 # HalfCauchy beta for local shrinkage

# --- Priors: Seasonality ---
PRIOR_GAMMA_SEASON_SIGMA = 0.3   # Normal sigma for seasonality

# --- Priors: Likelihood (Student-T) ---
PRIOR_SIGMA_OBS = 1.0  # HalfNormal sigma for observation noise. Calibrated for y_log with std ≈ 0.5-1.5
USE_STUDENT_T = True   # Use robust Student-T likelihood
PRIOR_NU_ALPHA = 2     # Gamma(alpha, beta) for degrees of freedom
PRIOR_NU_BETA = 0.5    # mean nu ≈ 4 (more robust to outliers)

# =============================================================================
# 7. BASELINE MODEL CONFIGURATION (Ridge Regression with Bayesian Optimization)
# =============================================================================

# Bayesian Hyperparameter Search (gp_minimize)
BAYESIAN_N_CALLS = 50
BAYESIAN_ADSTOCK_BOUNDS = (0.001, 0.5)   # log-uniform (conservative upper bound)
BAYESIAN_SATURATION_BOUNDS = (0.1, 0.5)   # increased lower bound to prevent binary-like behavior
BAYESIAN_ALPHA_BOUNDS = (0.1, 500.0)      # log-uniform

# Cross-Validation Settings
CV_GAP_WEEKS = 2  # Gap between train/test folds to prevent adstock leakage

# =============================================================================
# 8. EXPERIMENT TRACKING (MLflow)
# =============================================================================

MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = "MMM-Experiments"

# =============================================================================
# 9. UTILITIES & CONFIG CLASS
# =============================================================================

def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_run_id() -> str:
    """Generate unique run ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Paths
    raw_data_path: Path = field(default_factory=lambda: RAW_DATA_DIR / DATA_FILENAME)
    processed_data_path: Path = field(default_factory=lambda: PROCESSED_DATA_DIR / PROCESSED_FILENAME)
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    reports_dir: Path = field(default_factory=lambda: REPORTS_DIR)
    logs_dir: Path = field(default_factory=lambda: LOGS_DIR)

    # Metadata
    run_id: str = field(default_factory=generate_run_id)
    git_hash: str = field(default_factory=get_git_hash)

    # Columns
    target_column: str = TARGET_COL
    date_column: str = "DATE_DAY"
    region_column: str = "TERRITORY_NAME"
    channel_columns: list[str] = field(default_factory=lambda: SPEND_COLS.copy())

    # MCMC
    chains: int = MCMC_CHAINS
    draws: int = MCMC_DRAWS
    tune: int = MCMC_TUNE
    target_accept: float = MCMC_TARGET_ACCEPT
    max_treedepth: int = MCMC_MAX_TREEDEPTH

    # Validation
    holdout_weeks: int = HOLDOUT_WEEKS

    # Model
    baseline_region: str | None = None
    max_regions: int | None = None
    min_spend_threshold: float = MIN_SPEND_THRESHOLD
    use_custom_hierarchical: bool = True

    # Retry
    max_retries: int = 3
    retry_delay: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "run_id": self.run_id,
            "git_hash": self.git_hash,
            "processed_data_path": str(self.processed_data_path),
            "models_dir": str(self.models_dir),
            "target_column": self.target_column,
            "channels": len(self.channel_columns),
            "mcmc": {
                "chains": self.chains,
                "draws": self.draws,
                "tune": self.tune,
                "target_accept": self.target_accept,
            },
            "holdout_weeks": self.holdout_weeks,
        }

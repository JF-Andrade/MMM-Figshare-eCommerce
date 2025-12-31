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

# Target Variable
TARGET_COL = "ALL_PURCHASES_ORIGINAL_PRICE"
DATE_COL = "week"
GEO_COL = "geo"
MIN_WEEKS_PER_REGION = 52

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

# Share of Spend Columns
# TIKTOK_SHARE is dropped to avoid perfect multicollinearity
_ALL_SHARE_COLS = [c.replace("_SPEND", "_SHARE") for c in SPEND_COLS]
SHARE_COLS = _ALL_SHARE_COLS[:-1]  # Excludes TIKTOK_SHARE
SHARE_REFERENCE_COL = _ALL_SHARE_COLS[-1]  # "TIKTOK_SHARE"

# Control Variables (removed month, week_of_year, quarter - redundant with SEASON_COLS)
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

# Defined but excluded from the final model to avoid endogeneity (CTR, CPC) or data leakage (Customer metrics).
CTR_COLS = [
    "GOOGLE_PAID_SEARCH_CTR", "GOOGLE_SHOPPING_CTR", "GOOGLE_PMAX_CTR",
    "GOOGLE_DISPLAY_CTR", "GOOGLE_VIDEO_CTR", "META_FACEBOOK_CTR",
    "META_INSTAGRAM_CTR"
]

CPC_COLS = [
    "GOOGLE_PAID_SEARCH_CPC", "GOOGLE_SHOPPING_CPC", "GOOGLE_PMAX_CPC",
    "GOOGLE_DISPLAY_CPC", "GOOGLE_VIDEO_CPC", "META_FACEBOOK_CPC",
    "META_INSTAGRAM_CPC", "META_OTHER_CPC", "TIKTOK_CPC"
]

# Endogenous features explicitly excluded from modeling
ENDOGENOUS_COLS = CTR_COLS + CPC_COLS

# Final Feature Set for Modeling (excluding endogenous variables and SHARE_COLS)
# Note: SHARE_COLS removed - redundant with SPEND_COLS and causes multicollinearity
ALL_FEATURES = [
    col for col in (SPEND_COLS + TRAFFIC_COLS + CONTROL_COLS + SEASON_COLS)
    if col not in ENDOGENOUS_COLS
]

# Regional defaults (baseline model)
DEFAULT_CURRENCY = "GBP"
TARGET_TERRITORY = "UK"

# =============================================================================
# 4. PREPROCESSING & QUALITY
# =============================================================================

# Data Quality Filters
MIN_SPEND_THRESHOLD = 0.05
MIN_NONZERO_RATIO = 0.20

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
MCMC_DRAWS = 3000
MCMC_TUNE = 1500
MCMC_TARGET_ACCEPT = 0.85
MCMC_MAX_TREEDEPTH = 12
MCMC_SAMPLER = "numpyro"          # Options: "pymc", "numpyro" (requires JAX)

# --- Priors: Adstock (Geometric) ---
L_MAX =  6                     # Maximum lag weeks
PRIOR_ADSTOCK_ALPHA = 2        # Beta(alpha, beta) for decay rate
PRIOR_ADSTOCK_BETA = 2
PRIOR_SIGMA_ADSTOCK_TERRITORY = 0.2  # Regional variation in adstock (M1: increased from 0.1)

# --- Priors: Saturation (Hill Function) ---
PRIOR_SATURATION_L_SIGMA = 0.3       # HalfNormal sigma for L (half-saturation)
                                      # Calibrated for X_spend_norm in [0, 1]
PRIOR_SATURATION_K_ALPHA = 2         # Gamma(alpha, beta) for k (steepness)
PRIOR_SATURATION_K_BETA = 1
PRIOR_SIGMA_SATURATION_TERRITORY = 0.2 # Regional variation in L (M5: increased from 0.1)

# --- Priors: Hierarchical Intercepts ---
# PRIOR_SIGMA_CURRENCY removed: currency hierarchy eliminated (each territory -> one currency)
PRIOR_SIGMA_TERRITORY = 0.5  # Variation between territories (increased to absorb currency effect)

# --- Priors: Channel Effects ---
PRIOR_BETA_CHANNEL_SIGMA = 0.5     # HalfNormal sigma for channel betas
PRIOR_SIGMA_BETA_TERRITORY = 0.05  # Regional variation in channel effects

# --- Priors: Feature Effects (Regularized Horseshoe - Piironen & Vehtari, 2017) --- Lido em https://arxiv.org/abs/1707.01694
# Higher m0 = weaker regularization, lower m0 = stronger regularization
PRIOR_HORSESHOE_M0 = 5  # Expected ~5 relevant features out of ~10
PRIOR_HORSESHOE_LAMBDA_BETA = 1  # HalfCauchy beta for local shrinkage
# Note: tau0 is computed dynamically based on m0, D, and n in the model

# --- Priors: Seasonality ---
PRIOR_GAMMA_SEASON_SIGMA = 0.3   # Normal sigma for seasonality

# --- Priors: Likelihood (Student-T) ---
PRIOR_SIGMA_OBS = 1.0  # HalfNormal sigma for observation noise
                       # Calibrated for y_log with std ≈ 0.5-1.5
USE_STUDENT_T = True   # Use robust Student-T likelihood
PRIOR_NU_ALPHA = 2     # Gamma(alpha, beta) for degrees of freedom
PRIOR_NU_BETA = 0.5    # C1: Changed from 0.1 to 0.5 -> mean nu ≈ 4 (more robust to outliers)

# =============================================================================
# 7. BASELINE MODEL CONFIGURATION (Ridge Regression with Bayesian Optimization)
# =============================================================================

RIDGE_ALPHAS = [0.1, 1, 10, 50, 100, 500]  # Fallback for non-Bayesian

# Bayesian Hyperparameter Search (gp_minimize)
BAYESIAN_N_CALLS = 50
BAYESIAN_ADSTOCK_BOUNDS = (0.001, 0.85)   # Extended for brand/awareness channels
BAYESIAN_SATURATION_BOUNDS = (0.01, 0.5)  # log-uniform
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

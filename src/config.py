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

# Share Columns (from Spend)
SHARE_COLS = [c.replace("_SPEND", "_SHARE") for c in SPEND_COLS]

# Control Variables
CONTROL_COLS = ["trend", "is_holiday", "month", "week_of_year", "quarter", "is_q4"]

# Seasonality Terms (Fourier)
SEASON_COLS = ["sin_1", "cos_1", "sin_2", "cos_2"]

# Non-paid Traffic (Exogenous Demand)
TRAFFIC_COLS = [
    "DIRECT_CLICKS",
    "BRANDED_SEARCH_CLICKS",
    "ORGANIC_SEARCH_CLICKS",
    "EMAIL_CLICKS",
    "REFERRAL_CLICKS",
    "ALL_OTHER_CLICKS",
]

# NOTE: The following feature sets are defined but excluded from the final model 
# to avoid endogeneity (CTR, CPC) or data leakage (Customer metrics).
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

# Final Feature Set for Modeling
ALL_FEATURES = (
    SPEND_COLS + SHARE_COLS + TRAFFIC_COLS + 
    CONTROL_COLS + SEASON_COLS
)

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

HOLDOUT_WEEKS = 12

# 3 Fold Cross-Validation Configuration
CV_ENABLED = True
CV_MODE = "full"       # "full" = K independent training runs
CV_FOLDS = 3
CV_MIN_TRAIN_WEEKS = 52  # Minimum training weeks for first fold
CV_TEST_WEEKS = 12       # Fixed test window per fold
CV_SAVE_INTERMEDIATE = True  # Save idata per fold to disk
CV_CHECKPOINT_DIR = MODELS_DIR / "cv_checkpoints"
CV_RESUME_FROM_FOLD = 0  # Set > 0 to resume interrupted CV

# =============================================================================
# 6. BAYESIAN MODEL CONFIGURATION (Hierarchical)
# =============================================================================

# --- MCMC Settings ---
MCMC_CHAINS = 2
MCMC_DRAWS = 50
MCMC_TUNE = 50
MCMC_TARGET_ACCEPT = 0.85
MCMC_MAX_TREEDEPTH = 18

# --- Priors: Adstock (Geometric) ---
L_MAX = 12                     # Maximum lag weeks
PRIOR_ADSTOCK_ALPHA = 2        # Beta(alpha, beta) for decay rate
PRIOR_ADSTOCK_BETA = 2
PRIOR_SIGMA_ADSTOCK_TERRITORY = 0.1  # Regional variation in adstock

# --- Priors: Saturation (Hill Function) ---
PRIOR_SATURATION_L_SIGMA = 1.0       # HalfNormal sigma for L (half-saturation)
PRIOR_SATURATION_K_ALPHA = 2         # Gamma(alpha, beta) for k (steepness)
PRIOR_SATURATION_K_BETA = 1
PRIOR_SIGMA_SATURATION_TERRITORY = 0.1 # Regional variation in L

# --- Priors: Hierarchical Intercepts ---
PRIOR_SIGMA_CURRENCY = 0.5   # Variation between currencies
PRIOR_SIGMA_TERRITORY = 0.3  # Variation between territories within currency

# --- Priors: Channel Effects ---
PRIOR_BETA_CHANNEL_SIGMA = 0.5     # HalfNormal sigma for channel betas
PRIOR_SIGMA_BETA_TERRITORY = 0.05  # Regional variation in channel effects

# --- Priors: Feature Effects (Horseshoe Regularization) ---
PRIOR_HORSESHOE_TAU_BETA = 1     # HalfCauchy beta for global shrinkage
PRIOR_HORSESHOE_LAMBDA_BETA = 1  # HalfCauchy beta for local shrinkage

# --- Priors: Seasonality ---
PRIOR_GAMMA_SEASON_SIGMA = 0.3   # Normal sigma for seasonality

# --- Priors: Likelihood (Student-T) ---
PRIOR_SIGMA_OBS = 0.5  # HalfNormal sigma for observation noise
USE_STUDENT_T = True   # Use robust Student-T likelihood
PRIOR_NU_ALPHA = 2     # Gamma(alpha, beta) for degrees of freedom
PRIOR_NU_BETA = 0.1

# =============================================================================
# 7. BASELINE MODEL CONFIGURATION (freq. Ridge)
# =============================================================================

RIDGE_ALPHAS = [0.1, 1, 5, 10, 25, 40, 45, 50, 100, 500]
BASELINE_ADSTOCK_DECAY = [0.1, 0.3, 0.5, 0.7, 0.9]
BASELINE_SATURATION_HALF = [0.1, 0.3, 0.5, 0.7]

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

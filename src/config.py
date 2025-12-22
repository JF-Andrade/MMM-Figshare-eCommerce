# MMM Project Configuration

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pymc_marketing.prior import Prior

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# File names
DATA_FILENAME = "conjura_mmm_data.csv"
PROCESSED_FILENAME = "mmm_data.parquet"

# =============================================================================
# MMM Model Configuration
# =============================================================================

# Random seed for reproducibility
SEED = 1991

# Holdout for temporal validation
HOLDOUT_WEEKS = 12

# Minimum spend threshold (aggregate channels with < 5% of total spend)
MIN_SPEND_THRESHOLD = 0.05

# Minimum ratio of non-zero values for channel inclusion
MIN_NONZERO_RATIO = 0.20  # Channels with >80% zeros are filtered out

# Cross-validation settings (expanding window)
CV_FOLDS = 3           # Number of CV folds
CV_MIN_TRAIN = 52      # Minimum training weeks for first fold
CV_TEST_SIZE = 12      # Test size per fold (same as HOLDOUT_WEEKS)

# Currency to use (GBP has most data)
DEFAULT_CURRENCY = "GBP"

# Target territory (None = aggregate all territories with DEFAULT_CURRENCY)
# Set to specific territory name (e.g., "UK") for single-region analysis
TARGET_TERRITORY = "UK"

# Adstock settings
L_MAX = 12  # Maximum lag weeks for carryover effect (increased from 8)

# Yearly seasonality Fourier terms
YEARLY_SEASONALITY = 2

# =============================================================================
# Prior Distributions (PyMC-Marketing)
# =============================================================================
# O conhecimento de negócio pode ajudar a definir os priors. Caso não se tenham hipóteses sobre eles, pode-se utilizar dist. uniforme.

ADSTOCK_PRIORS = {
    # alpha: decay rate, values close to 0 = fast decay, close to 1 = slow decay
    "alpha": Prior("Beta", alpha=1, beta=1),
}

SATURATION_PRIORS = {
    # lam: controls saturation curve steepness
    # Gamma(3, 1) prior allows moderate saturation (mean = 3)
    "lam": Prior("Gamma", alpha=3, beta=1),
    # beta: channel effect magnitude
    # HalfNormal(2) prior allows positive effects with moderate magnitude
    "beta": Prior("HalfNormal", sigma=2),
}

# =============================================================================
# MCMC Sampling Configuration
# =============================================================================

MCMC_CHAINS = 4
MCMC_DRAWS = 2000
MCMC_TUNE = 1500
MCMC_TARGET_ACCEPT = 0.99
MCMC_MAX_TREEDEPTH = 15

# =============================================================================
# Channel Columns
# =============================================================================

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

# Target column (using ORIGINAL_PRICE to avoid data leakage from GROSS_DISCOUNT)
TARGET_COL = "ALL_PURCHASES_ORIGINAL_PRICE"

# Date column (after weekly aggregation)
DATE_COL = "week"

# Control columns for the model
CONTROL_COLS = ["trend", "is_holiday", "month", "week_of_year", "quarter", "is_q4"]

# Engineered features (57 features in total including channels and controls)
# Efficiency (CTR)
CTR_COLS = [
    "GOOGLE_PAID_SEARCH_CTR", "GOOGLE_SHOPPING_CTR", "GOOGLE_PMAX_CTR",
    "GOOGLE_DISPLAY_CTR", "GOOGLE_VIDEO_CTR", "META_FACEBOOK_CTR",
    "META_INSTAGRAM_CTR"
]

# Cost (CPC)
CPC_COLS = [
    "GOOGLE_PAID_SEARCH_CPC", "GOOGLE_SHOPPING_CPC", "GOOGLE_PMAX_CPC",
    "GOOGLE_DISPLAY_CPC", "GOOGLE_VIDEO_CPC", "META_FACEBOOK_CPC",
    "META_INSTAGRAM_CPC", "META_OTHER_CPC", "TIKTOK_CPC"
]

# Rolling/Momentum (7D Mean/Std)
ROLLING_COLS = [
    f"{c}_ROLLING_7D_MEAN" for c in SPEND_COLS
] + [
    f"{c}_ROLLING_7D_STD" for c in SPEND_COLS
]

# Share of Spend
SHARE_COLS = [
    c.replace("_SPEND", "_SHARE") for c in SPEND_COLS
]

# Customer behavior
CUSTOMER_COLS = [
    "AVG_ORDER_VALUE", "UNITS_PER_ORDER", "NEW_CUSTOMER_RATIO", "DISCOUNT_RATE"
]

# Seasonality Fourier terms
SEASON_COLS = ["sin_1", "cos_1", "sin_2", "cos_2"]

ALL_FEATURES = (
    SPEND_COLS + CONTROL_COLS + CTR_COLS + CPC_COLS + 
    ROLLING_COLS + SHARE_COLS + CUSTOMER_COLS + SEASON_COLS
)

# Legacy preprocessing parameters (kept for compatibility)
ADSTOCK_DECAY = 0.5
SATURATION_HALF_PCT = 0.5
LAG_PERIODS = [1, 2, 3, 7]

# =============================================================================
# Baseline (Ridge Regression) Configuration
# =============================================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
BASELINE_ADSTOCK_DECAY = 0.5
BASELINE_SATURATION_HALF = 0.5

# MLflow configuration
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()  # file:// URI for cross-platform compatibility
MLFLOW_EXPERIMENT_NAME = "MMM-Experiments"


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

    # Run metadata
    run_id: str = field(default_factory=generate_run_id)
    git_hash: str = field(default_factory=get_git_hash)

    # Target and columns
    target_column: str = TARGET_COL  # Uses config constant
    date_column: str = "DATE_DAY"
    region_column: str = "TERRITORY_NAME"

    channel_columns: list[str] = field(
        default_factory=lambda: [
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
    )

    # MCMC settings (production defaults)
    chains: int = 4
    draws: int = 2000
    tune: int = 1500
    target_accept: float = 0.98
    max_treedepth: int = 15

    # Holdout for validation
    holdout_weeks: int = 12

    # Model settings
    baseline_region: str | None = None  # Auto-select largest if None
    max_regions: int | None = None  # All valid regions if None
    min_spend_threshold: float = 0.01  # Exclude channels with < 1% of total
    use_custom_hierarchical: bool = True  # Use the custom nested model

    # Retry settings
    max_retries: int = 3
    retry_delay: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
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

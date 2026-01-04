"""
Core mathematical transformations and feature engineering utilities.

Low-level functions for MMM data processing.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_LOG_OFFSET,
    DEFAULT_ADSTOCK_DECAY,
    DEFAULT_ADSTOCK_LMAX,
    DEFAULT_SATURATION_SLOPE,
    DEFAULT_IMPUTE_VALUE,
    REGION_HOLIDAY_MAP,
    RAW_REGION_COL,
    RAW_CURRENCY_COL,
    RAW_DATE_COL,
    GEO_COL,
    MIN_NONZERO_RATIO,
)

# =============================================================================
# 1. CORE MATH TRANSFORMATIONS
# =============================================================================

# Used by: Both models (target transformation)
def log_transform(
    x: NDArray[np.floating] | pd.Series,
    offset: float = DEFAULT_LOG_OFFSET,
) -> NDArray[np.floating]:
    """Apply log(1 + x) transformation."""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.log(x_arr + offset)


# Used by: Both models (predictions back-transformation)
def inverse_log_transform(
    x: NDArray[np.floating] | pd.Series,
    offset: float = DEFAULT_LOG_OFFSET,
) -> NDArray[np.floating]:
    """Inverse of log transform."""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.exp(x_arr) - offset


# Used by: Baseline Ridge model (prepare_baseline_features)
def apply_adstock(
    x: NDArray[np.floating] | pd.Series,
    decay: float = DEFAULT_ADSTOCK_DECAY,
    l_max: int = DEFAULT_ADSTOCK_LMAX,
) -> NDArray[np.floating]:
    """Apply geometric adstock transformation using convolution."""
    weights = decay ** np.arange(l_max)
    x_arr = np.asarray(x, dtype=np.float64)
    adstocked = np.convolve(x_arr, weights, mode='full')[:len(x_arr)]
    return adstocked


# Used by: Baseline Ridge model (prepare_baseline_features)
def apply_saturation_with_max(
    x: NDArray[np.floating] | pd.Series,
    x_max: float,
    half_saturation: float,
    slope: float = DEFAULT_SATURATION_SLOPE,
) -> NDArray[np.floating]:
    """Apply Hill saturation using pre-computed max."""
    x_arr = np.asarray(x, dtype=np.float64)
    x_arr = np.maximum(x_arr, 0)
    x_norm = x_arr / (x_max + 1e-8)
    
    numerator = x_norm**slope
    denominator = half_saturation**slope + x_norm**slope
    
    return numerator / denominator


# =============================================================================
# 2. FEATURE ENGINEERING UTILITIES
# =============================================================================

# Used by: Both models (preprocessing)
def impute_missing_values(
    df: pd.DataFrame,
    fill_value: float = DEFAULT_IMPUTE_VALUE,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Impute missing values in specified columns."""
    result = df.copy()

    if columns is None:
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(fill_value)
    else:
        for col in columns:
            result[col] = result[col].fillna(fill_value)

    return result


# Used by: Both models (channel selection)
def filter_low_variance_channels(
    df: pd.DataFrame,
    spend_cols: list[str],
    min_nonzero_ratio: float = MIN_NONZERO_RATIO,
    verbose: bool = True,
) -> list[str]:
    """Filter out channels with insufficient variance."""
    existing_cols = [c for c in spend_cols if c in df.columns]
    nonzero_ratios = (df[existing_cols] > 0).mean()
    valid_channels = nonzero_ratios[nonzero_ratios >= min_nonzero_ratio].index.tolist()

    if verbose:
        dropped = set(existing_cols) - set(valid_channels)
        for c in dropped:
            print(f"Dropping {c}: {nonzero_ratios[c]:.1%} non-zero (min: {min_nonzero_ratio:.0%})")

    return valid_channels


# Used by: Both models (calendar features)
def add_seasonality_features(
    df: pd.DataFrame,
    date_col: str = RAW_DATE_COL,
) -> pd.DataFrame:
    """Add cyclic seasonality features (Sine/Cosine)."""
    df = df.copy()
    date = pd.to_datetime(df[date_col])
    
    # Week of Year (Period = 52)
    week = date.dt.isocalendar().week
    df["WEEK_SIN"] = np.sin(2 * np.pi * week / 52)
    df["WEEK_COS"] = np.cos(2 * np.pi * week / 52)
    df["WEEK_SIN_2"] = np.sin(4 * np.pi * week / 52)
    df["WEEK_COS_2"] = np.cos(4 * np.pi * week / 52)

    # Month (Period = 12)
    month = date.dt.month
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)
    df["MONTH_SIN_2"] = np.sin(4 * np.pi * month / 12)
    df["MONTH_COS_2"] = np.cos(4 * np.pi * month / 12)
    
    return df


# Used by: Both models (event flags)
def add_event_features(
    df: pd.DataFrame,
    date_col: str = RAW_DATE_COL,
) -> pd.DataFrame:
    """Add event features (Holidays, Black Friday, Q4)."""
    df = df.copy()
    date = pd.to_datetime(df[date_col])

    df["is_q4"] = (date.dt.quarter == 4).astype(int)
    df["is_black_friday"] = ((date.dt.month == 11) & (date.dt.day >= 23) & (date.dt.day <= 29)).astype(int)

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    if RAW_REGION_COL in df.columns:
        years = date.dt.year.unique()
        present_regions = df[RAW_REGION_COL].unique()
        
        for region in present_regions:
            country_code = REGION_HOLIDAY_MAP.get(region)
            if not country_code:
                continue
            try:
                country_hols = holidays.country_holidays(country_code, years=years)
                mask = df[RAW_REGION_COL] == region
                
                # Handle timezone if present
                region_dates = date[mask]
                if region_dates.dt.tz is not None:
                     is_hol = region_dates.dt.date.isin(country_hols).astype(int)
                else:
                     is_hol = region_dates.dt.date.isin(country_hols).astype(int)
                
                df.loc[mask, "is_holiday"] = is_hol
            except Exception:
                continue
    
    return df


# Used by: Hierarchical Bayesian model (currency normalization)
def normalize_spend_by_currency(
    df: pd.DataFrame,
    spend_cols: list[str],
    currency_col: str = RAW_CURRENCY_COL,
) -> pd.DataFrame:
    """Normalize spend features within each currency."""
    df = df.copy()
    for col in spend_cols:
        df[f"{col}_norm"] = df.groupby(currency_col)[col].transform(
            lambda x: x / (x.max() + 1e-8)
        )
    return df


# Used by: Hierarchical Bayesian model (territory indexing)
def create_hierarchy_indices(
    df: pd.DataFrame,
    geo_col: str = GEO_COL,
) -> tuple[NDArray, list[str]]:
    """Create integer indices for hierarchical model."""
    territory_cat = pd.Categorical(df[geo_col])
    territory_idx = territory_cat.codes
    territory_names = territory_cat.categories.tolist()
    return territory_idx, territory_names

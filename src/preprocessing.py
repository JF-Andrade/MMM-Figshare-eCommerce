"""
Preprocessing functions for MMM project.

Implements transformations:
- Adstock transformation
- Saturation (Hill function)
- Calendar features
- Seasonality (Fourier terms)
- Missing value imputation

ORGANIZATION:
1. Core Math Transformations (Shared Logic)
2. Feature Engineering Utilities (Shared Logic)
3. Data Preparation Pipelines (Shared Logic)
4. Baseline Model Specific Functions
5. Hierarchical Model Specific Functions
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import holidays
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.config import (
    REGION_HOLIDAY_MAP,
    CHANNELS,
    METRICS,
    MIN_NONZERO_RATIO,
    MIN_SPEND_THRESHOLD,
    DEFAULT_ADSTOCK_DECAY,
    DEFAULT_ADSTOCK_LMAX,
    DEFAULT_LOG_OFFSET,
    DEFAULT_SATURATION_SLOPE,
    DEFAULT_IMPUTE_VALUE,
    RAW_DATE_COL,
    RAW_REGION_COL,
    DATE_COL,
    GEO_COL,
    RAW_CURRENCY_COL,
    TARGET_COL,
    SPEND_COLS,
    TRAFFIC_COLS,
    CONTROL_COLS,
    SEASON_COLS,
    L_MAX,
    HOLDOUT_WEEKS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    PROJECT_ROOT,
    ALL_FEATURES,
    EPSILON,
)


# =============================================================================
# 1. CORE MATH TRANSFORMATIONS (Shared Logic)
# 
# Used by: Baseline & Hierarchical
# Purpose: Low-level mathematical operations (log, adstock, saturation).
# =============================================================================


def log_transform(
    x: NDArray[np.floating] | pd.Series,
    offset: float = DEFAULT_LOG_OFFSET,
) -> NDArray[np.floating]:
    """
    Apply log(1 + x) transformation to handle zeros and reduce skewness.

    Args:
        x: Input values.
        offset: Value added before log to handle zeros.

    Returns:
        Log-transformed values.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.log(x_arr + offset)


def inverse_log_transform(
    x: NDArray[np.floating] | pd.Series,
    offset: float = DEFAULT_LOG_OFFSET,
) -> NDArray[np.floating]:
    """
    Inverse of log transform for predictions.

    Args:
        x: Log-transformed values.
        offset: Same offset used in log_transform.

    Returns:
        Original scale values.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.exp(x_arr) - offset


def apply_adstock(
    x: NDArray[np.floating] | pd.Series,
    decay: float = DEFAULT_ADSTOCK_DECAY,
    l_max: int = DEFAULT_ADSTOCK_LMAX,
) -> NDArray[np.floating]:
    """
    Apply geometric adstock transformation using convolution (vectorized).

    Adstock models the carryover effect of advertising.

    Args:
        x: Spend or media variable array.
        decay: Decay rate between 0 and 1. Higher = longer carryover.
        l_max: Maximum lag window (default=DEFAULT_ADSTOCK_LMAX).

    Returns:
        Adstocked values.
    """
    weights = decay ** np.arange(l_max)
    # Ensure typed array
    x_arr = np.asarray(x, dtype=np.float64)
    adstocked = np.convolve(x_arr, weights, mode='full')[:len(x_arr)]
    return adstocked


def apply_saturation_with_max(
    x: NDArray[np.floating] | pd.Series,
    x_max: float,
    half_saturation: float,
    slope: float = DEFAULT_SATURATION_SLOPE,
) -> NDArray[np.floating]:
    """
    Apply Hill saturation using pre-computed max (avoids data leakage).
    
    Use for train/test splits where max should come from train only.
    
    Args:
        x: Input values (typically adstocked spend).
        x_max: Pre-computed maximum (from training data).
        half_saturation: Point where response reaches 50% of max.
        slope: Controls steepness of curve.
    
    Returns:
        Saturated values between 0 and 1.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_arr = np.maximum(x_arr, 0)
    x_norm = x_arr / (x_max + 1e-8)
    
    numerator = x_norm**slope
    denominator = half_saturation**slope + x_norm**slope
    
    return numerator / denominator


# =============================================================================
# 2. FEATURE ENGINEERING UTILITIES (Shared Logic)
# 
# Used by: Baseline & Hierarchical
# Purpose: Helpers for cleaning, imputation, and feature generation.
# =============================================================================


def impute_missing_values(
    df: pd.DataFrame,
    fill_value: float = DEFAULT_IMPUTE_VALUE,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Impute missing values in specified columns.

    TikTok data is 99.5% missing due to late channel launch.
    Impute with zeros for pre-launch period.

    Args:
        df: Input DataFrame.
        fill_value: Value to fill missing data with.
        columns: Specific columns to impute. If None, imputes all numeric.

    Returns:
        DataFrame with imputed values.
    """
    result = df.copy()

    if columns is None:
        # Impute all numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(fill_value)
    else:
        for col in columns:
            result[col] = result[col].fillna(fill_value)

    return result


def filter_low_variance_channels(
    df: pd.DataFrame,
    spend_cols: list[str],
    min_nonzero_ratio: float = MIN_NONZERO_RATIO,
    verbose: bool = True,
) -> list[str]:
    """
    Filter out channels with insufficient variance (too many zeros).

    Args:
        df: DataFrame with spend columns.
        spend_cols: List of spend column names to evaluate.
        min_nonzero_ratio: Minimum ratio of non-zero values (default 0.20).
        verbose: Print filtering info.

    Returns:
        List of channel names that pass the variance filter.
    """
    existing_cols = [c for c in spend_cols if c in df.columns]

    # Calculate non-zero ratios
    nonzero_ratios = (df[existing_cols] > 0).mean()

    # Identify valid channels
    valid_channels = nonzero_ratios[nonzero_ratios >= min_nonzero_ratio].index.tolist()

    if verbose:
        dropped = set(existing_cols) - set(valid_channels)
        for c in dropped:
            print(f"Dropping {c}: {nonzero_ratios[c]:.1%} non-zero (min: {min_nonzero_ratio:.0%})")

    return valid_channels


def apply_adstock_per_territory(
    df: pd.DataFrame,
    spend_cols: list[str],
    geo_col: str = GEO_COL,
    date_col: str = DATE_COL,
    alpha: float = DEFAULT_ADSTOCK_DECAY,
    l_max: int = DEFAULT_ADSTOCK_LMAX,
) -> pd.DataFrame:
    """
    Apply geometric adstock transformation per territory.
    
    Critical for panel data: adstock must be computed within each territory's
    time series, not across the concatenated DataFrame.
    """
    df = df.copy()
    
    # Ensure sorted by territory and date
    df = df.sort_values([geo_col, date_col]).reset_index(drop=True)
    
    for col in spend_cols:
        adstock_col = f"{col}_adstock"
        df[adstock_col] = 0.0

        for territory in df[geo_col].unique():
            mask = df[geo_col] == territory
            x = df.loc[mask, col].values
            df.loc[mask, adstock_col] = apply_adstock(x, decay=alpha, l_max=l_max)
    
    return df


def add_seasonality_features(
    df: pd.DataFrame,
    date_col: str = RAW_DATE_COL,
) -> pd.DataFrame:
    """Add cyclic seasonality features (Sine/Cosine)."""
    df = df.copy()
    date = pd.to_datetime(df[date_col])
    
    # Week of Year (Period = 52) - 1st Order
    week = date.dt.isocalendar().week
    df["WEEK_SIN"] = np.sin(2 * np.pi * week / 52)
    df["WEEK_COS"] = np.cos(2 * np.pi * week / 52)

    # Week of Year - 2nd Order (captures bi-annual patterns)
    df["WEEK_SIN_2"] = np.sin(4 * np.pi * week / 52)
    df["WEEK_COS_2"] = np.cos(4 * np.pi * week / 52)

    # Month (Period = 12) - 1st Order
    month = date.dt.month
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    # Month - 2nd Order (captures bi-monthly patterns)
    df["MONTH_SIN_2"] = np.sin(4 * np.pi * month / 12)
    df["MONTH_COS_2"] = np.cos(4 * np.pi * month / 12)
    
    return df


def add_event_features(
    df: pd.DataFrame,
    date_col: str = RAW_DATE_COL,
) -> pd.DataFrame:
    """Add event features (Holidays, Black Friday, Q4)."""
    df = df.copy()
    date = pd.to_datetime(df[date_col])

    # Quarter features (for Q4 sales boost)
    df["is_q4"] = (date.dt.quarter == 4).astype(int)

    # Black Friday Indicator (Nov 23-29 window)
    df["is_black_friday"] = ((date.dt.month == 11) & (date.dt.day >= 23) & (date.dt.day <= 29)).astype(int)

    # Holiday Indicator (Region-specific)
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    # Calculate holidays
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
                region_dates = date[mask]
                
                if region_dates.dt.tz is not None:
                    is_hol = region_dates.dt.date.isin(country_hols).astype(int)
                else:
                    is_hol = region_dates.dt.date.isin(country_hols).astype(int)
                    
                df.loc[mask, "is_holiday"] = is_hol
                
            except Exception:
                continue
    
    return df


def normalize_spend_by_currency(
    df: pd.DataFrame,
    spend_cols: list[str],
    currency_col: str = RAW_CURRENCY_COL,
) -> pd.DataFrame:
    """
    Normalize spend features within each currency (0-1 scaling).
    
    Prevents scale issues when combining territories with different currencies.
    """
    df = df.copy()
    
    for col in spend_cols:
        df[f"{col}_norm"] = df.groupby(currency_col)[col].transform(
            lambda x: x / (x.max() + 1e-8)
        )
    
    return df


def create_hierarchy_indices(
    df: pd.DataFrame,
    geo_col: str = GEO_COL,
) -> tuple["NDArray", list[str]]:
    """
    Create integer indices for hierarchical model.
    """
    # Territory indexing
    territory_cat = pd.Categorical(df[geo_col])
    territory_idx = territory_cat.codes
    territory_names = territory_cat.categories.tolist()
    
    return territory_idx, territory_names


# =============================================================================
# 3. DATA PREPARATION PIPELINES (Shared Logic)
# 
# Used by: Baseline & Hierarchical
# Purpose: High-level data aggregation and transformation pipelines.
# =============================================================================


def prepare_weekly_data(
    df: pd.DataFrame,
    region: str | None = None,
    date_col: str = RAW_DATE_COL,
    target_col: str | None = None,
    spend_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate daily data to weekly for MMM.

    Used by both baseline and hierarchical scripts.

    Args:
        df: Daily DataFrame with DATE_DAY column.
        region: Optional region filter. If None, aggregates all.
        date_col: Name of date column.
        target_col: Target variable column. If None, uses config.TARGET_COL.
        spend_cols: Spend columns to aggregate.

    Returns:
        Weekly aggregated DataFrame with trend and seasonality features.
    """
    if target_col is None:
        target_col = TARGET_COL
    
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col])

    if region:
        df = df[df[RAW_REGION_COL] == region].copy()

    df["week"] = df["date"].dt.to_period("W").dt.start_time

    if spend_cols is None:
        spend_cols = SPEND_COLS

    # 1. Start with REQUIRED columns
    agg_dict = {target_col: "sum"}

    # 2. Add OPTIONAL columns
    candidates = spend_cols + TRAFFIC_COLS + ["ALL_PURCHASES", "ALL_PURCHASES_ORIGINAL_PRICE"]
    for col in candidates:
        if col in df.columns:
            agg_dict[col] = "sum"

    # 3. Compute Daily Features FIRST to capture events correctly
    df = add_event_features(df, date_col="date")

    # 4. Add Control Variables to Aggregation
    control_vars = ["is_holiday", "is_black_friday", "is_q4"]
    for col in control_vars:
        if col in df.columns:
            agg_dict[col] = "max"

    df_weekly = df.groupby("week").agg(agg_dict).reset_index()
    df_weekly = df_weekly.sort_values("week").reset_index(drop=True)

    # Trend feature (normalized 0 to 1)
    df_weekly["trend"] = np.arange(len(df_weekly)) / len(df_weekly)

    # Compute seasonality (Cyclic features only)
    # No longer overwrites event flags!
    df_weekly = add_seasonality_features(df_weekly, date_col="week")

    # Log-transform target for better model fit
    df_weekly[f"{target_col}_log"] = np.log1p(df_weekly[target_col])

    # Normalize spend columns per channel
    for col in spend_cols:
        if col in df_weekly.columns:
            max_val = df_weekly[col].max()
            if max_val > 0:
                df_weekly[f"{col}_norm"] = df_weekly[col] / max_val

    return df_weekly


# =============================================================================
# 4. BASELINE MODEL SPECIFIC
# 
# Used by: Ridge Regression (mmm_baseline.py)
# Purpose: Feature generation specific to simple regression models.
# =============================================================================


def prepare_baseline_features(
    df_weekly: pd.DataFrame,
    adstock_decay: float,
    saturation_half: float,
    spend_cols: list[str],
    target_col: str,
    min_nonzero_ratio: float = MIN_NONZERO_RATIO,
    min_spend_threshold: float = MIN_SPEND_THRESHOLD,
    train_end_idx: int | None = None,
    control_cols: list[str] | None = None,
    season_cols: list[str] | None = None,
    traffic_cols: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str], float, dict, list[str]]:
    """
    Prepare features with adstock and saturation transforms for baseline model.

    Args:
        df_weekly: Weekly aggregated DataFrame.
        adstock_decay: Decay rate for adstock.
        saturation_half: Half-saturation point.
        spend_cols: List of spend column names.
        target_col: Target column name.
        min_nonzero_ratio: Minimum non-zero ratio for channels.
        min_spend_threshold: Minimum spend share threshold.
        train_end_idx: Index where training data ends (prevents leakage).
        verbose: Print channel info.

    Returns:
        (X, y, channels, y_mean, channel_max_dict)
        
        y_mean: Mean of target variable used for normalization.
               To recover original scale: y_original = y_normalized * y_mean
        other_spend_sources: List of channel names that were aggregated into OTHER_SPEND.
    """
    df = df_weekly.copy()
    channels = [c for c in spend_cols if c in df.columns]

    channels = filter_low_variance_channels(df, channels, min_nonzero_ratio, verbose=verbose)

    total_spend = sum(df[c].sum() for c in channels)
    
    if total_spend == 0:
        if verbose:
            print("Warning: Total spend is zero. Returning empty features.")
        return pd.DataFrame(), np.array([]), [], 1.0, {}
    
    channels_filtered = []
    other_spend_sources = []  # Track channels aggregated into OTHER_SPEND
    other_spend = pd.Series(0.0, index=df.index)

    for c in channels:
        spend_share = df[c].sum() / total_spend
        if spend_share >= min_spend_threshold:
            channels_filtered.append(c)
        else:
            other_spend += df[c].fillna(0)
            other_spend_sources.append(c)  # Track this channel

    if other_spend.sum() > 0:
        df["OTHER_SPEND"] = other_spend
        channels_filtered.append("OTHER_SPEND")

    if verbose:
        print(f"Channels: {channels_filtered}")

    feature_cols = []
    channel_max_dict = {}

    for c in channels_filtered:
        col_adstock = f"{c}_adstock"
        col_sat = f"{c}_sat"

        df[col_adstock] = apply_adstock(df[c].fillna(0).values, decay=adstock_decay)
        
        if train_end_idx is not None:
            train_max = df[col_adstock].iloc[:train_end_idx].max()
        else:
            train_max = df[col_adstock].max()
        
        df[col_sat] = apply_saturation_with_max(
            df[col_adstock].values, train_max, saturation_half
        )

        channel_max_dict[c] = train_max
        feature_cols.append(col_sat)
    
    # Add Other Features (Control, Seasonality, Traffic)
    # Use provided lists or defaults from config
    ctrl = control_cols if control_cols is not None else CONTROL_COLS
    seas = season_cols if season_cols is not None else SEASON_COLS
    traf = traffic_cols if traffic_cols is not None else TRAFFIC_COLS

    other_cols_candidates = ctrl + seas + traf
    
    for col in other_cols_candidates:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].fillna(0)

    y = df[target_col].values
    y_mean = y.mean()
    y = y / y_mean

    return X, y, channels_filtered, y_mean, channel_max_dict, other_spend_sources


# =============================================================================
# 5. HIERARCHICAL MODEL SPECIFIC
# 
# Used by: Bayesian Hierarchical Model (mmm_hierarchical.py)
# Purpose: Panel data preparation and tensor creation for PyMC.
# =============================================================================


def get_panel_holdout_indices(
    df: pd.DataFrame,
    geo_col: str,
    date_col: str,
    holdout_size: int,
) -> tuple[list[int], list[int]]:
    """
    Get train/test indices for panel data holdout split.
    
    CRITICAL: This function explicitly sorts data by Territory and Date 
    to prevent temporal leakage (training on future data).

    Note: Currently used primarily by the Hierarchical Model (mmm_hierarchical.py)
    to handle correct splitting of stacked panel data. Baseline models typically
    use standard TimeSeriesSplit on per-region data.
    
    Args:
        df: Panel DataFrame.
        geo_col: Column name for territory.
        date_col: Column name for date (required for safe sorting).
        holdout_size: Rows per territory for test.
    
    Returns:
        (train_indices, test_indices)
    
    Raises:
        ValueError: If a territory has insufficient data for the split.
    """
    # 1. FAIL SAFE: Force sort to guarantee chronological order
    # Using 'kind="stable"' to preserve existing order if already sorted
    df_sorted = df.sort_values(by=[geo_col, date_col], ascending=[True, True])
    
    train_indices = []
    test_indices = []
    
    for territory in df_sorted[geo_col].unique():
        mask = df_sorted[geo_col] == territory
        positions = df_sorted.index[mask].tolist()
        n_obs = len(positions)
        
        # 2. FAIL FAST: Check for data sufficiency
        if n_obs <= holdout_size:
            raise ValueError(
                f"Territory '{territory}' has {n_obs} observations, but holdout_size is {holdout_size}. "
                "Cannot perform valid train/test split."
            )
            
        train_indices.extend(positions[:-holdout_size])
        test_indices.extend(positions[-holdout_size:])
    
    return train_indices, test_indices


def prepare_hierarchical_data(
    df: pd.DataFrame,
    regions: list[str],
) -> tuple[pd.DataFrame, dict]:

    # 1. Prepare weekly data per region
    all_data = []
    for region in regions:
        df_weekly = prepare_weekly_data(df, region=region)
        df_weekly[GEO_COL] = region
        # Add currency from original data to normalize spend by currency
        currency = df[df['TERRITORY_NAME'] == region]['CURRENCY_CODE'].iloc[0]
        df_weekly['CURRENCY_CODE'] = currency
        all_data.append(df_weekly)

    df_combined = pd.concat(all_data, ignore_index=True)

    # 2. Add trend per region (normalized)
    for region in regions:
        mask = df_combined[GEO_COL] == region
        df_combined.loc[mask, "trend"] = np.arange(mask.sum()) / (mask.sum() + 1)

    # 3. Log-transform target
    df_combined['y_log'] = np.log1p(df_combined[TARGET_COL])

    # 4. Ensure consistent sorting before generating indices
    df_combined = df_combined.sort_values([GEO_COL, DATE_COL]).reset_index(drop=True)
    
    for geo in df_combined[GEO_COL].unique():
        geo_dates = df_combined.loc[df_combined[GEO_COL] == geo, DATE_COL]
        if not geo_dates.is_monotonic_increasing:
            raise ValueError(
                f"Dates not monotonic for territory {geo}. "
                "Adstock computation requires sorted data."
            )

    # 5. Create hierarchy indices (Global → Territory)
    territory_idx, territory_names = create_hierarchy_indices(df_combined, geo_col=GEO_COL)

    indices = {
        "territory_idx": territory_idx,
        "territory_names": territory_names,
    }

    print(f"Combined data: {len(df_combined)} rows, {len(regions)} regions")
    print(f"DataFrame columns: {len(df_combined.columns)} (includes raw/intermediate cols)")
    print(f"Hierarchy: {len(territory_names)} territories")

    return df_combined, indices


def prepare_model_data(
    df: pd.DataFrame,
    indices: dict,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
) -> dict[str, Any]:
    """
    Split data temporally and prepare X/y for Bayesian model.
    
    If train_indices/test_indices provided, use them (for CV).
    Otherwise, fall back to HOLDOUT_WEEKS split (backward compatibility).
    
    Spend normalization happens AFTER split,
    fitting max on train only to prevent data leakage.
    """
    
    # Seasonality Cyclic terms
    season_cols = [c for c in SEASON_COLS if c in df.columns]
    
    # All other features (Traffic, Controls - NOT spend or season)
    excluded = [TARGET_COL, "y_log", DATE_COL, GEO_COL, "CURRENCY_CODE"] + season_cols + SPEND_COLS
    other_feature_cols = [c for c in df.columns if c in ALL_FEATURES and c not in excluded]


    # Temporal Split
    if train_indices is None or test_indices is None:
        train_indices, test_indices = get_panel_holdout_indices(
            df, GEO_COL, DATE_COL, HOLDOUT_WEEKS
        )

    # FAIL FAST: Ensure we have data
    if not train_indices or len(train_indices) == 0:
        raise ValueError("train_indices is empty. Cannot prepare model data.")
    if not test_indices or len(test_indices) == 0:
        raise ValueError("test_indices is empty. Cannot prepare model data.")

    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    # Normalize spend AFTER split, fitting on train only
    spend_max_by_currency = {}  # Store max per (currency, channel) from train
    
    for col in SPEND_COLS:
        if col not in df_train.columns:
            continue
        # Compute max per currency from TRAIN only
        train_max = df_train.groupby("CURRENCY_CODE")[col].transform("max")
        df_train[f"{col}_norm"] = df_train[col] / (train_max + EPSILON)
        
        # Store max values for applying to test
        currency_max = df_train.groupby("CURRENCY_CODE")[col].max().to_dict()
        spend_max_by_currency[col] = currency_max
        
        # Apply SAME max to test (no data leakage)
        test_max = df_test["CURRENCY_CODE"].map(currency_max).fillna(EPSILON)
        df_test[f"{col}_norm"] = df_test[col] / (test_max + EPSILON)
    
    spend_norm_cols = [f"{c}_norm" for c in SPEND_COLS if f"{c}_norm" in df_train.columns]
    print(f"Normalized {len(spend_norm_cols)} spend columns using train-only max")

    # Scale features and seasonality (StandardScaler)    
    scaler_features = StandardScaler()
    scaler_season = StandardScaler()
    
    # Fit on train, transform both
    X_features_train = scaler_features.fit_transform(df_train[other_feature_cols].fillna(0).values)
    X_season_train = scaler_season.fit_transform(df_train[season_cols].fillna(0).values)
    
    X_features_test = scaler_features.transform(df_test[other_feature_cols].fillna(0).values)
    X_season_test = scaler_season.transform(df_test[season_cols].fillna(0).values)

    # Prepare Dictionary for Model Fitting
    # Fill any NaNs in spend normalized columns
    channel_names = [c.replace("_norm", "").replace("_SPEND", "") for c in spend_norm_cols]
    
    model_data = {
        "X_spend_train": np.ascontiguousarray(df_train[spend_norm_cols].fillna(0).values).astype(np.float64),
        "X_spend_test": np.ascontiguousarray(df_test[spend_norm_cols].fillna(0).values).astype(np.float64),
        "X_features_train": np.ascontiguousarray(X_features_train).astype(np.float64),
        "X_features_test": np.ascontiguousarray(X_features_test).astype(np.float64),
        "X_season_train": np.ascontiguousarray(X_season_train).astype(np.float64),
        "X_season_test": np.ascontiguousarray(X_season_test).astype(np.float64),
        "y_train": np.ascontiguousarray(df_train["y_log"].fillna(0).values).astype(np.float64),
        "y_test": np.ascontiguousarray(df_test["y_log"].fillna(0).values).astype(np.float64),
        "y_train_original": np.ascontiguousarray(df_train[TARGET_COL].fillna(0).values).astype(np.float64),
        "y_test_original": np.ascontiguousarray(df_test[TARGET_COL].fillna(0).values).astype(np.float64),
        "territory_idx_train": indices["territory_idx"][train_indices],
        "territory_idx_test": indices["territory_idx"][test_indices],
        "n_territories": len(indices["territory_names"]),
        "channel_names": channel_names,
        "spend_cols_raw": [c + "_SPEND" for c in channel_names],
        "feature_names": other_feature_cols,
        "df_train": df_train,
        "df_test": df_test,
    }


    print(f"\n{'='*60}")
    print(f"MODEL FEATURE SUMMARY")
    print(f"{'='*60}")
    print(f"\n1. SPEND CHANNELS (X_spend) - {model_data['X_spend_train'].shape[1]} channels:")
    print(f"   {model_data['channel_names']}")
    print(f"\n2. OTHER FEATURES (X_features) - {len(other_feature_cols)} cols:")
    print(f"   {other_feature_cols}")
    print(f"\n3. SEASONALITY (X_season) - {model_data['X_season_train'].shape[1]} cols:")
    print(f"   {season_cols}")
    print(f"\n{'='*60}")
    total_features = model_data['X_spend_train'].shape[1] + len(other_feature_cols) + model_data['X_season_train'].shape[1]
    print(f"TOTAL FEATURES: {total_features}")
    print(f"{'='*60}")
    
    # Data scale debug
    print(f"X_features: std (avg)={model_data['X_features_train'].std(axis=0).mean():.3f}")
    print(f"========================\n")
    
    # Add metadata for predictions deliverable
    model_data.update({
        "dates_train": df_train[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "dates_test": df_test[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "territories_train": df_train[GEO_COL].values,
        "territories_test": df_test[GEO_COL].values,
    })
    
    inspect_dir = Path("data/inspection")
    inspect_dir.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(inspect_dir / "hierarchical_train.parquet", index=False)
    df_test.to_parquet(inspect_dir / "hierarchical_test.parquet", index=False)
    print(f" - Saved inspection data to {inspect_dir}")
    return model_data

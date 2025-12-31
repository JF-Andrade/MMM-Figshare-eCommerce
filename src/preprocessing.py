"""
Preprocessing functions for MMM project.

Implements transformations from EDA findings:
- Adstock transformation
- Saturation (Hill function)
- Lag features
- Calendar features
- Missing value imputation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import holidays
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Channel configuration
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


def filter_low_variance_channels(
    df: pd.DataFrame,
    spend_cols: list[str],
    min_nonzero_ratio: float = 0.20,
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
    valid_channels = []
    for col in spend_cols:
        if col not in df.columns:
            continue
        nonzero_ratio = (df[col] > 0).sum() / len(df)
        if nonzero_ratio >= min_nonzero_ratio:
            valid_channels.append(col)
        elif verbose:
            print(f"Filtering {col}: {nonzero_ratio:.1%} non-zero (min: {min_nonzero_ratio:.0%})")
    return valid_channels


def apply_adstock(
    x: NDArray[np.floating] | pd.Series,
    decay: float = 0.5,
) -> NDArray[np.floating]:
    """
    Apply geometric adstock transformation.

    Adstock models the carryover effect of advertising:
    adstock[t] = x[t] + decay * adstock[t-1]

    Args:
        x: Spend or media variable array.
        decay: Decay rate between 0 and 1. Higher = longer carryover.

    Returns:
        Adstocked values.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x_arr)
    result[0] = x_arr[0]

    for t in range(1, len(x_arr)):
        result[t] = x_arr[t] + decay * result[t - 1]

    return result


def apply_saturation(
    x: NDArray[np.floating] | pd.Series,
    half_saturation: float,
    slope: float = 1.0,
) -> NDArray[np.floating]:
    """
    Apply Hill saturation function for diminishing returns.

    Hill function: x^slope / (half_saturation^slope + x^slope)

    Args:
        x: Input values (typically adstocked spend).
        half_saturation: Point where response reaches 50% of max.
        slope: Controls steepness of curve.

    Returns:
        Saturated values between 0 and 1.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_arr = np.maximum(x_arr, 0)  # Ensure non-negative

    numerator = x_arr**slope
    denominator = half_saturation**slope + x_arr**slope

    return numerator / denominator


def apply_saturation_with_max(
    x: NDArray[np.floating] | pd.Series,
    x_max: float,
    half_saturation: float,
    slope: float = 1.0,
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


def create_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.

    Args:
        df: Input DataFrame.
        columns: Column names to create lags for.
        lags: List of lag periods (e.g., [1, 2, 3, 7]).
        group_by: Optional column to group by before lagging.

    Returns:
        DataFrame with original and lagged columns.
    """
    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue

        for lag in lags:
            lag_col_name = f"{col}_lag{lag}"

            if group_by and group_by in result.columns:
                result[lag_col_name] = result.groupby(group_by)[col].shift(lag)
            else:
                result[lag_col_name] = result[col].shift(lag)

    return result


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "DATE",
    region_col: str | None = "REGION",
) -> pd.DataFrame:
    """
    Add calendar-based features for seasonality modeling.

    Features added:
    - day_of_week (0-6)
    - month (1-12)
    - quarter (1-4)
    - is_weekend (binary)
    - is_holiday (binary, region-specific if region_col provided)

    Args:
        df: Input DataFrame with date column.
        date_col: Name of date column.
        region_col: Optional region column for region-specific holidays.

    Returns:
        DataFrame with calendar features added.
    """
    result = df.copy()
    dates = pd.to_datetime(result[date_col])

    result["day_of_week"] = dates.dt.dayofweek
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)

    # Holiday detection
    if region_col and region_col in result.columns:
        result["is_holiday"] = 0

        for region, country_code in REGION_HOLIDAY_MAP.items():
            try:
                country_holidays = holidays.country_holidays(country_code)
                mask = result[region_col] == region
                region_dates = dates[mask]
                result.loc[mask, "is_holiday"] = region_dates.apply(
                    lambda d: 1 if d in country_holidays else 0
                )
            except NotImplementedError:
                # Country not supported, skip
                continue
    else:
        # Default to US holidays
        us_holidays = holidays.US()
        result["is_holiday"] = dates.apply(lambda d: 1 if d in us_holidays else 0)

    return result


def impute_missing_values(
    df: pd.DataFrame,
    fill_value: float = 0.0,
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
            if col in result.columns:
                result[col] = result[col].fillna(fill_value)

    return result


def log_transform(
    x: NDArray[np.floating] | pd.Series,
    offset: float = 1.0,
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
    offset: float = 1.0,
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


def transform_spend_columns(
    df: pd.DataFrame,
    decay: float = 0.5,
    half_saturation_pct: float = 0.5,
    apply_log: bool = False,
    group_by: str | None = "REGION",
) -> pd.DataFrame:
    """
    Apply full transformation pipeline to spend columns.

    Pipeline: Impute -> Adstock -> Saturation -> (optional Log)

    Args:
        df: Input DataFrame with spend columns.
        decay: Adstock decay rate.
        half_saturation_pct: Percentile of spend for half-saturation point.
        apply_log: Whether to apply log transform after saturation.
        group_by: Column to group by for transformations.

    Returns:
        DataFrame with transformed spend columns.
    """
    result = df.copy()
    spend_cols = [f"{ch}_SPEND" for ch in CHANNELS]

    for col in spend_cols:
        if col not in result.columns:
            continue

        # Impute missing
        result[col] = result[col].fillna(0)

        # Calculate half-saturation point from data
        half_sat = result[col].quantile(half_saturation_pct)
        if half_sat == 0:
            half_sat = result[col].mean() + 1e-6

        # Apply transformations per group
        if group_by and group_by in result.columns:
            adstocked = result.groupby(group_by)[col].transform(
                lambda x: apply_adstock(x.values, decay)
            )
            transformed = apply_saturation(adstocked.values, half_sat)
        else:
            adstocked = apply_adstock(result[col].values, decay)
            transformed = apply_saturation(adstocked, half_sat)

        result[f"{col}_transformed"] = transformed

        if apply_log:
            result[f"{col}_log"] = log_transform(result[col].values)

    return result


def preprocess_mmm_data(
    df: pd.DataFrame,
    date_col: str = "DATE",
    region_col: str = "REGION",
    target_col: str = "REVENUE",
    decay: float = 0.5,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for MMM data.

    Steps:
    1. Impute missing values (TikTok zeros)
    2. Add calendar features
    3. Transform spend columns (adstock + saturation)
    4. Create lag features
    5. Log-transform target

    Args:
        df: Raw DataFrame.
        date_col: Date column name.
        region_col: Region column name.
        target_col: Target variable column name.
        decay: Adstock decay rate.
        lags: Lag periods for spend features.

    Returns:
        Preprocessed DataFrame ready for modeling.
    """
    if lags is None:
        lags = [1, 2, 3, 7]

    # Step 1: Impute TikTok missing values
    tiktok_cols = [f"TIKTOK_{m}" for m in METRICS]
    result = impute_missing_values(df, fill_value=0.0, columns=tiktok_cols)

    # Step 2: Calendar features
    result = add_calendar_features(result, date_col=date_col, region_col=region_col)

    # Step 3: Transform spend columns
    result = transform_spend_columns(
        result,
        decay=decay,
        apply_log=True,
        group_by=region_col,
    )

    # Step 4: Create lag features for spend
    spend_cols = [f"{ch}_SPEND" for ch in CHANNELS if f"{ch}_SPEND" in result.columns]
    result = create_lag_features(result, spend_cols, lags, group_by=region_col)

    # Step 5: Log-transform target
    if target_col in result.columns:
        result[f"{target_col}_log"] = log_transform(result[target_col].values)

    # Sort by region and date
    result = result.sort_values([region_col, date_col]).reset_index(drop=True)

    return result


def prepare_weekly_data(
    df: pd.DataFrame,
    region: str | None = None,
    date_col: str = "DATE_DAY",
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
    from src.config import TARGET_COL
    
    # Use config default if not specified
    if target_col is None:
        target_col = TARGET_COL
    
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col])

    if region:
        df = df[df["TERRITORY_NAME"] == region].copy()

    df["week"] = df["date"].dt.to_period("W").dt.start_time

    # Default spend columns
    if spend_cols is None:
        spend_cols = [
            "GOOGLE_PAID_SEARCH_SPEND", "GOOGLE_SHOPPING_SPEND",
            "GOOGLE_PMAX_SPEND", "GOOGLE_DISPLAY_SPEND",
            "GOOGLE_VIDEO_SPEND", "META_FACEBOOK_SPEND",
            "META_INSTAGRAM_SPEND", "META_OTHER_SPEND", "TIKTOK_SPEND",
        ]

    # Columns to aggregate (spend + traffic + targets)
    from src.config import TRAFFIC_COLS
    sum_cols = [c for c in spend_cols + TRAFFIC_COLS + [
        target_col,
        "ALL_PURCHASES",
        "ALL_PURCHASES_ORIGINAL_PRICE",  # Include for alternative target
    ] if c in df.columns]

    control_cols = ["is_holiday"]

    agg_dict = {**{c: "sum" for c in sum_cols if c in df.columns}}
    if "is_holiday" in df.columns:
        agg_dict["is_holiday"] = "max"

    df_weekly = df.groupby("week").agg(agg_dict).reset_index()
    df_weekly = df_weekly.sort_values("week").reset_index(drop=True)

    # Trend feature (normalized 0 to 1)
    df_weekly["trend"] = np.arange(len(df_weekly)) / len(df_weekly)

    # Seasonality features
    df_weekly["month"] = df_weekly["week"].dt.month / 12
    df_weekly["week_of_year"] = df_weekly["week"].dt.isocalendar().week.astype(float) / 52
    df_weekly["month_sin"] = np.sin(2 * np.pi * df_weekly["week"].dt.month / 12)
    df_weekly["month_cos"] = np.cos(2 * np.pi * df_weekly["week"].dt.month / 12)

    # Quarter features (for Q4 sales boost)
    df_weekly["quarter"] = df_weekly["week"].dt.quarter / 4
    df_weekly["is_q4"] = (df_weekly["week"].dt.quarter == 4).astype(int)

    # Log-transform target for better model fit
    if target_col in df_weekly.columns:
        df_weekly[f"{target_col}_log"] = np.log1p(df_weekly[target_col])

    # Normalize spend columns per channel
    for col in spend_cols:
        if col in df_weekly.columns:
            max_val = df_weekly[col].max()
            if max_val > 0:
                df_weekly[f"{col}_norm"] = df_weekly[col] / max_val

    return df_weekly


# =============================================================================
# HIERARCHICAL MODEL PREPROCESSING FUNCTIONS
# =============================================================================


def normalize_spend_by_currency(
    df: pd.DataFrame,
    spend_cols: list[str],
    currency_col: str = "CURRENCY_CODE",
) -> pd.DataFrame:
    """
    Normalize spend features within each currency (0-1 scaling).
    
    Prevents scale issues when combining territories with different currencies.
    """
    df = df.copy()
    
    for col in spend_cols:
        if col not in df.columns:
            continue
        df[f"{col}_norm"] = df.groupby(currency_col)[col].transform(
            lambda x: x / (x.max() + 1e-8)
        )
    
    return df


def geometric_adstock(x: "NDArray", alpha: float, l_max: int = 8) -> "NDArray":
    """
    Apply geometric adstock transformation.
    
    adstock[t] = x[t] + alpha*x[t-1] + alpha^2*x[t-2] + ...
    
    Args:
        x: Spend values (1D array, must be sorted by time)
        alpha: Decay rate (0-1). Higher = longer carryover.
        l_max: Maximum lag to consider.
    
    Returns:
        Adstocked values.
    """
    if len(x) == 0:
        return x
    
    weights = alpha ** np.arange(l_max)
    adstocked = np.convolve(x, weights, mode='full')[:len(x)]
    return adstocked


def apply_adstock_per_territory(
    df: pd.DataFrame,
    spend_cols: list[str],
    geo_col: str = "geo",
    date_col: str = "week",
    alpha: float = 0.5,
    l_max: int = 8,
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
        if col not in df.columns:
            continue
        
        adstock_col = f"{col}_adstock"
        df[adstock_col] = 0.0
        
        for territory in df[geo_col].unique():
            mask = df[geo_col] == territory
            x = df.loc[mask, col].values
            df.loc[mask, adstock_col] = geometric_adstock(x, alpha, l_max)
    
    return df


def apply_saturation_transform(
    df: pd.DataFrame,
    adstock_cols: list[str],
    lam: float = 1.0,
) -> pd.DataFrame:
    """
    Apply logistic saturation transformation.
    
    saturation(x) = 1 - exp(-lam * x)
    
    Models diminishing returns of marketing spend.
    """
    df = df.copy()
    
    for col in adstock_cols:
        if col not in df.columns:
            continue
        
        sat_col = col.replace("_adstock", "_saturated")
        df[sat_col] = 1 - np.exp(-lam * df[col])
    
    return df


def create_hierarchy_indices(
    df: pd.DataFrame,
    geo_col: str = "geo",
) -> tuple["NDArray", list[str]]:
    """
    Create integer indices for hierarchical model.
    
    Returns:
        territory_idx: Territory index for each observation (n_obs,)
        territory_names: List of territory names
    """
    # Territory indexing
    territory_cat = pd.Categorical(df[geo_col])
    territory_idx = territory_cat.codes
    territory_names = territory_cat.categories.tolist()
    
    return territory_idx, territory_names


def add_fourier_seasonality(
    df: pd.DataFrame,
    date_col: str = "week",
    n_terms: int = 2,
) -> pd.DataFrame:
    """
    Add Fourier terms for yearly seasonality.
    
    Creates sin/cos pairs to capture annual patterns.
    """
    df = df.copy()
    
    week_of_year = df[date_col].dt.isocalendar().week.astype(float)
    
    for k in range(1, n_terms + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * week_of_year / 52)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * week_of_year / 52)
    
    return df


# =============================================================================
# BASELINE MODEL FEATURE PREPARATION
# =============================================================================


def prepare_baseline_features(
    df_weekly: pd.DataFrame,
    adstock_decay: float,
    saturation_half: float,
    spend_cols: list[str],
    target_col: str,
    min_nonzero_ratio: float = 0.20,
    min_spend_threshold: float = 0.01,
    train_end_idx: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str], float, dict]:
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
        (X, y, channels, y_scaler, channel_max_dict)
    """
    df = df_weekly.copy()
    channels = [c for c in spend_cols if c in df.columns]

    channels = filter_low_variance_channels(df, channels, min_nonzero_ratio, verbose=verbose)

    total_spend = sum(df[c].sum() for c in channels)
    channels_filtered = []
    other_spend = pd.Series(0.0, index=df.index)

    for c in channels:
        spend_share = df[c].sum() / total_spend
        if spend_share >= min_spend_threshold:
            channels_filtered.append(c)
        else:
            other_spend += df[c].fillna(0)

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

    from src.config import CONTROL_COLS, SEASON_COLS, TRAFFIC_COLS
    
    # Add Control Variables
    for col in CONTROL_COLS:
        if col in df.columns:
            feature_cols.append(col)
            
    # Add Seasonality Variables
    for col in SEASON_COLS:
        if col in df.columns:
            feature_cols.append(col)
            
    # Add Traffic Variables (exogenous demand)
    for col in TRAFFIC_COLS:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].fillna(0)

    y = df[target_col].values
    y_scaler = y.mean()
    y = y / y_scaler

    return X, y, channels_filtered, y_scaler, channel_max_dict


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def compute_temporal_features(
    df: pd.DataFrame,
    date_col: str = "DATE_DAY",
) -> pd.DataFrame:
    """
    Compute temporal features from date.

    Args:
        df: Input DataFrame.
        date_col: Date column name.

    Returns:
        DataFrame with temporal features added.
    """
    df = df.copy()
    
    if date_col in df.columns:
        date = pd.to_datetime(df[date_col])
        
        # Linear features (normalized 0-1)
        df["DAY_OF_WEEK"] = date.dt.dayofweek / 6
        df["QUARTER"] = date.dt.quarter / 4
        df["WEEK_OF_YEAR"] = date.dt.isocalendar().week / 52
        
        # Cyclic features (Sine/Cosine) - Better for seasonality
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

        
        # Black Friday Indicator (Week containing Nov 23-29 range)
        # Black Friday is the day after 4th Thursday. Earliest is Nov 23, Latest is Nov 29.
        # We flag the week that overlaps with this period.
        def check_black_friday(d):
            # Check if date is in November
            if d.month == 11:
                # Check for the specific date window
                if 23 <= d.day <= 29:
                    return 1
            return 0

        # Vectorized check is faster than apply for large data
        df["is_black_friday"] = ((date.dt.month == 11) & (date.dt.day >= 23) & (date.dt.day <= 29)).astype(int)
    
    return df


def compute_spend_share(
    df: pd.DataFrame,
    spend_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute channel spend share (media mix).

    Args:
        df: Input DataFrame.
        spend_cols: List of spend column names. If None, uses SPEND_COLS from config.

    Returns:
        DataFrame with _SHARE columns added.
    """
    from src.config import SPEND_COLS as DEFAULT_SPEND_COLS
    
    df = df.copy()
    
    if spend_cols is None:
        spend_cols = DEFAULT_SPEND_COLS
    
    available_spend = [c for c in spend_cols if c in df.columns]
    
    if not available_spend:
        return df
    
    total_spend = df[available_spend].sum(axis=1).replace(0, np.nan)
    
    for spend_col in available_spend:
        share_col = spend_col.replace("_SPEND", "_SHARE")
        df[share_col] = (df[spend_col] / total_spend).fillna(0)
    
    return df


def engineer_features(
    df: pd.DataFrame,
    date_col: str = "DATE_DAY",
    rolling_window: int = 7,
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Features created:
    - Temporal: day of week, quarter, week of year
    - Share: channel spend share

    Note: CTR/CPC features removed due to endogeneity.
          Customer metrics removed due to data leakage risk.

    Args:
        df: Input DataFrame.
        date_col: Date column name.
        rolling_window: Window for rolling features (not currently used).

    Returns:
        DataFrame with engineered features.
    """
    df = compute_temporal_features(df, date_col)
    df = compute_spend_share(df)
    
    return df

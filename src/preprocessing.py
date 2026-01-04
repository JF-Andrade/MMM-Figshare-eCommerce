"""
Preprocessing orchestration module for MMM project.

This module builds high-level data pipelines by coordinating:
1. Core transformations (from src.transformations)
2. Data aggregation (prepare_weekly_data)
3. Model-specific preparation (baseline and hierarchical)
4. Validation splits and fold transformations
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, TYPE_CHECKING
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.config import (
    CHANNELS,
    METRICS,
    MIN_NONZERO_RATIO,
    MIN_SPEND_THRESHOLD,
    DEFAULT_ADSTOCK_DECAY,
    DEFAULT_HALF_SATURATION_PCT,
    DEFAULT_ADSTOCK_LMAX,
    DEFAULT_LOG_OFFSET,
    DEFAULT_SATURATION_SLOPE,
    RAW_DATE_COL,
    RAW_REGION_COL,
    DATE_COL,
    GEO_COL,
    TARGET_COL,
    SPEND_COLS,
    TRAFFIC_COLS,
    CONTROL_COLS,
    SEASON_COLS,
    ALL_FEATURES,
    HOLDOUT_WEEKS,
    EPSILON,
)

from src.transformations import (
    log_transform,
    inverse_log_transform,
    apply_adstock,
    apply_saturation_with_max,
    impute_missing_values,
    filter_low_variance_channels,
    add_seasonality_features,
    add_event_features,
    create_hierarchy_indices,
)


# =============================================================================
# 1. DATA AGGREGATION PIPELINES (Shared Logic)
# =============================================================================

# Used by: Both models (data aggregation entry point)
def prepare_weekly_data(
    df: pd.DataFrame,
    region: str | None = None,
    date_col: str = RAW_DATE_COL,
    target_col: str | None = None,
    spend_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate daily data to weekly for MMM.

    Args:
        df: Daily DataFrame.
        region: Optional region filter.
        date_col: Name of date column.
        target_col: Target variable column.
        spend_cols: Spend columns to aggregate.

    Returns:
        Weekly aggregated DataFrame with engineered features.
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

    # Compute seasonality
    df_weekly = add_seasonality_features(df_weekly, date_col="week")

    # Log-transform target
    df_weekly[f"{target_col}_log"] = log_transform(df_weekly[target_col])

    # Normalize spend columns per channel
    for col in spend_cols:
        if col in df_weekly.columns:
            max_val = df_weekly[col].max()
            if max_val > 0:
                df_weekly[f"{col}_norm"] = df_weekly[col] / max_val

    return df_weekly


# =============================================================================
# 2. BASELINE MODEL ORCHESTRATION
# =============================================================================

# Used by: Baseline Ridge model (feature preparation pipeline)
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
    Prepare features for Baseline (Ridge) model.
    """
    df = df_weekly.copy()
    channels = [c for c in spend_cols if c in df.columns]

    channels = filter_low_variance_channels(df, channels, min_nonzero_ratio, verbose=verbose)

    total_spend = sum(df[c].sum() for c in channels)
    
    if total_spend == 0:
        if verbose:
            print("Warning: Total spend is zero. Returning empty features.")
        return pd.DataFrame(), np.array([]), [], 1.0, {}, []
    
    channels_filtered = []
    other_spend_sources = [] 
    other_spend = pd.Series(0.0, index=df.index)

    for c in channels:
        spend_share = df[c].sum() / total_spend
        if spend_share >= min_spend_threshold:
            channels_filtered.append(c)
        else:
            other_spend += df[c].fillna(0)
            other_spend_sources.append(c)

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


# Used by: Baseline Ridge model (test fold transformation)
def transform_test_fold(
    df_test: pd.DataFrame,
    channels: list[str],
    channel_max_dict: dict[str, float],
    y_mean: float,
    adstock_decay: float,
    saturation_half: float,
    target_col: str,
    other_spend_sources: list[str] | None = None,
    control_cols: list[str] | None = None,
    season_cols: list[str] | None = None,
    traffic_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Transform test fold using train statistics (prevent leakage).
    Moved from src.validation to keep orchestration logic together.
    """
    df = df_test.copy()
    feature_cols = []
    
    # Aggregate low-spend channels
    if "OTHER_SPEND" in channels and other_spend_sources:
        other_spend = pd.Series(0.0, index=df.index)
        for src_col in other_spend_sources:
            if src_col in df.columns:
                other_spend += df[src_col].fillna(0)
        df["OTHER_SPEND"] = other_spend
    
    for c in channels:
        if c not in df.columns:
            continue
            
        col_adstock = f"{c}_adstock"
        col_sat = f"{c}_sat"
        
        df[col_adstock] = apply_adstock(df[c].fillna(0).values, decay=adstock_decay)
        
        # Use TRAIN max
        train_max = channel_max_dict.get(c, df[col_adstock].max())
        df[col_sat] = apply_saturation_with_max(
            df[col_adstock].values,
            train_max,
            saturation_half
        )
        feature_cols.append(col_sat)
    
    ctrl = control_cols if control_cols is not None else CONTROL_COLS
    seas = season_cols if season_cols is not None else SEASON_COLS
    traf = traffic_cols if traffic_cols is not None else TRAFFIC_COLS

    for col in ctrl + seas + traf:
        if col in df.columns:
            feature_cols.append(col)
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].values / y_mean
    
    return X, y


# =============================================================================
# 3. HIERARCHICAL MODEL ORCHESTRATION
# =============================================================================

# Used by: Hierarchical Bayesian model (panel holdout split)
def get_panel_holdout_indices(
    df: pd.DataFrame,
    geo_col: str,
    date_col: str,
    holdout_size: int,
) -> tuple[list[int], list[int]]:
    """
    Get train/test indices for panel data holdout.
    Moved from src.validation.
    """
    df_sorted = df.sort_values(by=[geo_col, date_col], ascending=[True, True])
    
    train_indices = []
    test_indices = []
    
    for territory in df_sorted[geo_col].unique():
        mask = df_sorted[geo_col] == territory
        positions = df_sorted.index[mask].tolist()
        n_obs = len(positions)
        
        if n_obs <= holdout_size:
            raise ValueError(
                f"Territory '{territory}' has {n_obs} obs, holdout is {holdout_size}."
            )
            
        train_indices.extend(positions[:-holdout_size])
        test_indices.extend(positions[-holdout_size:])
    
    return train_indices, test_indices


# Used by: Hierarchical Bayesian model (multi-region stacking)
def prepare_hierarchical_data(
    df: pd.DataFrame,
    regions: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Prepare stacked DataFrame for hierarchical modeling."""
    all_data = []
    for region in regions:
        df_weekly = prepare_weekly_data(df, region=region)
        df_weekly[GEO_COL] = region
        # Add currency
        currency = df[df[RAW_REGION_COL] == region][RAW_CURRENCY_COL].iloc[0]
        df_weekly[RAW_CURRENCY_COL] = currency
        all_data.append(df_weekly)

    df_combined = pd.concat(all_data, ignore_index=True)

    # Add trend per region
    for region in regions:
        mask = df_combined[GEO_COL] == region
        df_combined.loc[mask, "trend"] = np.arange(mask.sum()) / (mask.sum() + 1)

    # Target
    df_combined['y_log'] = log_transform(df_combined[TARGET_COL])

    # Sort
    df_combined = df_combined.sort_values([GEO_COL, DATE_COL]).reset_index(drop=True)
    
    for geo in df_combined[GEO_COL].unique():
        geo_dates = df_combined.loc[df_combined[GEO_COL] == geo, DATE_COL]
        if not geo_dates.is_monotonic_increasing:
            raise ValueError(f"Dates not monotonic for {geo}")

    # Hierarchy indices
    territory_idx, territory_names = create_hierarchy_indices(df_combined, geo_col=GEO_COL)

    indices = {
        "territory_idx": territory_idx,
        "territory_names": territory_names,
    }

    print(f"Combined data: {len(df_combined)} rows, {len(regions)} regions")
    print(f"Hierarchy: {len(territory_names)} territories")

    return df_combined, indices


# Used by: Hierarchical Bayesian model (PyMC tensor preparation)
def prepare_model_data(
    df: pd.DataFrame,
    indices: dict,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Split data and prepare tensors for PyMC."""
    
    season_cols = [c for c in SEASON_COLS if c in df.columns]
    
    excluded = [TARGET_COL, "y_log", DATE_COL, GEO_COL, "CURRENCY_CODE"] + season_cols + SPEND_COLS
    
    # Use ALL_FEATURES from config to robustly identify feature columns
    # We intersect with df.columns to be safe
    other_feature_cols = [c for c in df.columns if c in ALL_FEATURES and c not in excluded]

    if train_indices is None or test_indices is None:
        train_indices, test_indices = get_panel_holdout_indices(
            df, GEO_COL, DATE_COL, HOLDOUT_WEEKS
        )

    if not train_indices:
        raise ValueError("train_indices is empty.")
    if not test_indices:
        raise ValueError("test_indices is empty.")

    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    spend_max_by_currency = {}
    
    # Normalize spend fitting on TRAIN only
    for col in SPEND_COLS:
        if col not in df_train.columns:
            continue
        train_max = df_train.groupby(RAW_CURRENCY_COL)[col].transform("max")
        df_train[f"{col}_norm"] = df_train[col] / (train_max + EPSILON)
        
        currency_max = df_train.groupby(RAW_CURRENCY_COL)[col].max().to_dict()
        spend_max_by_currency[col] = currency_max
        
        # Apply to test
        test_max = df_test[RAW_CURRENCY_COL].map(currency_max).fillna(EPSILON)
        df_test[f"{col}_norm"] = df_test[col] / (test_max + EPSILON)
    
    spend_norm_cols = [f"{c}_norm" for c in SPEND_COLS if f"{c}_norm" in df_train.columns]
    print(f"Normalized {len(spend_norm_cols)} spend columns using train-only max")

    scaler_features = StandardScaler()
    scaler_season = StandardScaler()
    
    X_features_train = scaler_features.fit_transform(df_train[other_feature_cols].fillna(0).values)
    X_season_train = scaler_season.fit_transform(df_train[season_cols].fillna(0).values)
    
    X_features_test = scaler_features.transform(df_test[other_feature_cols].fillna(0).values)
    X_season_test = scaler_season.transform(df_test[season_cols].fillna(0).values)

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
    print(f"\n1. SPEND CHANNELS (X_spend) - {len(channel_names)} channels")
    print(f"\n2. OTHER FEATURES (X_features) - {len(other_feature_cols)} cols")
    print(f"\n3. SEASONALITY (X_season) - {model_data['X_season_train'].shape[1]} cols")
    print(f"\n{'='*60}")
    
    return model_data

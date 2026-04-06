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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from src.config import (
    MIN_NONZERO_RATIO,
    MIN_SPEND_THRESHOLD,
    RAW_DATE_COL,
    RAW_REGION_COL,
    TARGET_COL,
    SPEND_COLS,
    TRAFFIC_COLS,
    CONTROL_COLS,
    SEASON_COLS,
)

from src.transformations import (
    log_transform,
    apply_adstock,
    apply_saturation_with_max,
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

    # FIX #9: REMOVED premature normalization that caused data leakage
    # Normalization is done in prepare_model_data() using train-only max
    # to prevent information from test set leaking into training

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




"""
Validation Module.

Time-series split utilities for Hierarchical MMM holdout validation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import CONTROL_COLS, SEASON_COLS, TRAFFIC_COLS
from src.preprocessing import apply_adstock, apply_saturation_with_max


def get_panel_holdout_indices(
    df: pd.DataFrame,
    geo_col: str,
    date_col: str,
    holdout_weeks: int,
) -> tuple[list[int], list[int]]:
    """
    Get train/test indices for panel data with temporal holdout.
    
    For each region, the last `holdout_weeks` observations are held out.
    This ensures temporal consistency while respecting panel structure.
    
    Args:
        df: Panel DataFrame with date and geo columns.
        geo_col: Name of the geography/region column.
        date_col: Name of the date column.
        holdout_weeks: Number of weeks to hold out for testing.
    
    Returns:
        Tuple of (train_indices, test_indices) as lists of integer positions.
    """
    train_indices = []
    test_indices = []
    
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    for region in df[geo_col].unique():
        region_mask = df[geo_col] == region
        region_df = df[region_mask].sort_values(date_col)
        
        n_obs = len(region_df)
        n_train = max(n_obs - holdout_weeks, 1)
        
        region_idx = region_df.index.tolist()
        train_indices.extend(region_idx[:n_train])
        test_indices.extend(region_idx[n_train:])
    
    return train_indices, test_indices


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
    Transform test fold using train statistics to prevent data leakage.
    
    Args:
        df_test: Test fold DataFrame.
        channels: List of channel names (from train fold, may include OTHER_SPEND).
        channel_max_dict: Max adstock values per channel (from train fold).
        y_mean: Mean of target variable (from train fold).
        adstock_decay: Adstock decay rate.
        saturation_half: Half-saturation point.
        target_col: Name of target column.
        other_spend_sources: List of original channel names that were aggregated
                            into OTHER_SPEND (if applicable).
        control_cols: Optional list of control columns.
        season_cols: Optional list of seasonality columns.
        traffic_cols: Optional list of traffic columns.
    
    Returns:
        (X_test, y_test) transformed using train statistics.
    """
    df = df_test.copy()
    feature_cols = []
    
    # Aggregate low-spend channels into OTHER_SPEND
    if "OTHER_SPEND" in channels and other_spend_sources:
        other_spend = pd.Series(0.0, index=df.index)
        for src_col in other_spend_sources:
            if src_col in df.columns:
                other_spend += df[src_col].fillna(0)
        df["OTHER_SPEND"] = other_spend
    
    for c in channels:
        # Skip if channel doesn't exist in test fold
        if c not in df.columns:
            continue
            
        col_adstock = f"{c}_adstock"
        col_sat = f"{c}_sat"
        
        # Apply adstock (causal transformation, no leakage issue)
        df[col_adstock] = apply_adstock(df[c].fillna(0).values, decay=adstock_decay)
        
        # Apply saturation using TRAIN max (prevents leakage)
        train_max = channel_max_dict.get(c, df[col_adstock].max())
        df[col_sat] = apply_saturation_with_max(
            df[col_adstock].values,
            train_max,
            saturation_half
        )
        feature_cols.append(col_sat)
    
    # Add control, seasonality, and traffic features
    ctrl = control_cols if control_cols is not None else CONTROL_COLS
    seas = season_cols if season_cols is not None else SEASON_COLS
    traf = traffic_cols if traffic_cols is not None else TRAFFIC_COLS

    for col in ctrl + seas + traf:
        if col in df.columns:
            feature_cols.append(col)
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].values / y_mean  # Use TRAIN y_mean
    
    return X, y


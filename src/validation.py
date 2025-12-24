"""
Validation Module.

Time-series split utilities for MMM holdout validation.
"""
from __future__ import annotations

import pandas as pd


def time_series_holdout_split(
    df: pd.DataFrame,
    holdout_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame keeping last N rows as holdout.
    
    Args:
        df: DataFrame sorted by time.
        holdout_size: Number of rows for test set.
    
    Returns:
        (train_df, test_df)
    """
    split_idx = len(df) - holdout_size
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def get_panel_holdout_indices(
    df: pd.DataFrame,
    geo_col: str,
    holdout_size: int,
) -> tuple[list[int], list[int]]:
    """
    Get train/test indices for panel data holdout split.
    
    Each territory contributes last N rows to test set.
    
    Args:
        df: Panel DataFrame (must be sorted by geo, then date).
        geo_col: Column name for territory.
        holdout_size: Rows per territory for test.
    
    Returns:
        (train_indices, test_indices)
    """
    train_indices = []
    test_indices = []
    
    for territory in df[geo_col].unique():
        mask = df[geo_col] == territory
        positions = df.index[mask].tolist()
        
        train_indices.extend(positions[:-holdout_size])
        test_indices.extend(positions[-holdout_size:])
    
    return train_indices, test_indices

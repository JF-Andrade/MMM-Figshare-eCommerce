"""
Validation Module.

Implements time-series cross-validation strategies for MMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


def expanding_window_cv(
    n_samples: int,
    n_splits: int = 3,
    min_train_size: int = 52,
    test_size: int = 12,
) -> Iterator[CVFold]:
    """
    Generate expanding window cross-validation splits.
    
    The training window expands with each fold while test window stays fixed.
    
    Example with 100 samples, 3 splits, min_train=52, test=12:
        Fold 1: train[0:52],  test[52:64]
        Fold 2: train[0:64],  test[64:76]
        Fold 3: train[0:76],  test[76:88]
    
    Args:
        n_samples: Total number of samples (e.g., weeks)
        n_splits: Number of CV folds
        min_train_size: Minimum training samples for first fold
        test_size: Fixed test size for each fold
    
    Yields:
        CVFold objects with train/test indices
    """
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        if test_end > n_samples:
            break

        yield CVFold(
            fold=i + 1,
            train_start=0,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
        )


def get_fold_data(
    df: pd.DataFrame,
    fold: CVFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe according to fold indices.
    
    Args:
        df: DataFrame to split (must be sorted by time)
        fold: CVFold object with indices
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df.iloc[fold.train_start : fold.train_end].copy()
    test_df = df.iloc[fold.test_start : fold.test_end].copy()
    return train_df, test_df


@dataclass
class CVResult:
    """Results from a single CV fold."""

    fold: int
    train_size: int
    test_size: int
    r2_train: float
    r2_test: float
    mape_train: float
    mape_test: float
    divergences: int


def aggregate_cv_results(results: list[CVResult]) -> dict:
    """
    Aggregate CV results across folds.
    
    Returns:
        Dict with mean and std of each metric
    """
    if not results:
        return {}

    metrics = ["r2_train", "r2_test", "mape_train", "mape_test"]
    agg = {}

    for metric in metrics:
        values = [getattr(r, metric) for r in results]
        agg[f"{metric}_mean"] = sum(values) / len(values)
        agg[f"{metric}_std"] = (
            sum((v - agg[f"{metric}_mean"]) ** 2 for v in values) / len(values)
        ) ** 0.5

    agg["n_folds"] = len(results)
    agg["total_divergences"] = sum(r.divergences for r in results)

    return agg


def validate_panel_for_cv(
    df: pd.DataFrame,
    geo_col: str,
    date_col: str,
    min_weeks: int,
) -> None:
    """
    Validate that panel data meets CV requirements.
    
    Raises:
        ValueError: If any territory has insufficient data.
    """
    weeks_per_territory = df.groupby(geo_col)[date_col].nunique()
    insufficient = weeks_per_territory[weeks_per_territory < min_weeks]
    
    if len(insufficient) > 0:
        msg = f"Territories with < {min_weeks} weeks: {insufficient.to_dict()}"
        raise ValueError(msg)


def panel_expanding_window_cv(
    df: pd.DataFrame,
    geo_col: str,
    date_col: str,
    n_splits: int = 3,
    min_train_weeks: int = 52,
    test_weeks: int = 12,
) -> "Iterator[CVFold]":
    """
    Generate expanding window splits for panel data.
    
    Each territory receives the SAME temporal split to prevent data leakage.
    
    Args:
        df: Panel DataFrame (must be sorted by geo, then date)
        geo_col: Column name for territory identifier
        date_col: Column name for date/week
        n_splits: Number of CV folds
        min_train_weeks: Minimum training weeks for fold 1
        test_weeks: Fixed test window size
        
    Yields:
        CVFold objects with week-based indices
        
    Raises:
        ValueError: If insufficient data for requested folds
    """
    total_weeks_needed = min_train_weeks + n_splits * test_weeks
    min_territory_weeks = df.groupby(geo_col)[date_col].nunique().min()
    
    if min_territory_weeks < total_weeks_needed:
        msg = (
            f"Insufficient data: need {total_weeks_needed} weeks, "
            f"smallest territory has {min_territory_weeks}"
        )
        raise ValueError(msg)
    
    for i in range(n_splits):
        train_end_week = min_train_weeks + i * test_weeks
        test_start_week = train_end_week
        test_end_week = train_end_week + test_weeks
        
        yield CVFold(
            fold=i + 1,
            train_start=0,
            train_end=train_end_week,
            test_start=test_start_week,
            test_end=test_end_week,
        )


def get_panel_fold_indices(
    df: pd.DataFrame,
    fold: CVFold,
    geo_col: str,
    date_col: str,
) -> tuple[list[int], list[int]]:
    """
    Get row indices for train/test split in panel data.
    
    Returns:
        (train_indices, test_indices) as lists of DataFrame row positions
        
    Raises:
        AssertionError: If data leakage detected (train dates >= test dates)
    """
    train_indices = []
    test_indices = []
    
    for territory in df[geo_col].unique():
        territory_mask = df[geo_col] == territory
        territory_df = df[territory_mask].sort_values(date_col)
        territory_positions = territory_df.index.tolist()
        
        train_rows = territory_positions[:fold.train_end]
        test_rows = territory_positions[fold.test_start:fold.test_end]
        
        train_indices.extend(train_rows)
        test_indices.extend(test_rows)
    
    # DATA LEAKAGE CHECK
    train_dates = df.loc[train_indices, date_col].max()
    test_dates = df.loc[test_indices, date_col].min()
    assert train_dates < test_dates, (
        f"Data leakage detected! Max train date {train_dates} >= min test date {test_dates}"
    )
    
    return train_indices, test_indices

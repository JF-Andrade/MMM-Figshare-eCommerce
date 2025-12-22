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

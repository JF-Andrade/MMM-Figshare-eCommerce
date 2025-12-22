"""
Data Loading Module.

Loads and validates MMM data from various sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config import TARGET_COL

if TYPE_CHECKING:
    pass

# Default configuration
DEFAULT_CURRENCY = "GBP"


def load_data(
    data_path: Path,
    currency: str | None = DEFAULT_CURRENCY,
) -> pd.DataFrame:
    """
    Load and validate MMM data with optional currency filtering.

    Args:
        data_path: Path to parquet or CSV file.
        currency: Currency code to filter (e.g., "GBP", "USD").
                  If None, loads all currencies without filtering.

    Returns:
        DataFrame with target variable (optionally filtered by currency).
    """
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    print(f"Loaded data: {df.shape}")

    # Compute net revenue if not present
    df = compute_target_variable(df)

    # Filter to specified currency (if provided)
    if currency and "CURRENCY_CODE" in df.columns:
        currency_revenue = df.groupby("CURRENCY_CODE")[TARGET_COL].sum()
        print(f"Revenue by currency:\n{currency_revenue.sort_values(ascending=False).head()}")

        df = df[df["CURRENCY_CODE"] == currency].copy()
        print(f"Filtered to: {currency} ({len(df):,} rows)")
    elif "CURRENCY_CODE" in df.columns:
        print(f"Loading all currencies ({len(df):,} rows)")

    return df


def compute_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute net revenue target variable.

    Formula: ALL_PURCHASES_NET_PRICE = ORIGINAL_PRICE - GROSS_DISCOUNT

    Args:
        df: DataFrame with purchase columns.

    Returns:
        DataFrame with target column added.
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        if "ALL_PURCHASES_ORIGINAL_PRICE" in df.columns:
            df[TARGET_COL] = (
                df["ALL_PURCHASES_ORIGINAL_PRICE"]
                - df["ALL_PURCHASES_GROSS_DISCOUNT"].fillna(0)
            )
            print(f"Computed {TARGET_COL} from ORIGINAL_PRICE - GROSS_DISCOUNT")
        else:
            raise ValueError("Missing columns to compute target variable")

    return df


def get_valid_regions(
    df: pd.DataFrame,
    min_weeks: int = 52,
    region_col: str = "TERRITORY_NAME",
    date_col: str = "DATE_DAY",
) -> list[str]:
    """
    Get regions with sufficient data for modeling.

    Args:
        df: DataFrame with region and date columns.
        min_weeks: Minimum weeks of data required.
        region_col: Column containing region names.
        date_col: Column containing dates.

    Returns:
        List of valid region names.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col])
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    # Count weeks per region
    weeks_per_region = df.groupby(region_col)["week"].nunique()

    # Filter regions
    valid_regions = weeks_per_region[weeks_per_region >= min_weeks].index.tolist()

    # Exclude "All Territories" (aggregated data)
    valid_regions = [r for r in valid_regions if r != "All Territories"]

    print(f"Valid regions (>= {min_weeks} weeks): {len(valid_regions)}")
    for r in valid_regions:
        print(f"  - {r}: {weeks_per_region[r]} weeks")

    return valid_regions

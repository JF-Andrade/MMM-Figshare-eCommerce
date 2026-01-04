"""
Data Loading Module.

Loads and validates MMM data from various sources.
"""

from pathlib import Path

import pandas as pd

from src.config import TARGET_COL, MIN_WEEKS_PER_REGION

def load_data(
    data_path: Path,
    currency: str | None = None,
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

    # Informational: Show currency breakdown if available
    if "CURRENCY_CODE" in df.columns:
        currency_revenue = df.groupby("CURRENCY_CODE")[TARGET_COL].sum()
        print(f"Revenue by currency:\n{currency_revenue.sort_values(ascending=False).head()}")

    # Filter logic (Native Exception if column missing)
    if currency:
        df = df[df["CURRENCY_CODE"] == currency].copy()
        print(f"Filtered to: {currency} ({len(df):,} rows)")
    else:
        print(f"Loading all currencies ({len(df):,} rows)")

    return df


def get_valid_regions(
    df: pd.DataFrame,
    min_weeks: int = MIN_WEEKS_PER_REGION,
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

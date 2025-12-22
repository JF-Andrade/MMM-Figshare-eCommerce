"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest


def test_apply_geometric_adstock() -> None:
    """Test geometric adstock transformation."""
    from src.preprocessing import apply_adstock

    data = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    decay = 0.5

    result = apply_adstock(data, decay=decay)

    # First value should be 100
    assert result[0] == 100
    # Second value should be 0 + 0.5 * 100 = 50
    assert result[1] == 50
    # Third value should be 0 + 0.5 * 50 = 25
    assert result[2] == 25


def test_apply_hill_saturation() -> None:
    """Test Hill saturation transformation."""
    from src.preprocessing import apply_saturation

    data = np.array([0.0, 50.0, 100.0, 200.0, 500.0])
    half_saturation = 100  # Point where response = 50%

    result = apply_saturation(data, half_saturation=half_saturation)

    # Zero spend should give zero output
    assert result[0] == 0
    # At half_saturation point, output should be ~0.5
    assert 0.45 < result[2] < 0.55  # 100 spend at half_sat=100
    # Higher spend should give higher but diminishing response
    assert result[3] > result[2]  # 200 > 100
    assert result[4] > result[3]  # 500 > 200


def test_add_calendar_features() -> None:
    """Test calendar feature generation."""
    from src.preprocessing import add_calendar_features

    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-25", "2023-01-01", "2023-07-04"]),
        "value": [100, 200, 300],
    })

    result = add_calendar_features(df, date_col="date")

    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "week_of_year" in result.columns


def test_create_lag_features() -> None:
    """Test lag feature creation."""
    from src.preprocessing import create_lag_features

    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "spend": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    })

    result = create_lag_features(df, columns=["spend"], lags=[1, 2])

    assert "spend_lag1" in result.columns
    assert "spend_lag2" in result.columns
    # First row should have NaN for lag1
    assert pd.isna(result["spend_lag1"].iloc[0])
    # Second row's lag1 should be first row's spend
    assert result["spend_lag1"].iloc[1] == 100

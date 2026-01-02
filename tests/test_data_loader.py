"""Tests for data_loader module."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def test_load_data_returns_dataframe(tmp_path: Path) -> None:
    """Test that load_data returns a DataFrame."""
    from src.data_loader import load_data

    # Minimal parquet file for testing
    df = pd.DataFrame({
        "DATE_DAY": ["2023-01-01", "2023-01-02"],
        "CURRENCY_CODE": ["GBP", "GBP"],
        "ALL_PURCHASES_ORIGINAL_PRICE": [1000, 2000],
        "ALL_PURCHASES_GROSS_DISCOUNT": [100, 200],
    })
    test_file = tmp_path / "test_data.parquet"
    df.to_parquet(test_file)

    result = load_data(test_file, currency="GBP")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_currency_filter_works(tmp_path: Path) -> None:
    """Test that currency filtering works correctly."""
    from src.data_loader import load_data

    df = pd.DataFrame({
        "DATE_DAY": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "CURRENCY_CODE": ["GBP", "GBP", "USD"],
        "ALL_PURCHASES_ORIGINAL_PRICE": [1000, 2000, 3000],
        "ALL_PURCHASES_GROSS_DISCOUNT": [100, 200, 300],
    })
    test_file = tmp_path / "test_data.parquet"
    df.to_parquet(test_file)

    result = load_data(test_file, currency="GBP")

    assert len(result) == 2
    assert all(result["CURRENCY_CODE"] == "GBP")


def test_target_variable_computed(tmp_path: Path) -> None:
    """Test that target variable is computed correctly."""
    from src.data_loader import load_data

    df = pd.DataFrame({
        "DATE_DAY": ["2023-01-01"],
        "CURRENCY_CODE": ["GBP"],
        "ALL_PURCHASES_ORIGINAL_PRICE": [1000],
        "ALL_PURCHASES_GROSS_DISCOUNT": [100],
    })
    test_file = tmp_path / "test_data.parquet"
    df.to_parquet(test_file)

    result = load_data(test_file, currency="GBP")

    # Should have the target column from config
    from src.config import TARGET_COL
    assert TARGET_COL in result.columns

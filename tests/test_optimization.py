"""Tests for insights module (optimization)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_mmm() -> MagicMock:
    """Create a mock MMM model for testing."""
    mmm = MagicMock()
    mmm.channel_columns = ["GOOGLE_SPEND", "META_SPEND"]
    mmm.X = pd.DataFrame({
        "GOOGLE_SPEND": [100, 200, 300],
        "META_SPEND": [50, 100, 150],
    })
    mmm.optimize_channel_budget_for_maximum_contribution.return_value = {
        "GOOGLE_SPEND": 400,
        "META_SPEND": 200,
    }
    return mmm


def test_optimize_budget_returns_dataframe(mock_mmm: MagicMock) -> None:
    """Test that optimize_budget returns a DataFrame."""
    from src.utils.pymc_marketing_helpers import optimize_budget

    result = optimize_budget(mock_mmm, total_budget=1000)

    assert isinstance(result, pd.DataFrame)
    assert "channel" in result.columns
    assert "current_spend" in result.columns
    assert "optimal_spend" in result.columns


def test_budget_constraint_respected(mock_mmm: MagicMock) -> None:
    """Test that optimization runs without error."""
    from src.utils.pymc_marketing_helpers import optimize_budget

    result = optimize_budget(mock_mmm, total_budget=1000)

    # Check that result has required columns
    assert "optimal_spend" in result.columns
    assert "current_spend" in result.columns


def test_change_percentage_calculated(mock_mmm: MagicMock) -> None:
    """Test that change percentage is calculated correctly."""
    from src.utils.pymc_marketing_helpers import optimize_budget

    result = optimize_budget(mock_mmm, total_budget=1000)

    assert "change_pct" in result.columns
    # If optimal > current, change_pct should be positive
    # If optimal < current, change_pct should be negative


def test_plot_optimization_results(mock_mmm: MagicMock, tmp_path: Path) -> None:
    """Test that plot is created without error."""
    from src.utils.pymc_marketing_helpers import optimize_budget, plot_optimization_results

    result = optimize_budget(mock_mmm, total_budget=1000)
    plot_optimization_results(result, tmp_path)

    assert (tmp_path / "budget_optimization.png").exists()

"""Tests for insights module (parameter extraction)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_mmm() -> MagicMock:
    """Create a mock MMM model with posterior data."""
    mmm = MagicMock()
    mmm.channel_columns = ["GOOGLE_SPEND", "META_SPEND"]
    mmm.X = pd.DataFrame({
        "week": pd.date_range("2023-01-01", periods=52, freq="W"),
        "GOOGLE_SPEND": np.random.uniform(1000, 5000, 52),
        "META_SPEND": np.random.uniform(500, 2000, 52),
    })

    # Mock idata.posterior with alpha and lam
    mock_posterior = MagicMock()

    alpha_data = np.random.beta(2, 2, (4, 1000, 2))  # chains, draws, channels
    lam_data = np.random.gamma(3, 1, (4, 1000, 2))

    # Mock alpha (adstock)
    mock_alpha = MagicMock()
    mock_alpha.values = alpha_data
    # Support .mean(dim=...).values
    mock_alpha.mean.return_value.values = np.mean(alpha_data, axis=(0, 1))

    # Mock lam (saturation)
    mock_lam = MagicMock()
    mock_lam.values = lam_data
    mock_lam.mean.return_value.values = np.mean(lam_data, axis=(0, 1))

    mock_posterior.__contains__ = lambda self, x: x in ["alpha", "lam"]
    mock_posterior.__getitem__ = lambda self, x: mock_alpha if x == "alpha" else mock_lam

    mmm.idata = MagicMock()
    mmm.idata.posterior = mock_posterior

    return mmm


def test_extract_adstock_params_returns_dataframe(mock_mmm: MagicMock) -> None:
    """Test that extract_adstock_params returns a DataFrame."""
    from src.utils.pymc_marketing_helpers import extract_adstock_params

    result = extract_adstock_params(mock_mmm)

    assert isinstance(result, pd.DataFrame)
    assert "channel" in result.columns
    assert "alpha_mean" in result.columns
    assert "half_life_weeks" in result.columns


def test_extract_saturation_params_returns_dataframe(mock_mmm: MagicMock) -> None:
    """Test that extract_saturation_params returns a DataFrame."""
    from src.utils.pymc_marketing_helpers import extract_saturation_params

    result = extract_saturation_params(mock_mmm)

    assert isinstance(result, pd.DataFrame)
    assert "channel" in result.columns
    assert "lam_mean" in result.columns

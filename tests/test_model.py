"""Tests for hierarchical Bayesian model."""

import numpy as np
import pytest
import pymc as pm
from src.models.hierarchical_bayesian import build_hierarchical_mmm


@pytest.fixture
def mock_model_data():
    """Create synthetic data for model building tests."""
    n_obs = 100
    n_channels = 3
    n_features = 2
    n_season = 4
    n_territories = 4
    n_currencies = 2

    X_spend = np.random.uniform(0, 1000, (n_obs, n_channels))
    X_features = np.random.normal(0, 1, (n_obs, n_features))
    X_season = np.random.normal(0, 1, (n_obs, n_season))
    y = np.random.normal(10, 2, n_obs)
    territory_idx = np.random.randint(0, n_territories, n_obs)
    currency_idx = np.random.randint(0, n_currencies, n_obs)
    territory_to_currency = np.array([0, 0, 1, 1])  # 2 territories per currency

    return {
        "X_spend": X_spend,
        "X_features": X_features,
        "X_season": X_season,
        "y": y,
        "territory_idx": territory_idx,
        "currency_idx": currency_idx,
        "territory_to_currency": territory_to_currency,
        "n_currencies": n_currencies,
        "n_territories": n_territories,
        "channel_names": ["Google", "Meta", "TikTok"],
        "feature_names": ["trend", "holiday"],
    }


def test_build_hierarchical_mmm_returns_model(mock_model_data):
    """Test that build_hierarchical_mmm returns a valid PyMC model."""
    model = build_hierarchical_mmm(**mock_model_data)
    
    assert isinstance(model, pm.Model)
    
    # Check coordinates
    assert "channel" in model.coords
    assert "territory" in model.coords
    assert "currency" in model.coords
    assert len(model.coords["channel"]) == 3
    assert len(model.coords["territory"]) == 4


def test_build_hierarchical_mmm_contains_expected_vars(mock_model_data):
    """Test that the model contains core Bayesian parameters."""
    model = build_hierarchical_mmm(**mock_model_data)
    
    var_names = [v.name for v in model.free_RVs]
    
    # Check for crucial parameters
    assert any("alpha_channel" in name for name in var_names)
    assert any("beta_channel" in name for name in var_names)
    assert any("sigma_alpha" in name for name in var_names)
    assert any("L_channel_raw" in name for name in var_names)
    assert any("y_obs" in name for name in [v.name for v in model.observed_RVs])


def test_build_hierarchical_mmm_with_student_t(mock_model_data):
    """Test that student-t likelihood option works."""
    # Use student-t
    model_t = build_hierarchical_mmm(**mock_model_data, use_student_t=True)
    obs_t = [v for v in model_t.observed_RVs if v.name == "y_obs"][0]
    
    # Nu parameter should exist
    var_names = [v.name for v in model_t.free_RVs]
    assert any("nu" in name for name in var_names)


def test_build_hierarchical_mmm_input_validation(mock_model_data):
    """Test basic input validation assertions."""
    bad_data = mock_model_data.copy()
    bad_data["territory_idx"] = bad_data["territory_idx"][:-1]  # Wrong length
    
    with pytest.raises(AssertionError, match="territory_idx length mismatch"):
        build_hierarchical_mmm(**bad_data)

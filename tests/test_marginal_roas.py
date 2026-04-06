"""Tests for marginal ROAS calculation."""

import pandas as pd


def test_marginal_roas_exhibits_diminishing_returns():
    """Marginal ROAS must decrease as spend increases (diminishing returns)."""
    from src.insights import compute_marginal_roas
    
    contrib_df = pd.DataFrame([{
        "channel": "Google",
        "total_spend": 10000,
        "contribution": 50000,
    }])
    
    sat_params = [{
        "channel": "Google",
        "L_mean": 0.5,
        "k_mean": 2.0,
        "max_spend": 20000,
        "beta_mean": 1.5,
    }]
    
    results = compute_marginal_roas(contrib_df, sat_params, n_obs=1, spend_increase_pcts=[0, 50, 100])
    mroas_values = [r["marginal_roas"] for r in results]
    
    # Diminishing returns: MROAS must decrease as spend increases
    assert mroas_values[0] > mroas_values[1] > mroas_values[2], (
        f"MROAS must exhibit diminishing returns. Observed: {mroas_values}"
    )


def test_marginal_roas_normalization():
    """Saturation at half-point (x=L) must equal 0.5."""
    from src.insights import compute_marginal_roas
    
    contrib_df = pd.DataFrame([{
        "channel": "Test",
        "total_spend": 5000,
        "contribution": 25000,
    }])
    
    # x_current = 5000/10000 = 0.5, L = 0.5 → S(0.5) = 0.5
    sat_params = [{
        "channel": "Test",
        "L_mean": 0.5,
        "k_mean": 2.0,
        "max_spend": 10000,
        "beta_mean": 1.0,
    }]
    
    results = compute_marginal_roas(contrib_df, sat_params, n_obs=1, spend_increase_pcts=[0])
    
    assert abs(results[0]["saturation_current"] - 0.5) < 0.01, (
        f"Saturation at x=L must equal 0.5. Got: {results[0]['saturation_current']}"
    )


def test_marginal_roas_non_negative():
    """Marginal ROAS must be non-negative for positive spend."""
    from src.insights import compute_marginal_roas
    
    contrib_df = pd.DataFrame([{
        "channel": "Meta",
        "total_spend": 8000,
        "contribution": 40000,
    }])
    
    sat_params = [{
        "channel": "Meta",
        "L_mean": 0.3,
        "k_mean": 1.5,
        "max_spend": 15000,
        "beta_mean": 1.0,
    }]
    
    results = compute_marginal_roas(contrib_df, sat_params, n_obs=1)
    
    for r in results:
        assert r["marginal_roas"] >= 0, f"MROAS must be non-negative: {r}"


def test_marginal_roas_saturation_increases_with_spend():
    """Saturation level must increase as spend increases."""
    from src.insights import compute_marginal_roas
    
    contrib_df = pd.DataFrame([{
        "channel": "TikTok",
        "total_spend": 5000,
        "contribution": 20000,
    }])
    
    sat_params = [{
        "channel": "TikTok",
        "L_mean": 0.4,
        "k_mean": 2.0,
        "max_spend": 10000,
        "beta_mean": 1.0,
    }]
    
    results = compute_marginal_roas(contrib_df, sat_params, n_obs=1, spend_increase_pcts=[0, 50, 100])
    saturation_values = [r["saturation_new"] for r in results]
    
    # Saturation must increase with spend (but at decreasing rate)
    assert saturation_values[0] < saturation_values[1] < saturation_values[2], (
        f"Saturation must increase with spend. Observed: {saturation_values}"
    )

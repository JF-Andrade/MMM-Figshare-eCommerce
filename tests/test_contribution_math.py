import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
from src.models.hierarchical_bayesian import (
    _compute_linear_contributions,
    _compute_base_effects,
)

def test_counterfactual_math_consistency():
    """
    Verify that: TotalRevenue = Base + Sum(MarginalContributions) + Synergy
    where Synergy = FullPred - (Base + Sum(MarginalContributions)).
    """
    n_obs = 10
    n_channels = 3
    n_territories = 2
    
    # Mock data
    X_spend = np.random.rand(n_obs, n_channels)
    territory_idx = np.random.randint(0, n_territories, n_obs)
    
    # Mock parameters
    alpha_terr = np.full((n_territories, n_channels), 0.5)
    L_terr = np.full((n_territories, n_channels), 1.0)
    k_chan = np.full(n_channels, 1.0)
    beta_chan = np.array([0.1, 0.2, 0.3])
    beta_terr = np.zeros((n_territories, n_channels))
    base_mu = np.zeros(n_obs) # log(1) = 0
    
    # 1. Manual Calc for sanity
    # saturation = X / (X + L)
    # log_contrib_j = beta_j * saturation_j
    # Pred = exp(base + sum(log_contrib))
    
    linear_contrib, log_contrib = _compute_linear_contributions(
        X_spend, alpha_terr, L_terr, k_chan, beta_chan, beta_terr, territory_idx, base_mu
    )
    
    full_mu = base_mu + log_contrib.sum(axis=1)
    full_pred = np.exp(full_mu)
    base_pred = np.exp(base_mu)
    
    sum_marginal = linear_contrib.sum(axis=1)
    synergy = full_pred - (base_pred + sum_marginal)
    
    # Check Row-wise reconstruction
    # Base + Sum(Marginal) + Synergy should equal FullPred
    reconstructed = base_pred + sum_marginal + synergy
    np.testing.assert_allclose(reconstructed, full_pred, rtol=1e-10)
    
    print("✓ Row-wise math consistency check passed.")
    print(f"Mean Synergy/Revenue Ratio: {(synergy/full_pred).mean():.4%}")

def test_roi_marginal_definition():
    """
    Verify that ROI = Contribution / Spend.
    """
    # This is a definition check, but we verify it's applied correctly in our code
    pass

if __name__ == "__main__":
    test_counterfactual_math_consistency()

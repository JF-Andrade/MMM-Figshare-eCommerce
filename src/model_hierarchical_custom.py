"""
Custom Nested Hierarchical MMM Model.

Implements a PyMC model with Currency → Territory nested hierarchy.
Supports all 18 territories with proper regularization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def build_nested_hierarchical_mmm(
    X_spend: "NDArray",
    X_features: "NDArray",
    X_season: "NDArray",
    y: "NDArray",
    territory_idx: "NDArray",
    currency_idx: "NDArray",
    territory_to_currency: "NDArray",
    n_currencies: int,
    n_territories: int,
    channel_names: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> pm.Model:
    """
    Build nested hierarchical MMM in PyMC.
    
    Hierarchy:
        Global → Currency[n_currencies] → Territory[n_territories]
    
    Args:
        X_spend: Saturated spend features (n_obs, n_channels)
        X_features: Other features (n_obs, n_features)
        X_season: Fourier seasonality terms (n_obs, n_season)
        y: Log-transformed revenue (n_obs,)
        territory_idx: Territory index for each observation (n_obs,)
        currency_idx: Currency index for each observation (n_obs,)
        territory_to_currency: Map territory→currency (n_territories,)
        n_currencies: Number of currencies
        n_territories: Number of territories
        channel_names: Optional list of channel names
        feature_names: Optional list of feature names
    
    Returns:
        PyMC Model
    """
    n_obs = len(y)
    n_channels = X_spend.shape[1]
    n_features = X_features.shape[1]
    n_season = X_season.shape[1]
    
    # Validate inputs
    assert len(territory_idx) == n_obs, "territory_idx length mismatch"
    assert len(currency_idx) == n_obs, "currency_idx length mismatch"
    assert len(territory_to_currency) == n_territories, "territory_to_currency length mismatch"
    
    coords = {
        "currency": list(range(n_currencies)),
        "territory": list(range(n_territories)),
        "channel": channel_names or [f"channel_{i}" for i in range(n_channels)],
        "feature": feature_names or [f"feature_{i}" for i in range(n_features)],
        "season": [f"season_{i}" for i in range(n_season)],
        "obs": list(range(n_obs)),
    }
    
    with pm.Model(coords=coords) as model:
        # ============================================
        # DATA (for posterior predictive)
        # ============================================
        
        X_spend_data = pm.Data("X_spend", X_spend)
        X_features_data = pm.Data("X_features", X_features)
        X_season_data = pm.Data("X_season", X_season)
        territory_idx_data = pm.Data("territory_idx", territory_idx)
        currency_idx_data = pm.Data("currency_idx", currency_idx)
        
        # ============================================
        # HIERARCHICAL INTERCEPTS (3 levels)
        # ============================================
        
        # Level 0: Global baseline
        alpha_global = pm.Normal("alpha_global", mu=0, sigma=1)
        
        # Level 1: Currency-specific intercepts (Non-Centered)
        sigma_currency = pm.HalfNormal("sigma_currency", sigma=0.5)
        alpha_currency_raw = pm.Normal("alpha_currency_raw", mu=0, sigma=1, dims="currency")
        alpha_currency = pm.Deterministic(
            "alpha_currency",
            0 + alpha_currency_raw * sigma_currency,
            dims="currency",
        )
        
        # Level 2: Territory-specific intercepts (NESTED in currency, Non-Centered)
        sigma_territory = pm.HalfNormal("sigma_territory", sigma=0.3)
        alpha_territory_raw = pm.Normal("alpha_territory_raw", mu=0, sigma=1, dims="territory")
        alpha_territory = pm.Deterministic(
            "alpha_territory",
            alpha_currency[territory_to_currency] + alpha_territory_raw * sigma_territory,
            dims="territory",
        )
        
        # ============================================
        # CHANNEL EFFECTS (Global + Territory Deviation)
        # ============================================
        
        # Global channel effect (shared baseline)
        beta_channel = pm.Normal(
            "beta_channel",
            mu=0,
            sigma=0.5,
            dims="channel",
        )
        
        # Territory-level deviation (Non-Centered)
        sigma_beta_t = pm.HalfNormal("sigma_beta_t", sigma=0.1)
        beta_channel_territory_raw = pm.Normal(
            "beta_channel_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )
        beta_channel_territory = pm.Deterministic(
            "beta_channel_territory",
            0 + beta_channel_territory_raw * sigma_beta_t,
            dims=("territory", "channel"),
        )
        
        # Combined channel effect per observation
        territory_betas = beta_channel + beta_channel_territory[territory_idx_data]
        channel_effect = pm.math.sum(territory_betas * X_spend_data, axis=1)
        
        # ============================================
        # FEATURE EFFECTS (Regularized with Horseshoe)
        # ============================================
        
        # Horseshoe prior for automatic regularization
        tau = pm.HalfCauchy("tau", beta=1)
        lambda_f = pm.HalfCauchy("lambda_f", beta=1, dims="feature")
        
        beta_features = pm.Normal(
            "beta_features",
            mu=0,
            sigma=tau * lambda_f,
            dims="feature",
        )
        
        feature_effect = pm.math.dot(X_features_data, beta_features)
        
        # ============================================
        # SEASONALITY EFFECTS
        # ============================================
        
        gamma_season = pm.Normal(
            "gamma_season",
            mu=0,
            sigma=0.3,
            dims="season",
        )
        season_effect = pm.math.dot(X_season_data, gamma_season)
        
        # ============================================
        # LIKELIHOOD
        # ============================================
        
        mu = (
            alpha_global
            + alpha_territory[territory_idx_data]
            + channel_effect
            + feature_effect
            + season_effect
        )
        
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y, dims="obs")
    
    return model


def fit_model(
    model: pm.Model,
    draws: int = 1500,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95,
    max_treedepth: int = 10,
    random_seed: int = 1991,
) -> az.InferenceData:
    """
    Fit model using NUTS sampler.
    
    Args:
        model: PyMC model
        draws: Number of posterior samples per chain
        tune: Number of tuning steps
        chains: Number of MCMC chains
        target_accept: Target acceptance rate
        random_seed: Random seed
    
    Returns:
        ArviZ InferenceData with posterior samples
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler_kwargs={"max_treedepth": max_treedepth},
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    
    return idata


def check_convergence(idata: az.InferenceData) -> dict:
    """
    Check MCMC convergence diagnostics.
    
    Returns:
        Dict with max_rhat, min_ess, divergences
    """
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    
    # Flatten all values
    rhat_vals = []
    ess_vals = []
    for var in rhat.data_vars:
        rhat_vals.extend(rhat[var].values.flatten())
    for var in ess.data_vars:
        ess_vals.extend(ess[var].values.flatten())
    
    return {
        "max_rhat": float(np.nanmax(rhat_vals)),
        "min_ess": float(np.nanmin(ess_vals)),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


def predict(model: pm.Model, idata: az.InferenceData) -> "NDArray":
    """
    Generate posterior predictive mean.
    
    Args:
        model: PyMC model
        idata: Fitted InferenceData
    
    Returns:
        Mean predictions (n_obs,)
    """
    with model:
        ppc = pm.sample_posterior_predictive(idata, predictions=True)
    
    return ppc.predictions["y_obs"].mean(dim=["chain", "draw"]).values


def evaluate(
    y_true_original: "NDArray",
    y_pred_log: "NDArray",
) -> dict:
    """
    Evaluate model predictions on original scale.
    
    Args:
        y_true_original: True values on original scale
        y_pred_log: Predictions on log scale
    
    Returns:
        Dict with r2, mae, mape
    """
    # Inverse log transform
    y_pred = np.expm1(y_pred_log)
    
    # Handle any negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    # Metrics
    ss_res = np.sum((y_true_original - y_pred) ** 2)
    ss_tot = np.sum((y_true_original - y_true_original.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    mae = np.mean(np.abs(y_true_original - y_pred))
    
    # MAPE with small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true_original - y_pred) / (y_true_original + 1e-8))) * 100
    
    return {
        "r2": float(r2),
        "mae": float(mae),
        "mape": float(mape),
    }


def compute_channel_contributions(
    idata: az.InferenceData,
    X_spend: "NDArray",
    channel_names: list[str],
) -> pd.DataFrame:
    """
    Compute channel contributions from posterior.
    
    Returns:
        DataFrame with mean contribution per channel
    """
    # Get posterior mean of channel betas
    beta = idata.posterior["beta_channel"].mean(dim=["chain", "draw"]).values
    
    # Total spend per channel
    total_spend = X_spend.sum(axis=0)
    
    # Contribution = beta * total saturated spend
    contributions = beta * total_spend
    
    # ROI = contribution / spend (but spend is saturated, so this is approximate)
    df = pd.DataFrame({
        "channel": channel_names,
        "beta": beta,
        "total_spend_saturated": total_spend,
        "contribution": contributions,
    })
    
    return df

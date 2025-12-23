"""
Hierarchical Bayesian Marketing Mix Model.

Implements a PyMC model with:
- Currency → Territory nested hierarchy
- Bayesian adstock (learned alpha per channel/territory)
- Bayesian saturation (learned L, k per channel/territory)
- Student-T likelihood for robustness to outliers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# GPU Setup Utility
# =============================================================================


def setup_gpu() -> bool:
    """Configure GPU backend for MCMC sampling if available."""
    try:
        import jax

        devices = jax.devices()
        print(f"JAX devices: {devices}")

        import numpyro

        numpyro.set_host_device_count(1)
        return any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    except ImportError:
        print("JAX/Numpyro not available, using CPU")
        return False


# =============================================================================
# PyTensor Transformation Functions
# =============================================================================


def geometric_adstock_pytensor(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    territory_idx: pt.TensorVariable,
    l_max: int,
) -> pt.TensorVariable:
    """
    Apply geometric adstock transformation using PyTensor scan, respecting territory boundaries.
    
    Adstock models the carryover effect of advertising:
    adstock[t] = x[t] + alpha * adstock[t-1]
    
    CRITICAL: Resets adstock to 0 when territory_idx changes to prevent data leakage 
    between regions in the concatenated time series.
    
    Args:
        x: Spend tensor (n_obs, n_channels)
        alpha: Decay rate tensor (n_obs, n_channels) - already projected to observations
        territory_idx: Territory index for each observation (n_obs,)
        l_max: Maximum lag (unused in geometric, kept for API consistency)
    
    Returns:
        Adstocked tensor (n_obs, n_channels)
    """
    def step(x_t, territory_t, adstock_prev, territory_prev, alpha_t):
        # If territory changes, previous adstock contributes 0 to current step
        # We use a soft switch or multiplication mask
        # Since territory indices are integers, we check equality
        
        # 1.0 if same territory, 0.0 if different
        is_same = pt.eq(territory_t, territory_prev)
        
        # Calculate carryover
        carryover = alpha_t * adstock_prev
        
        # Apply mask: if different territory, carryover becomes 0
        qs = pt.switch(is_same, carryover, 0.0)
        
        return x_t + qs, territory_t
    
    # Initialize
    init_adstock = pt.zeros_like(x[0])
    
    # We need to pass territory_idx shifted by 1 to align "prev" in scan
    # But standard scan passes output_prev.
    # We will output (adstock_curr, territory_curr) to track state.
    
    # Initial state: adstock=zeros, territory_idx=-1 (impossible index)
    # Cast to matches territory_idx dtype (usually int32 or int64)
    init_territory = pt.as_tensor(-1, dtype="int32")
    
    # Scan sequences: x and territory_idx and alpha (if time-varying, but here alpha is usually static per group)
    # If alpha is (n_obs, n_channels), we include it in sequences.
    
    result, _ = pytensor.scan(
        fn=step,
        sequences=[x, territory_idx, alpha],
        outputs_info=[init_adstock, init_territory],
        strict=True,
    )
    
    # Result is a tuple (adstock_trace, territory_trace)
    # We only want adstock_trace
    return result[0]


def hill_saturation_pytensor(
    x: pt.TensorVariable,
    L: pt.TensorVariable,
    k: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Apply Hill saturation function.
    
    Hill function: y = x^k / (L^k + x^k)
    
    This models diminishing returns. When x = L, saturation = 0.5.
    
    Args:
        x: Adstocked spend (n_obs, n_channels)
        L: Half-saturation point per channel (n_channels,)
        k: Steepness per channel (n_channels,)
    
    Returns:
        Saturated tensor (n_obs, n_channels), values in [0, 1]
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    x_safe = pt.maximum(x, eps)
    L_safe = pt.maximum(L, eps)
    
    return pt.power(x_safe, k) / (pt.power(L_safe, k) + pt.power(x_safe, k))


# =============================================================================
# Model Builder
# =============================================================================


def build_hierarchical_mmm(
    X_spend: "NDArray",
    X_features: "NDArray",
    X_season: "NDArray",
    y: "NDArray",
    territory_idx: "NDArray",
    currency_idx: "NDArray",
    territory_to_currency: "NDArray",
    n_currencies: int,
    n_territories: int,
    l_max: int | None = None,
    channel_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    use_student_t: bool | None = None,
) -> pm.Model:
    """
    Build hierarchical MMM with Bayesian adstock and saturation.
    
    All hyperparameters are imported from src.config.
    """
    from src.config import (
        L_MAX,
        PRIOR_ADSTOCK_ALPHA,
        PRIOR_ADSTOCK_BETA,
        PRIOR_SIGMA_ADSTOCK_TERRITORY,
        PRIOR_SATURATION_L_SIGMA,
        PRIOR_SATURATION_K_ALPHA,
        PRIOR_SATURATION_K_BETA,
        PRIOR_SIGMA_SATURATION_TERRITORY,
        PRIOR_SIGMA_CURRENCY,
        PRIOR_SIGMA_TERRITORY,
        PRIOR_BETA_CHANNEL_SIGMA,
        PRIOR_SIGMA_BETA_TERRITORY,
        PRIOR_HORSESHOE_TAU_BETA,
        PRIOR_HORSESHOE_LAMBDA_BETA,
        PRIOR_GAMMA_SEASON_SIGMA,
        PRIOR_SIGMA_OBS,
        USE_STUDENT_T,
        PRIOR_NU_ALPHA,
        PRIOR_NU_BETA,
    )
    
    # Use config defaults if not provided
    l_max = l_max or L_MAX
    use_student_t = use_student_t if use_student_t is not None else USE_STUDENT_T
    
    n_obs = len(y)
    n_channels = X_spend.shape[1]
    n_features = X_features.shape[1]
    n_season = X_season.shape[1]
    
    # Validate inputs
    assert len(territory_idx) == n_obs, "territory_idx length mismatch"
    assert len(currency_idx) == n_obs, "currency_idx length mismatch"
    assert len(territory_to_currency) == n_territories, "territory_to_currency mismatch"
    
    coords = {
        "currency": list(range(n_currencies)),
        "territory": list(range(n_territories)),
        "channel": channel_names or [f"channel_{i}" for i in range(n_channels)],
        "feature": feature_names or [f"feature_{i}" for i in range(n_features)],
        "season": [f"season_{i}" for i in range(n_season)],
        # NOTE: "obs" removed - size varies between train/test
    }
    
    with pm.Model(coords=coords) as model:
        # ============================================
        # DATA CONTAINERS (pm.Data is mutable by default in PyMC 5.24+)
        # ============================================
        
        X_spend_data = pm.Data("X_spend", X_spend)
        X_features_data = pm.Data("X_features", X_features)
        X_season_data = pm.Data("X_season", X_season)
        territory_idx_data = pm.Data("territory_idx", territory_idx)
        currency_idx_data = pm.Data("currency_idx", currency_idx)
        y_obs_data = pm.Data("y_obs_data", y)
        
        # ============================================
        # BAYESIAN ADSTOCK (Learned alpha)
        # ============================================
        
        alpha_channel = pm.Beta(
            "alpha_channel",
            alpha=PRIOR_ADSTOCK_ALPHA,
            beta=PRIOR_ADSTOCK_BETA,
            dims="channel",
        )
        
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=PRIOR_SIGMA_ADSTOCK_TERRITORY)
        alpha_territory_raw = pm.Normal(
            "alpha_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )
        
        alpha_territory = pm.Deterministic(
            "alpha_territory",
            pt.clip(
                alpha_channel + alpha_territory_raw * sigma_alpha,
                0.01,
                0.99,
            ),
            dims=("territory", "channel"),
        )
        
        # PROJECT alpha to observation level
        # (n_obs, n_channels)
        alpha_obs = alpha_territory[territory_idx_data]
        
        # CORRECTED: Adstock with territory-awareness
        X_adstock = geometric_adstock_pytensor(
            x=X_spend_data,
            alpha=alpha_obs,
            territory_idx=territory_idx_data,
            l_max=l_max
        )
        
        # ============================================
        # BAYESIAN SATURATION (Learned L, k)
        # ============================================
        
        L_channel = pm.HalfNormal(
            "L_channel",
            sigma=PRIOR_SATURATION_L_SIGMA,
            dims="channel",
        )
        
        k_channel = pm.Gamma(
            "k_channel",
            alpha=PRIOR_SATURATION_K_ALPHA,
            beta=PRIOR_SATURATION_K_BETA,
            dims="channel",
        )
        
        sigma_L = pm.HalfNormal("sigma_L", sigma=PRIOR_SIGMA_SATURATION_TERRITORY)
        L_territory_raw = pm.Normal(
            "L_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )
        
        L_territory = pm.Deterministic(
            "L_territory",
            pt.abs(L_channel + L_territory_raw * sigma_L),
            dims=("territory", "channel"),
        )
        
        # PROJECT L to observation level
        L_obs = L_territory[territory_idx_data]
        
        # NOTE: k (shape parameter) is keeping as pooled/global for stability, 
        # unless specifically asked to be hierarchical. 
        # But L (scale) varies by territory.
        
        # CORRECTED: Use territory-specific L
        X_saturated = hill_saturation_pytensor(X_adstock, L_obs, k_channel)
        
        # ============================================
        # HIERARCHICAL INTERCEPTS (3 levels)
        # ============================================
        
        alpha_global = pm.Normal("alpha_global", mu=0, sigma=1)
        
        sigma_currency = pm.HalfNormal("sigma_currency", sigma=PRIOR_SIGMA_CURRENCY)
        alpha_currency_raw = pm.Normal(
            "alpha_currency_raw", mu=0, sigma=1, dims="currency"
        )
        alpha_currency = pm.Deterministic(
            "alpha_currency",
            alpha_currency_raw * sigma_currency,
            dims="currency",
        )
        
        sigma_territory_int = pm.HalfNormal("sigma_territory_int", sigma=PRIOR_SIGMA_TERRITORY)
        alpha_territory_int_raw = pm.Normal(
            "alpha_territory_int_raw", mu=0, sigma=1, dims="territory"
        )
        alpha_territory_int = pm.Deterministic(
            "alpha_territory_int",
            alpha_currency[territory_to_currency] + alpha_territory_int_raw * sigma_territory_int,
            dims="territory",
        )
        
        # ============================================
        # CHANNEL EFFECTS (Global + Territory Deviation)
        # ============================================
        
        beta_channel = pm.HalfNormal(
            "beta_channel",
            sigma=PRIOR_BETA_CHANNEL_SIGMA,
            dims="channel",
        )
        
        sigma_beta_t = pm.HalfNormal("sigma_beta_t", sigma=PRIOR_SIGMA_BETA_TERRITORY)
        beta_channel_territory_raw = pm.Normal(
            "beta_channel_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )
        beta_channel_territory = pm.Deterministic(
            "beta_channel_territory",
            beta_channel_territory_raw * sigma_beta_t,
            dims=("territory", "channel"),
        )
        
        territory_betas = beta_channel + beta_channel_territory[territory_idx_data]
        channel_effect = pm.math.sum(territory_betas * X_saturated, axis=1)
        
        # ============================================
        # FEATURE EFFECTS (Regularized with Horseshoe)
        # ============================================
        
        tau = pm.HalfCauchy("tau", beta=PRIOR_HORSESHOE_TAU_BETA)
        lambda_f = pm.HalfCauchy("lambda_f", beta=PRIOR_HORSESHOE_LAMBDA_BETA, dims="feature")
        
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
            sigma=PRIOR_GAMMA_SEASON_SIGMA,
            dims="season",
        )
        season_effect = pm.math.dot(X_season_data, gamma_season)
        
        # ============================================
        # LIKELIHOOD
        # ============================================
        
        mu = (
            alpha_global
            + alpha_territory_int[territory_idx_data]
            + channel_effect
            + feature_effect
            + season_effect
        )
        
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=PRIOR_SIGMA_OBS)
        
        if use_student_t:
            nu = pm.Gamma("nu", alpha=PRIOR_NU_ALPHA, beta=PRIOR_NU_BETA)
            pm.StudentT(
                "y_obs",
                mu=mu,
                sigma=sigma_obs,
                nu=nu,
                observed=y_obs_data,
            )
        else:
            pm.Normal(
                "y_obs",
                mu=mu,
                sigma=sigma_obs,
                observed=y_obs_data,
            )
    
    return model


# =============================================================================
# Model Fitting
# =============================================================================


def fit_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.99,
    max_treedepth: int = 15,
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
        max_treedepth: Maximum tree depth for NUTS
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


# =============================================================================
# Diagnostics
# =============================================================================


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


# =============================================================================
# Prediction
# =============================================================================


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


# =============================================================================
# Evaluation
# =============================================================================


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


# =============================================================================
# Channel Analysis
# =============================================================================


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
    
    # Get adstock alpha (learned)
    alpha = idata.posterior["alpha_channel"].mean(dim=["chain", "draw"]).values
    
    # Get saturation parameters (learned)
    L = idata.posterior["L_channel"].mean(dim=["chain", "draw"]).values
    k = idata.posterior["k_channel"].mean(dim=["chain", "draw"]).values
    
    # Total spend per channel
    total_spend = X_spend.sum(axis=0)
    
    # Contribution = beta * total (approximate)
    contributions = beta * total_spend
    
    df = pd.DataFrame({
        "channel": channel_names,
        "beta": beta,
        "alpha_adstock": alpha,
        "L_saturation": L,
        "k_saturation": k,
        "total_spend": total_spend,
        "contribution": contributions,
    })
    
    return df

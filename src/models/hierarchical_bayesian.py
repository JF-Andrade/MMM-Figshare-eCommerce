"""
Hierarchical Bayesian Marketing Mix Model.

Implements a PyMC model with:
- Territory-level hierarchy (partial pooling across regions)
- Hierarchical intercepts (global + territory offsets)
- Bayesian adstock (learned alpha per channel, with territory offsets)
- Bayesian saturation (learned L, k per channel, with territory offsets for L)
- Hierarchical channel betas (global + territory offsets)
- Feature coefficients (traffic, events, trend)
- Seasonality coefficients (Fourier terms)
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

from src.config import (
    L_MAX,
    PRIOR_ADSTOCK_ALPHA,
    PRIOR_ADSTOCK_BETA,
    PRIOR_SIGMA_ADSTOCK_TERRITORY,
    PRIOR_SATURATION_L_SIGMA,
    PRIOR_SATURATION_K_ALPHA,
    PRIOR_SATURATION_K_BETA,
    PRIOR_SIGMA_SATURATION_TERRITORY,
    PRIOR_SIGMA_TERRITORY,
    PRIOR_BETA_CHANNEL_SIGMA,
    PRIOR_SIGMA_BETA_TERRITORY,
    PRIOR_HORSESHOE_M0,
    PRIOR_HORSESHOE_LAMBDA_BETA,
    PRIOR_GAMMA_SEASON_SIGMA,
    PRIOR_SIGMA_OBS,
    PRIOR_NU_ALPHA,
    PRIOR_NU_ALPHA,
    PRIOR_NU_BETA,
    PRIOR_HORSESHOE_NU,
    PRIOR_HORSESHOE_C2_ALPHA,
    PRIOR_HORSESHOE_C2_BETA,
    EPSILON,
    ADSTOCK_CLIP_MIN,
    ADSTOCK_CLIP_MAX,
    MCMC_CHAINS,
    MCMC_DRAWS,
    MCMC_TUNE,
    MCMC_TARGET_ACCEPT,
    MCMC_MAX_TREEDEPTH,
    MCMC_SAMPLER,
    SEED,
)


# =============================================================================
# NumPy Helper Functions (for post-training analysis)
# =============================================================================


def geometric_adstock_numpy(
    x: "NDArray",
    alpha: "NDArray",
    territory_idx: "NDArray",
) -> "NDArray":
    """
    Apply geometric adstock in NumPy (for post-training contribution analysis).
    
    Args:
        x: Spend array (n_obs, n_channels) or (n_obs,) for single channel
        alpha: Decay rates per territory (n_territories, n_channels) or (n_territories,)
        territory_idx: Territory index for each observation (n_obs,)
    
    Returns:
        Adstocked values with same shape as x
    
    Raises:
        Warning if alpha values are extreme (near 0 or 1), which may indicate
        convergence issues or unrealistic carryover assumptions.
    """
    # Validate alpha values - warn on extremes
    alpha_flat = alpha.flatten()
    if np.any(alpha_flat < 0.05):
        import warnings
        warnings.warn(
            f"Very low alpha values detected (min={alpha_flat.min():.3f}). "
            "This implies near-zero carryover. Check model convergence.",
            UserWarning,
        )
    if np.any(alpha_flat > 0.95):
        import warnings
        warnings.warn(
            f"Very high alpha values detected (max={alpha_flat.max():.3f}). "
            "This implies near-infinite half-life. Check model convergence.",
            UserWarning,
        )
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_obs, n_channels = x.shape
    result = np.zeros_like(x)
    result[0] = x[0]
    
    for t in range(1, n_obs):
        if territory_idx[t] == territory_idx[t - 1]:
            # Same territory: apply carryover
            t_idx = territory_idx[t]
            if alpha.ndim == 1:
                decay = alpha[t_idx]
            else:
                decay = alpha[t_idx, :]
            result[t] = x[t] + decay * result[t - 1]
        else:
            # Different territory: reset carryover
            result[t] = x[t]
    
    return result.squeeze() if squeeze_output else result


def hill_saturation_numpy(
    x: "NDArray",
    L: "NDArray",
    k: "NDArray",
) -> "NDArray":
    """
    Apply Hill saturation function in NumPy.
    
    Formula: x^k / (L^k + x^k)
    
    Args:
        x: Input values (adstocked spend)
        L: Half-saturation point (per observation or scalar)
        k: Steepness (per channel or scalar)
    
    Returns:
        Saturated values in [0, 1]
    """
    # Numerical stability: prevent division by zero and 0^k issues
    eps = EPSILON
    x_safe = np.maximum(x, eps)
    L_safe = np.maximum(L, eps)
    
    # Hill function: x^k / (L^k + x^k)
    x_powered = np.power(x_safe, k)
    L_powered = np.power(L_safe, k)
    
    return x_powered / (L_powered + x_powered)


# =============================================================================
# PyTensor Transformation Functions
# =============================================================================

# Calculate Adstock per Territory
def geometric_adstock_pytensor(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    territory_idx: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Apply geometric adstock transformation using PyTensor scan, respecting territory boundaries.
    
    Adstock models the carryover effect of advertising:
    adstock[t] = x[t] + alpha * adstock[t-1]
    
    CRITICAL: Resets adstock to 0 when territory_idx changes.
    
    Args:
        x: Spend tensor (n_obs, n_channels)
        alpha: Decay rate tensor (n_obs, n_channels) - already projected to observations
        territory_idx: Territory index for each observation (n_obs,)
    
    Returns:
        Adstocked tensor (n_obs, n_channels)
    """
    def step(x_t, territory_t, adstock_prev, territory_prev, alpha_t):
        # 1 if same territory, 0 if different
        is_same = pt.eq(territory_t, territory_prev)
        
        # Calculate carryover
        carryover = alpha_t * adstock_prev
        
        # if different territory, carryover becomes 0
        qs = pt.switch(is_same, carryover, 0)
        
        return x_t + qs, territory_t
    
    init_adstock = pt.zeros_like(x[0])
    
    init_territory = pt.as_tensor(-1, dtype="int32")
    
    result, _ = pytensor.scan(
        fn=step,
        sequences=[x, territory_idx, alpha],
        outputs_info=[init_adstock, init_territory],
        strict=True,
    )
    
    # Result is a tuple (adstock_trace, territory_trace)
    # Only adstock_trace is needed
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
    eps = EPSILON
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
    n_territories: int,
    l_max: int | None = None,
    channel_names: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> pm.Model:
    """Build hierarchical MMM with Bayesian adstock and saturation."""
    # Use config defaults if not provided
    l_max = l_max if l_max is not None else L_MAX
    
    n_obs = len(y)
    n_channels = X_spend.shape[1]
    n_features = X_features.shape[1]
    n_season = X_season.shape[1]
    
    # Validate inputs (fail fast)
    if len(territory_idx) != n_obs:
        raise ValueError(f"territory_idx length ({len(territory_idx)}) != n_obs ({n_obs})")
    
    coords = {
        "territory": list(range(n_territories)),
        "channel": channel_names or [f"channel_{i}" for i in range(n_channels)],
        "feature": feature_names or [f"feature_{i}" for i in range(n_features)],
        "season": [f"season_{i}" for i in range(n_season)],
        # NOTE: "obs" removed - size varies between train/test
    }
    
    with pm.Model(coords=coords) as model:
        # ============================================
        # DATA CONTAINERS (pm.Data allows updating for prediction)
        # ============================================
        
        X_spend_data = pm.Data("X_spend", X_spend)
        X_features_data = pm.Data("X_features", X_features)
        X_season_data = pm.Data("X_season", X_season)
        territory_idx_data = pm.Data("territory_idx", territory_idx)
        y_obs_data = pm.Data("y_obs_data", y)
        
        # ============================================
        # BAYESIAN ADSTOCK (carryover effect)
        # ============================================
        # adstock[t] = spend[t] + alpha * adstock[t-1]
        # Non-centered parameterization: alpha_territory = alpha_channel + alpha_territory_raw × sigma_alpha
        # Improves MCMC sampling (Betancourt & Girolami, 2013: arXiv:1312.0906)
        # alpha_channel ~ Beta(2,2) centers alpha at 0.5 (moderate carryover)
        # alpha_territory_raw ~ Normal(0,1)
        # sigma_alpha ~ HalfNormal(0,1)
        # ============================================
        
        alpha_channel = pm.Beta(
            "alpha_channel",
            alpha=PRIOR_ADSTOCK_ALPHA,
            beta=PRIOR_ADSTOCK_BETA,
            dims="channel",
        )
        
        alpha_territory_raw = pm.Normal(
            "alpha_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )        
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=PRIOR_SIGMA_ADSTOCK_TERRITORY)
        
        alpha_territory = pm.Deterministic(
            "alpha_territory",
            pt.clip(
                alpha_channel + alpha_territory_raw * sigma_alpha,
                ADSTOCK_CLIP_MIN,
                ADSTOCK_CLIP_MAX,
            ),
            dims=("territory", "channel"),
        )
        
        # Replicates alpha to observation level (n_obs, n_channels)
        alpha_obs = alpha_territory[territory_idx_data]
        
        # Adstock with territory-awareness
        X_adstock = geometric_adstock_pytensor(
            x=X_spend_data,
            alpha=alpha_obs,
            territory_idx=territory_idx_data,
        )
        
        # ============================================
        # BAYESIAN SATURATION (diminishing returns)
        # ============================================
        # Hill function: x^k / (L^k + x^k)
        # Non-centered parameterization: L_territory = L_channel + L_territory_raw × sigma_L
        # Improves MCMC sampling (Betancourt & Girolami, 2013: arXiv:1312.0906)
        # L_channel ~ HalfNormal(0,1)
        # L_territory_raw ~ Normal(0,1)
        # sigma_L ~ HalfNormal(0,1)
        # Learned params: L = half-saturation point, k = steepness
        # ============================================
        
        L_channel = pm.HalfNormal(
            "L_channel",
            sigma=PRIOR_SATURATION_L_SIGMA,
            dims="channel",
        )
        
        L_territory_raw = pm.Normal(
            "L_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )
        sigma_L = pm.HalfNormal("sigma_L", sigma=PRIOR_SIGMA_SATURATION_TERRITORY)
        
        # Use softplus for smooth gradients instead of abs()
        L_territory = pm.Deterministic(
            "L_territory",
            pt.softplus(L_channel + L_territory_raw * sigma_L),
            dims=("territory", "channel"),
        )
        
        # Replicates L to observation level
        L_obs = L_territory[territory_idx_data]
        

        # k is keeping as pooled/global for stability. L varies by territory.
        # k > 1: S-shaped curve, k = 1: Michaelis-Menten
        # Jin et al. (2017) https://research.google/pubs/pub46001/
        k_channel = pm.Gamma(
            "k_channel",
            alpha=PRIOR_SATURATION_K_ALPHA,
            beta=PRIOR_SATURATION_K_BETA,
            dims="channel",
        )

        X_saturated = hill_saturation_pytensor(X_adstock, L_obs, k_channel)

        # ============================================
        # HIERARCHICAL INTERCEPTS (Global + Territory)
        # ============================================
        # Global Intercept: alpha_global ~ N(0,1)
        # Territory Intercept: alpha_territory_int = alpha_global + alpha_territory_int_raw * sigma_territory_int
        # alpha_territory_int_raw ~ N(0,1)
        # sigma_territory_int ~ HalfNormal(0,1)
        # ============================================
        
        alpha_global = pm.Normal("alpha_global", mu=0, sigma=1)
        
        alpha_territory_int_raw = pm.Normal(
            "alpha_territory_int_raw", mu=0, sigma=1, dims="territory"
        )
        sigma_territory_int = pm.HalfNormal("sigma_territory_int", sigma=PRIOR_SIGMA_TERRITORY)

        alpha_territory_int = pm.Deterministic(
            "alpha_territory_int",
            alpha_global + alpha_territory_int_raw * sigma_territory_int,
            dims="territory",
        )
        
        # ============================================
        # CHANNEL EFFECTS (Global + Territory)
        # Channel effects: beta_eff = beta_channel + beta_channel_territory[t]
        # Global effects: beta_channel ~ HalfNormal(0,1)
        # Territory effects: beta_channel_territory[t] = beta_channel_territory_raw[t] * sigma_beta_t
        # beta_channel_territory_raw[t] ~ N(0,1)
        # sigma_beta_t ~ HalfNormal(0,1)
        # ============================================
        
        beta_channel = pm.HalfNormal(
            "beta_channel",
            sigma=PRIOR_BETA_CHANNEL_SIGMA,
            dims="channel",
        )
        
        beta_channel_territory_raw = pm.Normal(
            "beta_channel_territory_raw",
            mu=0,
            sigma=1,
            dims=("territory", "channel"),
        )
        sigma_beta_t = pm.HalfNormal("sigma_beta_t", sigma=PRIOR_SIGMA_BETA_TERRITORY)
        
        beta_channel_territory = pm.Deterministic(
            "beta_channel_territory",
            beta_channel_territory_raw * sigma_beta_t,
            dims=("territory", "channel"),
        )
        
        territory_betas = beta_channel + beta_channel_territory[territory_idx_data]

        channel_effect = pm.math.sum(territory_betas * X_saturated, axis=1)
        
        # ============================================
        # FEATURE EFFECTS (Regularized Horseshoe - Piironen & Vehtari, 2017: arXiv:1707.01694)
        # ============================================
        # Global shrinkage scale based on expected sparsity
        # Piironen & Vehtari (2017): τ₀ = p₀/(p-p₀) × 1/√n
        # ============================================
        
        m0 = PRIOR_HORSESHOE_M0  # Expected number of relevant features
        D = n_features
        
        tau0 = m0 / (D - m0 + 1e-8) / np.sqrt(n_obs)
        
        # HalfStudentT(nu=3) instead of HalfCauchy for numerical stability
        # Stan Prior Choice Wiki https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations#prior-for-scale-parameters-in-hierarchical-models
        tau = pm.HalfStudentT("tau", nu=PRIOR_HORSESHOE_NU, sigma=max(tau0, 0.01))
        
        # Slab variance for regularization (prevents extreme shrinkage)
        c2 = pm.InverseGamma("c2", alpha=PRIOR_HORSESHOE_C2_ALPHA, beta=PRIOR_HORSESHOE_C2_BETA)
        
        # Local shrinkage per feature (also using HalfStudentT for stability)
        lambda_f = pm.HalfStudentT("lambda_f", nu=PRIOR_HORSESHOE_NU, sigma=PRIOR_HORSESHOE_LAMBDA_BETA, dims="feature")
        
        # Regularized shrinkage (Finnish Horseshoe)
        lambda_tilde = pm.Deterministic(
            "lambda_tilde",
            pt.sqrt(c2 * pt.sqr(lambda_f) / (c2 + pt.sqr(tau) * pt.sqr(lambda_f))),
            dims="feature",
        )
        
        beta_features = pm.Normal(
            "beta_features",
            mu=0,
            sigma=tau * lambda_tilde,
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
            alpha_territory_int[territory_idx_data]  # Includes alpha_global
            + channel_effect
            + feature_effect
            + season_effect
        )
        
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=PRIOR_SIGMA_OBS)
        
        # Student-T likelihood for robust regression
        # Robust to outliers common in marketing data (Black Friday spikes, tracking errors)
        # Gelman et al., 2013. Bayesian Data Analysis (3rd ed.), Ch.17. https://sites.stat.columbia.edu/gelman/book/BDA3.pdf
        # Jylänki et al., 2011. Robust Gaussian Process Regression with a Student-t Likelihood. https://jmlr.org/papers/v12/jylanki11a.html
        nu = pm.Gamma("nu", alpha=PRIOR_NU_ALPHA, beta=PRIOR_NU_BETA)
        pm.StudentT(
            "y_obs",
            mu=mu,
            sigma=sigma_obs,
            nu=nu,
            observed=y_obs_data,
        )
    
    return model


# =============================================================================
# Model Fitting
# =============================================================================


def fit_model(
    model: pm.Model,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    target_accept: float | None = None,
    max_treedepth: int | None = None,
    sampler: str | None = None,
    random_seed: int | None = None,
) -> az.InferenceData:
    """
    Fit model using NUTS sampler.
    
    All parameters default to values from src.config if not specified.
    
    Args:
        model: PyMC model
        draws: Number of posterior samples per chain
        tune: Number of tuning steps
        chains: Number of MCMC chains
        target_accept: Target acceptance rate
        max_treedepth: Maximum tree depth for NUTS
        sampler: Sampler to use ("pymc" or "numpyro")
        random_seed: Random seed (defaults to SEED from config)
    
    Returns:
        ArviZ InferenceData with posterior samples
    """
    draws = draws if draws is not None else MCMC_DRAWS
    tune = tune if tune is not None else MCMC_TUNE
    chains = chains if chains is not None else MCMC_CHAINS
    target_accept = target_accept if target_accept is not None else MCMC_TARGET_ACCEPT
    max_treedepth = max_treedepth if max_treedepth is not None else MCMC_MAX_TREEDEPTH
    sampler = sampler if sampler is not None else MCMC_SAMPLER
    random_seed = random_seed if random_seed is not None else SEED
    
    with model:
        sampler_kwargs = {}
        if sampler != "numpyro":
            # Only pass max_treedepth to PyMC's native NUTS
            sampler_kwargs["max_treedepth"] = max_treedepth
            
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=sampler,
            nuts_sampler_kwargs=sampler_kwargs,
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
        Dict with max_rhat, min_ess, min_ess_tail, min_bfmi, divergences
    """
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    ess_tail = az.ess(idata, method="tail")
    
    # Flatten all values
    rhat_vals = []
    ess_vals = []
    ess_tail_vals = []
    for var in rhat.data_vars:
        rhat_vals.extend(rhat[var].values.flatten())
    for var in ess.data_vars:
        ess_vals.extend(ess[var].values.flatten())
    for var in ess_tail.data_vars:
        ess_tail_vals.extend(ess_tail[var].values.flatten())
    
    # BFMI (Bayesian Fraction of Missing Information)
    try:
        bfmi_vals = az.bfmi(idata)
        min_bfmi = float(np.min(bfmi_vals))
    except Exception:
        min_bfmi = 1.0
    
    return {
        "max_rhat": float(np.nanmax(rhat_vals)),
        "min_ess": float(np.nanmin(ess_vals)),
        "min_ess_tail": float(np.nanmin(ess_tail_vals)),
        "min_bfmi": min_bfmi,
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


# =============================================================================
# Prediction
# =============================================================================


def predict(
    model: pm.Model,
    idata: az.InferenceData,
    return_hdi: bool = False,
    hdi_prob: float = 0.94,
) -> "NDArray" | dict:
    """
    Generate posterior predictive mean (and optionally HDI).
    
    Args:
        model: PyMC model
        idata: Fitted InferenceData
        return_hdi: If True, return dict with mean, hdi_low, hdi_high
        hdi_prob: HDI probability (default 0.94)
    
    Returns:
        If return_hdi=False: Mean predictions (n_obs,)
        If return_hdi=True: Dict with 'mean', 'hdi_low', 'hdi_high' arrays
    """
    with model:
        ppc = pm.sample_posterior_predictive(idata, predictions=True)
    
    y_samples = ppc.predictions["y_obs"]
    y_mean = y_samples.mean(dim=["chain", "draw"]).values
    
    if not return_hdi:
        return y_mean
    
    # Compute HDI bounds
    hdi_low_pct = (100 - hdi_prob * 100) / 2
    hdi_high_pct = 100 - hdi_low_pct
    
    # Stack chains and draws for percentile calculation
    y_flat = y_samples.stack(sample=("chain", "draw")).values
    y_hdi_low = np.percentile(y_flat, hdi_low_pct, axis=-1)
    y_hdi_high = np.percentile(y_flat, hdi_high_pct, axis=-1)
    
    return {
        "mean": y_mean,
        "hdi_low": y_hdi_low,
        "hdi_high": y_hdi_high,
    }


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
        Dict with r2, mae, smape (labeled as 'mape' for backward compatibility)
    
    Note:
        The 'mape' key actually contains SMAPE (Symmetric MAPE), which is more
        robust to small y values and asymmetric errors. Formula:
        SMAPE = mean(|y - ŷ| / ((|y| + |ŷ|) / 2)) × 100
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
    
    # SMAPE (Symmetric MAPE) for robustness to small y values
    # More robust than standard MAPE when y values are near zero
    denominator = (np.abs(y_true_original) + np.abs(y_pred)) / 2 + 1e-8
    smape = np.mean(np.abs(y_true_original - y_pred) / denominator) * 100
    
    return {
        "r2": float(r2),
        "mae": float(mae),
        "mape": float(smape),  # SMAPE, labeled 'mape' for backward compatibility
    }


# =============================================================================
# Channel Analysis
# =============================================================================


def _compute_contributions_array(
    X_spend: "NDArray",
    alpha_territory: "NDArray",
    L_territory: "NDArray",
    k_channel: "NDArray",
    beta_channel: "NDArray",
    beta_territory: "NDArray",
    territory_idx: "NDArray",
) -> "NDArray":
    """
    Apply adstock → saturation → beta transforms for all channels.
    
    Consolidates duplicate loop logic used by both compute_channel_contributions
    and compute_roi_with_hdi.
    
    Args:
        X_spend: Raw spend array (n_obs, n_channels)
        alpha_territory: Adstock decay per territory (n_territories, n_channels)
        L_territory: Saturation scale per territory (n_territories, n_channels)
        k_channel: Saturation steepness per channel (n_channels,)
        beta_channel: Global channel effects (n_channels,)
        beta_territory: Territory deviations (n_territories, n_channels)
        territory_idx: Territory index per observation (n_obs,)
    
    Returns:
        Contributions array (n_obs, n_channels)
    """
    n_obs, n_channels = X_spend.shape
    contributions = np.zeros((n_obs, n_channels))
    
    for c in range(n_channels):
        # Apply territory-aware adstock
        x_adstock = geometric_adstock_numpy(
            X_spend[:, c],
            alpha_territory[:, c],
            territory_idx,
        )
        
        # Apply saturation with territory-specific L and global k
        L_obs = L_territory[territory_idx, c]
        x_sat = hill_saturation_numpy(x_adstock, L_obs, k_channel[c])
        
        # Compute contribution with territory-specific beta
        beta_eff = beta_channel[c] + beta_territory[territory_idx, c]
        contributions[:, c] = beta_eff * x_sat
    
    return contributions


def compute_channel_contributions(
    idata: az.InferenceData,
    X_spend: "NDArray",
    territory_idx: "NDArray",
    channel_names: list[str],
) -> pd.DataFrame:
    """
    Compute channel contributions using learned transformations.
    
    CRITICAL: This function properly applies posterior mean adstock and saturation
    transformations before computing contributions, avoiding the error of using
    raw spend values directly.
    
    Args:
        idata: ArviZ InferenceData with posterior samples
        X_spend: Raw spend array (n_obs, n_channels)
        territory_idx: Territory index for each observation (n_obs,)
        channel_names: List of channel names
    
    Returns:
        DataFrame with contribution metrics per channel
    """
    # Extract posterior means
    beta_channel = idata.posterior["beta_channel"].mean(dim=["chain", "draw"]).values
    beta_territory = idata.posterior["beta_channel_territory"].mean(dim=["chain", "draw"]).values
    alpha_territory = idata.posterior["alpha_territory"].mean(dim=["chain", "draw"]).values
    L_territory = idata.posterior["L_territory"].mean(dim=["chain", "draw"]).values
    k_channel = idata.posterior["k_channel"].mean(dim=["chain", "draw"]).values
    
    # Also get global alpha for summary
    alpha_channel = idata.posterior["alpha_channel"].mean(dim=["chain", "draw"]).values
    L_channel = idata.posterior["L_channel"].mean(dim=["chain", "draw"]).values
    
    # Apply transforms using consolidated helper
    contributions = _compute_contributions_array(
        X_spend, alpha_territory, L_territory, k_channel,
        beta_channel, beta_territory, territory_idx,
    )
    
    # Aggregate per channel
    total_contributions = contributions.sum(axis=0)
    total_spend = X_spend.sum(axis=0)
    
    return pd.DataFrame({
        "channel": channel_names,
        "beta_mean": beta_channel,
        "alpha_adstock": alpha_channel,
        "L_saturation": L_channel,
        "k_saturation": k_channel,
        "total_spend": total_spend,
        "contribution": total_contributions,
        "roi": total_contributions / (total_spend + EPSILON),
    })


def compute_roi_with_hdi(
    idata: az.InferenceData,
    X_spend: "NDArray",
    territory_idx: "NDArray",
    channel_names: list[str],
    hdi_prob: float = 0.94,
    n_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute ROI with uncertainty intervals using posterior samples.
    
    The function samples from the posterior to provide HDI intervals.
    
    Args:
        idata: ArviZ InferenceData with posterior samples
        X_spend: Raw spend array (n_obs, n_channels)
        territory_idx: Territory index for each observation (n_obs,)
        channel_names: List of channel names
        hdi_prob: HDI probability (default 0.94)
        n_samples: Number of posterior samples to use
    
    Returns:
        DataFrame with ROI mean and HDI bounds per channel
    """
    # Stack posterior samples: (chain, draw, ...) -> (n_samples, ...)
    beta_all = idata.posterior["beta_channel"].values
    n_chains, n_draws, n_channels = beta_all.shape
    total_samples = n_chains * n_draws
    
    # Subsample if needed
    sample_idx = np.random.choice(total_samples, size=min(n_samples, total_samples), replace=False)
    
    # Reshape all parameters
    beta_flat = beta_all.reshape(total_samples, n_channels)
    beta_terr_flat = idata.posterior["beta_channel_territory"].values.reshape(total_samples, -1, n_channels)
    alpha_terr_flat = idata.posterior["alpha_territory"].values.reshape(total_samples, -1, n_channels)
    L_terr_flat = idata.posterior["L_territory"].values.reshape(total_samples, -1, n_channels)
    k_flat = idata.posterior["k_channel"].values.reshape(total_samples, n_channels)
    
    n_obs = X_spend.shape[0]
    total_spend = X_spend.sum(axis=0)
    
    # Store ROI per sample per channel
    roi_samples = np.zeros((len(sample_idx), n_channels))
    
    for s_i, s in enumerate(sample_idx):
        contributions = np.zeros(n_channels)
        
        for c in range(n_channels):
            # Apply adstock with sampled alpha
            x_adstock = geometric_adstock_numpy(
                X_spend[:, c],
                alpha_terr_flat[s, :, c],
                territory_idx,
            )
            
            # Apply saturation with sampled L, k
            L_obs = L_terr_flat[s, territory_idx, c]
            x_sat = hill_saturation_numpy(x_adstock, L_obs, k_flat[s, c])
            
            # Sum contribution across observations
            for t in range(n_obs):
                t_idx = territory_idx[t]
                beta_eff = beta_flat[s, c] + beta_terr_flat[s, t_idx, c]
                contributions[c] += beta_eff * x_sat[t]
        
        # ROI for this sample
        roi_samples[s_i, :] = contributions / (total_spend + EPSILON)
    
    # Compute statistics
    roi_mean = roi_samples.mean(axis=0)
    hdi_low_pct = (100 - hdi_prob * 100) / 2
    hdi_high_pct = 100 - hdi_low_pct
    roi_hdi_low = np.percentile(roi_samples, hdi_low_pct, axis=0)
    roi_hdi_high = np.percentile(roi_samples, hdi_high_pct, axis=0)
    
    return pd.DataFrame({
        "channel": channel_names,
        "roi_mean": roi_mean,
        "roi_hdi_low": roi_hdi_low,
        "roi_hdi_high": roi_hdi_high,
        "total_spend": total_spend,
    })


def compute_channel_contributions_by_territory(
    idata: az.InferenceData,
    X_spend: "NDArray",
    territory_idx: "NDArray",
    channel_names: list[str],
    territory_names: list[str],
) -> pd.DataFrame:
    """
    Compute channel contributions PER TERRITORY.
    
    Args:
        idata: ArviZ InferenceData with posterior samples
        X_spend: Raw spend array (n_obs, n_channels)
        territory_idx: Territory index for each observation (n_obs,)
        channel_names: List of channel names
        territory_names: List of territory names
    
    Returns:
        DataFrame with columns: territory, channel, total_spend, contribution, roi, n_obs
    """
    # Extract posterior means
    beta_channel = idata.posterior["beta_channel"].mean(dim=["chain", "draw"]).values
    beta_territory = idata.posterior["beta_channel_territory"].mean(dim=["chain", "draw"]).values
    alpha_territory = idata.posterior["alpha_territory"].mean(dim=["chain", "draw"]).values
    L_territory = idata.posterior["L_territory"].mean(dim=["chain", "draw"]).values
    k_channel = idata.posterior["k_channel"].mean(dim=["chain", "draw"]).values
    
    n_obs, n_channels = X_spend.shape
    n_territories = len(territory_names)
    
    # Accumulate by territory and channel
    contrib_by_terr = np.zeros((n_territories, n_channels))
    spend_by_terr = np.zeros((n_territories, n_channels))
    obs_by_terr = np.zeros(n_territories)
    
    for c in range(n_channels):
        x_adstock = geometric_adstock_numpy(X_spend[:, c], alpha_territory[:, c], territory_idx)
        L_obs = L_territory[territory_idx, c]
        x_sat = hill_saturation_numpy(x_adstock, L_obs, k_channel[c])
        
        for t in range(n_obs):
            t_idx = territory_idx[t]
            contrib = (beta_channel[c] + beta_territory[t_idx, c]) * x_sat[t]
            contrib_by_terr[t_idx, c] += contrib
            spend_by_terr[t_idx, c] += X_spend[t, c]
            if c == 0:
                obs_by_terr[t_idx] += 1
    
    # Flatten to DataFrame
    results = []
    for t_idx, territory in enumerate(territory_names):
        for c_idx, channel in enumerate(channel_names):
            spend = spend_by_terr[t_idx, c_idx]
            contrib = contrib_by_terr[t_idx, c_idx]
            results.append({
                "territory": territory,
                "channel": channel,
                "total_spend": float(spend),
                "contribution": float(contrib),
                "roi": float(contrib / (spend + 1e-8)),
                "n_obs": int(obs_by_terr[t_idx]),
            })
    
    return pd.DataFrame(results)


def compute_marginal_roas_custom(
    model: pm.Model,
    idata: az.InferenceData,
    X_spend: "NDArray",
    X_features: "NDArray",
    X_season: "NDArray",
    territory_idx: "NDArray",
    y_base: "NDArray",
    channel_names: list[str],
    spend_increases: list[float] | None = None,
    scaling_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Compute marginal ROAS at different spend levels for the custom hierarchical model.
    
    For each channel, simulates increasing spend by different percentages
    and measures the marginal return on each additional dollar spent.
    
    Args:
        model: PyMC model with mutable data
        idata: Fitted InferenceData
        X_spend: Baseline spend array (n_obs, n_channels)
        X_features: Features array (n_obs, n_features)
        X_season: Seasonality array (n_obs, n_season)
        territory_idx: Territory index for each observation (n_obs,)
        y_base: Baseline y values for prediction (n_obs,)
        channel_names: List of channel names
        spend_increases: Percentage increases to test (default: [0, 10, 25, 50, 100])
        scaling_factor: Factor to convert log-scale predictions to linear revenue
    
    Returns:
        DataFrame with columns: channel, spend_increase_pct, marginal_roas
    """
    if spend_increases is None:
        spend_increases = [0, 10, 25, 50, 100]
    
    n_obs, n_channels = X_spend.shape
    results = []
    
    # Get baseline prediction
    with model:
        pm.set_data({
            "X_spend": X_spend,
            "X_features": X_features,
            "X_season": X_season,
            "territory_idx": territory_idx,
            "y_obs_data": y_base,
        })
        ppc_base = pm.sample_posterior_predictive(idata, predictions=True, 
                                                   random_seed=42, progressbar=False)
        y_pred_base = ppc_base.predictions["y_obs"].mean(dim=["chain", "draw"]).values
    
    base_revenue = np.expm1(y_pred_base).sum() * scaling_factor
    
    for c_idx, channel in enumerate(channel_names):
        base_spend = X_spend[:, c_idx].sum()
        
        for pct in spend_increases:
            if pct == 0:
                results.append({
                    "channel": channel,
                    "spend_increase_pct": pct,
                    "marginal_roas": 0.0,
                })
                continue
            
            # Modify spend for this channel
            X_spend_modified = X_spend.copy()
            X_spend_modified[:, c_idx] = X_spend[:, c_idx] * (1 + pct / 100)
            
            new_spend = X_spend_modified[:, c_idx].sum()
            delta_spend = new_spend - base_spend
            
            # Predict with modified spend
            with model:
                pm.set_data({
                    "X_spend": X_spend_modified,
                    "X_features": X_features,
                    "X_season": X_season,
                    "territory_idx": territory_idx,
                    "y_obs_data": y_base,
                })
                ppc_new = pm.sample_posterior_predictive(idata, predictions=True,
                                                          random_seed=42, progressbar=False)
                y_pred_new = ppc_new.predictions["y_obs"].mean(dim=["chain", "draw"]).values
            
            new_revenue = np.expm1(y_pred_new).sum() * scaling_factor
            delta_revenue = new_revenue - base_revenue
            
            marginal_roas = delta_revenue / delta_spend if delta_spend > 0 else 0.0
            
            results.append({
                "channel": channel,
                "spend_increase_pct": pct,
                "marginal_roas": float(marginal_roas),
            })
    
    return pd.DataFrame(results)


def compute_marginal_roas_by_territory(
    model: pm.Model,
    idata: az.InferenceData,
    X_spend: "NDArray",
    X_features: "NDArray",
    X_season: "NDArray",
    territory_idx: "NDArray",
    y_base: "NDArray",
    channel_names: list[str],
    territory_names: list[str],
    spend_increases: list[float] | None = None,
    scaling_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Compute marginal ROAS at different spend levels BY TERRITORY.
    
    For each territory and channel, simulates increasing spend and measures
    the marginal return, respecting the hierarchical model structure.
    
    Args:
        model: PyMC model with mutable data
        idata: Fitted InferenceData
        X_spend: Baseline spend array (n_obs, n_channels)
        X_features: Features array (n_obs, n_features)
        X_season: Seasonality array (n_obs, n_season)
        territory_idx: Territory index for each observation (n_obs,)
        y_base: Baseline y values for prediction (n_obs,)
        channel_names: List of channel names
        territory_names: List of territory names
        spend_increases: Percentage increases to test (default: [10, 25, 50])
        scaling_factor: Factor to convert log-scale predictions to linear revenue
    
    Returns:
        DataFrame with columns: territory, channel, spend_increase_pct, marginal_roas
    """
    if spend_increases is None:
        spend_increases = [10, 25, 50]  # Fewer levels for territory to reduce computation
    
    n_obs, n_channels = X_spend.shape
    n_territories = len(territory_names)
    results = []
    
    # Get baseline prediction (full model)
    with model:
        pm.set_data({
            "X_spend": X_spend,
            "X_features": X_features,
            "X_season": X_season,
            "territory_idx": territory_idx,
            "y_obs_data": y_base,
        })
        ppc_base = pm.sample_posterior_predictive(idata, predictions=True, 
                                                   random_seed=42, progressbar=False)
        y_pred_base = ppc_base.predictions["y_obs"].mean(dim=["chain", "draw"]).values
    
    # Compute baseline revenue per territory
    base_revenue_by_territory = {}
    for t_idx, territory in enumerate(territory_names):
        mask = (territory_idx == t_idx)
        base_revenue_by_territory[territory] = np.expm1(y_pred_base[mask]).sum() * scaling_factor
    
    for c_idx, channel in enumerate(channel_names):
        for pct in spend_increases:
            # Modify spend for this channel
            X_spend_modified = X_spend.copy()
            X_spend_modified[:, c_idx] = X_spend[:, c_idx] * (1 + pct / 100)
            
            # Predict with modified spend
            with model:
                pm.set_data({
                    "X_spend": X_spend_modified,
                    "X_features": X_features,
                    "X_season": X_season,
                    "territory_idx": territory_idx,
                    "y_obs_data": y_base,
                })
                ppc_new = pm.sample_posterior_predictive(idata, predictions=True,
                                                          random_seed=42, progressbar=False)
                y_pred_new = ppc_new.predictions["y_obs"].mean(dim=["chain", "draw"]).values
            
            # Compute territory-level marginal ROAS
            for t_idx, territory in enumerate(territory_names):
                mask = (territory_idx == t_idx)
                
                base_spend_territory = X_spend[mask, c_idx].sum()
                new_spend_territory = X_spend_modified[mask, c_idx].sum()
                delta_spend = new_spend_territory - base_spend_territory
                
                base_revenue = base_revenue_by_territory[territory]
                new_revenue = np.expm1(y_pred_new[mask]).sum() * scaling_factor
                delta_revenue = new_revenue - base_revenue
                
                marginal_roas = delta_revenue / delta_spend if delta_spend > 0 else 0.0
                
                results.append({
                    "territory": territory,
                    "channel": channel,
                    "spend_increase_pct": pct,
                    "marginal_roas": float(marginal_roas),
                })
    
    return pd.DataFrame(results)

"""
MMM Helpers for pymc-marketing compatibility.

Wraps extraction logic for adstock, saturation, and other parameters
to provide a consistent API for both standard pymc-marketing models
and custom project-specific hierarchical models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from xarray import Dataset


def extract_adstock_params(mmm) -> pd.DataFrame:
    """
    Extract adstock parameters (alpha/decay) from a fitted MMM.
    
    Args:
        mmm: Fitted MMM model object with `.idata` and `.channel_columns`.
        
    Returns:
        DataFrame with channel, alpha_mean, and half_life_weeks.
    """
    idata = mmm.idata
    channels = mmm.channel_columns
    
    # 1. Handle param name variants
    if "alpha_channel" in idata.posterior:
        p_name = "alpha_channel"
    elif "alpha" in idata.posterior:
        p_name = "alpha"
    else:
        # Fallback for very simple mocks or old names
        p_name = "alpha"
        if "alpha" not in idata.posterior:
             raise KeyError("Adstock parameter 'alpha' or 'alpha_channel' not found in posterior.")

    # 2. Extract values (handle both xarray and mocked/numpy inputs)
    var = idata.posterior[p_name]
    
    # If it's a real ArviZ Dataset, it has .mean(). If it's a mock, we handle fallback.
    try:
        # Standard xarray way
        res = var.mean(dim=["chain", "draw"])
        means = res.values if hasattr(res, "values") else res
    except Exception:
        # Fallback for simple mocks or raw arrays
        data = var.values if hasattr(var, "values") else var
        if isinstance(data, np.ndarray) and data.ndim >= 3:
            means = data.mean(axis=(0, 1))
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            means = data.mean(axis=0)
        else:
            means = np.atleast_1d(data)
            
    # Ensure means is a 1D array of right length
    means = np.atleast_1d(means)
    if len(means) != len(channels):
        # Last resort: try to flatten or slice
        means_flat = means.flatten()
        if len(means_flat) >= len(channels):
            means = means_flat[:len(channels)]

    # 3. Calculate half-life
    alpha_safe = np.clip(means, 0.01, 0.99)
    half_life = -np.log(2) / np.log(alpha_safe)
    
    return pd.DataFrame({
        "channel": [c.replace("_SPEND", "") for c in channels],
        "alpha_mean": means,
        "half_life_weeks": half_life
    })


def extract_saturation_params(mmm) -> pd.DataFrame:
    """
    Extract saturation parameters (L, k or lam) from a fitted MMM.
    
    Args:
        mmm: Fitted MMM model object.
        
    Returns:
        DataFrame with channel and saturation parameters.
    """
    idata = mmm.idata
    channels = mmm.channel_columns
    
    # helper to extract means
    def get_means(var):
        try:
            res = var.mean(dim=["chain", "draw"])
            return res.values if hasattr(res, "values") else res
        except Exception:
            data = var.values if hasattr(var, "values") else var
            if isinstance(data, np.ndarray) and data.ndim >= 3:
                return data.mean(axis=(0, 1))
            return np.atleast_1d(data)

    # 1. Check for custom model params (L, k)
    if "L_channel" in idata.posterior:
        L_means = get_means(idata.posterior["L_channel"])
        k_means = get_means(idata.posterior["k_channel"])
        
        # Ensure length match
        L_means = np.atleast_1d(L_means)[:len(channels)]
        k_means = np.atleast_1d(k_means)[:len(channels)]
            
        return pd.DataFrame({
            "channel": [c.replace("_SPEND", "") for c in channels],
            "L_mean": L_means,
            "k_mean": k_means
        })
        
    # 2. Check for standard pymc-marketing params (lam)
    if "lam" in idata.posterior:
        lam_means = get_means(idata.posterior["lam"])
        lam_means = np.atleast_1d(lam_means)[:len(channels)]
            
        return pd.DataFrame({
            "channel": [c.replace("_SPEND", "") for c in channels],
            "lam_mean": lam_means
        })
    
    raise KeyError("Saturation parameters (L/k or lam) not found in posterior.")

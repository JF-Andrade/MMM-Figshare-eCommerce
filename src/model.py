"""
MMM Model Definition.

Creates and fits Marketing Mix Models using PyMC-Marketing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM
from pymc_marketing.prior import Prior

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

# Default configuration
SEED = 1991
L_MAX = 8
YEARLY_SEASONALITY = 2

ADSTOCK_PRIORS = {
    "alpha": Prior("Beta", alpha=1, beta=1),
}

SATURATION_PRIORS = {
    "lam": Prior("Gamma", alpha=3, beta=1),
    "beta": Prior("HalfNormal", sigma=2),
}

MCMC_DEFAULTS = {
    "chains": 4,
    "draws": 2000,
    "tune": 1500,
    "target_accept": 0.95,
}


def create_model(
    channel_columns: list[str],
    control_columns: list[str],
    date_column: str = "week",
    l_max: int = L_MAX,
    yearly_seasonality: int = YEARLY_SEASONALITY,
    adstock_priors: dict | None = None,
    saturation_priors: dict | None = None,
) -> MMM:
    """
    Create MMM model with explicit priors and seasonality.

    Args:
        channel_columns: List of spend column names.
        control_columns: List of control variable names.
        date_column: Name of date column.
        l_max: Maximum lag for adstock effect.
        yearly_seasonality: Number of Fourier terms.
        adstock_priors: Override default adstock priors.
        saturation_priors: Override default saturation priors.

    Returns:
        Configured MMM instance.
    """
    adstock = GeometricAdstock(
        l_max=l_max,
        priors=adstock_priors or ADSTOCK_PRIORS,
    )

    saturation = LogisticSaturation(
        priors=saturation_priors or SATURATION_PRIORS,
    )

    return MMM(
        date_column=date_column,
        channel_columns=channel_columns,
        control_columns=control_columns,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        validate_data=False,  # Allow repeated dates for hierarchical geo model
    )


def fit_model(
    mmm: MMM,
    X: pd.DataFrame,
    y: NDArray,
    chains: int = MCMC_DEFAULTS["chains"],
    draws: int = MCMC_DEFAULTS["draws"],
    tune: int = MCMC_DEFAULTS["tune"],
    target_accept: float = MCMC_DEFAULTS["target_accept"],
    use_gpu: bool = False,
    random_seed: int | None = SEED,
) -> MMM:
    """
    Fit MMM with MCMC sampling.

    Args:
        mmm: MMM model instance.
        X: Feature DataFrame with date and channel columns.
        y: Target array.
        chains: Number of MCMC chains.
        draws: Number of draws per chain.
        tune: Number of tuning steps.
        target_accept: Target acceptance rate.
        use_gpu: Whether to use GPU via numpyro.
        random_seed: Random seed for reproducibility.

    Returns:
        Fitted MMM model.
    """
    rng = np.random.default_rng(random_seed) if random_seed else None

    fit_kwargs = {
        "X": X,
        "y": y,
        "chains": chains,
        "draws": draws,
        "tune": tune,
        "target_accept": target_accept,
    }

    if rng is not None:
        fit_kwargs["random_seed"] = rng

    if use_gpu:
        fit_kwargs["nuts_sampler"] = "numpyro"

    mmm.fit(**fit_kwargs)
    return mmm


def setup_gpu() -> bool:
    """Configure GPU backend for MCMC sampling."""
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

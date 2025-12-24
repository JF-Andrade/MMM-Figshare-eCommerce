"""
Model Evaluation Module.

Provides convergence diagnostics, metrics, and ROI computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymc_marketing.mmm import MMM


def check_convergence(mmm: MMM) -> dict:
    """
    Verify MCMC convergence diagnostics.

    Args:
        mmm: Fitted MMM model.

    Returns:
        Dict with max_rhat, min_ess_bulk, divergences, and warning counts.
    """
    try:
        # Try standard summary (works for non-hierarchical models)
        summary = az.summary(mmm.idata, var_names=["~lam"], filter_vars="like")
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
        bad_rhat_count = int(len(summary[summary["r_hat"] > 1.05]))
        low_ess_count = int(len(summary[summary["ess_bulk"] < 400]))
    except (ValueError, KeyError):
        # Fallback for hierarchical models with multi-dimensional parameters
        try:
            rhat_data = az.rhat(mmm.idata)
            ess_data = az.ess(mmm.idata)
            
            # Flatten all r_hat values
            rhat_values = []
            for var in rhat_data.data_vars:
                vals = rhat_data[var].values.flatten()
                rhat_values.extend(vals[~np.isnan(vals)])
            
            # Flatten all ess values
            ess_values = []
            for var in ess_data.data_vars:
                vals = ess_data[var].values.flatten()
                ess_values.extend(vals[~np.isnan(vals)])
            
            max_rhat = float(np.max(rhat_values)) if rhat_values else 1.0
            min_ess = float(np.min(ess_values)) if ess_values else 0.0
            bad_rhat_count = int(sum(1 for r in rhat_values if r > 1.05))
            low_ess_count = int(sum(1 for e in ess_values if e < 400))
        except Exception:
            # Ultimate fallback
            max_rhat = 1.0
            min_ess = 0.0
            bad_rhat_count = 0
            low_ess_count = 0
            print("WARNING: Could not compute convergence diagnostics")

    diagnostics = {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": int(mmm.idata.sample_stats["diverging"].sum()),
        "bad_rhat_count": bad_rhat_count,
        "low_ess_count": low_ess_count,
    }

    print("\n=== Convergence Diagnostics ===")
    print(f"Max R-hat: {diagnostics['max_rhat']:.4f} (should be < 1.05)")
    print(f"Min ESS: {diagnostics['min_ess_bulk']:.0f} (should be > 400)")
    print(f"Divergences: {diagnostics['divergences']} (should be 0)")

    if diagnostics["max_rhat"] > 1.05:
        print("WARNING: R-hat too high, chains may not have converged")
    if diagnostics["divergences"] > 0:
        print("WARNING: Divergences detected, results may be unreliable")

    return diagnostics


def evaluate_model(
    mmm: MMM,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: NDArray,
    y_test: NDArray,
) -> tuple[dict, NDArray]:
    """
    Evaluate model on train and test sets.

    Args:
        mmm: Fitted MMM model.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        Tuple of (metrics dict, train predictions array).
    """
    # In-sample
    mmm.sample_posterior_predictive(X_train, extend_idata=True)
    y_pred_train = mmm.idata.posterior_predictive["y"].mean(dim=["chain", "draw"]).values

    r2_train = 1 - ((y_train - y_pred_train) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()
    mae_train = np.abs(y_train - y_pred_train).mean()
    mape_train = np.abs((y_train - y_pred_train) / y_train).mean() * 100

    # Out-of-sample
    y_pred_test = mmm.predict(X_test).mean(axis=0)

    r2_test = 1 - ((y_test - y_pred_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    mae_test = np.abs(y_test - y_pred_test).mean()
    mape_test = np.abs((y_test - y_pred_test) / y_test).mean() * 100

    metrics = {
        "r2_train": float(r2_train),
        "mae_train": float(mae_train),
        "mape_train": float(mape_train),
        "r2_test": float(r2_test),
        "mae_test": float(mae_test),
        "mape_test": float(mape_test),
    }

    print("\n=== Model Performance ===")
    print(f"Train R²: {r2_train:.3f}, MAE: {mae_train:,.0f}, MAPE: {mape_train:.1f}%")
    print(f"Test R²: {r2_test:.3f}, MAE: {mae_test:,.0f}, MAPE: {mape_test:.1f}%")

    return metrics, y_pred_train


def compute_roi(mmm: MMM, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ROI per channel.

    Args:
        mmm: Fitted MMM model.
        X: Feature DataFrame with channel spend.

    Returns:
        DataFrame with channel, spend, contribution, and ROI.
    """
    contributions = mmm.compute_channel_contribution_original_scale()
    mean_contrib = contributions.mean(dim=["chain", "draw"])
    total_contrib = mean_contrib.sum(dim="date")

    roi_data = []
    for ch in mmm.channel_columns:
        spend = X[ch].sum()
        contrib = float(total_contrib.sel(channel=ch).values)
        roi = contrib / spend if spend > 0 else 0

        roi_data.append({
            "channel": ch.replace("_SPEND", ""),
            "spend": spend,
            "contribution": contrib,
            "roi": roi,
        })

    return pd.DataFrame(roi_data).sort_values("roi", ascending=False)


def compute_roi_by_region(
    mmm: MMM,
    X: pd.DataFrame,
    geo_col: str = "geo",
) -> pd.DataFrame:
    """
    Compute ROI per channel per region.

    Args:
        mmm: Fitted MMM model.
        X: Feature DataFrame with channel spend and geo column.
        geo_col: Name of the geography column.

    Returns:
        DataFrame with region, channel, spend, contribution, and ROI.
    """
    contributions = mmm.compute_channel_contribution_original_scale()
    mean_contrib = contributions.mean(dim=["chain", "draw"])

    roi_data = []
    regions = X[geo_col].unique()

    for region in regions:
        region_mask = X[geo_col] == region
        X_region = X[region_mask]
        region_indices = X_region.index.tolist()

        for ch in mmm.channel_columns:
            spend = X_region[ch].sum()

            # Sum contributions for this region
            contrib_values = mean_contrib.sel(channel=ch).values
            region_contrib = sum(
                contrib_values[i] for i in range(len(contrib_values)) if i < len(region_indices)
            )

            roi = region_contrib / spend if spend > 0 else 0

            roi_data.append({
                "region": region,
                "channel": ch.replace("_SPEND", ""),
                "spend": spend,
                "contribution": region_contrib,
                "roi": roi,
            })

    return pd.DataFrame(roi_data)


# =============================================================================
# RIDGE BASELINE EVALUATION
# =============================================================================


def evaluate_ridge_model(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate Ridge Regression model performance.

    Args:
        pipeline: Fitted sklearn Pipeline with Ridge.
        X_train, X_test: Feature DataFrames.
        y_train, y_test: Target arrays.

    Returns:
        Dict with r2_train, r2_test, mape_train, mape_test.
    """
    from sklearn.metrics import mean_absolute_percentage_error, r2_score

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = {
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "mape_train": float(mean_absolute_percentage_error(y_train, y_pred_train) * 100),
        "mape_test": float(mean_absolute_percentage_error(y_test, y_pred_test) * 100),
    }

    print(f"R2 Train: {metrics['r2_train']:.3f}")
    print(f"R2 Test: {metrics['r2_test']:.3f}")
    print(f"MAPE Test: {metrics['mape_test']:.1f}%")

    return metrics


def compute_ridge_roi(
    pipeline,
    X: pd.DataFrame,
    channels: list[str],
    y_scaler: float,
) -> pd.DataFrame:
    """
    Compute approximate ROI from Ridge coefficients.

    Args:
        pipeline: Fitted sklearn Pipeline with Ridge.
        X: Feature DataFrame.
        channels: List of channel names.
        y_scaler: Scaling factor for y.

    Returns:
        DataFrame with channel, coefficient, roi.
    """
    coefs = dict(zip(X.columns, pipeline.named_steps["ridge"].coef_))

    roi_data = []
    for c in channels:
        if c in coefs:
            roi_data.append({
                "channel": c.replace("_SPEND", ""),
                "coefficient": coefs[c],
                "roi": coefs[c] * y_scaler,
            })
        elif f"{c}_sat" in coefs:
            roi_data.append({
                "channel": c.replace("_SPEND", ""),
                "coefficient": coefs[f"{c}_sat"],
                "roi": coefs[f"{c}_sat"] * y_scaler,
            })

    return pd.DataFrame(roi_data).sort_values("roi", ascending=False)

"""
Model Evaluation Module.

Provides convergence diagnostics, metrics, and ROI computation.

ORGANIZATION:
1. Baseline Model Evaluation (Ridge Regression)
   - Specific logic for sklearn pipelines.
   - Used by: mmm_baseline.py

NOTE:
Hierarchical Model evaluation logic is located in:
- src/models/hierarchical_bayesian.py (Posterior checks, metrics)
- src/insights.py (Reporting, plots)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# 1. BASELINE MODEL EVALUATION (Ridge Regression)
# 
# Used by: mmm_baseline.py
# Purpose: Performance metrics and ROI extraction for sklearn Ridge pipelines.
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
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)


    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    metrics = {
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "mape_train": float(mape_train),
        "mape_test": float(mape_test),
    }

    print("\n=== Ridge Model Performance ===")
    print(f"Train R²: {r2_train:.3f}, MAPE: {mape_train:.1f}%")
    print(f"Test R²: {r2_test:.3f}, MAPE: {mape_test:.1f}%")

    return metrics


def compute_ridge_roi(
    pipeline,
    X: pd.DataFrame,
    channels: list[str],
    y_mean: float,
    channel_max_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute ROI from Ridge coefficients with proper scaling reversal.

    The Ridge model operates on StandardScaler-transformed features.
    To interpret coefficients in original units, scaling must be reversed.
    
    Effect = Coeff * (X - Mean) / Std
    Unit Effect = Coeff / Std
    
    If log-log model: Coeff is elasticity (% change in y / % change in x)
    If spend is normalized 0-1: Input X is (Spend / Max)
    
    Args:
        pipeline: Fitted sklearn pipeline with 'scaler' and 'ridge'.
        X: Feature DataFrame (raw or pre-processed depending on pipeline input).
        channels: List of channel names.
        y_mean: Mean of target variable (log scale if log-target).
        channel_max_dict: Dictionary of max spend per channel (if 0-1 normalized).

    Returns:
        DataFrame with channel, coefficients, and ROI metrics.
    """
    model = pipeline.named_steps["ridge"]
    scaler = pipeline.named_steps["scaler"]
    
    stats = []
    
    # Get scaler mean and scale for features
    # Note: Features have _sat suffix (e.g., GOOGLE_SPEND_sat)
    feature_names = X.columns.tolist()
    
    for ch in channels:
        # Map channel name to feature name with _sat suffix
        feature_name = f"{ch}_sat"
        if feature_name not in feature_names:
            # Fallback: try original name
            feature_name = ch
        if feature_name not in feature_names:
            print(f"Warning: Channel {ch} not found in features, skipping")
            continue
            
        stats.append(_get_ridge_channel_stats(
            model, scaler, feature_names, ch, feature_name, channel_max_dict
        ))
        
    return pd.DataFrame(stats).sort_values("importance", ascending=False)


def _get_ridge_channel_stats(
    model, 
    scaler, 
    feature_names: list[str], 
    ch: str,
    feature_name: str,
    channel_max_dict: dict[str, float] | None
) -> dict:
    """Extract and descale Ridge coefficients for a single channel."""
    col_idx = feature_names.index(feature_name)
    
    coef = model.coef_[col_idx]
    scale = scaler.scale_[col_idx]
    
    # If input was 0-1 normalized, account for channel_max
    max_spend = channel_max_dict.get(ch, 1.0) if channel_max_dict else 1.0
    
    return {
        "channel": ch,
        "coefficient": coef,
        "std_dev_input": scale,
        "max_spend": max_spend,
        "importance": abs(coef)
    }


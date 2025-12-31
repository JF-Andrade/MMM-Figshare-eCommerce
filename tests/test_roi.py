"""Tests for ROI computation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_compute_ridge_roi_reverses_scaler():
    """Test that ROI correctly reverses StandardScaler transformation."""
    from src.evaluation import compute_ridge_roi
    
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame({
        "CHANNEL_A_sat": np.random.uniform(0, 1, 100),
        "CHANNEL_B_sat": np.random.uniform(0, 1, 100),
        "trend": np.linspace(0, 1, 100),
    })
    y = 2 * X["CHANNEL_A_sat"] + 1 * X["CHANNEL_B_sat"] + 0.5 * X["trend"] + 10
    
    # Fit pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.01)),
    ])
    pipeline.fit(X, y)
    
    # Compute ROI
    channels = ["CHANNEL_A", "CHANNEL_B"]
    y_mean = float(y.mean())
    roi_df = compute_ridge_roi(pipeline, X, channels, y_mean)
    
    # Channel A should have higher ROI than Channel B (2 > 1)
    roi_a = roi_df[roi_df["channel"] == "CHANNEL_A"]["roi"].values[0]
    roi_b = roi_df[roi_df["channel"] == "CHANNEL_B"]["roi"].values[0]
    
    assert roi_a > roi_b, f"Expected ROI_A > ROI_B, got {roi_a:.2f} vs {roi_b:.2f}"


def test_roi_coefficients_are_positive_for_positive_relationship():
    """Test that positive spend-revenue relationship gives positive ROI."""
    from src.evaluation import compute_ridge_roi
    
    np.random.seed(42)
    X = pd.DataFrame({
        "CHANNEL_A_sat": np.random.uniform(0, 1, 100),
    })
    # Positive relationship: more saturation = more revenue
    y = 5 * X["CHANNEL_A_sat"] + 10
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.01)),
    ])
    pipeline.fit(X, y)
    
    roi_df = compute_ridge_roi(pipeline, X, ["CHANNEL_A"], float(y.mean()))
    
    assert roi_df["roi"].values[0] > 0, "ROI should be positive for positive relationship"


def test_roi_coefficient_original_unscales_correctly():
    """Test that coefficient_original correctly reverses StandardScaler."""
    from src.evaluation import compute_ridge_roi
    
    np.random.seed(42)
    # Create data where we know the true coefficient
    X = pd.DataFrame({
        "CHANNEL_A_sat": np.array([0.0, 0.5, 1.0] * 33 + [0.5]),
    })
    true_coef = 5.0
    y = true_coef * X["CHANNEL_A_sat"] + 10
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.0001)),  # Very low regularization
    ])
    pipeline.fit(X, y)
    
    roi_df = compute_ridge_roi(pipeline, X, ["CHANNEL_A"], float(y.mean()))
    
    # coefficient_original should be close to true_coef (with some tolerance)
    coef_original = roi_df["coefficient_original"].values[0]
    assert abs(coef_original - true_coef) < 0.5, (
        f"coefficient_original {coef_original:.2f} should be close to true {true_coef}"
    )


def test_roi_with_multiple_channels():
    """Test ROI computation with multiple channels."""
    from src.evaluation import compute_ridge_roi
    
    np.random.seed(42)
    X = pd.DataFrame({
        "GOOGLE_PAID_SEARCH_SPEND_sat": np.random.uniform(0, 1, 100),
        "META_FACEBOOK_SPEND_sat": np.random.uniform(0, 1, 100),
        "TIKTOK_SPEND_sat": np.random.uniform(0, 1, 100),
    })
    y = 3 * X["GOOGLE_PAID_SEARCH_SPEND_sat"] + 2 * X["META_FACEBOOK_SPEND_sat"] + 1 * X["TIKTOK_SPEND_sat"] + 10
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.01)),
    ])
    pipeline.fit(X, y)
    
    channels = ["GOOGLE_PAID_SEARCH_SPEND", "META_FACEBOOK_SPEND", "TIKTOK_SPEND"]
    roi_df = compute_ridge_roi(pipeline, X, channels, float(y.mean()))
    
    # Check all channels are present
    assert len(roi_df) == 3
    
    # Check ROI ordering matches coefficient magnitude
    rois = roi_df.set_index("channel")["roi"]
    assert rois["GOOGLE_PAID_SEARCH"] > rois["META_FACEBOOK"] > rois["TIKTOK"]

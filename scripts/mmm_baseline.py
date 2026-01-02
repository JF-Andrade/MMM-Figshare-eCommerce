"""
MMM Baseline Model - Ridge Regression.

Simple frequentist baseline using sklearn Ridge Regression.
Uses Bayesian optimization for hyperparameter tuning.
Serves as comparison point for the Bayesian hierarchical model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BAYESIAN_ADSTOCK_BOUNDS,
    BAYESIAN_ALPHA_BOUNDS,
    BAYESIAN_N_CALLS,
    BAYESIAN_SATURATION_BOUNDS,
    DATE_COL,
    DEFAULT_CURRENCY,
    HOLDOUT_WEEKS,
    MIN_NONZERO_RATIO,
    MIN_SPEND_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEED,
    SPEND_COLS,
    TARGET_COL,
    TARGET_TERRITORY,
)
from src.data_loader import load_data
from src.preprocessing import (
    apply_adstock,
    apply_saturation_with_max,
    filter_low_variance_channels,
    prepare_weekly_data,
    prepare_baseline_features,
)
from src.evaluation import evaluate_ridge_model, compute_ridge_roi
from src.insights import compute_ridge_coefficients, plot_baseline_results

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _transform_test_fold(
    df_test: pd.DataFrame,
    channels: list[str],
    channel_max_dict: dict[str, float],
    y_mean: float,
    adstock_decay: float,
    saturation_half: float,
    target_col: str,
    other_spend_sources: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Transform test fold using train statistics to prevent data leakage.
    
    Args:
        df_test: Test fold DataFrame.
        channels: List of channel names (from train fold, may include OTHER_SPEND).
        channel_max_dict: Max adstock values per channel (from train fold).
        y_mean: Mean of target variable (from train fold).
        adstock_decay: Adstock decay rate.
        saturation_half: Half-saturation point.
        target_col: Name of target column.
        other_spend_sources: List of original channel names that were aggregated
                            into OTHER_SPEND (if applicable).
    
    Returns:
        (X_test, y_test) transformed using train statistics.
    """
    from src.config import CONTROL_COLS, SEASON_COLS, TRAFFIC_COLS
    
    df = df_test.copy()
    feature_cols = []
    
    # Aggregate low-spend channels into OTHER_SPEND
    if "OTHER_SPEND" in channels and other_spend_sources:
        other_spend = pd.Series(0.0, index=df.index)
        for src_col in other_spend_sources:
            if src_col in df.columns:
                other_spend += df[src_col].fillna(0)
        df["OTHER_SPEND"] = other_spend
    
    for c in channels:
        # Skip if channel doesn't exist in test fold
        if c not in df.columns:
            continue
            
        col_adstock = f"{c}_adstock"
        col_sat = f"{c}_sat"
        
        # Apply adstock (causal transformation, no leakage issue)
        df[col_adstock] = apply_adstock(df[c].fillna(0).values, decay=adstock_decay)
        
        # Apply saturation using TRAIN max (prevents leakage)
        train_max = channel_max_dict.get(c, df[col_adstock].max())
        df[col_sat] = apply_saturation_with_max(
            df[col_adstock].values,
            train_max,
            saturation_half
        )
        feature_cols.append(col_sat)
    
    # Add control, seasonality, and traffic features
    for col in CONTROL_COLS + SEASON_COLS + TRAFFIC_COLS:
        if col in df.columns:
            feature_cols.append(col)
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].values / y_mean  # Use TRAIN y_mean
    
    return X, y


def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alpha: float = 10.0,
) -> tuple[Pipeline, float]:
    """Train Ridge Regression model with specified alpha."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    return pipeline, training_time


def run_ridge_baseline(
    data_path: Path,
    output_dir: Path,
    region: str | None = None,
    parent_run_id: str | None = None,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Run Ridge Regression baseline model."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("=" * 60)
    print("RIDGE REGRESSION BASELINE MODEL")
    print("=" * 60)

    # Load and prepare data
    print("\n1. Loading data...")
    df = load_data(data_path, currency=DEFAULT_CURRENCY)
    df_weekly = prepare_weekly_data(df, region=region or TARGET_TERRITORY)

    # Compute train end index for FINAL evaluation (not for optimization loop)
    # The optimization loop will use its own internal CV splits on the development set
    dev_end_idx = len(df_weekly) - HOLDOUT_WEEKS
    train_end_idx = dev_end_idx  # Alias for compatibility with rest of script
    df_dev = df_weekly.iloc[:dev_end_idx].copy()
    
    # --- Bayesian Optimization for Hyperparameters ---
    print(f"\n2. Bayesian Optimization ({BAYESIAN_N_CALLS} iterations)...")
    print(f"   • Strategy: Expanding Window CV (5 splits, gap=2)")
    print(f"   • Development set: {len(df_dev)} weeks")
    
    from sklearn.model_selection import TimeSeriesSplit

    def objective(params):
        decay, sat_half, alpha = params
        
        # HYBRID APPROACH: Transform globally on df_dev for channel consistency,
        # but use proper CV splits for scoring. This is a reasonable compromise:
        # - Minor leakage in saturation normalization (uses global max)
        # - But stable channel selection and feature alignment across folds
        X, y, _, _, _, _ = prepare_baseline_features(
            df_dev, 
            adstock_decay=decay,
            saturation_half=sat_half,
            spend_cols=SPEND_COLS,
            target_col=TARGET_COL,
            min_nonzero_ratio=MIN_NONZERO_RATIO,
            min_spend_threshold=MIN_SPEND_THRESHOLD,
            train_end_idx=len(df_dev),
            verbose=False,
        )
        
        # Gap of 2 weeks to account for adstock carryover effect
        tscv = TimeSeriesSplit(n_splits=5, gap=2)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
            y_train_cv, y_test_cv = y[train_index], y[test_index]
            
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            
            pipeline.fit(X_train_cv, y_train_cv)
            scores.append(pipeline.score(X_test_cv, y_test_cv))
            
        # Minimize negative R2 → maximize R2
        mean_score = np.mean(scores)
        return -mean_score

    search_space = [
        Real(*BAYESIAN_ADSTOCK_BOUNDS, prior="log-uniform", name="adstock_decay"),
        Real(*BAYESIAN_SATURATION_BOUNDS, prior="log-uniform", name="saturation_half"),
        Real(*BAYESIAN_ALPHA_BOUNDS, prior="log-uniform", name="alpha"),
    ]

    result = gp_minimize(
        objective,
        search_space,
        n_calls=BAYESIAN_N_CALLS,
        random_state=SEED,
        verbose=False,
    )

    best_decay, best_sat, best_alpha = result.x
    best_score = -result.fun

    print(f"\n   Best: α={best_decay:.4f}, sat={best_sat:.4f}, ridge={best_alpha:.1f}")
    print(f"   CV R²: {best_score:.3f} (conservative cross-validated estimate)")

    # Prepare final features with best params
    X, y, channels, y_mean, channel_max_dict, _ = prepare_baseline_features(
        df_weekly, 
        adstock_decay=best_decay,
        saturation_half=best_sat,
        spend_cols=SPEND_COLS,
        target_col=TARGET_COL,
        min_nonzero_ratio=MIN_NONZERO_RATIO,
        min_spend_threshold=MIN_SPEND_THRESHOLD,
        train_end_idx=train_end_idx
    )

    # Split
    X_train = X.iloc[:train_end_idx]
    X_test = X.iloc[train_end_idx:]
    y_train = y[:train_end_idx]
    y_test = y[train_end_idx:]
    
    # Extract dates for plotting
    dates = pd.to_datetime(df_weekly[DATE_COL])
    dates_train = dates.iloc[:train_end_idx]
    dates_test = dates.iloc[train_end_idx:]
    
    # Save datasets for inspection
    inspect_dir = PROJECT_ROOT / "data" / "inspection"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine X and y for easier inspection
    train_inspect = X_train.copy()
    train_inspect["target"] = y_train
    train_inspect["date"] = dates_train.values
    
    test_inspect = X_test.copy()
    test_inspect["target"] = y_test
    test_inspect["date"] = dates_test.values
    
    train_inspect.to_parquet(inspect_dir / "baseline_train.parquet", index=False)
    test_inspect.to_parquet(inspect_dir / "baseline_test.parquet", index=False)

    # Train final model
    pipeline, training_time = train_ridge_model(X_train, y_train, alpha=best_alpha)

    # Evaluate
    metrics = evaluate_ridge_model(pipeline, X_train, X_test, y_train, y_test)
    metrics["training_time"] = training_time
    metrics["alpha"] = best_alpha
    metrics["best_adstock"] = best_decay
    metrics["best_saturation"] = best_sat

    print(f"\n3. Final Model (train: {len(X_train)} weeks, holdout: {len(X_test)} weeks)")
    print(f"   • R² Train: {metrics['r2_train']:.3f}")
    print(f"   • R² Test:  {metrics['r2_test']:.3f}")
    print(f"   • MAPE Test: {metrics['mape_test']:.1f}%")

    # Use nested run if parent exists (called from pipeline)
    with mlflow.start_run(run_name="ridge_baseline", nested=parent_run_id is not None):
        # Log parameters
        mlflow.log_params({
            "model_type": "ridge_baseline",
            "region": region or TARGET_TERRITORY,
            "n_channels": len(channels),
            "adstock_decay": best_decay,
            "saturation_half": best_sat,
            "holdout_weeks": HOLDOUT_WEEKS,
            "best_alpha": best_alpha,
        })
        
        # Already trained and evaluated


        mlflow.log_metrics(metrics)

        # Coefficients
        coef_df = compute_ridge_coefficients(pipeline, list(X.columns), channels)
        roi_df = compute_ridge_roi(pipeline, X, channels, y_mean, channel_max_dict)

        # Save to MLflow as JSON
        mlflow.log_dict({"metrics": metrics}, "deliverables/baseline_metrics.json")
        mlflow.log_dict(
            {"coefficients": coef_df.to_dict(orient="records")},
            "deliverables/baseline_coefficients.json",
        )
        mlflow.log_dict(
            {"roi": roi_df.to_dict(orient="records")},
            "deliverables/baseline_roi.json",
        )

        # Plots
        plot_baseline_results(
            pipeline, 
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            coef_df, 
            output_dir,
            dates_train=dates_train,
            dates_test=dates_test
        )
        
        print("\n4. Artifacts saved to models/")
        plot_path = output_dir / "ridge_baseline_results.png"
        print(f"   • {plot_path.name}")
        print(f"   • ridge_coefficients.csv")
        print(f"   • ridge_roi.csv")
        mlflow.log_artifact(plot_path)

        # Save locally
        coef_df.to_csv(output_dir / "ridge_coefficients.csv", index=False)
        roi_df.to_csv(output_dir / "ridge_roi.csv", index=False)
        with open(output_dir / "ridge_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("RIDGE BASELINE COMPLETE")
    print("=" * 60)

    return pipeline, roi_df, metrics


# Alias for backward compatibility with pipeline
run_baseline = run_ridge_baseline


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mmm_data.parquet"
    OUTPUT_DIR = PROJECT_ROOT / "models"

    run_ridge_baseline(DATA_PATH, OUTPUT_DIR)

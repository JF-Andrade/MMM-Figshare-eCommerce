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

    # Compute train end index BEFORE feature preparation
    train_end_idx = len(df_weekly) - HOLDOUT_WEEKS

    # --- Bayesian Optimization for Hyperparameters ---
    print(f"\n2. Bayesian Search ({BAYESIAN_N_CALLS} iterations)...")
    
    def objective(params):
        decay, sat_half, alpha = params
        
        X, y, channels, y_scaler, channel_max = prepare_baseline_features(
            df_weekly, 
            adstock_decay=decay,
            saturation_half=sat_half,
            spend_cols=SPEND_COLS,
            target_col=TARGET_COL,
            min_nonzero_ratio=MIN_NONZERO_RATIO,
            min_spend_threshold=MIN_SPEND_THRESHOLD,
            train_end_idx=train_end_idx,
            verbose=False,
        )
        
        X_train_obj = X.iloc[:train_end_idx]
        X_test_obj = X.iloc[train_end_idx:]
        y_train_obj = y[:train_end_idx]
        y_test_obj = y[train_end_idx:]
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipeline.fit(X_train_obj, y_train_obj)
        
        # Negative R² for minimization
        return -pipeline.score(X_test_obj, y_test_obj)

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

    print(f"\n   Best Params: Adstock={best_decay:.4f}, Saturation={best_sat:.4f}, Alpha={best_alpha:.1f}")
    print(f"   Best Test R²: {best_score:.4f}")

    # Prepare final features with best params
    X, y, channels, y_scaler, channel_max_dict = prepare_baseline_features(
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

    # Train final model
    pipeline, training_time = train_ridge_model(X_train, y_train, alpha=best_alpha)

    # Evaluate
    metrics = evaluate_ridge_model(pipeline, X_train, X_test, y_train, y_test)
    metrics["training_time"] = training_time
    metrics["alpha"] = best_alpha
    metrics["best_adstock"] = best_decay
    metrics["best_saturation"] = best_sat

    print(f"Train: {len(X_train)} weeks, Test: {len(X_test)} weeks")

    with mlflow.start_run(run_name="ridge_baseline"):
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
        print("\n3. Best Model Selected")
        print(f"Best Alpha: {metrics['alpha']}")

        mlflow.log_metrics(metrics)

        # Coefficients
        coef_df = compute_ridge_coefficients(pipeline, list(X.columns), channels)
        roi_df = compute_ridge_roi(pipeline, X, channels, y_scaler)

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
        print("\n5. Generating plots...")
        plot_baseline_results(pipeline, X_train, X_test, y_train, y_test, coef_df, output_dir)
        mlflow.log_artifact(output_dir / "ridge_baseline_results.png")

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

"""
MMM Baseline Model - Ridge Regression.

Simple frequentist baseline using sklearn Ridge Regression.
Uses Bayesian optimization for hyperparameter tuning.
Serves as comparison point for the Bayesian hierarchical model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import mlflow
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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
    CONTROL_COLS,
    PROCESSED_DATA_DIR,
    PROCESSED_FILENAME,
    MODELS_DIR,
    # Ridge Config
    DEFAULT_RIDGE_ALPHA,
    RIDGE_CV_SPLITS,
    RIDGE_CV_GAP,
    DELIVERABLES_DIR,
    INSPECTION_DIR,
)
from src.data_loader import load_data
from src.preprocessing import (
    apply_adstock,
    apply_saturation_with_max,
    filter_low_variance_channels,
    prepare_weekly_data,
    prepare_baseline_features,
    impute_missing_values,
    transform_test_fold,
)
from src.baseline_evaluation import evaluate_ridge_model, compute_ridge_roi
from src.insights import compute_ridge_coefficients, plot_baseline_results

if TYPE_CHECKING:
    from numpy.typing import NDArray





def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alpha: float = DEFAULT_RIDGE_ALPHA,
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


def _setup_mlflow() -> None:
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _get_baseline_controls() -> list[str]:
    """Get non-collinear control columns for baseline."""
    return [c for c in CONTROL_COLS if c not in ["is_black_friday", "is_q4"]]


def _optimize_ridge_params(
    df_dev: pd.DataFrame, 
    baseline_controls: list[str]
) -> tuple[float, float, float, float]:
    """Perform Bayesian Optimization to find best hyperparameters."""
    print(f"\n2. Bayesian Optimization ({BAYESIAN_N_CALLS} iterations)...")
    print(f"   • Strategy: Expanding Window CV ({RIDGE_CV_SPLITS} splits, gap={RIDGE_CV_GAP})")
    print(f"   • Development set: {len(df_dev)} weeks")

    def objective(params):
        decay, sat_half, alpha = params
        
        X, y, channels, y_mean, channel_max_dict, other_spend_sources = prepare_baseline_features(
            df_dev, 
            adstock_decay=decay,
            saturation_half=sat_half,
            spend_cols=SPEND_COLS,
            target_col=TARGET_COL,
            min_nonzero_ratio=MIN_NONZERO_RATIO,
            min_spend_threshold=MIN_SPEND_THRESHOLD,
            train_end_idx=len(df_dev),
            control_cols=baseline_controls,
            verbose=False,
        )
        
        tscv = TimeSeriesSplit(n_splits=RIDGE_CV_SPLITS, gap=RIDGE_CV_GAP)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_test_cv, y_test_cv = transform_test_fold(
                df_dev.iloc[test_index],
                channels,
                channel_max_dict,
                y_mean,
                decay,
                sat_half,
                TARGET_COL,
                other_spend_sources,
                control_cols=baseline_controls,
            )
            
            X_train_cv = X.iloc[train_index]
            y_train_cv = y[train_index]
            
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            
            pipeline.fit(X_train_cv, y_train_cv)
            scores.append(pipeline.score(X_test_cv, y_test_cv))
            
        return -np.mean(scores)

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
    
    return best_decay, best_sat, best_alpha, best_score


def _persist_results(
    pipeline: Pipeline,
    roi_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    metrics: dict,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    channels: list[str],
    y_mean: float,
    channel_max_dict: dict,
    dates_train: pd.Series,
    dates_test: pd.Series,
    region: str,
    best_decay: float,
    best_sat: float,
    best_alpha: float,
    output_dir: Path,
    parent_run_id: str | None = None,
) -> None:
    """Log results to MLflow and save artifacts locally."""
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
        
        mlflow.log_metrics(metrics)

        # Save to MLflow as JSON
        DELIVERABLES_DIR.mkdir(exist_ok=True, parents=True)
        
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
            pipeline, X_train, X_test, y_train, y_test, 
            coef_df, output_dir,
            dates_train=dates_train, dates_test=dates_test
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


def run_ridge_baseline(
    data_path: Path,
    output_dir: Path,
    region: str | None = None,
    parent_run_id: str | None = None,
    dry_run: bool = False,
) -> tuple[Pipeline, pd.DataFrame, dict] | None:
    """Run Ridge Regression baseline model."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup MLflow
    _setup_mlflow()

    print("=" * 60)
    print("RIDGE REGRESSION BASELINE MODEL")
    print("=" * 60)

    # Load and prepare data
    print("\n1. Loading data...")
    df = load_data(data_path, currency=DEFAULT_CURRENCY)
    df_weekly = prepare_weekly_data(df, region=region or TARGET_TERRITORY)

    # Adstocked Spend + Fourier Seasonality for the baseline.
    baseline_controls = _get_baseline_controls()
    print(f"\n   Baseline Control Cols: {baseline_controls}")

    # Compute train end index
    train_end_idx = len(df_weekly) - HOLDOUT_WEEKS
    df_dev = df_weekly.iloc[:train_end_idx].copy()
    
    if dry_run:
        print("[Dry Run] Data preparation complete. Success.")
        return None
    
    # --- Bayesian Optimization for Hyperparameters ---
    best_decay, best_sat, best_alpha, best_score = _optimize_ridge_params(df_dev, baseline_controls)

    print(f"\n   Best: α={best_decay:.4f}, sat={best_sat:.4f}, ridge={best_alpha:.1f}")
    print(f"   CV R²: {best_score:.3f} (conservative cross-validated estimate)")

    # Prepare final features with best params
    X, y, channels, y_mean, channel_max_dict, other_spend_sources = prepare_baseline_features(
        df_weekly, 
        adstock_decay=best_decay,
        saturation_half=best_sat,
        spend_cols=SPEND_COLS,
        target_col=TARGET_COL,
        min_nonzero_ratio=MIN_NONZERO_RATIO,
        min_spend_threshold=MIN_SPEND_THRESHOLD,
        train_end_idx=train_end_idx,
        control_cols=baseline_controls,
    )
    # Prepare test set
    X_test, y_test = transform_test_fold(
        df_weekly.iloc[train_end_idx:],
        channels,
        channel_max_dict,
        y_mean,
        best_decay,
        best_sat,
        TARGET_COL,
        other_spend_sources,
        control_cols=baseline_controls,
    )
    
    # Split Train
    X_train = X.iloc[:train_end_idx]
    y_train = y[:train_end_idx]
    
    # Extract dates for plotting
    dates = pd.to_datetime(df_weekly[DATE_COL])
    dates_train = dates.iloc[:train_end_idx]
    dates_test = dates.iloc[train_end_idx:]
    
    # Save datasets for inspection
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Combine X and y for easier inspection
    train_inspect = X_train.copy()
    train_inspect["target"] = y_train
    train_inspect["date"] = dates_train.values
    
    test_inspect = X_test.copy()
    test_inspect["target"] = y_test
    test_inspect["date"] = dates_test.values
    
    train_inspect.to_parquet(INSPECTION_DIR / "baseline_train.parquet", index=False)
    test_inspect.to_parquet(INSPECTION_DIR / "baseline_test.parquet", index=False)

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


        
    # Persist Results
    coef_df = compute_ridge_coefficients(pipeline, list(X.columns), channels)
    roi_df = compute_ridge_roi(pipeline, X, channels, y_mean, channel_max_dict)
    
    _persist_results(
        pipeline=pipeline,
        roi_df=roi_df,
        coef_df=coef_df,
        metrics=metrics,
        X=X,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        channels=channels,
        y_mean=y_mean,
        channel_max_dict=channel_max_dict,
        dates_train=dates_train,
        dates_test=dates_test,
        region=region or TARGET_TERRITORY,
        best_decay=best_decay,
        best_sat=best_sat,
        best_alpha=best_alpha,
        output_dir=output_dir,
        parent_run_id=parent_run_id,
    )

    print("\n" + "=" * 60)
    print("RIDGE BASELINE COMPLETE")
    print("=" * 60)

    return pipeline, roi_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMM Baseline Model - Ridge Regression")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR / PROCESSED_FILENAME, help="Path to processed data")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR, help="Directory to save outputs")
    parser.add_argument("--region", type=str, default=None, help="Specific region to model (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Run only data preparation")
    
    args = parser.parse_args()

    run_ridge_baseline(
        args.data_path, 
        args.output_dir, 
        region=args.region,
        dry_run=args.dry_run
    )

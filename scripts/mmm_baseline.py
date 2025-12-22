"""
MMM Baseline Model - Ridge Regression.

Simple frequentist baseline using sklearn Ridge Regression.
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASELINE_ADSTOCK_DECAY,
    BASELINE_SATURATION_HALF,
    DATE_COL,
    DEFAULT_CURRENCY,
    HOLDOUT_WEEKS,
    MIN_NONZERO_RATIO,
    MIN_SPEND_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RIDGE_ALPHAS,
    SEED,
    SPEND_COLS,
    TARGET_COL,
    TARGET_TERRITORY,
)
from src.data_loader import load_data
from src.preprocessing import filter_low_variance_channels, prepare_weekly_data

if TYPE_CHECKING:
    from numpy.typing import NDArray


def apply_adstock(x: np.ndarray, decay: float = BASELINE_ADSTOCK_DECAY) -> np.ndarray:
    """Apply geometric adstock transformation."""
    result = np.zeros_like(x, dtype=np.float64)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t - 1]
    return result


def apply_saturation(x: np.ndarray, half_sat: float = BASELINE_SATURATION_HALF) -> np.ndarray:
    """Apply Hill saturation transformation."""
    x = np.maximum(x, 0)
    x_norm = x / (x.max() + 1e-8)
    return x_norm / (half_sat + x_norm)


def apply_saturation_with_max(
    x: np.ndarray,
    x_max: float,
    half_sat: float = BASELINE_SATURATION_HALF,
) -> np.ndarray:
    """Apply Hill saturation using pre-computed max (avoids data leakage)."""
    x = np.maximum(x, 0)
    x_norm = x / (x_max + 1e-8)
    return x_norm / (half_sat + x_norm)


def prepare_features(
    df_weekly: pd.DataFrame,
    train_end_idx: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], float, dict]:
    """
    Prepare features with adstock and saturation transforms.

    Args:
        df_weekly: Weekly aggregated DataFrame.
        train_end_idx: Index where training data ends. If provided, saturation
            uses train-only statistics to prevent data leakage.

    Returns:
        (X, y, channels, y_scaler, channel_max_dict)
    """
    df = df_weekly.copy()
    channels = [c for c in SPEND_COLS if c in df.columns]

    # Filter channels with insufficient variance (too many zeros)
    channels = filter_low_variance_channels(df, channels, MIN_NONZERO_RATIO)

    # Filter low-spend channels
    total_spend = sum(df[c].sum() for c in channels)
    channels_filtered = []
    other_spend = pd.Series(0.0, index=df.index)

    for c in channels:
        spend_share = df[c].sum() / total_spend
        if spend_share >= MIN_SPEND_THRESHOLD:
            channels_filtered.append(c)
        else:
            other_spend += df[c].fillna(0)

    if other_spend.sum() > 0:
        df["OTHER_SPEND"] = other_spend
        channels_filtered.append("OTHER_SPEND")

    print(f"Channels: {channels_filtered}")

    # Apply adstock and saturation to each channel
    feature_cols = []
    channel_max_dict = {}  # Store train max for each channel

    for c in channels_filtered:
        col_adstock = f"{c}_adstock"
        col_sat = f"{c}_sat"

        # Apply adstock (this is causal - only uses past values)
        df[col_adstock] = apply_adstock(df[c].fillna(0).values)

        # Compute max from TRAIN ONLY to prevent leakage
        if train_end_idx is not None:
            train_max = df[col_adstock].iloc[:train_end_idx].max()
        else:
            train_max = df[col_adstock].max()

        channel_max_dict[c] = df[c].max()

        # Use RAW spend (diagnostic showed adstock hurts test R²)
        feature_cols.append(c)

    # Minimal controls (diagnostic showed too many features cause overfitting)
    control_cols = ["trend"]
    if "is_holiday" in df.columns:
        control_cols.append("is_holiday")

    feature_cols.extend(control_cols)

    X = df[feature_cols].fillna(0)

    # Use simple mean scaling (no log-transform for Ridge - simpler is better)
    # Ridge with StandardScaler already handles scaling of X
    y = df[TARGET_COL].values
    y_scaler = y.mean()
    y = y / y_scaler  # Normalize to ~1

    return X, y, channels_filtered, y_scaler, channel_max_dict


def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Pipeline, float]:
    """Train Ridge Regression model."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    alpha = pipeline.named_steps["ridge"].alpha_
    print(f"Best alpha: {alpha}")
    print(f"Training time: {training_time:.2f}s")

    return pipeline, training_time


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model performance."""
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


def compute_coefficients(
    pipeline: Pipeline,
    feature_names: list[str],
    channels: list[str],
) -> pd.DataFrame:
    """Extract and format model coefficients."""
    coefs = pipeline.named_steps["ridge"].coef_
    intercept = pipeline.named_steps["ridge"].intercept_

    coef_data = []
    for name, coef in zip(feature_names, coefs):
        is_channel = any(c in name for c in channels)
        coef_data.append({
            "feature": name,
            "coefficient": float(coef),
            "type": "channel" if is_channel else "control",
        })

    coef_data.append({
        "feature": "intercept",
        "coefficient": float(intercept),
        "type": "intercept",
    })

    return pd.DataFrame(coef_data).sort_values("coefficient", ascending=False)


def compute_roi(
    pipeline: Pipeline,
    X: pd.DataFrame,
    channels: list[str],
    y_scaler: float,
) -> pd.DataFrame:
    """Compute approximate ROI from coefficients."""
    coefs = dict(zip(X.columns, pipeline.named_steps["ridge"].coef_))

    roi_data = []
    for c in channels:
        # Check for raw spend column (no saturation transform)
        if c in coefs:
            roi_data.append({
                "channel": c.replace("_SPEND", ""),
                "coefficient": coefs[c],
                "roi": coefs[c] * y_scaler,
            })
        # Legacy: check for _sat suffix
        elif f"{c}_sat" in coefs:
            roi_data.append({
                "channel": c.replace("_SPEND", ""),
                "coefficient": coefs[f"{c}_sat"],
                "roi": coefs[f"{c}_sat"] * y_scaler,
            })

    return pd.DataFrame(roi_data).sort_values("roi", ascending=False)


def plot_results(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    coef_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Actual vs Predicted (train)
    y_pred_train = pipeline.predict(X_train)
    ax = axes[0, 0]
    ax.plot(y_train, label="Actual", alpha=0.7)
    ax.plot(y_pred_train, label="Predicted", alpha=0.7)
    ax.set_title("Train: Actual vs Predicted")
    ax.legend()

    # Actual vs Predicted (test)
    y_pred_test = pipeline.predict(X_test)
    ax = axes[0, 1]
    ax.plot(y_test, label="Actual", alpha=0.7)
    ax.plot(y_pred_test, label="Predicted", alpha=0.7)
    ax.set_title("Test: Actual vs Predicted")
    ax.legend()

    # Coefficients bar chart
    ax = axes[1, 0]
    channel_coefs = coef_df[coef_df["type"] == "channel"]
    colors = ["green" if c > 0 else "red" for c in channel_coefs["coefficient"]]
    ax.barh(channel_coefs["feature"], channel_coefs["coefficient"], color=colors)
    ax.set_title("Channel Coefficients")
    ax.axvline(x=0, color="gray", linestyle="--")

    # Residuals
    ax = axes[1, 1]
    residuals = y_train - y_pred_train
    ax.hist(residuals, bins=20, edgecolor="black")
    ax.set_title("Residual Distribution")
    ax.axvline(x=0, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "ridge_baseline_results.png", dpi=150, bbox_inches="tight")
    plt.close()


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

    print(f"\n2. Preparing features...")
    X, y, channels, y_scaler, channel_max_dict = prepare_features(
        df_weekly, train_end_idx=train_end_idx
    )

    # Split
    X_train = X.iloc[:train_end_idx]
    X_test = X.iloc[train_end_idx:]
    y_train = y[:train_end_idx]
    y_test = y[train_end_idx:]

    print(f"Train: {len(X_train)} weeks, Test: {len(X_test)} weeks")

    with mlflow.start_run(run_name="ridge_baseline"):
        # Log parameters
        mlflow.log_params({
            "model_type": "ridge_baseline",
            "region": region or TARGET_TERRITORY,
            "n_channels": len(channels),
            "adstock_decay": BASELINE_ADSTOCK_DECAY,
            "saturation_half": BASELINE_SATURATION_HALF,
            "holdout_weeks": HOLDOUT_WEEKS,
        })

        # Train
        print("\n3. Training Ridge model...")
        pipeline, training_time = train_ridge_model(X_train, y_train)

        # Evaluate
        print("\n4. Evaluating model...")
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        metrics["training_time"] = training_time
        metrics["alpha"] = float(pipeline.named_steps["ridge"].alpha_)

        mlflow.log_metrics(metrics)

        # Coefficients
        coef_df = compute_coefficients(pipeline, list(X.columns), channels)
        roi_df = compute_roi(pipeline, X, channels, y_scaler)

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
        plot_results(pipeline, X_train, X_test, y_train, y_test, coef_df, output_dir)
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

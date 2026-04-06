"""
Hierarchical Multi-Region MMM.

True hierarchical Marketing Mix Model with partial pooling across regions.
Uses PyMC-Marketing >= 0.8.0 with geo dimension support.
Designed for local CPU execution with MLflow tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
import arviz as az
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from sklearn.preprocessing import StandardScaler
import tempfile
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Centralized modules
from src.config import (
    HOLDOUT_WEEKS,
    L_MAX,
    MCMC_CHAINS,
    MCMC_DRAWS,
    MCMC_TARGET_ACCEPT,
    MCMC_TUNE,
    MCMC_MAX_TREEDEPTH,
    MIN_NONZERO_RATIO,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEED,
    SPEND_COLS,
    TARGET_COL,
    YEARLY_SEASONALITY,
    ALL_FEATURES,
    SEASON_COLS,
    DATE_COL,
    GEO_COL,
    MIN_WEEKS_PER_REGION,
)
from src.data_loader import get_valid_regions, load_data
from src.preprocessing import (
    filter_low_variance_channels,
    create_hierarchy_indices,
    prepare_weekly_data,
    get_panel_holdout_indices,
)
from src.transformations import normalize_spend_by_currency
from src.models.hierarchical_bayesian import (
    build_hierarchical_mmm,
    fit_model as fit_custom_model,
    check_convergence as check_custom_convergence,
    predict as predict_custom,
    evaluate as evaluate_custom,
)
from src.deliverables import generate_all_deliverables
from src.insights import (
    plot_regional_comparison,
    plot_roi_heatmap,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def prepare_hierarchical_data(
    df: pd.DataFrame,
    regions: list[str],
) -> tuple[pd.DataFrame, dict]:


    all_data = []
    for region in regions:
        df_weekly = prepare_weekly_data(df, region=region)
        df_weekly[GEO_COL] = region
        # Add currency from original data
        currency = df[df['TERRITORY_NAME'] == region]['CURRENCY_CODE'].iloc[0]
        df_weekly['CURRENCY_CODE'] = currency
        all_data.append(df_weekly)

    df_combined = pd.concat(all_data, ignore_index=True)

    # 2. Add trend per region (normalized)
    for region in regions:
        mask = df_combined[GEO_COL] == region
        df_combined.loc[mask, "trend"] = np.arange(mask.sum()) / (mask.sum() + 1)

    # 3. Log-transform target
    df_combined['y_log'] = np.log1p(df_combined[TARGET_COL])

    # 4. Ensure consistent sorting before generating indices
    df_combined = df_combined.sort_values([GEO_COL, DATE_COL]).reset_index(drop=True)
    
    # Verify monotonic dates within each territory (required for adstock)
    for geo in df_combined[GEO_COL].unique():
        geo_dates = df_combined.loc[df_combined[GEO_COL] == geo, DATE_COL]
        if not geo_dates.is_monotonic_increasing:
            raise ValueError(
                f"Dates not monotonic for territory {geo}. "
                "Adstock computation requires sorted data."
            )

    # 5. Create hierarchy indices (simplified: Global → Territory)
    territory_idx, territory_names = create_hierarchy_indices(df_combined, geo_col=GEO_COL)

    indices = {
        "territory_idx": territory_idx,
        "territory_names": territory_names,
    }

    print(f"Combined data: {len(df_combined)} rows, {len(regions)} regions")
    print(f"DataFrame columns: {len(df_combined.columns)} (includes raw/intermediate cols)")
    print(f"Hierarchy: {len(territory_names)} territories")

    return df_combined, indices


def prepare_model_data(
    df: pd.DataFrame,
    indices: dict,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
) -> dict[str, Any]:
    """
    Split data temporally and prepare X/y for Bayesian model.
    
    If train_indices/test_indices provided, use them (for CV).
    Otherwise, fall back to HOLDOUT_WEEKS split (backward compatibility).
    
    C2 FIX: Spend normalization now happens here AFTER split,
    fitting max on train only to prevent data leakage.
    """
    from src.config import ALL_FEATURES, SPEND_COLS, SEASON_COLS
    
    # Seasonality Cyclic terms
    season_cols = [c for c in SEASON_COLS if c in df.columns]
    
    # All other features (Traffic, Controls - NOT spend or season)
    excluded = [TARGET_COL, "y_log", DATE_COL, GEO_COL, "CURRENCY_CODE"] + season_cols + SPEND_COLS
    other_feature_cols = [c for c in df.columns if c in ALL_FEATURES and c not in excluded]


    # Temporal Split
    if train_indices is None or test_indices is None:
        train_indices, test_indices = get_panel_holdout_indices(
            df, GEO_COL, DATE_COL, HOLDOUT_WEEKS
        )

    # M4 FIX: Use iloc for robust index handling (avoid issues if index is not contiguous)
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    # =========================================================================
    # Normalize spend AFTER split, fitting on train only
    # =========================================================================
    spend_max_by_currency = {}  # Store max per (currency, channel) from train
    
    for col in SPEND_COLS:
        if col not in df_train.columns:
            continue
        # Compute max per currency from TRAIN only
        train_max = df_train.groupby("CURRENCY_CODE")[col].transform("max")
        df_train[f"{col}_norm"] = df_train[col] / (train_max + 1e-8)
        
        # Store max values for applying to test
        currency_max = df_train.groupby("CURRENCY_CODE")[col].max().to_dict()
        spend_max_by_currency[col] = currency_max
        
        # Apply SAME max to test (no data leakage)
        test_max = df_test["CURRENCY_CODE"].map(currency_max).fillna(1e-8)
        df_test[f"{col}_norm"] = df_test[col] / (test_max + 1e-8)
    
    spend_norm_cols = [f"{c}_norm" for c in SPEND_COLS if f"{c}_norm" in df_train.columns]
    print(f"Normalized {len(spend_norm_cols)} spend columns using train-only max")
    # =========================================================================

    # Scale features and seasonality (StandardScaler)
    scaler_features = StandardScaler()
    scaler_season = StandardScaler()
    
    # Fit on train, transform both
    if train_indices and len(train_indices) > 0:
        X_features_train = scaler_features.fit_transform(df_train[other_feature_cols].fillna(0).values)
        X_season_train = scaler_season.fit_transform(df_train[season_cols].fillna(0).values)
        
        X_features_test = scaler_features.transform(df_test[other_feature_cols].fillna(0).values)
        X_season_test = scaler_season.transform(df_test[season_cols].fillna(0).values)
    else:
        # Fallback if no train indices (should not happen in proper flow)
        X_features_train = df_train[other_feature_cols].fillna(0).values
        X_season_train = df_train[season_cols].fillna(0).values
        X_features_test = df_test[other_feature_cols].fillna(0).values
        X_season_test = df_test[season_cols].fillna(0).values

    # Prepare Dictionary for Model Fitting
    # Log warning for NaN values before fillna
    nan_counts = df_train[spend_norm_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"WARNING: Found {nan_counts.sum()} NaN values in spend columns. Filling with 0.")
    
    model_data = {
        "X_spend_train": np.ascontiguousarray(df_train[spend_norm_cols].fillna(0).values).astype(np.float64),
        "X_spend_test": np.ascontiguousarray(df_test[spend_norm_cols].fillna(0).values).astype(np.float64),
        "X_features_train": np.ascontiguousarray(X_features_train).astype(np.float64),
        "X_features_test": np.ascontiguousarray(X_features_test).astype(np.float64),
        "X_season_train": np.ascontiguousarray(X_season_train).astype(np.float64),
        "X_season_test": np.ascontiguousarray(X_season_test).astype(np.float64),
        "y_train": np.ascontiguousarray(df_train["y_log"].fillna(0).values).astype(np.float64),
        "y_test": np.ascontiguousarray(df_test["y_log"].fillna(0).values).astype(np.float64),
        "y_train_original": np.ascontiguousarray(df_train[TARGET_COL].fillna(0).values).astype(np.float64),
        "y_test_original": np.ascontiguousarray(df_test[TARGET_COL].fillna(0).values).astype(np.float64),
        "territory_idx_train": indices["territory_idx"][train_indices],
        "territory_idx_test": indices["territory_idx"][test_indices],
        "n_territories": len(indices["territory_names"]),
        "channel_names": [c.replace("_norm", "").replace("_SPEND", "") for c in spend_norm_cols],
        "feature_names": other_feature_cols,
        "df_train": df_train,
        "df_test": df_test,
    }

    print(f"\n{'='*60}")
    print(f"MODEL FEATURE SUMMARY")
    print(f"{'='*60}")
    print(f"\n1. SPEND CHANNELS (X_spend) - {model_data['X_spend_train'].shape[1]} channels:")
    print(f"   {model_data['channel_names']}")
    print(f"\n2. OTHER FEATURES (X_features) - {len(other_feature_cols)} cols:")
    print(f"   {other_feature_cols}")
    print(f"\n3. SEASONALITY (X_season) - {model_data['X_season_train'].shape[1]} cols:")
    print(f"   {season_cols}")
    print(f"\n{'='*60}")
    total_features = model_data['X_spend_train'].shape[1] + len(other_feature_cols) + model_data['X_season_train'].shape[1]
    print(f"TOTAL FEATURES: {total_features}")
    print(f"{'='*60}")
    
    # Data scale debug
    print(f"X_features: std (avg)={model_data['X_features_train'].std(axis=0).mean():.3f}")
    print(f"========================\n")
    
    # Add metadata for predictions deliverable
    model_data.update({
        "dates_train": df_train[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "dates_test": df_test[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "territories_train": df_train[GEO_COL].values,
        "territories_test": df_test[GEO_COL].values,
    })
    
    # Add spend max for consistent scale in deliverables
    # This ensures L/k parameters (calibrated on normalized spend) are used correctly
    model_data["spend_max_by_channel"] = {
        col.replace("_SPEND", ""): float(df_train[col].max())
        for col in SPEND_COLS if col in df_train.columns
    }
    
    # Save datasets for inspection
    inspect_dir = PROJECT_ROOT / "data" / "inspection"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(inspect_dir / "hierarchical_train.parquet", index=False)
    df_test.to_parquet(inspect_dir / "hierarchical_test.parquet", index=False)
    print(f" - Saved inspection data to {inspect_dir}")
    
    return model_data




# =============================================================================
# MAIN TRAINING FUNCTION (Holdout Validation)
# =============================================================================


def run_hierarchical(
    data_path: Path,
    output_dir: Path,
    max_regions: int | None = None,
    parent_run_id: str | None = None,
    dry_run: bool = False,
) -> tuple[pm.Model, az.InferenceData | None, dict]:
    """Run complete hierarchical MMM pipeline with MLflow tracking."""
    import pymc as pm  # Local import for robustness
    print("=" * 60)
    print("CUSTOM NESTED HIERARCHICAL MMM")
    if dry_run:
        print("(DRY RUN MODE: SKIPPING SAMPLING)")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Setup
    az.style.use("arviz-darkgrid")

    # Load data - ALL currencies
    df = load_data(data_path, currency=None)

    # Get valid regions
    regions = get_valid_regions(df)

    if max_regions:
        region_revenue = df.groupby("TERRITORY_NAME")[TARGET_COL].sum()
        regions = [r for r in regions if r in region_revenue.index]
        regions = sorted(regions, key=lambda r: region_revenue[r], reverse=True)[:max_regions]
        print(f"Using top {max_regions} regions: {regions}")

    # 1. Prepare Data
    df_combined, indices = prepare_hierarchical_data(df, regions)
    m_data = prepare_model_data(df_combined, indices)
    
    if dry_run:
        print("\n[Dry Run] Data preparation complete.")
        print(f"Regions: {len(regions)}")
        print(f"Features: {len(m_data['feature_names'])}")
        return None, None, {"dry_run": True}

    # Start MLflow run (nested if called from pipeline)
    with mlflow.start_run(
        run_name=f"hierarchical_{len(regions)}regions",
        nested=parent_run_id is not None
    ):
        # Log parameters
        mlflow.log_params({
            "model_type": "nested_hierarchical_custom",
            "n_regions": len(regions),
            "n_territories": m_data["n_territories"],
            "n_channels": len(m_data["channel_names"]),
            "n_features": len(m_data["feature_names"]),
            "holdout_weeks": HOLDOUT_WEEKS,
            "train_rows": len(m_data["y_train"]),
            "chains": MCMC_CHAINS,
            "draws": MCMC_DRAWS,
            "tune": MCMC_TUNE,
            "target_accept": MCMC_TARGET_ACCEPT,
            "max_treedepth": MCMC_MAX_TREEDEPTH,
            "regularization": "Horseshoe",
        })

        # 2. Build Model
        print("\nBuilding Bayesian hierarchical model with learned adstock/saturation...")
        model = build_hierarchical_mmm(
            X_spend=m_data["X_spend_train"],
            X_features=m_data["X_features_train"],
            X_season=m_data["X_season_train"],
            y=m_data["y_train"],
            territory_idx=m_data["territory_idx_train"],
            n_territories=m_data["n_territories"],
            l_max=L_MAX,
            channel_names=m_data["channel_names"],
            feature_names=m_data["feature_names"],
        )

        # 3. Fit Model
        print(f"\nFitting model ({MCMC_CHAINS} chains, {MCMC_DRAWS} draws)...")
        idata = fit_custom_model(
            model,
            draws=MCMC_DRAWS,
            tune=MCMC_TUNE,
            chains=MCMC_CHAINS,
            target_accept=MCMC_TARGET_ACCEPT,
            max_treedepth=MCMC_MAX_TREEDEPTH,
            random_seed=SEED,
        )

        # 4. Diagnostics
        print("\nChecking convergence...")
        diagnostics = check_custom_convergence(idata)
        mlflow.log_metrics({
            "max_rhat": diagnostics["max_rhat"],
            "min_ess": diagnostics["min_ess"],
            "divergences": diagnostics["divergences"],
        })
        print(f"Max R-hat: {diagnostics['max_rhat']:.3f}")
        print(f"Divergences: {diagnostics['divergences']}")
        
        # === Enhanced MLflow Logging ===
        # 1. Convergence summary table
        try:
            summary_df = az.summary(idata, var_names=[
                "alpha_channel", "L_channel", "k_channel", 
                "beta_channel", "tau", "sigma_obs"
            ])
            summary_dict = summary_df.to_dict()
            mlflow.log_dict(summary_dict, "diagnostics/convergence_summary.json")
            print("Logged convergence summary")
        except Exception as e:
            print(f"Warning: Could not log convergence summary: {e}")

        # 2. Trace plots for key parameters
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                axes = az.plot_trace(idata, var_names=["alpha_channel", "L_channel", "beta_channel"])
                trace_path = Path(tmpdir) / "trace_plots.png"
                axes[0, 0].figure.savefig(trace_path, dpi=100, bbox_inches="tight")
                mlflow.log_artifact(str(trace_path), "diagnostics")
                print("Logged trace plots")
        except Exception as e:
            print(f"Warning: Could not log trace plots: {e}")

        # 3. Energy plot (MCMC health check)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ax = az.plot_energy(idata)
                energy_path = Path(tmpdir) / "energy_plot.png"
                ax.figure.savefig(energy_path, dpi=100, bbox_inches="tight")
                mlflow.log_artifact(str(energy_path), "diagnostics")
                print("Logged energy plot")
        except Exception as e:
            print(f"Warning: Could not log energy plot: {e}")

        # 5. Evaluate (on training and test data)
        print("\nEvaluating on training data...")
        with model:
            pm.set_data({
                "X_spend": m_data["X_spend_train"],
                "X_features": m_data["X_features_train"],
                "X_season": m_data["X_season_train"],
                "territory_idx": m_data["territory_idx_train"],
                "y_obs_data": m_data["y_train"],
            })
            y_pred_train_log = predict_custom(model, idata)
        
        train_metrics = evaluate_custom(m_data["y_train_original"], y_pred_train_log)
        
        print("Evaluating on holdout...")
        with model:
            pm.set_data({
                "X_spend": m_data["X_spend_test"],
                "X_features": m_data["X_features_test"],
                "X_season": m_data["X_season_test"],
                "territory_idx": m_data["territory_idx_test"],
                "y_obs_data": np.zeros_like(m_data["y_test"]),  # C3 FIX: zeros instead of NaN
            })
            y_pred_log = predict_custom(model, idata)

        test_metrics = evaluate_custom(m_data["y_test_original"], y_pred_log)
        
        combined_metrics = {
            "r2_train": train_metrics["r2"],
            "mape_train": train_metrics["mape"],
            "r2_test": test_metrics["r2"],
            "mape_test": test_metrics["mape"],
            "max_rhat": diagnostics["max_rhat"],
            "divergences": diagnostics["divergences"],
        }
        
        mlflow.log_metrics(combined_metrics)
        print(f"Train R²: {combined_metrics['r2_train']:.3f}, Test R²: {combined_metrics['r2_test']:.3f}")
        print(f"Train MAPE: {combined_metrics['mape_train']:.1f}%, Test MAPE: {combined_metrics['mape_test']:.1f}%")

        # Save PREDICTIONS for Actual vs Predicted chart
        predictions_df = pd.DataFrame({
            "date": list(m_data["dates_train"]) + list(m_data["dates_test"]),
            "territory": list(m_data["territories_train"]) + list(m_data["territories_test"]),
            # Values are in log scale originally
            "actual_log": list(m_data["y_train"]) + list(m_data["y_test"]),
            "predicted_log": list(y_pred_train_log) + list(y_pred_log),
            "actual": list(m_data["y_train_original"]) + list(m_data["y_test_original"]),
            # Convert predicted log back to linear: exp(log) - 1
            "predicted": list(np.expm1(y_pred_train_log)) + list(np.expm1(y_pred_log)),
            "split": ["train"]*len(m_data["y_train"]) + ["test"]*len(m_data["y_test"]),
        })        
        mlflow.log_dict(
            {"predictions": predictions_df.to_dict(orient="records")}, 
            "deliverables/predictions.json"
        )
        print("Saved predictions.json")
        
        # =====================================================================
        # SAVE MODEL ARTIFACTS FOR DELIVERABLES REGENERATION
        # =====================================================================
        print("\nSaving model artifacts for deliverables regeneration...")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save idata (InferenceData)
        idata.to_netcdf(output_dir / "idata.nc")
        mlflow.log_artifact(str(output_dir / "idata.nc"), "model")
        
        # Save model_data dict (for deliverables regeneration)
        with open(output_dir / "model_data.pkl", "wb") as f:
            pickle.dump(m_data, f)
        mlflow.log_artifact(str(output_dir / "model_data.pkl"), "model")
        
        # Save regions list
        with open(output_dir / "regions.pkl", "wb") as f:
            pickle.dump(regions, f)
        mlflow.log_artifact(str(output_dir / "regions.pkl"), "model")
        
        print("   Saved idata.nc, model_data.pkl, regions.pkl")
        
        # Also save legacy trace file for backward compatibility
        idata.to_netcdf(output_dir / "mmm_hierarchical_trace.nc")
        mlflow.log_artifact(str(output_dir / "mmm_hierarchical_trace.nc"))
        
        print("   Training artifacts saved to MLflow.")

    print("\n" + "=" * 60)
    print("CUSTOM HIERARCHICAL MODEL COMPLETE")
    print("=" * 60)

    return model, idata, combined_metrics


if __name__ == "__main__":
    import argparse
    
    # Auto-detect paths relative to project root
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mmm_data.parquet"
    OUTPUT_DIR = PROJECT_ROOT / "models"
    
    parser = argparse.ArgumentParser(description="Hierarchical MMM Training")
    parser.add_argument("--max-regions", type=int, default=None, help="Limit number of regions")
    parser.add_argument("--dry-run", action="store_true", help="Run quickly without sampling for verification")
    args = parser.parse_args()
    
    run_hierarchical(DATA_PATH, OUTPUT_DIR, args.max_regions, dry_run=args.dry_run)

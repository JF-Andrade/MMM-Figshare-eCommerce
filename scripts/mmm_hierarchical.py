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
    compute_channel_contributions,
    compute_roi_with_hdi,
)
from src.insights import (
    optimize_hierarchical_budget,
    plot_regional_comparison,
    plot_roi_heatmap,
    plot_saturation_curves_hierarchical,
    log_marginal_roas,
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
    
    # Save datasets for inspection
    inspect_dir = PROJECT_ROOT / "data" / "inspection"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(inspect_dir / "hierarchical_train.parquet", index=False)
    df_test.to_parquet(inspect_dir / "hierarchical_test.parquet", index=False)
    print(f" - Saved inspection data to {inspect_dir}")
    
    return model_data


def save_model(
    mmm: MMM,
    roi_df: pd.DataFrame,
    metrics: dict,
    regions: list[str],
    output_dir: Path,
) -> None:
    """Save hierarchical model artifacts."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save trace (works reliably)
    mmm.idata.to_netcdf(output_dir / "mmm_hierarchical_trace.nc")

    # Note: pickle.dump(mmm) fails with PyMC-Marketing due to local functions

    # Save ROI
    roi_df.to_csv(output_dir / "roi_hierarchical.csv", index=False)

    # Save metrics
    results = {"regions": regions, "n_regions": len(regions), "metrics": metrics}
    pd.DataFrame([results]).to_json(output_dir / "metrics_hierarchical.json", orient="records")

    print(f"\nSaved to {output_dir}")


def validate_and_save_deliverables(
    run_id: str,
    metrics: dict,
    roi_df: pd.DataFrame,
    regions: list[str],
    channels: list[str],
) -> dict:
    """
    Validate deliverables using Pydantic schemas.
    
    Args:
        run_id: MLflow run ID.
        metrics: Model metrics dict.
        roi_df: ROI by channel/region DataFrame.
        regions: List of region names.
        channels: List of channel names.
        
    Returns:
        Validated deliverables as dict.
    """
    from datetime import datetime
    from src.schemas import (
        ModelMetadata,
        ChannelROI,
    )
    
    # Pydantic metadata for validated output
    metadata = ModelMetadata(
        run_id=run_id,
        model_type="hierarchical",
        timestamp=datetime.now(),
        n_regions=len(regions),
        n_channels=len(channels),
        r2_train=metrics.get("r2_train", 0.0),
        r2_test=metrics.get("r2_test", 0.0),
        mape_train=metrics.get("mape_train", 0.0),
        mape_test=metrics.get("mape_test", 0.0),
        max_rhat=metrics.get("max_rhat", 1.0),
        divergences=metrics.get("divergences", 0),
    )
    
    # Validate ROI entries
    roi_list = []
    for _, row in roi_df.iterrows():
        roi_entry = ChannelROI(
            channel=row.get("channel", ""),
            region=row.get("region", None),
            spend=float(row.get("spend", 0)),
            contribution=float(row.get("contribution", 0)),
            roi=float(row.get("roi", 0)),
        )
        roi_list.append(roi_entry.model_dump())
    
    deliverables = {
        "metadata": metadata.model_dump(),
        "roi": roi_list,
    }
    
    print(f"Validated {len(roi_list)} ROI entries")
    return deliverables


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
        
        # 6. Contributions and ROI
        print("\nComputing contributions...")
        contrib_df = compute_channel_contributions(
            idata, 
            m_data["X_spend_train"],
            m_data["territory_idx_train"],
            m_data["channel_names"]
        )
        
        # FIX: Convert contributions from log scale to linear $ scale
        # Model predicts y_log, so contributions are in log units
        total_revenue = m_data["df_train"][TARGET_COL].sum()
        mean_log_revenue = m_data["df_train"]["y_log"].mean()
        n_obs_train = len(m_data["df_train"])
        
        contrib_df["contribution_log"] = contrib_df["contribution"]
        contrib_df["contribution"] = contrib_df["contribution_log"] * (total_revenue / (mean_log_revenue * n_obs_train + 1e-8))
        
        # Override total_spend with raw spend (not normalized)
        spend_cols_raw = [c + "_SPEND" for c in m_data["channel_names"]]
        for i, raw_col in enumerate(spend_cols_raw):
            if raw_col in m_data["df_train"].columns:
                contrib_df.loc[i, "total_spend"] = m_data["df_train"][raw_col].sum()
        
        # Recalculate ROI with correct scale
        contrib_df["roi"] = contrib_df["contribution"] / (contrib_df["total_spend"] + 1e-8)
        
        # Add contribution percentage
        contrib_df["contribution_pct"] = contrib_df["contribution"] / contrib_df["contribution"].sum()
        
        print(f"   Total revenue: {total_revenue:,.0f}")
        print(f"   Contribution range: {contrib_df['contribution'].min():,.0f} to {contrib_df['contribution'].max():,.0f}")
        
        # Log contributions
        mlflow.log_dict({"contributions": contrib_df.to_dict(orient="records")}, "deliverables/contributions.json")

        # Log ROI (re-using columns from contrib_df)
        roi_data = contrib_df[["channel", "roi", "total_spend", "contribution"]].to_dict(orient="records")
        mlflow.log_dict({"roi": roi_data}, "deliverables/roi.json")
        
        # Note: log_marginal_roas called after saturation_params is computed
        
        # Iterate over actual regions to compute specific metrics
        print("\nComputing regional metrics...")
        regional_data_list = []
        
        # Scale factors for log→linear conversion (region level)
        scale_factor = total_revenue / (mean_log_revenue * n_obs_train + 1e-8)
        
        # In build logic, 'territory_idx' maps 0..N-1 to regions list order.
        for r_idx, region_name in enumerate(regions):
            try:
                # Isolate observations for this territory
                mask = (m_data["territory_idx_train"] == r_idx)
                if not np.any(mask):
                    continue
                    
                # Filter data
                X_sub = m_data["X_spend_train"][mask]
                idx_sub = m_data["territory_idx_train"][mask]
                df_sub = m_data["df_train"].iloc[np.where(mask)[0]]
                
                # Compute contributions for this region
                # compute_channel_contributions handles the beta_territory lookup using idx_sub
                reg_contrib_df = compute_channel_contributions(
                    idata,
                    X_sub,
                    idx_sub,
                    m_data["channel_names"]
                )
                
                # FIX: Convert log→linear and use raw spend for ROI
                reg_contrib_df["contribution"] = reg_contrib_df["contribution"] * scale_factor
                
                for i, raw_col in enumerate(spend_cols_raw):
                    if raw_col in df_sub.columns:
                        reg_contrib_df.loc[i, "total_spend"] = df_sub[raw_col].sum()
                
                reg_contrib_df["roi"] = reg_contrib_df["contribution"] / (reg_contrib_df["total_spend"] + 1e-8)
                
                # Add percentage
                if reg_contrib_df["contribution"].sum() > 0:
                    reg_contrib_df["contribution_pct"] = reg_contrib_df["contribution"] / reg_contrib_df["contribution"].sum()
                else:
                    reg_contrib_df["contribution_pct"] = 0.0
                
                # FIX: Flatten structure (avoid nested 'channels' that causes [object Object])
                for ch_row in reg_contrib_df.to_dict(orient="records"):
                    regional_data_list.append({
                        "region": region_name,
                        **ch_row
                    })
            except Exception as e:
                print(f"Warning: Failed to compute metrics for region {region_name}: {e}")

        mlflow.log_dict({"regional": regional_data_list}, "deliverables/regional.json")
        
        # Log Adstock/Saturation Params (Global + Territory)
        # NOTE: Model uses L_channel (half-saturation) and k_channel (steepness)
        summary = az.summary(idata, var_names=["alpha_channel", "L_channel", "k_channel"])
        adstock_params = []
        saturation_params = []
        
        for channel in m_data["channel_names"]:
            try:
                # az.summary uses channel names as index, not numeric indices
                alpha = summary.loc[f"alpha_channel[{channel}]", "mean"]
                L = summary.loc[f"L_channel[{channel}]", "mean"]
                k = summary.loc[f"k_channel[{channel}]", "mean"]
                
                # Calculate half-life from alpha: t_half = log(0.5) / log(alpha)
                half_life = float(np.log(0.5) / (np.log(alpha) + 1e-8)) if alpha > 0 else 0.0
                
                # Compute max_spend from training data raw spend column
                raw_spend_col = f"{channel}_SPEND"
                if raw_spend_col in m_data["df_train"].columns:
                    max_spend = float(m_data["df_train"][raw_spend_col].max())
                else:
                    max_spend = None
                
                adstock_params.append({
                    "channel": channel, 
                    "alpha_mean": float(alpha),
                    "half_life_weeks": half_life
                })
                saturation_params.append({
                    "channel": channel, 
                    "L_mean": float(L),
                    "k_mean": float(k),
                    "lam_mean": float(1.0 / (L + 1e-8)),
                    "max_spend": max_spend
                })
            except KeyError as e:
                print(f"Warning: Could not find parameter for {channel}: {e}")
        
        print(f"   Extracted global params for {len(adstock_params)} channels")
        
        # Extract TERRITORY-LEVEL parameters
        alpha_territory = idata.posterior["alpha_territory"].mean(dim=["chain", "draw"]).values
        L_territory = idata.posterior["L_territory"].mean(dim=["chain", "draw"]).values
        k_channel_values = idata.posterior["k_channel"].mean(dim=["chain", "draw"]).values
        
        adstock_territory_params = []
        saturation_territory_params = []
        
        for t_idx, territory in enumerate(regions):
            for c_idx, channel in enumerate(m_data["channel_names"]):
                adstock_territory_params.append({
                    "territory": territory,
                    "channel": channel,
                    "alpha_mean": float(alpha_territory[t_idx, c_idx]),
                })
                saturation_territory_params.append({
                    "territory": territory,
                    "channel": channel,
                    "L_mean": float(L_territory[t_idx, c_idx]),
                    "k_mean": float(k_channel_values[c_idx]),
                })
        
        # Save global params
        mlflow.log_dict({"adstock": adstock_params}, "deliverables/adstock.json")
        mlflow.log_dict({"saturation": saturation_params}, "deliverables/saturation.json")
        
        # Save territory params
        mlflow.log_dict({"adstock_territory": adstock_territory_params}, "deliverables/adstock_territory.json")
        mlflow.log_dict({"saturation_territory": saturation_territory_params}, "deliverables/saturation_territory.json")
        print(f"Saved territory parameters for {len(regions)} regions x {len(m_data['channel_names'])} channels")

        # Log marginal ROAS analysis (must be after saturation_params is computed)
        log_marginal_roas(contrib_df, saturation_params)

        # 6. Compute Efficiency Metrics (iROAS, CAC, Attribution)
        # Uses the model's contribution estimates + AOV approach
        print("\nComputing Channel Efficiency Metrics...")
        from src.insights import compute_channel_metrics, compute_blended_metrics
        
        # Calculate AOV from training data (total_revenue already computed in section 5)
        # AOV = Total Revenue / Total Transactions
        total_transactions = m_data["df_train"]["ALL_PURCHASES"].sum()
        aov = total_revenue / total_transactions if total_transactions > 0 else 0
        print(f"Global AOV: {aov:.2f}")
        
        # Compute channel metrics using AOV
        metrics_df = compute_channel_metrics(contrib_df, aov)
        
        # Save metrics
        print("\nTop Channels by iROAS:")
        print(metrics_df[["channel", "iroas", "cac"]].head())
        
        mlflow.log_dict(
            {"channel_metrics": metrics_df.to_dict(orient="records")},
            "deliverables/channel_metrics.json"
        )
        
        # Compute blended (aggregate) metrics
        blended_metrics = compute_blended_metrics(metrics_df)
        print(f"Blended ROAS: {blended_metrics['blended_roas']:.2f}")
        print(f"Blended CAC: {blended_metrics['blended_cac']:.2f}")
        
        mlflow.log_dict(blended_metrics, "deliverables/blended_metrics.json")

        # 7. ROI HDI (Probabilistic)
        # ROI with HDI already comes from compute_roi_with_hdi in hierarchical_bayesian.py
        print("\nComputing ROI HDI...")
        roi_hdi_df = compute_roi_with_hdi(
            idata=idata,
            X_spend=m_data["X_spend_train"],
            territory_idx=m_data["territory_idx_train"],
            channel_names=m_data["channel_names"],
            hdi_prob=0.94
        )
        mlflow.log_dict(
            {"roi_hdi": roi_hdi_df.to_dict(orient="records")},
            "deliverables/roi_hdi.json"
        )
        print(f"ROI HDI computed for {len(roi_hdi_df)} channels")

        
        # 8. Saturation Curves Visualization
        print("\nGenerating saturation curves plot...")
        sat_curves_path = output_dir / "saturation_curves.png"
        plot_saturation_curves_hierarchical(
            saturation_params=saturation_params,
            output_path=sat_curves_path,
        )
        mlflow.log_artifact(str(sat_curves_path), "diagnostics")

        # Log Optimization
        # Calculate total spend from contribution dataframe
        total_budget_current = contrib_df["total_spend"].sum()
        n_obs_train = len(m_data["X_spend_train"])
        
        # Optimize budget (reallocate within +/- 30% bounds)
        from src.insights import optimize_hierarchical_budget
        optimization_result = optimize_hierarchical_budget(
            contrib_df=contrib_df,
            saturation_params=saturation_params,
            total_budget=total_budget_current,
            n_obs=n_obs_train,
            budget_bounds_pct=(0.70, 1.30)
        )
        
        # Extract results
        optimization_data = optimization_result["allocation"]
        lift_metrics = optimization_result["metrics"]
        
        mlflow.log_dict({"optimization": optimization_data}, "deliverables/optimization.json")
        mlflow.log_dict({"revenue_lift": lift_metrics}, "deliverables/revenue_lift.json")

        # 9b. Optimize Budget BY TERRITORY
        print("\nComputing optimization by territory...")
        from src.insights import optimize_budget_by_territory
        from src.models.hierarchical_bayesian import compute_channel_contributions_by_territory
        
        # Get contributions by territory
        contrib_by_territory_df = compute_channel_contributions_by_territory(
            idata,
            m_data["X_spend_train"],
            m_data["territory_idx_train"],
            m_data["channel_names"],
            regions,
        )
        
        # Convert territory contributions from log scale to linear $ scale
        contrib_by_territory_df["contribution_log"] = contrib_by_territory_df["contribution"]
        contrib_by_territory_df["contribution"] = contrib_by_territory_df["contribution_log"] * scale_factor
        
        # Override total_spend with raw spend per territory (not normalized)
        spend_cols_raw = [c + "_SPEND" for c in m_data["channel_names"]]
        df_train = m_data["df_train"]
        
        for idx, row in contrib_by_territory_df.iterrows():
            territory = row["territory"]
            channel = row["channel"]
            raw_col = f"{channel}_SPEND"
            if raw_col in df_train.columns:
                terr_mask = df_train[GEO_COL] == territory
                raw_spend = df_train.loc[terr_mask, raw_col].sum()
                contrib_by_territory_df.loc[idx, "total_spend"] = float(raw_spend)
        
        # Recalculate ROI with correct scale
        contrib_by_territory_df["roi"] = contrib_by_territory_df["contribution"] / (contrib_by_territory_df["total_spend"] + 1e-8)
        
        mlflow.log_dict(
            {"contributions_territory": contrib_by_territory_df.to_dict(orient="records")},
            "deliverables/contributions_territory.json"
        )
        
        # Optimize for each territory
        optimization_by_territory = []
        lift_by_territory = []
        
        for territory in regions:
            terr_contrib = contrib_by_territory_df[contrib_by_territory_df["territory"] == territory]
            terr_sat_params = [p for p in saturation_territory_params if p["territory"] == territory]
            
            terr_opt = optimize_budget_by_territory(
                contrib_territory_df=terr_contrib,
                saturation_params=terr_sat_params,
                territory=territory,
            )
            optimization_by_territory.extend(terr_opt["allocation"])
            if terr_opt["metrics"].get("success"):
                lift_by_territory.append(terr_opt["metrics"])
        
        mlflow.log_dict({"optimization_territory": optimization_by_territory}, "deliverables/optimization_territory.json")
        mlflow.log_dict({"lift_by_territory": lift_by_territory}, "deliverables/lift_by_territory.json")
        print(f"Optimization completed for {len(lift_by_territory)} territories")

        # Save deliverables
        run_id = mlflow.active_run().info.run_id
        
        # Aggregate ROI/Contribution for schema validation
        # The schema expects spend, contribution, roi per channel (now computed correctly)
        validated = validate_and_save_deliverables(
            run_id=run_id,
            metrics=combined_metrics,
            roi_df=contrib_df.assign(region="Global", spend=contrib_df["total_spend"]),
            regions=regions,
            channels=m_data["channel_names"],
        )
        mlflow.log_dict(validated, "deliverables/validated.json")

        # Save trace
        output_dir.mkdir(exist_ok=True, parents=True)
        idata.to_netcdf(output_dir / "mmm_hierarchical_trace.nc")
        mlflow.log_artifact(output_dir / "mmm_hierarchical_trace.nc")

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

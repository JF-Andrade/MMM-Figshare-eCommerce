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
    HOLDOUT_WEEKS,
    ALL_FEATURES,
    SEASON_COLS,
    DATE_COL,
    GEO_COL,
    MIN_WEEKS_PER_REGION,
)
from src.data_loader import get_valid_regions, load_data
from src.preprocessing import (
    filter_low_variance_channels,
    normalize_spend_by_currency,
    apply_adstock_per_territory,
    apply_saturation_transform,
    create_hierarchy_indices,
    apply_adstock_per_territory,
    apply_saturation_transform,
    create_hierarchy_indices,
    compute_temporal_features,
    prepare_weekly_data,
)
from src.validation import get_panel_holdout_indices
from src.evaluation import check_convergence, compute_roi_by_region, evaluate_model
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
    extract_adstock_params,
    extract_saturation_params,
    plot_adstock_decay,
    plot_channel_contributions_waterfall,
    plot_saturation_curves,
    compute_marginal_roas,
    compute_revenue_lift,
    optimize_budget,
    plot_marginal_roas_curves,
    plot_optimization_results,
    plot_regional_comparison,
    plot_roi_heatmap,
    plot_saturation_curves_hierarchical,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def prepare_hierarchical_data(
    df: pd.DataFrame,
    regions: list[str],
) -> tuple[pd.DataFrame, dict]:
    from src.preprocessing import engineer_features

    all_data = []
    for region in regions:
        df_weekly = prepare_weekly_data(df, region=region)
        df_weekly[GEO_COL] = region
        # Add currency from original data
        currency = df[df['TERRITORY_NAME'] == region]['CURRENCY_CODE'].iloc[0]
        df_weekly['CURRENCY_CODE'] = currency
        all_data.append(df_weekly)

    df_combined = pd.concat(all_data, ignore_index=True)

    # 1. Apply FULL feature engineering (Efficiency, Cost, Customer, Temporal, Rolling, Share)
    df_combined = engineer_features(df_combined, date_col=DATE_COL)

    # 2. Add trend per region (normalized)
    for region in regions:
        mask = df_combined[GEO_COL] == region
        df_combined.loc[mask, "trend"] = np.arange(mask.sum()) / (mask.sum() + 1)

    # 3. Add Temporal features (Cyclic Sin/Cos for Week and Month)
    df_combined = compute_temporal_features(df_combined, date_col=DATE_COL)

    # C2 FIX: Removed normalize_spend_by_currency() here
    # Normalization now happens in prepare_model_data() AFTER train/test split
    # to prevent data leakage (max from test influencing train normalization)
    # Just ensure SPEND_COLS exist for later normalization

    # 5. Log-transform target
    df_combined['y_log'] = np.log1p(df_combined[TARGET_COL])

    # 6. Ensure consistent sorting before generating indices
    df_combined = df_combined.sort_values([GEO_COL, DATE_COL]).reset_index(drop=True)
    
    # VALIDATION: Verify monotonic dates within each territory (required for adstock)
    for geo in df_combined[GEO_COL].unique():
        geo_dates = df_combined.loc[df_combined[GEO_COL] == geo, DATE_COL]
        if not geo_dates.is_monotonic_increasing:
            raise ValueError(
                f"Dates not monotonic for territory {geo}. "
                "Adstock computation requires sorted data."
            )

    # 7. Create hierarchy indices (simplified: Global → Territory)
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
            df, GEO_COL, HOLDOUT_WEEKS
        )

    # M4 FIX: Use iloc for robust index handling (avoid issues if index is not contiguous)
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    # =========================================================================
    # C2 FIX: Normalize spend AFTER split, fitting on train only
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
    print(f"C2 FIX: Normalized {len(spend_norm_cols)} spend columns using train-only max")
    # =========================================================================

    # Scale features and seasonality (StandardScaler)
    # This helps NUTS sampler convergence
    from sklearn.preprocessing import StandardScaler
    
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
    print(f"\n=== DATA SCALE DEBUG ===")
    print(f"y_log: mean={model_data['y_train'].mean():.3f}, std={model_data['y_train'].std():.3f}")
    print(f"X_spend: max={model_data['X_spend_train'].max():.3f}")
    print(f"X_features: std (avg)={model_data['X_features_train'].std(axis=0).mean():.3f}")
    print(f"========================\n")
    
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
    
    # Create metadata
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
) -> tuple[pm.Model, az.InferenceData, dict]:
    """Run complete hierarchical MMM pipeline with MLflow tracking."""
    import pymc as pm  # Local import for robustness
    print("=" * 60)
    print("CUSTOM NESTED HIERARCHICAL MMM")
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
            use_student_t=True,
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
        import tempfile
        
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

        # 6. Contributions and ROI
        print("\nComputing contributions...")
        contrib_df = compute_channel_contributions(
            idata, 
            m_data["X_spend_train"],
            m_data["territory_idx_train"],
            m_data["channel_names"]
        )
        # Add contribution percentage
        contrib_df["contribution_pct"] = contrib_df["contribution"] / contrib_df["contribution"].sum()
        
        # Log contributions
        mlflow.log_dict({"contributions": contrib_df.to_dict(orient="records")}, "deliverables/contributions.json")

        # Log ROI (re-using columns from contrib_df)
        roi_data = contrib_df[["channel", "roi"]].to_dict(orient="records")
        mlflow.log_dict({"roi": roi_data}, "deliverables/roi.json")
        
        # Log Regional Data
        # Iterate over actual regions to compute specific metrics
        print("\nComputing regional metrics...")
        regional_data_list = []
        
        # In build logic, 'territory_idx' maps 0..N-1 to regions list order.
        for r_idx, region_name in enumerate(regions):
            try:
                # Create mask for this region
                mask = (m_data["territory_idx_train"] == r_idx)
                if not np.any(mask):
                    continue
                    
                # Filter data
                X_sub = m_data["X_spend_train"][mask]
                idx_sub = m_data["territory_idx_train"][mask]
                
                # Compute contributions for this region
                # compute_channel_contributions handles the beta_territory lookup using idx_sub
                reg_contrib_df = compute_channel_contributions(
                    idata,
                    X_sub,
                    idx_sub,
                    m_data["channel_names"]
                )
                
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
        
        # Log Adstock/Saturation Params
        # NOTE: Model uses L_channel (half-saturation) and k_channel (steepness)
        summary = az.summary(idata, var_names=["alpha_channel", "L_channel", "k_channel"])
        adstock_params = []
        saturation_params = []
        
        for i, channel in enumerate(m_data["channel_names"]):
            try:
                alpha = summary.loc[f"alpha_channel[{i}]", "mean"]
                L = summary.loc[f"L_channel[{i}]", "mean"]
                k = summary.loc[f"k_channel[{i}]", "mean"]
                
                adstock_params.append({"channel": channel, "alpha_mean": float(alpha)})
                # Store L and k for saturation. For optimization, we use L as the scale factor.
                saturation_params.append({
                    "channel": channel, 
                    "L_mean": float(L),  # Half-saturation point
                    "k_mean": float(k),  # Steepness
                    "lam_mean": float(1.0 / (L + 1e-8))  # Approximate lambda for old interface compatibility
                })
            except KeyError:
                pass
                
        mlflow.log_dict({"adstock": adstock_params}, "deliverables/adstock.json")
        mlflow.log_dict({"saturation": saturation_params}, "deliverables/saturation.json")

        # 7. ROI with Uncertainty (HDI)
        print("\nComputing ROI with uncertainty intervals...")
        roi_hdi_df = compute_roi_with_hdi(
            idata,
            m_data["X_spend_train"],
            m_data["territory_idx_train"],
            m_data["channel_names"],
            hdi_prob=0.94,
            n_samples=500,
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

        # Save deliverables (validated)
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
    args = parser.parse_args()
    
    run_hierarchical(DATA_PATH, OUTPUT_DIR, args.max_regions)

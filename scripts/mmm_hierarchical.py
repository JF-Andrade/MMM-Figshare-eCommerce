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
    SHARE_COLS,
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
    add_fourier_seasonality,
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

    # 3. Add Fourier seasonality (replaced simple sine/cosine)
    df_combined = add_fourier_seasonality(df_combined, date_col=DATE_COL, n_terms=2)

    # 4. Normalize spend within currency
    df_combined = normalize_spend_by_currency(df_combined, SPEND_COLS)

    # NOTE: Adstock and saturation are now applied INSIDE the Bayesian model
    # as learned parameters. We pass raw normalized spend to the model.
    # The model will learn alpha (adstock decay) and L, k (saturation) per channel.

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

    # 7. Create hierarchy indices
    territory_idx, currency_idx, territory_to_currency, territory_names, currency_names = create_hierarchy_indices(
        df_combined, geo_col=GEO_COL, currency_col='CURRENCY_CODE'
    )

    indices = {
        "territory_idx": territory_idx,
        "currency_idx": currency_idx,
        "territory_to_currency": territory_to_currency,
        "territory_names": territory_names,
        "currency_names": currency_names,
    }

    print(f"Combined data: {len(df_combined)} rows, {len(regions)} regions")
    print(f"Features: {len(df_combined.columns)} columns")
    print(f"Hierarchy: {len(currency_names)} currencies, {len(territory_names)} territories")

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
    """
    from src.config import ALL_FEATURES, SPEND_COLS

    # Group columns by type for the model
    # RAW normalized spend (adstock/saturation applied inside model)
    spend_norm_cols = [f"{c}_norm" for c in SPEND_COLS if f"{c}_norm" in df.columns]
    
    # Seasonality Fourier terms
    season_cols = [c for c in ["sin_1", "cos_1", "sin_2", "cos_2"] if c in df.columns]
    
    # All other features (Efficiency, Cost, Customer, Trend, Holiday)
    excluded = [TARGET_COL, "y_log", DATE_COL, GEO_COL, "CURRENCY_CODE"] + spend_norm_cols + season_cols
    other_feature_cols = [c for c in df.columns if c in ALL_FEATURES and c not in excluded]

    # Temporal Split
    if train_indices is None or test_indices is None:
        train_indices, test_indices = get_panel_holdout_indices(
            df, GEO_COL, HOLDOUT_WEEKS
        )

    df_train = df.loc[train_indices].copy()
    df_test = df.loc[test_indices].copy()

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
        "currency_idx_train": indices["currency_idx"][train_indices],
        "currency_idx_test": indices["currency_idx"][test_indices],
        "territory_to_currency": indices["territory_to_currency"],
        "n_currencies": len(indices["currency_names"]),
        "n_territories": len(indices["territory_names"]),
        "channel_names": [c.replace("_norm", "").replace("_SPEND", "") for c in spend_norm_cols],
        "feature_names": other_feature_cols,
        "df_train": df_train,
        "df_test": df_test,
    }

    print(f"Model Data Prepared:")
    print(f" - Spend: {model_data['X_spend_train'].shape}")
    print(f" - Features: {model_data['X_features_train'].shape}")
    print(f" - Season: {model_data['X_season_train'].shape}")
    
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

    # Start MLflow run
    with mlflow.start_run(run_name=f"hierarchical_custom_{len(regions)}regions"):
        # Log parameters
        mlflow.log_params({
            "model_type": "nested_hierarchical_custom",
            "n_regions": len(regions),
            "n_currencies": m_data["n_currencies"],
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
            currency_idx=m_data["currency_idx_train"],
            territory_to_currency=m_data["territory_to_currency"],
            n_currencies=m_data["n_currencies"],
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
                "currency_idx": m_data["currency_idx_train"],
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
                "currency_idx": m_data["currency_idx_test"],
                "y_obs_data": np.full_like(m_data["y_test"], np.nan),  # NaN for holdout
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
        # ROI is now computed correctly inside compute_channel_contributions
        mlflow.log_dict({"contributions": contrib_df.to_dict(orient="records")}, "deliverables/contributions.json")

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

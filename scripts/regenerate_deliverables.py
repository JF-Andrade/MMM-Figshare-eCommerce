"""
Regenerate Dashboard Deliverables from Saved Model.

This script loads a saved model trace (idata) from MLflow and regenerates
all deliverables required by the dashboard, including the new territory-level
parameters and optimizations.

Usage:
    python scripts/regenerate_deliverables.py --run-id <RUN_ID>
    python scripts/regenerate_deliverables.py --latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.models.hierarchical_bayesian import (
    compute_channel_contributions,
    compute_channel_contributions_by_territory,
    compute_roi_with_hdi,
)
from src.insights import (
    optimize_hierarchical_budget,
    optimize_budget_by_territory,
)


def get_latest_hierarchical_run() -> str:
    """Get the latest hierarchical model run ID that has a trace file."""
    from mlflow import MlflowClient
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_type LIKE '%hierarchical%'",
        order_by=["start_time DESC"],
        max_results=20,
    )
    
    if not runs:
        raise ValueError("No hierarchical model runs found")
    
    # Find first run with trace file
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        has_trace = any("trace.nc" in a.path for a in artifacts)
        if has_trace:
            return run.info.run_id
    
    raise ValueError("No runs with trace file found. Re-run the pipeline to generate one.")


def list_valid_runs():
    """List all runs that have a trace file available."""
    from mlflow import MlflowClient
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_type LIKE '%hierarchical%'",
        order_by=["start_time DESC"],
        max_results=20,
    )
    
    print("\nRuns with trace file available:")
    print("-" * 80)
    
    found = 0
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        has_trace = any("trace.nc" in a.path for a in artifacts)
        if has_trace:
            found += 1
            r2 = run.data.metrics.get("r2_test", "N/A")
            name = run.info.run_name or run.info.run_id[:8]
            print(f"  {run.info.run_id}  R²={r2:.3f}  {name}")
    
    if found == 0:
        print("  No runs with trace file found. Re-run the pipeline.")
    else:
        print(f"\nTotal: {found} runs available")


def load_idata_from_run(run_id: str) -> az.InferenceData:
    """Download and load idata from MLflow artifacts."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # Check if trace exists
    artifacts = client.list_artifacts(run_id)
    trace_artifact = None
    for a in artifacts:
        if "trace.nc" in a.path:
            trace_artifact = a.path
            break
    
    if not trace_artifact:
        raise ValueError(
            f"Run {run_id} does not have a trace file.\n"
            "Use --list to see runs with available traces."
        )
    
    # Download the trace file
    try:
        local_path = client.download_artifacts(run_id, trace_artifact)
        print(f"Loaded idata from: {local_path}")
        return az.from_netcdf(local_path)
    except MlflowException as e:
        raise ValueError(f"Failed to download trace: {e}")



def load_model_data(data_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Load prepared model data from hierarchical_train.parquet.
    
    This uses the SAME normalized data that the model was trained on,
    ensuring consistency in ROI calculations.
    """
    from src.preprocessing import create_hierarchy_indices
    from src.config import SPEND_COLS, GEO_COL
    
    # Use pre-prepared training data (saved by pipeline)
    inspection_dir = data_path.parent.parent / "inspection"
    train_parquet = inspection_dir / "hierarchical_train.parquet"
    
    if not train_parquet.exists():
        raise FileNotFoundError(
            f"Prepared training data not found: {train_parquet}\n"
            "Run the hierarchical pipeline first to generate this file."
        )
    
    df_train = pd.read_parquet(train_parquet)
    print(f"Loaded prepared training data: {df_train.shape}")
    
    # Get normalized spend columns (used by model)
    spend_norm_cols = [f"{c}_norm" for c in SPEND_COLS if f"{c}_norm" in df_train.columns]
    
    if not spend_norm_cols:
        raise ValueError("No normalized spend columns found. Check pipeline output.")
    
    print(f"Using {len(spend_norm_cols)} normalized spend columns")
    
    # Get territories from data
    region_col = GEO_COL if GEO_COL in df_train.columns else "TERRITORY_NAME"
    territory_idx, territory_names = create_hierarchy_indices(df_train, geo_col=region_col)
    
    # Extract normalized spend
    X_spend_train = df_train[spend_norm_cols].fillna(0).values
    
    # Channel names without _norm suffix
    channel_names = [c.replace("_norm", "").replace("_SPEND", "") for c in spend_norm_cols]
    
    return df_train, {
        "X_spend_train": X_spend_train,
        "territory_idx_train": territory_idx,
        "channel_names": channel_names,
        "regions": territory_names,
        "n_obs_train": len(X_spend_train),
        "df_train": df_train,  # Keep for spend total calculation
        "spend_cols_raw": [c.replace("_norm", "") for c in spend_norm_cols],
    }


def regenerate_deliverables(
    idata: az.InferenceData,
    m_data: dict,
    run_id: str,
) -> dict:
    """Regenerate all deliverables from idata."""
    
    deliverables = {}
    regions = m_data["regions"]
    channel_names = m_data["channel_names"]
    X_spend_train = m_data["X_spend_train"]  # Normalized [0,1]
    territory_idx_train = m_data["territory_idx_train"]
    n_obs_train = m_data["n_obs_train"]
    df_train = m_data["df_train"]
    spend_cols_raw = m_data["spend_cols_raw"]  # Raw spend column names
    
    print("\n1. Computing channel contributions (global)...")
    contrib_df = compute_channel_contributions(
        idata,
        X_spend_train,
        territory_idx_train,
        channel_names,
    )
    
    # CRITICAL FIX: Model predicts y_log (log1p transformed revenue)
    # Contributions are in LOG SCALE - they represent log-revenue increments
    # To convert to linear $ scale: exp(contribution) - 1 gives approximate revenue lift
    # ROI requires actual spend for comparison
    
    # Get total original revenue for scaling
    if "ALL_PURCHASES_ORIGINAL_PRICE" in df_train.columns:
        total_revenue = df_train["ALL_PURCHASES_ORIGINAL_PRICE"].sum()
        mean_log_revenue = df_train["y_log"].mean()
    else:
        total_revenue = np.expm1(df_train["y_log"].sum())
        mean_log_revenue = df_train["y_log"].mean()
    
    print(f"   Total revenue: {total_revenue:,.0f}")
    print(f"   Mean log revenue: {mean_log_revenue:.2f}")
    
    # Override total_spend with RAW spend values
    for i, raw_col in enumerate(spend_cols_raw):
        if raw_col in df_train.columns:
            contrib_df.loc[i, "total_spend"] = df_train[raw_col].sum()
    
    # Convert contribution from log scale to approximate $ contribution
    # Interpretation: each unit of beta*x_sat adds that much to log(revenue)
    # Approximate $ contribution = total_revenue * (exp(contrib_log/n_obs) - 1) * n_obs
    # Simplified: contrib_$ ≈ contribution_log * (revenue / mean_log_y)
    contrib_df["contribution_log"] = contrib_df["contribution"]
    contrib_df["contribution"] = contrib_df["contribution_log"] * (total_revenue / (mean_log_revenue * n_obs_train + 1e-8))
    
    # Recalculate ROI with linear scale contribution and raw spend
    contrib_df["roi"] = contrib_df["contribution"] / (contrib_df["total_spend"] + 1e-8)
    
    deliverables["contributions"] = contrib_df.to_dict(orient="records")
    
    print("2. Computing ROI...")
    roi_data = []
    for _, row in contrib_df.iterrows():
        roi_data.append({
            "channel": row["channel"],
            "roi": float(row["roi"]),
            "total_spend": float(row["total_spend"]),
            "contribution": float(row["contribution"]),
        })
    deliverables["roi"] = roi_data
    
    print("3. Computing ROI with HDI intervals...")
    roi_hdi_df = compute_roi_with_hdi(
        idata,
        X_spend_train,
        territory_idx_train,
        channel_names,
        hdi_prob=0.94,
        n_samples=500,
    )
    deliverables["roi_hdi"] = roi_hdi_df.to_dict(orient="records")
    
    print("4. Extracting global adstock/saturation parameters...")
    summary = az.summary(idata, var_names=["alpha_channel", "L_channel", "k_channel"])
    
    adstock_params = []
    saturation_params = []
    for i, channel in enumerate(channel_names):
        try:
            # Fix: Use channel names for indexing instead of numeric indices
            # depending on how ArviZ/PyMC 5.x created the summary
            try:
                alpha = summary.loc[f"alpha_channel[{channel}]", "mean"]
                L = summary.loc[f"L_channel[{channel}]", "mean"]
                k = summary.loc[f"k_channel[{channel}]", "mean"]
            except KeyError:
                # Fallback to numeric if needed, but usually it's by name now
                alpha = summary.loc[f"alpha_channel[{i}]", "mean"]
                L = summary.loc[f"L_channel[{i}]", "mean"]
                k = summary.loc[f"k_channel[{i}]", "mean"]
            
            # Calculate half-life
            half_life = float(np.log(0.5) / (np.log(alpha) + 1e-8)) if alpha > 0 else 0.0

            adstock_params.append({
                "channel": channel, 
                "alpha_mean": float(alpha),
                "half_life_weeks": half_life
            })
            saturation_params.append({
                "channel": channel,
                "L_mean": float(L),
                "k_mean": float(k),
                "lam_mean": float(1.0 / (L + 1e-8))
            })
        except KeyError as e:
            print(f"Warning: Could not find parameter for {channel}: {e}")
    
    deliverables["adstock"] = adstock_params
    deliverables["saturation"] = saturation_params
    
    print("5. Extracting TERRITORY-LEVEL parameters...")
    alpha_territory = idata.posterior["alpha_territory"].mean(dim=["chain", "draw"]).values
    L_territory = idata.posterior["L_territory"].mean(dim=["chain", "draw"]).values
    k_channel_values = idata.posterior["k_channel"].mean(dim=["chain", "draw"]).values
    
    adstock_territory_params = []
    saturation_territory_params = []
    
    for t_idx, territory in enumerate(regions):
        for c_idx, channel in enumerate(channel_names):
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
    
    deliverables["adstock_territory"] = adstock_territory_params
    deliverables["saturation_territory"] = saturation_territory_params
    
    print("6. Computing contributions BY TERRITORY...")
    contrib_by_territory_df = compute_channel_contributions_by_territory(
        idata,
        X_spend_train,
        territory_idx_train,
        channel_names,
        regions,
    )
    deliverables["contributions_territory"] = contrib_by_territory_df.to_dict(orient="records")
    
    # Regional data (flat format)
    deliverables["regional"] = contrib_by_territory_df.rename(columns={"territory": "region"}).to_dict(orient="records")
    
    print("7. Running budget optimization BY TERRITORY...")
    optimization_by_territory = []
    lift_by_territory = []
    
    # Track aggregated results for global consistency (Bottom-Up Approach)
    global_current_spend = 0.0
    global_opt_spend = 0.0
    global_current_contrib = 0.0
    global_opt_contrib = 0.0
    
    # Store per-channel global aggregation
    global_channel_alloc = {}
    
    for territory in regions:
        terr_contrib = contrib_by_territory_df[contrib_by_territory_df["territory"] == territory]
        terr_sat_params = [p for p in saturation_territory_params if p["territory"] == territory]
        
        # Optimize for this territory
        terr_opt = optimize_budget_by_territory(
            contrib_territory_df=terr_contrib,
            saturation_params=terr_sat_params,
            territory=territory,
        )
        
        # Collect results
        optimization_by_territory.extend(terr_opt["allocation"])
        
        # Aggregate logic
        for alloc in terr_opt["allocation"]:
            ch = alloc["channel"]
            if ch not in global_channel_alloc:
                global_channel_alloc[ch] = {
                    "channel": ch, "current_spend": 0.0, "optimized_spend": 0.0,
                    "current_contribution": 0.0, "optimized_contribution": 0.0
                }
            
            global_channel_alloc[ch]["current_spend"] += alloc["current_spend"]
            global_channel_alloc[ch]["optimized_spend"] += alloc["optimized_spend"]
            # Metrics might be missing if opt failed, assume proportional if so, but terr_opt usually has them
            
        metrics = terr_opt["metrics"]
        if metrics.get("success"):
            lift_by_territory.append(metrics)
            global_current_spend += metrics.get("current_spend", 0)
            global_opt_spend += metrics.get("optimized_spend", 0)
            global_current_contrib += metrics.get("current_contribution", 0)
            global_opt_contrib += metrics.get("projected_contribution", 0)

    deliverables["optimization_territory"] = optimization_by_territory
    deliverables["lift_by_territory"] = lift_by_territory
    
    print("8. Aggregating to create GLOBAL optimization results (Bottom-Up)...")
    # Create global optimization list from aggregation
    global_optimization = [
        {
            "channel": v["channel"],
            "current_spend": v["current_spend"],
            "optimized_spend": v["optimized_spend"],
            # Recalculate PCT change based on aggregated sums
            "delta_spend_pct": ((v["optimized_spend"] / v["current_spend"]) - 1) if v["current_spend"] > 0 else 0
        }
        for v in global_channel_alloc.values()
    ]
    deliverables["optimization"] = global_optimization
    
    # Create global lift metrics from aggregation
    global_lift_abs = global_opt_contrib - global_current_contrib
    global_lift_pct = (global_lift_abs / global_current_contrib * 100) if global_current_contrib > 0 else 0
    
    deliverables["revenue_lift"] = {
        "current_spend": global_current_spend,
        "optimized_spend": global_opt_spend,
        "current_contribution": global_current_contrib,
        "projected_contribution": global_opt_contrib,
        "lift_absolute": global_lift_abs,
        "lift_pct": global_lift_pct,
        "success": True
    }
    
    print(f"   aggregated global lift: {global_lift_pct:.2f}% (${global_lift_abs:,.0f})")

    print(f"\nGenerated {len(deliverables)} deliverables")
    return deliverables


def save_deliverables_to_mlflow(deliverables: dict, run_id: str) -> None:
    """Save deliverables to the existing MLflow run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    with mlflow.start_run(run_id=run_id):
        for name, data in deliverables.items():
            mlflow.log_dict({name: data}, f"deliverables/{name}.json")
            print(f"  Saved: deliverables/{name}.json")


def main():
    parser = argparse.ArgumentParser(description="Regenerate dashboard deliverables")
    parser.add_argument("--run-id", type=str, help="MLflow run ID to use")
    parser.add_argument("--latest", action="store_true", help="Use latest hierarchical run with trace")
    parser.add_argument("--list", action="store_true", help="List runs with trace files available")
    parser.add_argument("--data-path", type=str, default="data/processed/mmm_data.parquet")
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_valid_runs()
        return
    
    # Determine run ID
    if args.run_id:
        run_id = args.run_id
    elif args.latest:
        run_id = get_latest_hierarchical_run()
        print(f"Using latest run: {run_id}")
    else:
        print("Error: Specify --run-id <ID>, --latest, or --list")
        sys.exit(1)
    
    # Load idata
    print(f"\nLoading model from run: {run_id}")
    idata = load_idata_from_run(run_id)
    
    # Load data
    data_path = PROJECT_ROOT / args.data_path
    print(f"Loading data from: {data_path}")
    df, m_data = load_model_data(data_path)
    
    # Regenerate
    deliverables = regenerate_deliverables(idata, m_data, run_id)
    
    # Save
    print("\nSaving deliverables to MLflow...")
    save_deliverables_to_mlflow(deliverables, run_id)
    
    print("\nDone! Refresh the dashboard to see updated data.")


if __name__ == "__main__":
    main()

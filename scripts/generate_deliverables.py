"""
Regenerate deliverables from existing model artifacts.

This script loads a trained model's artifacts (idata, model_data) and 
regenerates all deliverables without re-running MCMC sampling.

Usage:
    python scripts/generate_deliverables.py [--run-id RUN_ID] [--new-run]
    
If --run-id not provided, uses the latest hierarchical run.
If --new-run is specified, creates a new MLflow run for the regenerated deliverables.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import arviz as az
import mlflow

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.deliverables import generate_all_deliverables


def load_artifacts_from_run(run_id: str, client) -> tuple:
    """
    Load model artifacts from MLflow run.
    
    Args:
        run_id: MLflow run ID.
        client: MLflow client instance.
    
    Returns:
        Tuple of (idata, m_data, regions).
    """
    print(f"Loading artifacts from run: {run_id}")
    
    # Load idata (InferenceData)
    try:
        idata_path = client.download_artifacts(run_id, "model/idata.nc")
        idata = az.from_netcdf(idata_path)
        print(f"   Loaded idata from model/idata.nc")
    except Exception:
        # Fallback to old path
        idata_path = client.download_artifacts(run_id, "mmm_hierarchical_trace.nc")
        idata = az.from_netcdf(idata_path)
        print(f"   Loaded idata from mmm_hierarchical_trace.nc")
    
    # Load model_data
    try:
        m_data_path = client.download_artifacts(run_id, "model/model_data.pkl")
        with open(m_data_path, "rb") as f:
            m_data = pickle.load(f)
        print(f"   Loaded model_data from model/model_data.pkl")
    except Exception as e:
        raise FileNotFoundError(
            f"model_data.pkl not found in run {run_id}. "
            "This run may have been created before the refactoring. "
            f"Error: {e}"
        )
    
    # Load regions
    try:
        regions_path = client.download_artifacts(run_id, "model/regions.pkl")
        with open(regions_path, "rb") as f:
            regions = pickle.load(f)
        print(f"   Loaded regions from model/regions.pkl")
    except Exception as e:
        raise FileNotFoundError(
            f"regions.pkl not found in run {run_id}. "
            f"Error: {e}"
        )
    
    return idata, m_data, regions


def get_latest_hierarchical_run(client):
    """Get the most recent hierarchical model run."""
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_type LIKE '%hierarchical%'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    
    if not runs:
        raise ValueError("No hierarchical model runs found")
    
    return runs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate deliverables from existing model artifacts."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID. If not provided, uses latest hierarchical run."
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Create a new MLflow run for regenerated deliverables."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to models/."
    )
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    
    # Get run ID
    if args.run_id:
        run_id = args.run_id
    else:
        run = get_latest_hierarchical_run(client)
        run_id = run.info.run_id
        print(f"Using latest hierarchical run: {run_id}")
    
    # Load artifacts
    idata, m_data, regions = load_artifacts_from_run(run_id, client)
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate deliverables
    if args.new_run:
        print("\nCreating new MLflow run for deliverables...")
        with mlflow.start_run(run_name=f"deliverables_regen_{run_id[:8]}"):
            # Log reference to original run
            mlflow.log_param("source_run_id", run_id)
            mlflow.log_param("model_type", "deliverables_regeneration")
            
            deliverables = generate_all_deliverables(
                idata=idata,
                m_data=m_data,
                regions=regions,
                output_dir=output_dir,
                log_to_mlflow=True,
            )
            new_run_id = mlflow.active_run().info.run_id
            print(f"\nDeliverables logged to new run: {new_run_id}")
    else:
        print(f"\nUpdating deliverables in existing run: {run_id}")
        with mlflow.start_run(run_id=run_id):
            deliverables = generate_all_deliverables(
                idata=idata,
                m_data=m_data,
                regions=regions,
                output_dir=output_dir,
                log_to_mlflow=True,
            )
    
    print("\n✅ Deliverables regeneration complete!")
    print(f"   Generated {len(deliverables)} deliverable groups")
    
    return deliverables


if __name__ == "__main__":
    main()

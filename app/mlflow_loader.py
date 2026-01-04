"""
MLflow data loader for Streamlit dashboard.

Loads deliverables from MLflow artifacts for dashboard visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

import sys

# Default MLflow configuration
# Add path to allow importing src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import centralized config
try:
    from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME as EXPERIMENT_NAME
except ImportError:
    # Fallback if running outside of project structure (unlikely for streamlit)
    MLFLOW_TRACKING_URI = "file:./mlruns"
    EXPERIMENT_NAME = "MMM-Experiments"


def get_mlflow_client(tracking_uri: str | None = None) -> MlflowClient:
    """Get MLflow client configured for the project."""
    uri = tracking_uri or MLFLOW_TRACKING_URI
    return MlflowClient(tracking_uri=uri)


def get_experiment_id(client: MlflowClient, experiment_name: str = EXPERIMENT_NAME) -> str:
    """Get experiment ID by name."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return experiment.experiment_id


def get_latest_hierarchical_run(client: MlflowClient | None = None) -> dict:
    """
    Get the latest successful hierarchical model run.

    Returns:
        Dict with run_id, start_time, metrics, and params.
    """
    if client is None:
        client = get_mlflow_client()

    experiment_id = get_experiment_id(client)

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.model_type LIKE '%hierarchical%'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No hierarchical model runs found")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
    }


def get_all_runs(client: MlflowClient | None = None, model_type: str | None = None) -> list[dict]:
    """
    Get all runs, optionally filtered by model type.

    Args:
        client: MLflow client.
        model_type: Optional filter for 'ridge_baseline' or 'hierarchical'.

    Returns:
        List of run dicts.
    """
    if client is None:
        client = get_mlflow_client()

    try:
        experiment_id = get_experiment_id(client)
    except ValueError:
        return []

    filter_string = ""
    if model_type:
        if model_type == "hierarchical":
            filter_string = "params.model_type LIKE '%hierarchical%'"
        else:
            filter_string = f"params.model_type = '{model_type}'"

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=50,
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or f"run_{run.info.run_id[:8]}",
            "start_time": run.info.start_time,
            "status": run.info.status,
            "model_type": run.data.params.get("model_type", "unknown"),
            "r2_test": run.data.metrics.get("r2_test"),
            "mape_test": run.data.metrics.get("mape_test"),
            "training_time": run.data.metrics.get("training_time"),
        }
        for run in runs
    ]


def load_deliverable(run_id: str, name: str, client: MlflowClient | None = None) -> Any:
    """
    Load a deliverable from MLflow artifacts, adapting to pipeline paths.

    Args:
        run_id: MLflow run ID.
        name: Deliverable key expected by App (e.g. 'contributions', 'adstock').
        client: Optional MLflow client.

    Returns:
        Parsed data (dict or list), or None if not found/mapped.
    """
    if client is None:
        client = get_mlflow_client()

    # 1. Define Mapping: App Key -> Pipeline Artifact Path
    # "None" means the artifact does not exist in the pipeline yet.
    ARTIFACT_MAPPING = {
        # Direct Mappings
        "predictions": "deliverables/predictions.json",
        
        # Path/Name Adaptations
        "contributions": "metrics/global_contributions.json",
        "roi_hdi": "metrics/roi_hdi.json",
        "regional": "metrics/regional_contributions.json",
        "optimization": "deliverables/budget_optimization_global.json",
        "optimization_territory": "deliverables/budget_optimization_regional.json",
        "marginal_roas": "metrics/marginal_roas_global.json",
        
        # Merged Files (Handled specially below)
        "adstock": "metrics/parameter_summary.json",
        "adstock_territory": "metrics/parameter_summary.json",
        "saturation": "metrics/parameter_summary.json",
        "saturation_territory": "metrics/parameter_summary.json",
        
        # Derived/Missing
        "roi": "metrics/roi_hdi.json",  # Will extract simplified version
        "contributions_territory": "metrics/regional_contributions.json", # Will extract if needed
        "channel_metrics": "deliverables/channel_metrics.json",
        "blended_metrics": "deliverables/blended_metrics.json",
        "revenue_lift": "deliverables/budget_optimization_global.json", # Extract lift metrics
        "lift_by_territory": "deliverables/budget_optimization_regional.json", # Extract lift metrics
    }

    artifact_path = ARTIFACT_MAPPING.get(name)

    if not artifact_path:
        # Artifact known to be missing or unmapped
        return None

    try:
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path) as f:
            data = json.load(f)
            
        # 2. Handle Merged/Derived Data Logic
        
        # Parameter Splitting
        if name == "adstock":
            return data.get("adstock_params")
        elif name == "adstock_territory":
            return data.get("adstock_territory_params")
        elif name == "saturation":
            return data.get("saturation_params")
        elif name == "saturation_territory":
            return data.get("saturation_territory_params")
            
        # ROI Simplification
        elif name == "roi":
            # App expects simple list of dicts with 'roi' key
            # roi_hdi has 'roi_mean', 'roi_hdi_low', etc.
            if isinstance(data, list):
                return [
                    {
                        "channel": item.get("channel"),
                        "roi": item.get("roi_mean"),
                        "contribution": item.get("contribution"), # If available
                        "spend": item.get("total_spend") # If available
                    }
                    for item in data
                ]
            return data

        # Lift Extraction
        elif name == "revenue_lift":
             # Optimization file contains "metrics" key with lift info
             if isinstance(data, dict) and "metrics" in data:
                 return data["metrics"]
             return None
             
        elif name == "lift_by_territory":
             # Regional optimization is a dict of territory -> result
             # We need to aggregate the "metrics" from each territory
             if isinstance(data, dict):
                 lift_list = []
                 for terr, result in data.items():
                     if "metrics" in result:
                         m = result["metrics"]
                         m["territory"] = terr
                         lift_list.append(m)
                 return lift_list
             return None

        # Regional Contributions Flattening (if needed)
        elif name == "contributions_territory":
            # If regional file is already flat list, return it
            # If it's something else, adapt. Currently it's likely a flat list from log_regional_metrics
            return data

        return data

    except Exception as e:
        # print(f"Warning: Could not load '{name}' from {artifact_path}: {e}")
        return None


def load_all_deliverables(run_id: str, client: MlflowClient | None = None) -> dict[str, Any]:
    """
    Load all deliverables for a run.

    Args:
        run_id: MLflow run ID.
        client: Optional MLflow client.

    Returns:
        Dict with all deliverables.
    """
    if client is None:
        client = get_mlflow_client()

    deliverable_names = [
        "roi",
        "roi_hdi",
        "saturation",
        "saturation_territory",
        "adstock",
        "adstock_territory",
        "contributions",
        "contributions_territory",
        "optimization",
        "optimization_territory",
        "marginal_roas",
        "regional",
        "revenue_lift",
        "lift_by_territory",
        "predictions",
        # Missing but requested by App
        "channel_metrics",
        "blended_metrics" 
    ]

    deliverables = {}
    for name in deliverable_names:
        # load_deliverable now handles safe loading and adaption
        data = load_deliverable(run_id, name, client)
        if data is not None:
            deliverables[name] = data

    return deliverables


def get_run_metrics(run_id: str, client: MlflowClient | None = None) -> dict:
    """Get all metrics for a run."""
    if client is None:
        client = get_mlflow_client()

    run = client.get_run(run_id)
    return dict(run.data.metrics)


def get_run_params(run_id: str, client: MlflowClient | None = None) -> dict:
    """Get all parameters for a run."""
    if client is None:
        client = get_mlflow_client()

    run = client.get_run(run_id)
    return dict(run.data.params)


def list_artifacts(run_id: str, path: str = "", client: MlflowClient | None = None) -> list[str]:
    """List all artifacts in a run."""
    if client is None:
        client = get_mlflow_client()

    artifacts = client.list_artifacts(run_id, path)
    return [a.path for a in artifacts]

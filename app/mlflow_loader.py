"""
MLflow Loader for Streamlit Dashboard.

Loads and transforms artifacts from MLflow runs for dashboard visualization.
Handles wrapper extraction, format adaptation, and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME as EXPERIMENT_NAME


# =============================================================================
# ARTIFACT MAPPING: App Key -> Pipeline Path
# =============================================================================

# Maps dashboard artifact keys to their MLflow paths
ARTIFACT_MAPPING = {
    # Model Parameters
    "adstock": "deliverables/adstock.json",
    "adstock_territory": "deliverables/adstock_territory.json",
    "saturation": "deliverables/saturation.json",
    "saturation_territory": "deliverables/saturation_territory.json",
    
    # Contributions & ROI
    "contributions": "deliverables/contributions.json",
    "contributions_territory": "deliverables/contributions_territory.json",
    "roi": "deliverables/roi.json",
    "roi_hdi": "deliverables/roi_hdi.json",
    "regional": "deliverables/regional.json",
    
    # Optimization
    "optimization": "deliverables/optimization.json",
    "optimization_territory": "deliverables/optimization_territory.json",
    "revenue_lift": "deliverables/revenue_lift.json",
    "lift_by_territory": "deliverables/lift_by_territory.json",
    
    # Predictions & Metrics
    "predictions": "deliverables/predictions.json",
    "channel_metrics": "deliverables/channel_metrics.json",
    "blended_metrics": "deliverables/blended_metrics.json",
    "marginal_roas": "deliverables/marginal_roas.json",
}

# Maps artifact keys to their JSON wrapper keys (None = flat dict)
WRAPPER_KEYS = {
    "adstock": "adstock",
    "adstock_territory": "adstock_territory",
    "saturation": "saturation",
    "saturation_territory": "saturation_territory",
    "contributions": "contributions",
    "contributions_territory": "contributions_territory",
    "roi": "roi",
    "roi_hdi": "roi_hdi",
    "regional": "regional",
    "optimization": "optimization",
    "optimization_territory": "optimization_territory",
    "revenue_lift": "revenue_lift",
    "lift_by_territory": "lift_by_territory",
    "predictions": "predictions",
    "channel_metrics": "channel_metrics",
    "blended_metrics": None,  # Flat dict, no wrapper
    "marginal_roas": "marginal_roas",
}


# =============================================================================
# CLIENT & EXPERIMENT HELPERS
# =============================================================================

def get_mlflow_client(tracking_uri: str | None = None) -> MlflowClient:
    """Return MLflow client configured with project tracking URI."""
    uri = tracking_uri or MLFLOW_TRACKING_URI
    return MlflowClient(tracking_uri=uri)


def get_experiment_id(client: MlflowClient, experiment_name: str = EXPERIMENT_NAME) -> str:
    """Return experiment ID for the given experiment name."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return experiment.experiment_id


# =============================================================================
# RUN DISCOVERY
# =============================================================================

def get_latest_hierarchical_run(client: MlflowClient | None = None) -> dict:
    """Return metadata for the latest hierarchical model run."""
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
    """Return list of runs, optionally filtered by model type."""
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


# =============================================================================
# ARTIFACT LOADING & TRANSFORMATION
# =============================================================================

def load_deliverable(run_id: str, name: str, client: MlflowClient | None = None) -> Any:
    """
    Load and transform a single deliverable artifact.
    
    Handles path mapping, wrapper extraction, and format normalization.
    Returns None if artifact not found or loading fails.
    """
    if client is None:
        client = get_mlflow_client()

    artifact_path = ARTIFACT_MAPPING.get(name)
    if not artifact_path:
        return None

    try:
        # Download and parse JSON
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path) as f:
            data = json.load(f)
            
        # Extract from wrapper if needed
        if isinstance(data, dict):
            wrapper_key = WRAPPER_KEYS.get(name)
            
            if wrapper_key is None:
                return data  # Flat dict (e.g., blended_metrics)
            
            if wrapper_key in data:
                data = data[wrapper_key]
            elif len(data) == 1:
                data = next(iter(data.values()))  # Fallback: single-key wrapper
        
        # Normalize legacy ROI format (roi_mean -> roi)
        if name == "roi" and isinstance(data, list) and len(data) > 0:
            if "roi_mean" in data[0]:
                return [
                    {
                        "channel": item.get("channel"),
                        "roi": item.get("roi_mean"),
                        "contribution": item.get("contribution"),
                        "spend": item.get("total_spend")
                    }
                    for item in data
                ]

        return data

    except Exception:
        return None


def load_all_deliverables(run_id: str, client: MlflowClient | None = None) -> dict[str, Any]:
    """Load all dashboard-required deliverables for a run."""
    if client is None:
        client = get_mlflow_client()

    deliverables = {}
    for name in ARTIFACT_MAPPING.keys():
        data = load_deliverable(run_id, name, client)
        if data is not None:
            deliverables[name] = data

    return deliverables


# =============================================================================
# RUN METADATA HELPERS
# =============================================================================

def get_run_metrics(run_id: str, client: MlflowClient | None = None) -> dict:
    """Return all logged metrics for a run."""
    if client is None:
        client = get_mlflow_client()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def get_run_params(run_id: str, client: MlflowClient | None = None) -> dict:
    """Return all logged parameters for a run."""
    if client is None:
        client = get_mlflow_client()
    run = client.get_run(run_id)
    return dict(run.data.params)


def list_artifacts(run_id: str, path: str = "", client: MlflowClient | None = None) -> list[str]:
    """Return list of artifact paths for a run."""
    if client is None:
        client = get_mlflow_client()
    artifacts = client.list_artifacts(run_id, path)
    return [a.path for a in artifacts]

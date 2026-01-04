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
    ARTIFACT_MAPPING = {
        # Direct Mappings (Split Artifacts)
        "adstock": "deliverables/adstock.json",
        "adstock_territory": "deliverables/adstock_territory.json",
        "saturation": "deliverables/saturation.json",
        "saturation_territory": "deliverables/saturation_territory.json",
        "contributions": "deliverables/contributions.json",
        "contributions_territory": "deliverables/contributions_territory.json",
        "roi": "deliverables/roi.json",
        "roi_hdi": "deliverables/roi_hdi.json",
        "regional": "deliverables/regional.json",
        "optimization": "deliverables/optimization.json",
        "optimization_territory": "deliverables/optimization_territory.json",
        "revenue_lift": "deliverables/revenue_lift.json",
        "lift_by_territory": "deliverables/lift_by_territory.json",
        "predictions": "deliverables/predictions.json",
        "channel_metrics": "deliverables/channel_metrics.json",
        "blended_metrics": "deliverables/blended_metrics.json",
        "marginal_roas": "deliverables/marginal_roas.json", # Attempt mapping if exists
    }

    artifact_path = ARTIFACT_MAPPING.get(name)

    if not artifact_path:
        return None

    try:
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path) as f:
            data = json.load(f)
            
        # 2. Extract Inner Data from Wrapper
        # All artifacts follow {"key_name": actual_data} pattern
        # Map artifact name to its wrapper key explicitly
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
        
        if isinstance(data, dict):
            wrapper_key = WRAPPER_KEYS.get(name)
            
            # blended_metrics is flat dict (no wrapper)
            if wrapper_key is None:
                return data
            
            # Extract using explicit wrapper key
            if wrapper_key in data:
                data = data[wrapper_key]
            elif len(data) == 1:
                # Fallback: single key wrapper
                data = next(iter(data.values()))
        
        # 3. Additional transformations for specific formats
        
        # ROI: ensure list format with 'roi' key (some old formats use 'roi_mean')
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

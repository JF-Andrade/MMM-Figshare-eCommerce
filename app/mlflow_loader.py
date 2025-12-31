"""
MLflow data loader for Streamlit dashboard.

Loads deliverables from MLflow artifacts for dashboard visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

# Default MLflow configuration
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


def load_deliverable(run_id: str, name: str, client: MlflowClient | None = None) -> dict:
    """
    Load a deliverable JSON from MLflow artifacts.

    Args:
        run_id: MLflow run ID.
        name: Deliverable name (without .json extension).
        client: Optional MLflow client.

    Returns:
        Parsed JSON data.
    """
    if client is None:
        client = get_mlflow_client()

    artifact_path = f"deliverables/{name}.json"

    try:
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Could not load deliverable '{name}' from run {run_id}: {e}")


def load_all_deliverables(run_id: str, client: MlflowClient | None = None) -> dict[str, Any]:
    """
    Load all 8 deliverables for a run.

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
        "adstock",
        "contributions",
        "optimization",
        "marginal_roas",
        "regional",
        "revenue_lift",
    ]

    deliverables = {}
    for name in deliverable_names:
        try:
            data = load_deliverable(run_id, name, client)
            # Handle different JSON structures:
            # Some are {name: [...]} and some are just [...]
            if isinstance(data, dict) and name in data:
                deliverables[name] = data[name]
            elif isinstance(data, dict) and len(data) == 1:
                # Single key dict - extract the value
                deliverables[name] = list(data.values())[0]
            else:
                deliverables[name] = data
        except Exception as e:
            # Silent fail for optional deliverables
            deliverables[name] = None

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

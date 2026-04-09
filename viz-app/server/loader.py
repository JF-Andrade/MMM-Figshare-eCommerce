"""
MLflow Loader for viz-app Backend.
Bridge between MLflow artifacts and the FastAPI service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

# Add project root to path to import src.config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import constants if needed, or define locally to avoid circular deps
# We'll use the environment variables or defaults
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = "MMM-Experiments"

ARTIFACT_MAPPING = {
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
    "marginal_roas": "deliverables/marginal_roas.json",
    "adstock": "deliverables/adstock.json",
    "saturation": "deliverables/saturation.json",
}

def get_mlflow_client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def get_experiment_id(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
    return experiment.experiment_id

def get_all_runs(client: MlflowClient) -> list[dict]:
    try:
        experiment_id = get_experiment_id(client)
    except ValueError:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=50,
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or f"run_{run.info.run_id[:8]}",
            "start_time": run.info.start_time,
            "status": run.info.status,
            "model_type": run.data.params.get("model_type", "hierarchical"),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }
        for run in runs
    ]

def load_deliverable(run_id: str, name: str, client: MlflowClient) -> Any:
    artifact_path = ARTIFACT_MAPPING.get(name)
    if not artifact_path:
        return None

    try:
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path) as f:
            data = json.load(f)
            
        # Extract from wrapper if needed (logic matches app/mlflow_loader.py)
        if isinstance(data, dict):
            if name in data:
                return data[name]
            elif len(data) == 1:
                return next(iter(data.values()))
        
        return data
    except Exception:
        return None

def load_all_deliverables(run_id: str, client: MlflowClient) -> dict[str, Any]:
    deliverables = {}
    for name in ARTIFACT_MAPPING.keys():
        data = load_deliverable(run_id, name, client)
        if data is not None:
            deliverables[name] = data
    return deliverables

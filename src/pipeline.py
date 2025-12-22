"""
MMM Pipeline Orchestrator.

Production-grade pipeline with:
- Modular stages
- Structured logging
- Error handling with retries
- Metadata tracking for reproducibility
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import PipelineConfig, generate_run_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""

    LOAD = auto()
    PREPROCESS = auto()
    TRAIN_BASELINE = auto()
    TRAIN_HIERARCHICAL = auto()
    EVALUATE = auto()
    EXPORT = auto()


@dataclass
class PipelineState:
    """Holds pipeline execution state and artifacts."""

    raw_data: pd.DataFrame | None = None
    processed_data: pd.DataFrame | None = None
    baseline_model: Any = None
    hierarchical_model: Any = None
    baseline_roi: pd.DataFrame | None = None
    hierarchical_roi: pd.DataFrame | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    hierarchical_metrics: dict[str, Any] = field(default_factory=dict)
    current_stage: PipelineStage | None = None
    completed_stages: list[PipelineStage] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def stage_timer(stage_name: str, timings: dict[str, float]):
    """Context manager for timing stage execution."""
    logger.info(f"[START] {stage_name}")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        timings[stage_name] = elapsed
        logger.info(f"[DONE] {stage_name} ({elapsed:.1f}s)")


def with_retry(max_retries: int = 3, delay: int = 5):
    """Decorator for retry logic with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_exception

        return wrapper

    return decorator


class MMMPipeline:
    """
    Production-grade pipeline orchestrator for MMM project.

    Features:
    - Modular stage execution
    - Structured logging with timing
    - Error handling with retries
    - Metadata tracking for reproducibility
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.state = PipelineState()
        self._start_time: float | None = None

    def run(
        self,
        stages: list[PipelineStage] | None = None,
        skip_stages: list[PipelineStage] | None = None,
    ) -> PipelineState:
        """
        Execute pipeline stages.

        Args:
            stages: Specific stages to run. If None, runs all.
            skip_stages: Stages to skip.

        Returns:
            Pipeline state with all artifacts.
        """
        self._start_time = time.time()

        logger.info("=" * 60)
        logger.info("MMM PIPELINE")
        logger.info(f"Run ID: {self.config.run_id}")
        logger.info(f"Git: {self.config.git_hash}")
        logger.info("=" * 60)

        all_stages = list(PipelineStage)
        stages = stages or all_stages

        if skip_stages:
            stages = [s for s in stages if s not in skip_stages]

        logger.info(f"Stages to run: {[s.name for s in stages]}")

        for stage in stages:
            self._run_stage(stage)

        total_time = time.time() - self._start_time
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE ({total_time:.1f}s)")
        logger.info(f"Completed: {[s.name for s in self.state.completed_stages]}")
        logger.info("=" * 60)

        return self.state

    def _run_stage(self, stage: PipelineStage) -> None:
        """Execute a single pipeline stage with timing and error handling."""
        self.state.current_stage = stage

        stage_methods = {
            PipelineStage.LOAD: self._load_data,
            PipelineStage.PREPROCESS: self._preprocess,
            PipelineStage.TRAIN_BASELINE: self._train_baseline,
            PipelineStage.TRAIN_HIERARCHICAL: self._train_hierarchical,
            PipelineStage.EVALUATE: self._evaluate,
            PipelineStage.EXPORT: self._export_results,
        }

        method = stage_methods.get(stage)
        if method:
            with stage_timer(stage.name, self.state.stage_timings):
                method()
            self.state.completed_stages.append(stage)

    def _load_data(self) -> None:
        """Load data from processed parquet or raw CSV."""
        processed_path = self.config.processed_data_path

        if processed_path.exists():
            logger.info(f"Loading processed data: {processed_path}")
            self.state.processed_data = pd.read_parquet(processed_path)
        else:
            raw_path = self.config.raw_data_path
            logger.info(f"Loading raw data: {raw_path}")
            self.state.raw_data = pd.read_csv(raw_path)

        df = self.state.processed_data if self.state.processed_data is not None else self.state.raw_data
        logger.info(f"Loaded {len(df):,} rows")

    def _preprocess(self) -> None:
        """Preprocess raw data."""
        if self.state.processed_data is not None:
            logger.info("Processed data loaded, skipping preprocessing")
            return

        if self.state.raw_data is None:
            raise ValueError("No data loaded. Run LOAD stage first.")

        logger.info("Preprocessing data...")
        df = self.state.raw_data.copy()
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])

        for col in self.config.channel_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if "is_holiday" not in df.columns:
            df["is_holiday"] = 0

        self.state.processed_data = df

        output_path = self.config.processed_data_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info(f"Saved: {output_path}")

    @with_retry(max_retries=2, delay=10)
    def _train_baseline(self) -> None:
        """Train baseline model with retry."""
        if self.state.processed_data is None:
            raise ValueError("No data. Run LOAD/PREPROCESS first.")

        logger.info("Training baseline model...")

        from scripts.mmm_baseline import run_baseline

        mmm, roi_df, metrics = run_baseline(
            self.config.processed_data_path,
            self.config.models_dir,
            region=self.config.baseline_region,
        )

        self.state.baseline_model = mmm
        self.state.baseline_roi = roi_df
        self.state.baseline_metrics = metrics

    @with_retry(max_retries=2, delay=10)
    def _train_hierarchical(self) -> None:
        """Train hierarchical model with retry."""
        if self.state.processed_data is None:
            raise ValueError("No data. Run LOAD/PREPROCESS first.")

        logger.info("Training hierarchical model...")

        from scripts.mmm_hierarchical import run_hierarchical

        model, idata, metrics = run_hierarchical(
            self.config.processed_data_path,
            self.config.models_dir,
            max_regions=self.config.max_regions,
        )

        self.state.hierarchical_model = idata  # Store trace
        self.state.hierarchical_metrics = metrics

    def _evaluate(self) -> None:
        """Summarize evaluation results."""
        logger.info("Summarizing results...")

        summary = {
            "baseline": self.state.baseline_metrics,
            "hierarchical": self.state.hierarchical_metrics,
        }

        for model_type, metrics in summary.items():
            if metrics:
                logger.info(f"{model_type}: R²={metrics.get('r2_test', 'N/A')}")

    def _export_results(self) -> None:
        """Export pipeline metadata and results."""
        metadata = {
            "run_id": self.config.run_id,
            "git_hash": self.config.git_hash,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "stages_completed": [s.name for s in self.state.completed_stages],
            "stage_timings": self.state.stage_timings,
            "baseline_metrics": self.state.baseline_metrics,
            "hierarchical_metrics": self.state.hierarchical_metrics,
        }

        output_path = self.config.reports_dir / f"run_{self.config.run_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Exported: {output_path}")


def create_pipeline(**kwargs) -> MMMPipeline:
    """Factory function to create pipeline with custom config."""
    config = PipelineConfig(**kwargs)
    return MMMPipeline(config)

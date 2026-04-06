#!/usr/bin/env python
"""
MMM Pipeline Runner.

Production CLI for executing pipeline stages.

Usage:
    python scripts/run_pipeline.py                       # Run all stages
    python scripts/run_pipeline.py --baseline-only       # Skip hierarchical
    python scripts/run_pipeline.py --stages load train_baseline
    python scripts/run_pipeline.py --dry-run             # Preview execution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.pipeline import MMMPipeline, PipelineStage


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MMM Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                    # All stages
  python scripts/run_pipeline.py --baseline-only    # Skip hierarchical
  python scripts/run_pipeline.py --dry-run          # Preview
        """,
    )

    # Stage selection
    stage_choices = [s.name.lower() for s in PipelineStage]
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=stage_choices,
        metavar="STAGE",
        help=f"Stages to run: {stage_choices}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=stage_choices,
        metavar="STAGE",
        help="Stages to skip",
    )

    # Shortcuts
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline model",
    )
    parser.add_argument(
        "--hierarchical-only",
        action="store_true",
        help="Run only hierarchical model",
    )
    parser.add_argument(
        "--deliverables-only",
        action="store_true",
        help="Run only deliverables generation (requires prior training)",
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to processed data parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for models",
    )

    # Model settings
    parser.add_argument(
        "--max-regions",
        type=int,
        help="Max regions for hierarchical model",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="Specific region for baseline model",
    )

    # Execution
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Build config
    config_kwargs = {}

    if args.data_path:
        config_kwargs["processed_data_path"] = args.data_path

    if args.output_dir:
        config_kwargs["models_dir"] = args.output_dir

    if args.max_regions:
        config_kwargs["max_regions"] = args.max_regions

    if args.region:
        config_kwargs["baseline_region"] = args.region

    config = PipelineConfig(**config_kwargs)

    # Determine stages
    stages = None
    if args.stages:
        stages = [PipelineStage[s.upper()] for s in args.stages]

    skip_stages = []
    if args.skip:
        skip_stages = [PipelineStage[s.upper()] for s in args.skip]

    if args.baseline_only:
        skip_stages.append(PipelineStage.TRAIN_HIERARCHICAL)
        skip_stages.append(PipelineStage.GENERATE_DELIVERABLES)

    if args.hierarchical_only:
        skip_stages.append(PipelineStage.TRAIN_BASELINE)

    if args.deliverables_only:
        # Only run GENERATE_DELIVERABLES stage
        stages = [PipelineStage.GENERATE_DELIVERABLES]

    # Compute effective stages
    effective_stages = stages or list(PipelineStage)
    effective_stages = [s for s in effective_stages if s not in skip_stages]

    # Dry run
    if args.dry_run:
        print("=" * 50)
        print("DRY RUN - Would execute:")
        print("=" * 50)
        print(f"Run ID:     {config.run_id}")
        print(f"Git:        {config.git_hash}")
        print(f"Data:       {config.processed_data_path}")
        print(f"Output:     {config.models_dir}")
        print(f"Stages:     {[s.name for s in effective_stages]}")
        print("=" * 50)
        return 0

    # Run pipeline
    pipeline = MMMPipeline(config)

    try:
        pipeline.run(
            stages=stages,
            skip_stages=skip_stages or None,
        )
        return 0
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

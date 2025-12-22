"""
Model comparison utilities.

Compare baseline (Ridge) vs hierarchical (Bayesian) model performance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compare_models(
    baseline_metrics: dict,
    hierarchical_metrics: dict,
) -> pd.DataFrame:
    """
    Compare baseline vs hierarchical model metrics.

    Args:
        baseline_metrics: Dict with r2_test, mape_test, training_time.
        hierarchical_metrics: Dict with r2_test, mape_test, training_time.

    Returns:
        DataFrame with comparison.
    """
    metrics_to_compare = ["r2_test", "mape_test", "training_time"]

    comparison = []
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, None)
        hier_val = hierarchical_metrics.get(metric, None)

        # Determine winner
        if baseline_val is None or hier_val is None:
            winner = "N/A"
        elif metric == "r2_test":
            winner = "Hierarchical" if hier_val > baseline_val else "Baseline"
        elif metric == "mape_test":
            winner = "Hierarchical" if hier_val < baseline_val else "Baseline"
        elif metric == "training_time":
            winner = "Baseline" if baseline_val < hier_val else "Hierarchical"
        else:
            winner = "N/A"

        comparison.append({
            "metric": metric,
            "baseline": baseline_val,
            "hierarchical": hier_val,
            "winner": winner,
        })

    return pd.DataFrame(comparison)


def compute_improvement(
    baseline_metrics: dict,
    hierarchical_metrics: dict,
) -> dict:
    """
    Compute improvement percentages.

    Returns:
        Dict with r2_improvement, mape_improvement, speed_ratio.
    """
    result = {}

    # R2 improvement (absolute difference)
    b_r2 = baseline_metrics.get("r2_test")
    h_r2 = hierarchical_metrics.get("r2_test")
    if b_r2 is not None and h_r2 is not None:
        result["r2_improvement"] = h_r2 - b_r2
        result["r2_improvement_pct"] = ((h_r2 - b_r2) / abs(b_r2) * 100) if b_r2 != 0 else 0

    # MAPE improvement (lower is better)
    b_mape = baseline_metrics.get("mape_test")
    h_mape = hierarchical_metrics.get("mape_test")
    if b_mape is not None and h_mape is not None:
        result["mape_improvement"] = b_mape - h_mape
        result["mape_improvement_pct"] = ((b_mape - h_mape) / b_mape * 100) if b_mape != 0 else 0

    # Speed ratio
    b_time = baseline_metrics.get("training_time")
    h_time = hierarchical_metrics.get("training_time")
    if b_time is not None and h_time is not None and b_time > 0:
        result["speed_ratio"] = h_time / b_time

    return result


def generate_comparison_insight(comparison_df: pd.DataFrame, improvement: dict) -> str:
    """
    Generate human-readable insight from comparison.

    Returns:
        Insight string.
    """
    insights = []

    # R2 comparison
    r2_row = comparison_df[comparison_df["metric"] == "r2_test"]
    if not r2_row.empty:
        winner = r2_row.iloc[0]["winner"]
        if winner == "Hierarchical":
            pct = improvement.get("r2_improvement_pct", 0)
            insights.append(f"Hierarchical model improves R2 by {abs(pct):.1f}%")
        else:
            insights.append("Baseline model has competitive R2 performance")

    # MAPE comparison
    mape_row = comparison_df[comparison_df["metric"] == "mape_test"]
    if not mape_row.empty:
        winner = mape_row.iloc[0]["winner"]
        if winner == "Hierarchical":
            pct = improvement.get("mape_improvement_pct", 0)
            insights.append(f"Reduces MAPE by {abs(pct):.1f}%")

    # Speed comparison
    speed_ratio = improvement.get("speed_ratio", 1)
    if speed_ratio > 1:
        insights.append(f"Baseline is {speed_ratio:.0f}x faster to train")

    return ". ".join(insights) + "." if insights else "Models have similar performance."


def format_metric_value(metric: str, value: float | None) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"

    if metric == "r2_test":
        return f"{value:.3f}"
    elif metric == "mape_test":
        return f"{value:.1f}%"
    elif metric == "training_time":
        if value < 60:
            return f"{value:.1f}s"
        else:
            return f"{value / 60:.1f}min"
    return f"{value:.2f}"

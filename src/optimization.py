"""
Budget Optimization Module.

Provides budget allocation and marginal ROAS computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM


def optimize_budget(
    mmm: MMM,
    total_budget: float,
    budget_bounds: dict[str, tuple[float, float]] | None = None,
    num_periods: int = 1,
) -> pd.DataFrame:
    """
    Optimize budget allocation across channels.

    Uses PyMC-Marketing's optimize_channel_budget_for_maximum_contribution.

    Args:
        mmm: Fitted MMM model.
        total_budget: Total budget to allocate.
        budget_bounds: Optional min/max bounds per channel.
        num_periods: Number of periods to optimize over.

    Returns:
        DataFrame with channel, current_spend, optimal_spend, and change_pct.
    """
    # Get current allocation from training data
    X_train = mmm.X  # type: ignore
    current_allocation = {ch: X_train[ch].sum() for ch in mmm.channel_columns}
    current_total = sum(current_allocation.values())

    # Scale to target budget
    scale_factor = total_budget / current_total if current_total > 0 else 1.0
    scaled_current = {ch: v * scale_factor for ch, v in current_allocation.items()}

    # Try different API methods (changed across pymc-marketing versions)
    optimizer_result = None
    try:
        if hasattr(mmm, 'optimize_channel_budget_for_maximum_contribution'):
            optimizer_result = mmm.optimize_channel_budget_for_maximum_contribution(
                total_budget=total_budget,
                budget_bounds=budget_bounds,
                num_periods=num_periods,
            )
        elif hasattr(mmm, 'optimize_budget'):
            optimizer_result = mmm.optimize_budget(
                total_budget=total_budget,
            )
        elif hasattr(mmm, 'allocator'):
            optimizer_result = mmm.allocator.allocate_budget(
                total_budget=total_budget,
            )
        else:
            print("WARNING: Budget optimization not available in this pymc-marketing version")
            optimizer_result = None
    except Exception as e:
        print(f"WARNING: Budget optimization failed: {e}")
        optimizer_result = None

    # Fallback if optimization not available
    if optimizer_result is None:
        return pd.DataFrame([
            {
                "channel": ch.replace("_SPEND", ""),
                "current_spend": scaled_current.get(ch, 0),
                "optimal_spend": scaled_current.get(ch, 0),
                "change_pct": 0,
            }
            for ch in mmm.channel_columns
        ])

    # Handle different return types (dict or tuple with DataFrame)
    if isinstance(optimizer_result, dict):
        optimal_allocation = optimizer_result
    elif isinstance(optimizer_result, tuple):
        # Newer versions return (allocation_dict, something_else) or DataFrame
        if isinstance(optimizer_result[0], dict):
            optimal_allocation = optimizer_result[0]
        elif hasattr(optimizer_result[0], "to_dict"):
            # It's a DataFrame
            df_result = optimizer_result[0]
            optimal_allocation = dict(zip(df_result["channel"], df_result["optimal_spend"]))
        else:
            # Fallback: use current allocation
            optimal_allocation = scaled_current
    elif hasattr(optimizer_result, "to_dict"):
        # It's a DataFrame directly
        optimal_allocation = dict(zip(optimizer_result["channel"], optimizer_result["optimal_spend"]))
    else:
        optimal_allocation = scaled_current

    # Build result DataFrame
    results = []
    for ch in mmm.channel_columns:
        current = scaled_current.get(ch, 0)
        optimal = optimal_allocation.get(ch, optimal_allocation.get(ch.replace("_SPEND", ""), 0))
        change_pct = ((optimal - current) / current * 100) if current > 0 else 0

        results.append({
            "channel": ch.replace("_SPEND", ""),
            "current_spend": current,
            "optimal_spend": optimal,
            "change_pct": change_pct,
        })

    return pd.DataFrame(results).sort_values("optimal_spend", ascending=False)


def compute_marginal_roas(
    mmm: MMM,
    X: pd.DataFrame,
    spend_increases: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute marginal ROAS at different spend levels.

    Args:
        mmm: Fitted MMM model.
        X: Feature DataFrame.
        spend_increases: Percentage increases to test (default: [0, 10, 25, 50, 100]).

    Returns:
        DataFrame with channel, spend_level, marginal_roas.
    """
    if spend_increases is None:
        spend_increases = [0, 10, 25, 50, 100]

    results = []

    for ch in mmm.channel_columns:
        base_spend = X[ch].sum()

        for pct in spend_increases:
            new_spend = base_spend * (1 + pct / 100)

            # Create modified X
            X_modified = X.copy()
            X_modified[ch] = X_modified[ch] * (1 + pct / 100)

            # Predict with modified spend
            y_pred_base = mmm.predict(X).mean(axis=0).sum()
            y_pred_new = mmm.predict(X_modified).mean(axis=0).sum()

            # Marginal ROAS = delta_revenue / delta_spend
            delta_revenue = y_pred_new - y_pred_base
            delta_spend = new_spend - base_spend
            marginal_roas = delta_revenue / delta_spend if delta_spend > 0 else 0

            results.append({
                "channel": ch.replace("_SPEND", ""),
                "spend_increase_pct": pct,
                "marginal_roas": marginal_roas,
            })

    return pd.DataFrame(results)


def plot_optimization_results(
    optimization_df: pd.DataFrame,
    output_dir: Path,
    title: str = "Budget Optimization",
) -> None:
    """
    Generate visualization of optimization results.

    Args:
        optimization_df: DataFrame from optimize_budget.
        output_dir: Directory to save plots.
        title: Plot title.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Current vs Optimal allocation
    ax = axes[0]
    x = np.arange(len(optimization_df))
    width = 0.35

    ax.bar(x - width / 2, optimization_df["current_spend"], width, label="Current", color="steelblue")
    ax.bar(x + width / 2, optimization_df["optimal_spend"], width, label="Optimal", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(optimization_df["channel"], rotation=45, ha="right")
    ax.set_ylabel("Budget")
    ax.set_title("Current vs Optimal Allocation")
    ax.legend()

    # Change percentage
    ax = axes[1]
    colors = ["green" if v > 0 else "red" for v in optimization_df["change_pct"]]
    ax.barh(optimization_df["channel"], optimization_df["change_pct"], color=colors)
    ax.axvline(x=0, color="gray", linestyle="--")
    ax.set_xlabel("Change (%)")
    ax.set_title("Budget Reallocation")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "budget_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_marginal_roas_curves(
    marginal_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot marginal ROAS curves for each channel.

    Args:
        marginal_df: DataFrame from compute_marginal_roas.
        output_dir: Directory to save plot.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for channel in marginal_df["channel"].unique():
        ch_data = marginal_df[marginal_df["channel"] == channel]
        ax.plot(
            ch_data["spend_increase_pct"],
            ch_data["marginal_roas"],
            marker="o",
            label=channel,
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", label="Break-even")
    ax.set_xlabel("Spend Increase (%)")
    ax.set_ylabel("Marginal ROAS")
    ax.set_title("Marginal ROAS by Channel")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "marginal_roas_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def compute_revenue_lift(
    mmm: MMM,
    X: pd.DataFrame,
    optimization_df: pd.DataFrame,
) -> dict:
    """
    Compute projected revenue lift from optimal budget allocation.

    Args:
        mmm: Fitted MMM model.
        X: Feature DataFrame.
        optimization_df: DataFrame from optimize_budget with current and optimal spend.

    Returns:
        Dict with current_revenue, optimal_revenue, lift_absolute, lift_pct.
    """
    # Current revenue prediction
    y_current = mmm.predict(X).mean(axis=0).sum()

    # Create X with optimal allocation
    X_optimal = X.copy()

    # Get channel mapping
    for _, row in optimization_df.iterrows():
        ch_name = row["channel"]
        # Find matching column
        for col in mmm.channel_columns:
            if ch_name in col:
                current_total = X[col].sum()
                optimal_total = row["optimal_spend"]
                if current_total > 0:
                    scale_factor = optimal_total / current_total
                    X_optimal[col] = X_optimal[col] * scale_factor
                break

    # Optimal revenue prediction
    y_optimal = mmm.predict(X_optimal).mean(axis=0).sum()

    lift_absolute = y_optimal - y_current
    lift_pct = (lift_absolute / y_current * 100) if y_current > 0 else 0

    result = {
        "current_revenue": float(y_current),
        "optimal_revenue": float(y_optimal),
        "lift_absolute": float(lift_absolute),
        "lift_pct": float(lift_pct),
    }

    print("\n=== Revenue Lift Projection ===")
    print(f"Current Revenue: {y_current:,.0f}")
    print(f"Optimal Revenue: {y_optimal:,.0f}")
    print(f"Lift: {lift_absolute:,.0f} ({lift_pct:.1f}%)")

    return result


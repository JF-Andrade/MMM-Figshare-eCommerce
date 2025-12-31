"""
Insights Module.

Extracts learned parameters from fitted MMM and generates business-facing
visualizations and budget optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================


def extract_adstock_params(mmm: MMM) -> pd.DataFrame:
    """
    Extract adstock decay rates (alpha) from posterior.

    Args:
        mmm: Fitted MMM model.

    Returns:
        DataFrame with channel, alpha_mean, alpha_std, alpha_hdi_3%, alpha_hdi_97%.
    """
    posterior = mmm.idata.posterior

    if "alpha" not in posterior:
        raise ValueError("Model posterior does not contain 'alpha' parameter")

    alpha = posterior["alpha"]

    if isinstance(alpha, tuple):
        alpha = alpha[0] if len(alpha) > 0 else None
        if alpha is None:
            raise ValueError("Could not extract alpha from tuple")

    results = []
    for i, ch in enumerate(mmm.channel_columns):
        try:
            if hasattr(alpha, 'dims') and "channel" in alpha.dims:
                alpha_ch = alpha.sel(channel=ch)
            elif hasattr(alpha, 'isel'):
                alpha_ch = alpha.isel(channel=i)
            else:
                values = np.array(alpha).flatten()
                alpha_ch = None
        except Exception:
            values = np.array(alpha).flatten()
            alpha_ch = None

        if alpha_ch is not None:
            values = alpha_ch.values.flatten()

        results.append({
            "channel": ch.replace("_SPEND", ""),
            "alpha_mean": float(values.mean()),
            "alpha_std": float(values.std()),
            "alpha_hdi_3%": float(np.percentile(values, 3)),
            "alpha_hdi_97%": float(np.percentile(values, 97)),
            "half_life_weeks": float(-np.log(2) / np.log(values.mean())) if values.mean() > 0 and values.mean() < 1 else np.inf,
        })

    return pd.DataFrame(results).sort_values("alpha_mean", ascending=False)


def extract_saturation_params(mmm: MMM) -> pd.DataFrame:
    """
    Extract saturation parameters (lambda) from posterior.

    Args:
        mmm: Fitted MMM model.

    Returns:
        DataFrame with channel, lam_mean, lam_std, lam_hdi_3%, lam_hdi_97%.
    """
    posterior = mmm.idata.posterior

    if "lam" not in posterior:
        raise ValueError("Model posterior does not contain 'lam' parameter")

    lam = posterior["lam"]

    if isinstance(lam, tuple):
        lam = lam[0] if len(lam) > 0 else None
        if lam is None:
            raise ValueError("Could not extract lam from tuple")

    results = []
    for i, ch in enumerate(mmm.channel_columns):
        try:
            if hasattr(lam, 'dims') and "channel" in lam.dims:
                lam_ch = lam.sel(channel=ch)
            elif hasattr(lam, 'isel'):
                lam_ch = lam.isel(channel=i)
            else:
                values = np.array(lam).flatten()
                lam_ch = None
        except Exception:
            values = np.array(lam).flatten()
            lam_ch = None

        if lam_ch is not None:
            values = lam_ch.values.flatten()

        results.append({
            "channel": ch.replace("_SPEND", ""),
            "lam_mean": float(values.mean()),
            "lam_std": float(values.std()),
            "lam_hdi_3%": float(np.percentile(values, 3)),
            "lam_hdi_97%": float(np.percentile(values, 97)),
        })

    return pd.DataFrame(results).sort_values("lam_mean", ascending=False)


# =============================================================================
# VISUALIZATION: SATURATION & ADSTOCK
# =============================================================================


def plot_saturation_curves(
    mmm: MMM,
    output_dir: Path,
    max_spend_multiplier: float = 2.0,
) -> None:
    """
    Plot saturation curves for each channel.

    Args:
        mmm: Fitted MMM model.
        output_dir: Directory to save plot.
        max_spend_multiplier: Maximum spend as multiple of observed max.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    saturation_df = extract_saturation_params(mmm)

    fig, ax = plt.subplots(figsize=(12, 6))

    X = mmm.X
    max_spend = max(X[ch].max() for ch in mmm.channel_columns)
    spend_range = np.linspace(0, max_spend * max_spend_multiplier, 200)

    for _, row in saturation_df.iterrows():
        lam = row["lam_mean"]
        saturation = 1 - np.exp(-lam * spend_range / max_spend)
        ax.plot(spend_range, saturation, label=row["channel"], linewidth=2)

    ax.set_xlabel("Spend")
    ax.set_ylabel("Saturation Effect (0-1)")
    ax.set_title("Channel Saturation Curves (Diminishing Returns)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "saturation_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_adstock_decay(
    mmm: MMM,
    output_dir: Path,
    max_weeks: int = 12,
) -> None:
    """
    Plot adstock decay curves for each channel.

    Args:
        mmm: Fitted MMM model.
        output_dir: Directory to save plot.
        max_weeks: Maximum weeks to show decay.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    adstock_df = extract_adstock_params(mmm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    weeks = np.arange(max_weeks + 1)

    ax = axes[0]
    for _, row in adstock_df.iterrows():
        alpha = row["alpha_mean"]
        decay = alpha ** weeks
        ax.plot(weeks, decay, label=row["channel"], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Weeks Since Spend")
    ax.set_ylabel("Retention Rate")
    ax.set_title("Adstock Decay (Carryover Effect)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% retention")

    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(adstock_df)))
    bars = ax.barh(adstock_df["channel"], adstock_df["half_life_weeks"], color=colors)
    ax.set_xlabel("Half-Life (weeks)")
    ax.set_title("Adstock Half-Life by Channel")

    for bar, val in zip(bars, adstock_df["half_life_weeks"]):
        if np.isfinite(val):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}w", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "adstock_decay.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_channel_contributions_waterfall(
    mmm: MMM,
    output_dir: Path,
) -> None:
    """
    Plot waterfall chart of channel contributions.

    Args:
        mmm: Fitted MMM model.
        output_dir: Directory to save plot.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    contributions = mmm.compute_channel_contribution_original_scale()
    mean_contrib = contributions.mean(dim=["chain", "draw"])
    total_contrib = mean_contrib.sum(dim="date")

    channels = [ch.replace("_SPEND", "") for ch in mmm.channel_columns]
    values = [float(total_contrib.sel(channel=ch).values) for ch in mmm.channel_columns]

    sorted_data = sorted(zip(channels, values), key=lambda x: x[1], reverse=True)
    channels, values = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    cumsum = 0
    for i, (ch, val) in enumerate(zip(channels, values)):
        ax.barh(ch, val, left=cumsum, color=plt.cm.viridis(i / len(channels)))
        cumsum += val

    ax.set_xlabel("Total Contribution")
    ax.set_title("Channel Contributions (Waterfall)")

    cumsum = 0
    for ch, val in zip(channels, values):
        ax.text(cumsum + val / 2, ch, f"{val:,.0f}", va="center", ha="center", fontsize=9, color="white")
        cumsum += val

    plt.tight_layout()
    plt.savefig(output_dir / "channel_contributions_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# BUDGET OPTIMIZATION
# =============================================================================


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
    X_train = mmm.X
    current_allocation = {ch: X_train[ch].sum() for ch in mmm.channel_columns}
    current_total = sum(current_allocation.values())

    scale_factor = total_budget / current_total if current_total > 0 else 1.0
    scaled_current = {ch: v * scale_factor for ch, v in current_allocation.items()}

    optimizer_result = None
    try:
        if hasattr(mmm, 'optimize_channel_budget_for_maximum_contribution'):
            optimizer_result = mmm.optimize_channel_budget_for_maximum_contribution(
                total_budget=total_budget,
                budget_bounds=budget_bounds,
                num_periods=num_periods,
            )
        elif hasattr(mmm, 'optimize_budget'):
            optimizer_result = mmm.optimize_budget(total_budget=total_budget)
        elif hasattr(mmm, 'allocator'):
            optimizer_result = mmm.allocator.allocate_budget(total_budget=total_budget)
        else:
            print("WARNING: Budget optimization not available in this pymc-marketing version")
    except Exception as e:
        print(f"WARNING: Budget optimization failed: {e}")

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

    if isinstance(optimizer_result, dict):
        optimal_allocation = optimizer_result
    elif isinstance(optimizer_result, tuple):
        if isinstance(optimizer_result[0], dict):
            optimal_allocation = optimizer_result[0]
        elif hasattr(optimizer_result[0], "to_dict"):
            df_result = optimizer_result[0]
            optimal_allocation = dict(zip(df_result["channel"], df_result["optimal_spend"]))
        else:
            optimal_allocation = scaled_current
    elif hasattr(optimizer_result, "to_dict"):
        optimal_allocation = dict(zip(optimizer_result["channel"], optimizer_result["optimal_spend"]))
    else:
        optimal_allocation = scaled_current

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

            X_modified = X.copy()
            X_modified[ch] = X_modified[ch] * (1 + pct / 100)

            y_pred_base = mmm.predict(X).mean(axis=0).sum()
            y_pred_new = mmm.predict(X_modified).mean(axis=0).sum()

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
    y_current = mmm.predict(X).mean(axis=0).sum()

    X_optimal = X.copy()

    for _, row in optimization_df.iterrows():
        ch_name = row["channel"]
        for col in mmm.channel_columns:
            if ch_name in col:
                current_total = X[col].sum()
                optimal_total = row["optimal_spend"]
                if current_total > 0:
                    scale_factor = optimal_total / current_total
                    X_optimal[col] = X_optimal[col] * scale_factor
                break

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


# =============================================================================
# RIDGE BASELINE INSIGHTS
# =============================================================================


def compute_ridge_coefficients(
    pipeline,
    feature_names: list[str],
    channels: list[str],
) -> pd.DataFrame:
    """
    Extract and format Ridge model coefficients.

    Args:
        pipeline: Fitted sklearn Pipeline with Ridge.
        feature_names: List of feature names.
        channels: List of channel names.

    Returns:
        DataFrame with feature, coefficient, type.
    """
    coefs = pipeline.named_steps["ridge"].coef_
    intercept = pipeline.named_steps["ridge"].intercept_

    coef_data = []
    for name, coef in zip(feature_names, coefs):
        is_channel = any(c in name for c in channels)
        coef_data.append({
            "feature": name,
            "coefficient": float(coef),
            "type": "channel" if is_channel else "control",
        })

    coef_data.append({
        "feature": "intercept",
        "coefficient": float(intercept),
        "type": "intercept",
    })

    return pd.DataFrame(coef_data).sort_values("coefficient", ascending=False)


def plot_baseline_results(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    coef_df: pd.DataFrame,
    output_dir: Path,
    dates_train: pd.Series | None = None,
    dates_test: pd.Series | None = None,
) -> None:
    """
    Generate visualization plots for Ridge baseline model.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_train, X_test: Feature DataFrames.
        y_train, y_test: Target arrays.
        coef_df: Coefficients DataFrame from compute_ridge_coefficients.
        output_dir: Directory to save plots.
        dates_train: Optional Series with training dates.
        dates_test: Optional Series with testing dates.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    y_pred_train = pipeline.predict(X_train)
    ax = axes[0, 0]
    
    if dates_train is not None:
        ax.plot(dates_train, y_train, label="Actual", alpha=0.7)
        ax.plot(dates_train, y_pred_train, label="Predicted", alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.plot(y_train, label="Actual", alpha=0.7)
        ax.plot(y_pred_train, label="Predicted", alpha=0.7)
        
    ax.set_title("Train: Actual vs Predicted")
    ax.legend()

    y_pred_test = pipeline.predict(X_test)
    ax = axes[0, 1]
    
    if dates_test is not None:
        ax.plot(dates_test, y_test, label="Actual", alpha=0.7)
        ax.plot(dates_test, y_pred_test, label="Predicted", alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.plot(y_test, label="Actual", alpha=0.7)
        ax.plot(y_pred_test, label="Predicted", alpha=0.7)
        
    ax.set_title("Test: Actual vs Predicted")
    ax.legend()

    ax = axes[1, 0]
    channel_coefs = coef_df[coef_df["type"] == "channel"]
    colors = ["green" if c > 0 else "red" for c in channel_coefs["coefficient"]]
    ax.barh(channel_coefs["feature"], channel_coefs["coefficient"], color=colors)
    ax.set_title("Channel Coefficients")
    ax.axvline(x=0, color="gray", linestyle="--")

    ax = axes[1, 1]
    residuals = y_train - y_pred_train
    ax.hist(residuals, bins=20, edgecolor="black")
    ax.set_title("Residual Distribution")
    ax.axvline(x=0, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "ridge_baseline_results.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# HIERARCHICAL MODEL INSIGHTS
# =============================================================================


def plot_regional_comparison(
    roi_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate regional comparison visualizations for hierarchical model.

    Args:
        roi_df: DataFrame with column 'region', 'channel', 'roi', 'contribution'.
        output_dir: Directory to save plots.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    pivot_roi = roi_df.pivot(index="channel", columns="region", values="roi")
    pivot_roi.plot(kind="barh", ax=ax)
    ax.axvline(x=1.0, color="gray", linestyle="--", label="Break-even")
    ax.set_xlabel("ROI")
    ax.set_title("Channel ROI by Region")
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax = axes[1]
    pivot_contrib = roi_df.pivot(index="channel", columns="region", values="contribution")
    pivot_contrib.plot(kind="barh", ax=ax)
    ax.set_xlabel("Total Contribution")
    ax.set_title("Channel Contributions by Region")
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "hierarchical_regional_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roi_heatmap(
    roi_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate ROI heatmap across channels and regions.

    Args:
        roi_df: DataFrame with 'channel', 'region', 'roi'.
        output_dir: Directory to save plot.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    pivot = roi_df.pivot(index="channel", columns="region", values="roi")

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 2.5 or val < 1 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="ROI")
    ax.set_title("Channel ROI by Region (Hierarchical Model)")

    plt.savefig(output_dir / "hierarchical_roi_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# HIERARCHICAL OPTIMIZATION (Standalone)
# =============================================================================

def optimize_hierarchical_budget(
    contrib_df: pd.DataFrame,
    saturation_params: list[dict],
    total_budget: float,
    budget_bounds_pct: tuple[float, float] = (0.70, 1.30),
) -> list[dict]:
    """
    Optimize budget allocation using learned attributes.
    
    Without the full PyMC model object available for optimization,
    response curves are reconstructed using:
    1. Current Spend & Contribution -> Estimate Scale Factor (Beta equivalent)
    2. Saturation Lambda -> Shape of curve
    
    Model: Contribution = Scale * (1 - exp(-lambda * spend))
    => Scale = Contribution / (1 - exp(-lambda * spend))
    
    Args:
        contrib_df: DataFrame with 'channel', 'total_spend', 'contribution'.
        saturation_params: List of dicts with 'channel', 'lam_mean'.
        total_budget: Total money to allocate.
        budget_bounds_pct: Min/Max multiplier for individual channel spend (safety bounds).
        
    Returns:
        List of dicts with 'channel', 'current_spend', 'optimal_spend', 'change_pct'.
    """
    from scipy.optimize import minimize, LinearConstraint, Bounds
    
    # 1. Prepare Data
    model_data = []
    lam_map = {p['channel']: p['lam_mean'] for p in saturation_params}
    
    channels = contrib_df['channel'].tolist()
    
    for _, row in contrib_df.iterrows():
        ch = row['channel']
        spend = row['total_spend']
        contribution = row['contribution']
        lam = lam_map.get(ch)
        
        if lam is None or spend <= 0 or contribution <= 0:
            # Fallback for channels with no data/params
            model_data.append({
                'channel': ch,
                'scale': 0,
                'lam': 0,
                'current_spend': spend
            })
            continue
            
        # Estimate Scale Factor A
        # Contrib = A * (1 - exp(-lam * spend))
        # A = Contrib / (1 - exp(-lam * spend))
        # Note: If model uses different saturation (e.g. Hill), this needs adjustment. 
        # The custom model uses exponential saturation.
        saturation_factor = 1 - np.exp(-lam * spend)
        scale = contribution / (saturation_factor + 1e-9)
        
        model_data.append({
            'channel': ch,
            'scale': scale,
            'lam': lam,
            'current_spend': spend
        })
        
    # 2. Define Objective Function (Generic)
    # Maximize Total Contribution (Minimize Negative)
    def objective(spends):
        total_contrib = 0
        for i, x in enumerate(spends):
            m = model_data[i]
            if m['scale'] > 0:
                total_contrib += m['scale'] * (1 - np.exp(-m['lam'] * x))
        return -total_contrib

    # 3. Constraints & Bounds
    x0 = [m['current_spend'] for m in model_data]
    
    # Sum of spend = total_budget
    # LinearConstraint: lb <= A.dot(x) <= ub
    # A = [1, 1, ..., 1]
    A_eq = np.ones((1, len(x0)))
    linear_constraint = LinearConstraint(A_eq, [total_budget], [total_budget])
    
    # Channel Bounds
    bounds_list = []
    for m in model_data:
        current = m['current_spend']
        if current > 0:
            lb = current * budget_bounds_pct[0]
            ub = current * budget_bounds_pct[1]
        else:
            lb = 0
            ub = 0 # Don't activate huge spend on zero-spend channels blindly
        bounds_list.append((lb, ub))
        
    # 4. Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=[linear_constraint],
        bounds=bounds_list,
        options={'disp': False, 'ftol': 1e-4}
    )
    
    # 5. Format Results
    optimized_allocation = []
    
    # Calculate totals for lift
    current_total_contrib = 0
    optimal_total_contrib = 0
    
    for i, m in enumerate(model_data):
        opt_spend = result.x[i] if result.success else m['current_spend']
        current = m['current_spend']
        
        # Calculate contributions
        if m['scale'] > 0:
            curr_c = m['scale'] * (1 - np.exp(-m['lam'] * current))
            opt_c = m['scale'] * (1 - np.exp(-m['lam'] * opt_spend))
        else:
            curr_c = 0
            opt_c = 0
            
        current_total_contrib += curr_c
        optimal_total_contrib += opt_c
        
        change_pct = ((opt_spend - current) / current * 100) if current > 0 else 0.0
        
        optimized_allocation.append({
            "channel": m['channel'],
            "current_spend": float(current),
            "optimal_spend": float(opt_spend),
            "change_pct": change_pct
        })
        
    lift_absolute = optimal_total_contrib - current_total_contrib
    lift_pct = (lift_absolute / current_total_contrib * 100) if current_total_contrib > 0 else 0.0
        
    return {
        "allocation": optimized_allocation,
        "metrics": {
            "current_contribution": float(current_total_contrib),
            "projected_contribution": float(optimal_total_contrib),
            "lift_absolute": float(lift_absolute),
            "lift_pct": float(lift_pct)
        }
    }

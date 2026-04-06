"""
Insights Module.

Extracts learned parameters, generates visualizations, and acts as the
primary reporting engine for both Baseline and Hierarchical models.

ORGANIZATION:
1. Baseline Model Insights (Ridge)
2. Hierarchical Model - Standard API (pymc-marketing helpers)
3. Hierarchical Model - Custom Implementation (Project-specific logic)
   3a. Core Optimization Logic
   3b. Visualization
   3c. Orchestration & Logging
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import LinearConstraint, minimize

from src.config import EPSILON
from src.models.hierarchical_bayesian import (
    check_convergence,
    compute_channel_contributions,
    compute_channel_contributions_by_territory,
    compute_roi_with_hdi,
    evaluate,
    hill_saturation_numpy,
    predict,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# 1. BASELINE MODEL INSIGHTS (Ridge Regression)
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
        DataFrame with feature, coefficient, type, and optional warning.
    """
    coefs = pipeline.named_steps["ridge"].coef_
    intercept = pipeline.named_steps["ridge"].intercept_

    coef_data = []
    warnings_issued = []

    for name, coef in zip(feature_names, coefs):
        is_channel = any(c in name for c in channels)
        warning = None

        # Validate: channel coefficients should be positive
        if is_channel and "_sat" in name and coef < 0:
            warning = "negative_coefficient"
            warnings_issued.append(
                f"⚠️ Negative coefficient for {name}: {coef:.4f}. "
                "This may indicate multicollinearity or identification issues."
            )

        coef_data.append({
            "feature": name,
            "coefficient": float(coef),
            "type": "channel" if is_channel else "control",
            "warning": warning,
        })

    # Print warnings
    for warning_msg in warnings_issued:
        print(warning_msg)

    coef_data.append({
        "feature": "intercept",
        "coefficient": float(intercept),
        "type": "intercept",
        "warning": None,
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
        ax.tick_params(axis="x", rotation=45)
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
        ax.tick_params(axis="x", rotation=45)
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
# 2. HIERARCHICAL MODEL - STANDARD API (pymc-marketing helpers)
#
# MIGRATED: These functions have been moved to:
#   src/utils/pymc_marketing_helpers.py
#
# Functions available there:
#   - extract_adstock_params(mmm)
#   - extract_saturation_params(mmm)
#   - plot_saturation_curves(mmm, output_dir)
#   - plot_adstock_decay(mmm, output_dir)
#   - plot_channel_contributions_waterfall(mmm, output_dir)
#   - optimize_budget(mmm, total_budget)
#   - compute_marginal_roas(mmm, X)
#   - plot_optimization_results(optimization_df, output_dir)
#   - plot_marginal_roas_curves(marginal_df, output_dir)
#   - compute_revenue_lift(mmm, X, optimization_df)
# =============================================================================



# =============================================================================
# 3. HIERARCHICAL MODEL - CUSTOM IMPLEMENTATION
#
# Custom logic for Hierarchical Bayesian Model (Custom PyMC).
# Handles complex territory hierarchy, custom plots, and specific optimization.
# =============================================================================


# -----------------------------------------------------------------------------
# 3a. Core Optimization Logic
# -----------------------------------------------------------------------------


def optimize_hierarchical_budget(
    contrib_df: pd.DataFrame,
    saturation_params: list[dict],
    total_budget: float,
    n_obs: int,
    budget_bounds_pct: tuple[float, float] = (0.70, 1.30),
    marginal_roas_data: list[dict] | None = None,
) -> dict:
    """
    Optimize budget allocation using learned Hill saturation parameters.
    
    Computes lift using exact Hill saturation difference (not marginal ROAS
    point estimate) for accurate projections.

    Args:
        contrib_df: DataFrame with 'channel', 'total_spend', 'contribution'.
        saturation_params: List of dicts with 'channel', 'L_mean', 'k_mean', 'max_spend'.
            CRITICAL: max_spend must match the normalization used during training.
        total_budget: Total money to allocate (sum across all obs).
        n_obs: Number of observations (weeks) - for scale normalization.
        budget_bounds_pct: Min/Max multiplier for individual channel spend.
        marginal_roas_data: Pre-computed marginal ROAS (optional, used for reporting).

    Returns:
        Dict with 'allocation' and 'metrics'.
    
    Raises:
        ValueError: If saturation_params missing required 'max_spend' field.
    """
    def hill_saturation(x, L, k):
        """Wrapper for scalar inputs using imported function."""
        return float(hill_saturation_numpy(np.asarray(x), np.asarray(L), np.asarray(k)))

    for p in saturation_params:
        if p.get("max_spend", 0) <= 0:
            raise ValueError(f"Channel '{p.get('channel', 'unknown')}' missing valid 'max_spend'.")
    
    # 1. Prepare Data - extract L and k from saturation params
    model_data = []
    param_map = {p["channel"]: p for p in saturation_params}

    for _, row in contrib_df.iterrows():
        ch = row["channel"]
        total_spend = row["total_spend"]
        contribution = row["contribution"]
        params = param_map.get(ch, {})

        L = params.get("L_mean", 0.3)
        k = params.get("k_mean", 2.0)
        max_spend = params["max_spend"]

        avg_spend = total_spend / n_obs if n_obs > 0 else total_spend
        avg_spend_norm = avg_spend / max_spend if max_spend > 0 else 0.5
        beta = params.get("beta_mean", 1.0)
        sat_current = hill_saturation(avg_spend_norm, L, k)
        
        is_fixed = total_spend <= 0 or contribution <= 0 or beta <= 0

        model_data.append({
            "channel": ch,
            "beta": beta,
            "L": L,
            "k": k,
            "current_spend": total_spend,
            "avg_spend": avg_spend,
            "max_spend": max_spend,
            "is_fixed": is_fixed,
            "sat_current": sat_current,
        })

    # 2. Objective Function using Multiplicative formulation
    def objective(avg_spends):
        log_multiplier_sum = 0.0
        for i, avg_x in enumerate(avg_spends):
            m = model_data[i]
            if not m["is_fixed"]:
                max_sp = m.get("max_spend", 1)
                avg_x_norm = avg_x / max_sp if max_sp > 0 else avg_x
                sat = hill_saturation(avg_x_norm, m["L"], m["k"])
                log_multiplier_sum += np.log1p(m["beta"] * sat)
            else:
                log_multiplier_sum += np.log1p(m["beta"] * m["sat_current"])
                
        return -log_multiplier_sum

    # 3. Constraints & Bounds
    x0 = [m["avg_spend"] for m in model_data]
    avg_budget = total_budget / n_obs

    A_eq = np.ones((1, len(x0)))
    linear_constraint = LinearConstraint(A_eq, [avg_budget], [avg_budget])

    bounds_list = []
    for m in model_data:
        current_avg = m["avg_spend"]
        if m.get("is_fixed", False):
            lb, ub = current_avg, current_avg
        elif current_avg > 0:
            lb = current_avg * budget_bounds_pct[0]
            ub = current_avg * budget_bounds_pct[1]
        else:
            lb, ub = 0, 0
        bounds_list.append((lb, ub))

    # 4. Optimize
    try:
        initial_fun = objective(x0)
        
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=[linear_constraint],
            bounds=bounds_list,
            options={"disp": False, "ftol": 1e-6, "maxiter": 100},
        )
        
        # Accept if result is successful or improves objective
        if result.success or result.fun < initial_fun:
            success = True
            optimized_x = result.x
        else:
            print(f"Warning: Optimization failed ({result.message}). Using current allocation.")
            success = False
            optimized_x = x0

    except Exception as e:
        print(f"Warning: Optimization error: {e}")
        success = False
        optimized_x = x0

    # 5. Format Results and Calculate Lift using Marginal ROAS
    optimized_allocation = []
    
    # Use fallback if optimization failed
    final_spends = optimized_x if success else x0
    optimized_allocation = []
    
    # Build marginal ROAS lookup: {channel: {spend_increase_pct: marginal_roas}}
    mroas_lookup = {}
    if marginal_roas_data:
        for item in marginal_roas_data:
            ch = item["channel"]
            pct = item["spend_increase_pct"]
            mroas = item["marginal_roas"]
            if ch not in mroas_lookup:
                mroas_lookup[ch] = {}
            mroas_lookup[ch][pct] = mroas

    # Extract total model predicted revenue for accurate lift scaling
    current_total_contrib = contrib_df["contribution"].sum()
    total_predicted = contrib_df.attrs.get("total_predicted_revenue", current_total_contrib * 2)

    # Compute current and new total log multipliers
    log_curr = 0.0
    log_new = 0.0

    for i, m in enumerate(model_data):
        opt_avg = final_spends[i]
        opt_total = opt_avg * n_obs
        current_total = m["current_spend"]
        delta_spend = opt_total - current_total

        change_pct = (
            (delta_spend / current_total * 100) if current_total > 0 else 0.0
        )
        
        max_sp = m.get("max_spend", 1)
        L, k = m["L"], m["k"]
        
        x_curr_norm = m["avg_spend"] / max_sp if max_sp > 0 else 0.5
        x_opt_norm = opt_avg / max_sp if max_sp > 0 else 0.5
        x_opt_norm = min(x_opt_norm, 2.0) # Clamp
        
        s_curr = hill_saturation(x_curr_norm, L, k)
        s_opt = hill_saturation(x_opt_norm, L, k)
        
        log_curr += np.log1p(m["beta"] * s_curr)
        log_new += np.log1p(m["beta"] * s_opt)

        optimized_allocation.append({
            "channel": m["channel"],
            "current_spend": float(current_total),
            "optimal_spend": float(opt_total),
            "change_pct": float(change_pct),
        })

    # Multiplicative Lift AC-5: R_new = R_old * (prod_new / prod_curr)
    # Ratio of products = exp(log_new - log_curr)
    multiplier_ratio = np.exp(log_new - log_curr)
    optimal_total_revenue = total_predicted * multiplier_ratio
    lift_absolute = optimal_total_revenue - total_predicted

    optimal_total_contrib = current_total_contrib + lift_absolute
    lift_pct = (
        (lift_absolute / current_total_contrib * 100) if current_total_contrib > 0 else 0.0
    )

    return {
        "allocation": optimized_allocation,
        "metrics": {
            "current_contribution": float(current_total_contrib),
            "projected_contribution": float(optimal_total_contrib),
            "lift_absolute": float(lift_absolute),
            "lift_pct": float(lift_pct),
        },
    }


def optimize_budget_by_territory(
    contrib_territory_df: pd.DataFrame,
    saturation_params: list[dict],
    territory: str,
    budget_bounds_pct: tuple[float, float] = (0.70, 1.30),
) -> dict:
    """
    Optimize budget allocation for a SINGLE TERRITORY.
    
    Uses exact Hill saturation difference for lift calculation.
    
    Raises:
        ValueError: If n_obs column missing or invalid.
    """
    def hill_saturation(x, L, k):
        """Wrapper for scalar inputs using imported function."""
        return float(hill_saturation_numpy(np.asarray(x), np.asarray(L), np.asarray(k)))

    if contrib_territory_df.empty or not saturation_params:
        return {"allocation": [], "metrics": {"territory": territory, "success": False}}

    # FIX #5: Validate n_obs is present and valid
    if "n_obs" not in contrib_territory_df.columns:
        raise ValueError(
            f"Territory '{territory}' contrib_df missing 'n_obs' column. "
            "Ensure compute_channel_contributions_by_territory() output is used."
        )
    n_obs = int(contrib_territory_df["n_obs"].iloc[0])
    if n_obs <= 0:
        raise ValueError(f"Territory '{territory}' has n_obs={n_obs}, must be > 0")
    
    param_map = {p["channel"]: p for p in saturation_params}

    model_data = []
    for _, row in contrib_territory_df.iterrows():
        ch = row["channel"]
        total_spend = row["total_spend"]
        params = param_map.get(ch, {})

        L = params.get("L_mean", 0.3)
        k = params.get("k_mean", 2.0)
        # Use max_spend from params with fallback to total_spend
        max_spend = params.get("max_spend", total_spend) if params.get("max_spend", 0) > 0 else total_spend
        
        beta = params.get("beta_mean", 1.0)
        
        avg_spend = total_spend / n_obs if n_obs > 0 else total_spend
        avg_spend_norm = avg_spend / max_spend if max_spend > 0 else 0.5

        is_fixed = False
        sat_current = hill_saturation(avg_spend_norm, L, k)

        if total_spend <= 0 or beta <= 0:
            is_fixed = True

        model_data.append({
            "channel": ch,
            "beta": beta,
            "L": L,
            "k": k,
            "current_spend": total_spend,
            "avg_spend": avg_spend,
            "max_spend": max_spend,
            "is_fixed": is_fixed,
            "sat_current": sat_current,
        })

    def objective(avg_spends):
        log_multiplier_sum = 0.0
        for i, avg_x in enumerate(avg_spends):
            m = model_data[i]
            if not m["is_fixed"]:
                max_sp = m.get("max_spend", 1)
                avg_x_norm = avg_x / max_sp if max_sp > 0 else avg_x
                sat = hill_saturation(avg_x_norm, m["L"], m["k"])
                log_multiplier_sum += np.log1p(m["beta"] * sat)
            else:
                log_multiplier_sum += np.log1p(m["beta"] * m["sat_current"])
        return -log_multiplier_sum

    x0 = [m["avg_spend"] for m in model_data]
    avg_budget = sum(x0)

    if avg_budget <= 0:
        return {"allocation": [], "metrics": {"territory": territory, "success": False}}

    A_eq = np.ones((1, len(x0)))
    linear_constraint = LinearConstraint(A_eq, [avg_budget], [avg_budget])

    bounds_list = []
    for m in model_data:
        current_avg = m["avg_spend"]
        if current_avg > 0:
            lb = current_avg * budget_bounds_pct[0]
            ub = current_avg * budget_bounds_pct[1]
        else:
            lb, ub = 0, 0
        bounds_list.append((lb, ub))

    result = minimize(
        objective, x0, method="SLSQP", constraints=[linear_constraint], bounds=bounds_list
    )

    allocation = []
    
    current_total_contrib = contrib_territory_df["contribution"].sum()
    total_predicted = contrib_territory_df.attrs.get("total_predicted_revenue", current_total_contrib * 2)

    log_curr = 0.0
    log_new = 0.0

    for i, m in enumerate(model_data):
        opt_avg = result.x[i] if result.success else m["avg_spend"]
        opt_total = opt_avg * n_obs
        change_pct = (
            ((opt_total - m["current_spend"]) / m["current_spend"] * 100)
            if m["current_spend"] > 0
            else 0.0
        )

        max_sp = m.get("max_spend", 1)
        curr_norm = m["avg_spend"] / max_sp if max_sp > 0 else 0.5
        opt_norm = opt_avg / max_sp if max_sp > 0 else 0.5
        opt_norm = min(opt_norm, 2.0)
        
        s_curr = hill_saturation(curr_norm, m["L"], m["k"])
        s_opt = hill_saturation(opt_norm, m["L"], m["k"])
        
        log_curr += np.log1p(m["beta"] * s_curr)
        log_new += np.log1p(m["beta"] * s_opt)

        allocation.append({
            "territory": territory,
            "channel": m["channel"],
            "current_spend": float(m["current_spend"]),
            "optimal_spend": float(opt_total),
            "change_pct": float(change_pct),
        })

    multiplier_ratio = np.exp(log_new - log_curr)
    optimal_total_revenue = total_predicted * multiplier_ratio
    lift_absolute = optimal_total_revenue - total_predicted
    
    optimal_total_contrib = current_total_contrib + lift_absolute

    lift_pct = (
        ((optimal_total_contrib - current_total_contrib) / current_total_contrib * 100)
        if current_total_contrib > 0
        else 0.0
    )

    return {
        "allocation": allocation,
        "metrics": {
            "territory": territory,
            "success": bool(result.success),
            "current_contribution": float(current_total_contrib),
            "projected_contribution": float(optimal_total_contrib),
            "lift_pct": float(lift_pct),
        },
    }


# -----------------------------------------------------------------------------
# 3b. Visualization
# -----------------------------------------------------------------------------


def plot_regional_comparison(
    roi_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate regional comparison visualizations for hierarchical model."""
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
    """Generate ROI heatmap across channels and regions."""
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


def plot_saturation_curves_hierarchical(
    saturation_params: list[dict],
    output_path: Path,
    spend_range: tuple[float, float] = (0.0, 1.0),
    n_points: int = 100,
) -> None:
    """Plot Hill saturation curves for each channel (Custom Hierarchy)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(spend_range[0], spend_range[1], n_points)

    colors = plt.cm.tab10.colors

    for i, params in enumerate(saturation_params):
        L = params.get("L_mean", 0.3)
        k = params.get("k_mean", 2.0)
        channel = params.get("channel", f"Channel_{i}")

        # Hill saturation: x^k / (L^k + x^k)
        y = (x**k) / (L**k + x**k + 1e-8)

        ax.plot(
            x,
            y,
            label=f"{channel} (L={L:.2f}, k={k:.1f})",
            color=colors[i % len(colors)],
            linewidth=2,
        )

        # Mark half-saturation point
        ax.axvline(x=L, color=colors[i % len(colors)], linestyle="--", alpha=0.3)

    ax.set_xlabel("Normalized Spend (0-1)", fontsize=12)
    ax.set_ylabel("Saturation Effect (0-1)", fontsize=12)
    ax.set_title("Channel Saturation Curves (Hill Function)", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(spend_range)
    ax.set_ylim(0, 1.05)

    # Add annotation
    ax.text(
        0.02,
        0.98,
        "Dashed lines = half-saturation points (L)",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        alpha=0.7,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved saturation curves to {output_path}")


# -----------------------------------------------------------------------------
# 3c. Orchestration & Logging
# -----------------------------------------------------------------------------


def log_diagnostic_artifacts(idata: az.InferenceData) -> dict:
    """Log convergence diagnostics and plots to MLflow."""
    diagnostics = check_convergence(idata)
    mlflow.log_metrics({
        "max_rhat": diagnostics["max_rhat"],
        "min_ess": diagnostics["min_ess"],
        "divergences": diagnostics["divergences"],
    })
    print(f"Max R-hat: {diagnostics['max_rhat']:.3f}")
    print(f"Divergences: {diagnostics['divergences']}")

    summary_df = az.summary(
        idata,
        var_names=[
            "alpha_channel",
            "L_channel",
            "k_channel",
            "beta_channel",
            "tau",
            "sigma_obs",
        ],
    )
    mlflow.log_dict(summary_df.to_dict(), "diagnostics/convergence_summary.json")
    print("Logged convergence summary")

    with tempfile.TemporaryDirectory() as tmpdir:
        axes = az.plot_trace(idata, var_names=["alpha_channel", "L_channel", "beta_channel"])
        trace_path = Path(tmpdir) / "trace_plots.png"
        axes[0, 0].figure.savefig(trace_path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(str(trace_path), "diagnostics")
        print("Logged trace plots")

    with tempfile.TemporaryDirectory() as tmpdir:
        ax = az.plot_energy(idata)
        energy_path = Path(tmpdir) / "energy_plot.png"
        ax.figure.savefig(energy_path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(str(energy_path), "diagnostics")
        print("Logged energy plot")

    return diagnostics


def evaluate_model_splits(
    model: pm.Model,
    idata: az.InferenceData,
    m_data: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model on train and test splits, return metrics and predictions."""
    print("\nEvaluating on training data...")
    with model:
        pm.set_data({
            "X_spend": m_data["X_spend_train"],
            "X_features": m_data["X_features_train"],
            "X_season": m_data["X_season_train"],
            "territory_idx": m_data["territory_idx_train"],
            "y_obs_data": m_data["y_train"],
        })
        y_pred_train_log = predict(model, idata)

    train_metrics = evaluate(m_data["y_train_original"], y_pred_train_log)

    print("Evaluating on holdout...")
    with model:
        pm.set_data({
            "X_spend": m_data["X_spend_test"],
            "X_features": m_data["X_features_test"],
            "X_season": m_data["X_season_test"],
            "territory_idx": m_data["territory_idx_test"],
            "y_obs_data": np.zeros_like(m_data["y_test"]),
        })
        y_pred_log = predict(model, idata)

    test_metrics = evaluate(m_data["y_test_original"], y_pred_log)

    return (
        {
            "r2_train": train_metrics["r2"],
            "mape_train": train_metrics["mape"],
            "r2_test": test_metrics["r2"],
            "mape_test": test_metrics["mape"],
        },
        y_pred_train_log,
        y_pred_log,
    )














# Used by: Hierarchical Bayesian model (marginal ROAS analysis)
def compute_marginal_roas(
    contributions_df: pd.DataFrame,
    saturation_params: list[dict],
    n_obs: int,
    spend_increase_pcts: list[float] | None = None,
) -> list[dict]:
    """
    Compute marginal ROAS at varying spend levels.
    
    Marginal ROAS quantifies the incremental revenue return per additional 
    unit of advertising expenditure. The calculation employs the first 
    derivative of the Hill saturation function.
    
    Mathematical formulation:
        Given R(S) = β × S^k / (L^k + S^k), the marginal response is:
        
        dR/dS = β × k × L^k × S^(k-1) / (L^k + S^k)^2
        
        For normalized spend (S_norm = S_raw / S_max):
        MROAS = dR/dS_norm × (1 / S_max)
    
    Args:
        contributions_df: DataFrame containing 'channel', 'total_spend', 'contribution'.
        saturation_params: List of dicts with 'channel', 'L_mean', 'k_mean', 'max_spend'.
        n_obs: Number of observations (weeks) used to compute average spend.
        spend_increase_pcts: Spend variation percentages to evaluate (default: -30 to +100).
    
    Returns:
        List of dicts with 'channel', 'spend_increase_pct', 'marginal_roas'.
    """
    if spend_increase_pcts is None:
        spend_increase_pcts = [-30, -20, -10, 0, 10, 20, 30, 50, 75, 100]
    
    param_lookup = {p["channel"]: p for p in saturation_params}
    results = []
    
    for _, row in contributions_df.iterrows():
        channel = row["channel"]
        base_spend = row["total_spend"]
        base_contribution = row["contribution"]
        
        params = param_lookup.get(channel)
        if not params:
            continue
        
        L = params["L_mean"]
        k = params["k_mean"]
        max_spend = params.get("max_spend", base_spend)
        
        # Compute avg spend per observation (week)
        avg_spend = base_spend / n_obs if n_obs > 0 else base_spend
        
        # Normalize to [0,1] scale (L and k were calibrated on normalized data)
        x_current = avg_spend / max_spend if max_spend > 0 else 0.5
        
        # Extract full model expected revenue (if available via counterfactual math)
        # to correctly scale the multiplicative derivative into linear dollars.
        # Fallback to base_contribution (legacy behavior) if not found.
        total_predicted = contributions_df.attrs.get("total_predicted_revenue", base_contribution * 2) 
        R_avg = total_predicted / n_obs if n_obs > 0 else total_predicted
        
        # Use true posterior beta (AC-4 requirement)
        beta_mean = params.get("beta_mean", 1.0)
        
        s_current = (x_current ** k) / (L ** k + x_current ** k + 1e-8)
        
        for pct in spend_increase_pcts:
            multiplier = 1 + pct / 100.0
            new_spend = base_spend * multiplier
            
            # Compute new normalized spend
            x_new = x_current * multiplier
            
            # FIX #3: Clamp to avoid extrapolation beyond valid domain
            x_new_clamped = np.clip(x_new, 1e-6, 2.0)  # Allow up to 2x max
            
            # Hill derivative: dS/dx = k × L^k × x^(k-1) / (L^k + x^k)^2
            # FIX #3: Apply epsilon to BASE before power for numerical stability
            x_safe = x_new_clamped + 1e-12
            x_pow_k_minus_1 = np.power(x_safe, k - 1)
            L_pow_k = np.power(L + 1e-12, k)
            x_pow_k = np.power(x_safe, k)
            
            numerator = k * L_pow_k * x_pow_k_minus_1
            denominator = np.power(L_pow_k + x_pow_k, 2) + 1e-8
            dS_dx = numerator / denominator
            
            # Marginal ROAS: dR/dS_raw = R_avg × β × dS/dx × (1/max_spend)
            # This is the exact partial derivative of the expected multiplicative model
            mroas = R_avg * beta_mean * dS_dx / max_spend if max_spend > 0 else 0
            
            # Compute saturation at new spend level
            s_new = x_pow_k / (L_pow_k + x_pow_k + 1e-8)
            
            results.append({
                "channel": channel,
                "spend_increase_pct": pct,
                "current_spend": float(base_spend),
                "new_spend": float(new_spend),
                "marginal_roas": float(mroas),
                "saturation_current": float(s_current),
                "saturation_new": float(s_new),
            })
    
    return results










# =============================================================================
# 3d. CHANNEL EFFICIENCY METRICS (Used by Streamlit App)
# =============================================================================


def compute_channel_metrics(
    contrib_df: pd.DataFrame,
    aov: float,
) -> pd.DataFrame:
    """
    Compute detailed channel efficiency metrics (pure computation).
    
    Uses the Contribution / AOV approach to estimate attributed conversions
    since the model predicts revenue, not conversions.
    
    Args:
        contrib_df: DataFrame with columns: channel, contribution (or contribution_mean), total_spend.
        aov: Global Average Order Value (Revenue / Transactions).
    
    Returns:
        DataFrame with channel efficiency metrics.
    """
    # Normalize column names (handle both naming conventions)
    contrib_col = "contribution_mean" if "contribution_mean" in contrib_df.columns else "contribution"
    spend_col = "total_spend"
    
    metrics = []
    for _, row in contrib_df.iterrows():
        channel = row["channel"]
        spend = row[spend_col]
        contribution = row[contrib_col]
        
        # Derive attributed conversions using AOV
        attributed_conversions = contribution / aov if aov > EPSILON else 0
        
        # Calculate CAC (Cost per Acquisition)
        cac = spend / attributed_conversions if attributed_conversions > EPSILON else 0
        
        # Calculate iROAS
        iroas = contribution / spend if spend > EPSILON else 0
        
        metrics.append({
            "channel": channel,
            "spend": float(spend),
            "revenue_contribution": float(contribution),
            "attributed_conversions": float(attributed_conversions),
            "cac": float(cac),
            "iroas": float(iroas),
        })
    
    return pd.DataFrame(metrics)


def compute_blended_metrics(
    channel_metrics_df: pd.DataFrame,
) -> dict:
    """
    Compute blended (aggregate) efficiency metrics (pure computation).
    
    Args:
        channel_metrics_df: DataFrame from compute_channel_metrics with 
            spend, revenue_contribution, attributed_conversions columns.
    
    Returns:
        Dict with blended CAC and ROAS.
    """
    total_spend = channel_metrics_df["spend"].sum()
    total_contribution = channel_metrics_df["revenue_contribution"].sum()
    total_conversions = channel_metrics_df["attributed_conversions"].sum()
    
    return {
        "total_spend": float(total_spend),
        "total_contribution": float(total_contribution),
        "total_conversions": float(total_conversions),
        "blended_cac": float(total_spend / total_conversions) if total_conversions > EPSILON else 0,
        "blended_roas": float(total_contribution / total_spend) if total_spend > EPSILON else 0,
    }







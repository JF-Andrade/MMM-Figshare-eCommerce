"""
Deliverables generation module.

Generates all dashboard deliverables from trained model artifacts.
Can be called independently of training for quick iteration.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import pandas as pd

from src.config import TARGET_COL, GEO_COL
from src.models.hierarchical_bayesian import (
    compute_channel_contributions,
    compute_roi_with_hdi,
    compute_channel_contributions_by_territory,
)
from src.insights import (
    compute_marginal_roas,
    compute_channel_metrics,
    compute_blended_metrics,
    optimize_hierarchical_budget,
    optimize_budget_by_territory,
    plot_saturation_curves_hierarchical,
)

if TYPE_CHECKING:
    import arviz as az


def _extract_posterior_parameters(
    idata: "az.InferenceData",
    channel_names: list[str],
    regions: list[str],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Extract adstock and saturation parameters from posterior.
    
    Returns:
        Tuple of (adstock_params, saturation_params, 
                  adstock_territory_params, saturation_territory_params)
    """
    posterior = idata.posterior
    
    # Channel-level parameters
    alpha_channel = posterior["alpha_channel"].mean(dim=["chain", "draw"]).values
    L_channel = posterior["L_channel"].mean(dim=["chain", "draw"]).values
    k_channel = posterior["k_channel"].mean(dim=["chain", "draw"]).values
    beta_channel = posterior["beta_channel"].mean(dim=["chain", "draw"]).values
    
    # Territory-level parameters (read directly, not as offsets)
    alpha_territory = posterior["alpha_territory"].mean(dim=["chain", "draw"]).values
    alpha_territory = np.clip(alpha_territory, 0.01, 0.99)
    
    L_territory = posterior["L_territory"].mean(dim=["chain", "draw"]).values
    L_territory = np.clip(L_territory, 0.01, None)
    
    # We add beta_channel_territory as offset
    beta_channel_territory = posterior["beta_channel_territory"].mean(dim=["chain", "draw"]).values
    
    # Build global params
    adstock_params = []
    saturation_params = []
    
    for c_idx, channel in enumerate(channel_names):
        alpha_val = float(alpha_channel[c_idx])
        half_life = np.log(0.5) / np.log(alpha_val) if alpha_val > 0 else 0
        
        adstock_params.append({
            "channel": channel,
            "alpha_mean": alpha_val,
            "half_life_weeks": float(half_life),
        })
        saturation_params.append({
            "channel": channel,
            "L_mean": float(L_channel[c_idx]),
            "k_mean": float(k_channel[c_idx]),
            "beta_mean": float(beta_channel[c_idx]),
        })
    
    # Build territory params
    adstock_territory_params = []
    saturation_territory_params = []
    
    for t_idx, territory in enumerate(regions):
        for c_idx, channel in enumerate(channel_names):
            adstock_territory_params.append({
                "territory": territory,
                "channel": channel,
                "alpha_mean": float(alpha_territory[t_idx, c_idx]),
            })
            
            # The territory beta effect is global beta + territory offset
            beta_terr = float(beta_channel[c_idx] + beta_channel_territory[t_idx, c_idx])
            
            saturation_territory_params.append({
                "territory": territory,
                "channel": channel,
                "L_mean": float(L_territory[t_idx, c_idx]),
                "k_mean": float(k_channel[c_idx]),
                "beta_mean": beta_terr,
            })
    
    return adstock_params, saturation_params, adstock_territory_params, saturation_territory_params


def _compute_contributions(
    idata: "az.InferenceData",
    m_data: dict,
) -> pd.DataFrame:
    """Compute counterfactual channel contributions in linear dollars."""
    contrib_df = compute_channel_contributions(
        idata,
        m_data["X_spend_train"],
        m_data["territory_idx_train"],
        m_data["channel_names"],
        X_features=m_data.get("X_features_train"),
        X_season=m_data.get("X_season_train"),
    )
    
    # Override total_spend with raw spend (not normalized)
    spend_cols_raw = [c + "_SPEND" for c in m_data["channel_names"]]
    for i, raw_col in enumerate(spend_cols_raw):
        if raw_col in m_data["df_train"].columns:
            contrib_df.loc[i, "total_spend"] = m_data["df_train"][raw_col].sum()
    
    # Recalculate ROI with correct scale
    contrib_df["roi"] = contrib_df["contribution"] / (contrib_df["total_spend"] + 1e-8)
    contrib_df["contribution_pct"] = contrib_df["contribution"] / contrib_df["contribution"].sum()
    
    return contrib_df


def _compute_regional_metrics(
    idata: "az.InferenceData",
    m_data: dict,
    regions: list[str],
) -> list[dict]:
    """
    Compute per-region channel metrics using counterfactual contributions.
    """
    # Computes contributions per territory correctly holding all other effects constant
    reg_contrib_df = compute_channel_contributions_by_territory(
        idata,
        m_data["X_spend_train"],
        m_data["territory_idx_train"],
        m_data["channel_names"],
        regions,
        X_features=m_data.get("X_features_train"),
        X_season=m_data.get("X_season_train"),
    )
    
    # Override total_spend with raw spend (not normalized) by territory
    df_train = m_data["df_train"]
    for idx, row in reg_contrib_df.iterrows():
        territory = row["territory"]
        channel = row["channel"]
        raw_col = f"{channel}_SPEND"
        if raw_col in df_train.columns:
            terr_mask = df_train[GEO_COL] == territory
            raw_spend = df_train.loc[terr_mask, raw_col].sum()
            reg_contrib_df.loc[idx, "total_spend"] = float(raw_spend)
            
    # Recalculate ROI and percentages
    reg_contrib_df["roi"] = reg_contrib_df["contribution"] / (reg_contrib_df["total_spend"] + 1e-8)
    
    # Calculate percentages grouped by territory
    reg_contrib_df["contribution_pct"] = reg_contrib_df.groupby("territory")["contribution"].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    
    return reg_contrib_df.to_dict(orient="records")


def generate_all_deliverables(
    idata: "az.InferenceData",
    m_data: dict,
    regions: list[str],
    output_dir: Path | None = None,
    log_to_mlflow: bool = True,
) -> dict:
    """
    Generate all deliverables from trained model.
    
    This function extracts all dashboard-required artifacts from a trained
    hierarchical model. It can be called independently of training to
    regenerate deliverables with updated calculation logic.
    
    Args:
        idata: ArviZ InferenceData from MCMC sampling.
        m_data: Model data dict from prepare_model_data().
        regions: List of territory names.
        output_dir: Directory for saving plots. If None, plots are skipped.
        log_to_mlflow: Whether to log artifacts to MLflow active run.
    
    Returns:
        Dict containing all generated deliverables.
    """
    deliverables = {}
    
    print("\n" + "=" * 60)
    print("GENERATING DELIVERABLES")
    print("=" * 60)
    
    # =========================================================================
    # 1. CONTRIBUTIONS AND ROI
    # =========================================================================
    print("\n[1/8] Computing contributions...")
    contrib_df = _compute_contributions(idata, m_data)
    
    deliverables["contributions"] = contrib_df.to_dict(orient="records")
    deliverables["roi"] = contrib_df[["channel", "roi", "total_spend", "contribution"]].to_dict(orient="records")
    
    # Extract internlas from attrs for simulator
    if "base_effects" in contrib_df.attrs:
        model_internals = {
            "base_effects": contrib_df.attrs["base_effects"].tolist(),
            "beta_eff_matrix": contrib_df.attrs["beta_eff_matrix"].tolist(),
            "total_predicted_revenue": contrib_df.attrs["total_predicted_revenue"],
            "channels": m_data["channel_names"],
        }
        deliverables["model_internals"] = model_internals
    else:
        model_internals = {}

    if log_to_mlflow:
        mlflow.log_dict({"contributions": deliverables["contributions"]}, "deliverables/contributions.json")
        mlflow.log_dict({"roi": deliverables["roi"]}, "deliverables/roi.json")
        if model_internals:
            mlflow.log_dict(model_internals, "deliverables/model_internals.json")
    
    print(f"   Contribution range: {contrib_df['contribution'].min():,.0f} to {contrib_df['contribution'].max():,.0f}")
    
    # =========================================================================
    # 2. REGIONAL METRICS
    # =========================================================================
    print("[2/8] Computing regional metrics...")
    regional_data = _compute_regional_metrics(idata, m_data, regions)  # FIX #1
    deliverables["regional"] = regional_data
    
    if log_to_mlflow:
        mlflow.log_dict({"regional": regional_data}, "deliverables/regional.json")
    
    print(f"   Computed for {len(regions)} regions")
    
    # =========================================================================
    # 3. ADSTOCK AND SATURATION PARAMETERS
    # =========================================================================
    print("[3/8] Extracting model parameters...")
    adstock_params, saturation_params, adstock_territory_params, saturation_territory_params = \
        _extract_posterior_parameters(idata, m_data["channel_names"], regions)
    
    # Add max_spend to saturation params for marginal ROAS calculation
    # Use pre-computed max from training for consistency with normalization
    channel_max_spend = {}  # Store for territory params
    for i, param in enumerate(saturation_params):
        channel = param["channel"]
        raw_col = f"{channel}_SPEND"
        
        # Prefer spend_max from training (consistent with L/k calibration)
        if "spend_max_by_channel" in m_data and channel in m_data["spend_max_by_channel"]:
            max_spend = float(m_data["spend_max_by_channel"][channel])
        elif raw_col in m_data["df_train"].columns:
            # Fallback for backward compatibility
            max_spend = float(m_data["df_train"][raw_col].max())
        else:
            max_spend = float(contrib_df.loc[contrib_df["channel"] == channel, "total_spend"].iloc[0])
        
        param["max_spend"] = max_spend
        channel_max_spend[channel] = max_spend
    
    # FIX #8: Add max_spend to territory params using global channel max
    # For territory optimization, we use the global max (consistent with training normalization)
    for tp in saturation_territory_params:
        channel = tp["channel"]
        tp["max_spend"] = channel_max_spend.get(channel, 1.0)
    
    deliverables["adstock"] = adstock_params
    deliverables["saturation"] = saturation_params
    deliverables["adstock_territory"] = adstock_territory_params
    deliverables["saturation_territory"] = saturation_territory_params
    
    if log_to_mlflow:
        mlflow.log_dict({"adstock": adstock_params}, "deliverables/adstock.json")
        mlflow.log_dict({"saturation": saturation_params}, "deliverables/saturation.json")
        mlflow.log_dict({"adstock_territory": adstock_territory_params}, "deliverables/adstock_territory.json")
        mlflow.log_dict({"saturation_territory": saturation_territory_params}, "deliverables/saturation_territory.json")
    
    print(f"   Extracted parameters for {len(m_data['channel_names'])} channels x {len(regions)} territories")
    
    # =========================================================================
    # 4. MARGINAL ROAS
    # =========================================================================
    print("[4/8] Computing marginal ROAS...")
    n_obs_train = len(m_data["X_spend_train"])
    marginal_roas_data = compute_marginal_roas(contrib_df, saturation_params, n_obs=n_obs_train)
    deliverables["marginal_roas"] = marginal_roas_data
    
    if log_to_mlflow:
        mlflow.log_dict({"marginal_roas": marginal_roas_data}, "deliverables/marginal_roas.json")
    
    print(f"   Computed for {len(set(d['channel'] for d in marginal_roas_data))} channels")
    
    # =========================================================================
    # 5. CHANNEL EFFICIENCY METRICS
    # =========================================================================
    print("[5/8] Computing channel efficiency metrics...")
    
    # Calculate AOV from training data
    transaction_col = [c for c in m_data["df_train"].columns if "transaction" in c.lower()]
    total_revenue = m_data["df_train"][TARGET_COL].sum()
    total_transactions = m_data["df_train"][transaction_col[0]].sum() if transaction_col else 1
    aov = total_revenue / total_transactions if total_transactions > 0 else 100
    
    channel_metrics_df = compute_channel_metrics(contrib_df, aov=aov)
    blended = compute_blended_metrics(channel_metrics_df)
    
    deliverables["channel_metrics"] = channel_metrics_df.to_dict(orient="records")
    deliverables["blended_metrics"] = blended
    
    if log_to_mlflow:
        mlflow.log_dict({"channel_metrics": deliverables["channel_metrics"]}, "deliverables/channel_metrics.json")
        mlflow.log_dict(blended, "deliverables/blended_metrics.json")
    
    print(f"   Blended ROAS: {blended.get('blended_roas', 0):.2f}x")
    
    # =========================================================================
    # 6. ROI HDI (PROBABILISTIC)
    # =========================================================================
    print("[6/8] Computing ROI HDI...")
    roi_hdi_df = compute_roi_with_hdi(
        idata=idata,
        X_spend=m_data["X_spend_train"],
        territory_idx=m_data["territory_idx_train"],
        channel_names=m_data["channel_names"],
        hdi_prob=0.94,
    )
    deliverables["roi_hdi"] = roi_hdi_df.to_dict(orient="records")
    
    if log_to_mlflow:
        mlflow.log_dict({"roi_hdi": deliverables["roi_hdi"]}, "deliverables/roi_hdi.json")
    
    print(f"   ROI HDI computed for {len(roi_hdi_df)} channels")
    
    # =========================================================================
    # 7. BUDGET OPTIMIZATION (GLOBAL)
    # =========================================================================
    print("[7/8] Computing budget optimization...")
    total_budget_current = contrib_df["total_spend"].sum()
    n_obs_train = len(m_data["X_spend_train"])
    
    optimization_result = optimize_hierarchical_budget(
        contrib_df=contrib_df,
        saturation_params=saturation_params,
        total_budget=total_budget_current,
        n_obs=n_obs_train,
        budget_bounds_pct=(0.70, 1.30),
        marginal_roas_data=marginal_roas_data,
    )
    
    deliverables["optimization"] = optimization_result["allocation"]
    deliverables["revenue_lift"] = optimization_result["metrics"]
    
    if log_to_mlflow:
        mlflow.log_dict({"optimization": deliverables["optimization"]}, "deliverables/optimization.json")
        mlflow.log_dict({"revenue_lift": deliverables["revenue_lift"]}, "deliverables/revenue_lift.json")
    
    lift_pct = optimization_result["metrics"].get("lift_pct", 0)
    print(f"   Projected lift: {lift_pct:.1f}%")
    
    # =========================================================================
    # 8. BUDGET OPTIMIZATION BY TERRITORY
    # =========================================================================
    print("[8/8] Computing territory-level optimization...")
    
    # Get contributions by territory
    contrib_by_territory_df = compute_channel_contributions_by_territory(
        idata,
        m_data["X_spend_train"],
        m_data["territory_idx_train"],
        m_data["channel_names"],
        regions,
        X_features=m_data.get("X_features_train"),
        X_season=m_data.get("X_season_train"),
    )
    
    # Override total_spend with raw spend per territory
    df_train = m_data["df_train"]
    for idx, row in contrib_by_territory_df.iterrows():
        territory = row["territory"]
        channel = row["channel"]
        raw_col = f"{channel}_SPEND"
        if raw_col in df_train.columns:
            terr_mask = df_train[GEO_COL] == territory
            raw_spend = df_train.loc[terr_mask, raw_col].sum()
            contrib_by_territory_df.loc[idx, "total_spend"] = float(raw_spend)
    
    # Recalculate ROI
    contrib_by_territory_df["roi"] = contrib_by_territory_df["contribution"] / (contrib_by_territory_df["total_spend"] + 1e-8)
    
    deliverables["contributions_territory"] = contrib_by_territory_df.to_dict(orient="records")
    
    if log_to_mlflow:
        mlflow.log_dict(
            {"contributions_territory": deliverables["contributions_territory"]},
            "deliverables/contributions_territory.json"
        )
    
    # Optimize for each territory
    optimization_by_territory = []
    lift_by_territory = []
    
    for territory in regions:
        terr_contrib = contrib_by_territory_df[contrib_by_territory_df["territory"] == territory]
        terr_sat_params = [p for p in saturation_territory_params if p["territory"] == territory]
        
        terr_opt = optimize_budget_by_territory(
            contrib_territory_df=terr_contrib,
            saturation_params=terr_sat_params,
            territory=territory,
        )
        optimization_by_territory.extend(terr_opt["allocation"])
        if terr_opt["metrics"].get("success"):
            lift_by_territory.append(terr_opt["metrics"])
    
    deliverables["optimization_territory"] = optimization_by_territory
    deliverables["lift_by_territory"] = lift_by_territory
    
    if log_to_mlflow:
        mlflow.log_dict({"optimization_territory": optimization_by_territory}, "deliverables/optimization_territory.json")
        mlflow.log_dict({"lift_by_territory": lift_by_territory}, "deliverables/lift_by_territory.json")
    
    print(f"   Optimization completed for {len(lift_by_territory)} territories")
    
    # =========================================================================
    # 9. SATURATION CURVES PLOT (OPTIONAL)
    # =========================================================================
    if output_dir is not None:
        print("\nGenerating saturation curves plot...")
        sat_curves_path = output_dir / "saturation_curves.png"
        plot_saturation_curves_hierarchical(
            saturation_params=saturation_params,
            output_path=sat_curves_path,
        )
        if log_to_mlflow:
            mlflow.log_artifact(str(sat_curves_path), "diagnostics")
    
    print("\n" + "=" * 60)
    print("DELIVERABLES GENERATION COMPLETE")
    print("=" * 60)
    
    return deliverables

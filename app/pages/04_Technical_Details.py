"""
Technical Details page.

Consolidated technical view with tabs for model parameters, comparison, and diagnostics.
For data scientists and advanced users.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config, get_selected_territory
from app.components import adstock_decay_chart, saturation_curves_chart, roi_with_uncertainty_chart
from app.mlflow_loader import get_mlflow_client, get_all_runs, get_run_metrics, get_run_params, load_deliverable
from src.comparison import compare_models, generate_comparison_insight

init_page_config("Technical Details")


def main():
    deliverables = shared_sidebar()
    page_header("Technical Details", "Model parameters, comparison, and diagnostics")

    run_id = st.session_state.get("run_id")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    territory = get_selected_territory()
    context_label = f" ({territory})" if territory else " (Global)"

    # ==========================================================================
    # MODEL SUMMARY KPIs
    # ==========================================================================
    if run_id:
        try:
            metrics = get_run_metrics(run_id)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Test", f"{metrics.get('r2_test', 0):.3f}")
            with col2:
                st.metric("MAPE Test", f"{metrics.get('mape_test', 0):.1f}%")
            with col3:
                st.metric("Max R-hat", f"{metrics.get('max_rhat', 1.0):.3f}")
            with col4:
                st.metric("Divergences", f"{int(metrics.get('divergences', 0))}")
        except Exception:
            pass

    st.markdown("---")

    # ==========================================================================
    # TABBED INTERFACE
    # ==========================================================================
    tab_params, tab_compare, tab_diag = st.tabs(["Parameters", "Model Comparison", "Diagnostics"])

    # ==========================================================================
    # TAB 1: PARAMETERS
    # ==========================================================================
    with tab_params:
        # Adstock Section
        st.subheader(f"Adstock Decay Parameters{context_label}")
        
        if territory:
            adstock_territory = deliverables.get("adstock_territory", [])
            adstock_df = pd.DataFrame(adstock_territory)
            if not adstock_df.empty and "territory" in adstock_df.columns:
                adstock_data = adstock_df[adstock_df["territory"] == territory].to_dict(orient="records")
            else:
                adstock_data = deliverables.get("adstock", [])
        else:
            adstock_data = deliverables.get("adstock", [])

        if adstock_data:
            col1, col2 = st.columns([2, 1])
            with col1:
                adstock_decay_chart(adstock_data)
            with col2:
                st.markdown("**Parameter Values**")
                adstock_df = pd.DataFrame(adstock_data)
                display_cols = [c for c in ["channel", "alpha_mean", "half_life_weeks"] if c in adstock_df.columns]
                st.dataframe(adstock_df[display_cols], use_container_width=True)
        else:
            st.info("Adstock data not available.")

        st.markdown("---")

        # Saturation Section
        st.subheader(f"Saturation Parameters{context_label}")

        if territory:
            sat_territory = deliverables.get("saturation_territory", [])
            sat_df = pd.DataFrame(sat_territory)
            if not sat_df.empty and "territory" in sat_df.columns:
                saturation_data = sat_df[sat_df["territory"] == territory].to_dict(orient="records")
            else:
                saturation_data = deliverables.get("saturation", [])
        else:
            saturation_data = deliverables.get("saturation", [])

        if saturation_data:
            col1, col2 = st.columns([2, 1])
            with col1:
                saturation_curves_chart(saturation_data)
            with col2:
                st.markdown("**Parameter Values**")
                sat_df = pd.DataFrame(saturation_data)
                display_cols = [c for c in ["channel", "L_mean", "k_mean", "max_spend"] if c in sat_df.columns]
                if "max_spend" in sat_df.columns:
                    sat_df["max_spend"] = sat_df["max_spend"].apply(lambda x: f"${x:,.0f}" if x else "N/A")
                st.dataframe(sat_df[display_cols], use_container_width=True)
        else:
            st.info("Saturation data not available.")

        st.markdown("---")

        # Marginal ROAS
        st.subheader("Marginal ROAS Analysis")
        marginal = deliverables.get("marginal_roas")
        if marginal:
            marginal_df = pd.DataFrame(marginal)
            fig = px.line(
                marginal_df,
                x="spend_increase_pct",
                y="marginal_roas",
                color="channel",
                title="Marginal ROAS at Different Spend Levels",
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Marginal ROAS data not available.")

    # ==========================================================================
    # TAB 2: MODEL COMPARISON
    # ==========================================================================
    with tab_compare:
        st.subheader("Baseline vs Hierarchical Comparison")

        try:
            client = get_mlflow_client()
            
            # Load baseline runs
            all_runs = get_all_runs(client)
            baseline_runs = [r for r in all_runs if r.get("model_type") == "ridge_baseline"]
            hier_runs = get_all_runs(client, model_type="hierarchical")

            if not baseline_runs:
                st.warning("No baseline runs found. Run `python scripts/mmm_baseline.py` first.")
            elif not hier_runs:
                st.warning("No hierarchical runs found.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    baseline_options = {r["run_name"]: r["run_id"] for r in baseline_runs}
                    baseline_label = st.selectbox("Baseline Run", list(baseline_options.keys()))
                    baseline_run_id = baseline_options[baseline_label]
                
                with col2:
                    hier_options = {r["run_name"]: r["run_id"] for r in hier_runs}
                    hier_label = st.selectbox("Hierarchical Run", list(hier_options.keys()))
                    hier_run_id = hier_options[hier_label]

                # Load metrics
                baseline_metrics = get_run_metrics(baseline_run_id)
                hier_metrics = get_run_metrics(hier_run_id)

                # Comparison table
                st.markdown("### Metrics Comparison")
                comparison_data = []
                for key in ["r2_test", "mape_test", "r2_train", "mape_train"]:
                    baseline_val = baseline_metrics.get(key, 0)
                    hier_val = hier_metrics.get(key, 0)
                    comparison_data.append({
                        "Metric": key.replace("_", " ").title(),
                        "Baseline": f"{baseline_val:.3f}" if baseline_val else "N/A",
                        "Hierarchical": f"{hier_val:.3f}" if hier_val else "N/A",
                        "Improvement": f"{((hier_val - baseline_val) / baseline_val * 100):+.1f}%" if baseline_val else "N/A",
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

                # Predictions comparison
                st.markdown("### Predictions Comparison")
                hier_predictions = deliverables.get("predictions", [])
                if hier_predictions:
                    pred_df = pd.DataFrame(hier_predictions)
                    if "actual" in pred_df.columns and "predicted" in pred_df.columns:
                        fig = px.scatter(
                            pred_df,
                            x="actual",
                            y="predicted",
                            color="split",
                            title="Actual vs Predicted (Hierarchical)",
                        )
                        fig.add_trace(px.line(x=[0, pred_df["actual"].max()], y=[0, pred_df["actual"].max()]).data[0])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Predictions data not available.")

        except Exception as e:
            st.error(f"Error loading comparison data: {e}")

    # ==========================================================================
    # TAB 3: DIAGNOSTICS
    # ==========================================================================
    with tab_diag:
        st.subheader("Model Diagnostics")

        if run_id:
            try:
                metrics = get_run_metrics(run_id)
                params = get_run_params(run_id)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Convergence Metrics")
                    conv_data = []
                    for key in ["max_rhat", "divergences", "training_time"]:
                        val = metrics.get(key)
                        if val is not None:
                            conv_data.append({"Metric": key, "Value": f"{val:.3f}" if isinstance(val, float) else val})
                    st.dataframe(pd.DataFrame(conv_data), use_container_width=True)

                    # Status indicators
                    max_rhat = metrics.get("max_rhat", 1.0)
                    divergences = metrics.get("divergences", 0)
                    
                    if max_rhat < 1.01 and divergences == 0:
                        st.success("Model converged well")
                    elif max_rhat < 1.05:
                        st.warning("Model converged with minor issues")
                    else:
                        st.error("Model may have convergence issues")

                with col2:
                    st.markdown("### Training Parameters")
                    params_df = pd.DataFrame([
                        {"Parameter": k, "Value": v}
                        for k, v in params.items()
                    ])
                    st.dataframe(params_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading diagnostics: {e}")

        # ROI Uncertainty Chart
        st.markdown("---")
        st.subheader("ROI Uncertainty (HDI)")
        roi_hdi = deliverables.get("roi_hdi")
        if roi_hdi:
            roi_with_uncertainty_chart(roi_hdi)
        else:
            st.info("ROI HDI data not available.")


if __name__ == "__main__":
    main()
else:
    main()

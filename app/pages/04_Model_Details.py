"""
Model Details page.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config
from app.components import adstock_decay_chart, saturation_curves_chart
from app.mlflow_loader import get_run_metrics, get_run_params

init_page_config("Model Details")


def main():
    deliverables = shared_sidebar()
    page_header("Model Details", "Technical parameters and learned coefficients")

    run_id = st.session_state.get("run_id")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    # ==========================================================================
    # GLOBAL TERRITORY SELECTOR (Applies to ALL sections)
    # ==========================================================================
    adstock_territory = deliverables.get("adstock_territory")
    saturation_territory = deliverables.get("saturation_territory")
    
    territories = []
    if saturation_territory:
        sat_terr_df = pd.DataFrame(saturation_territory)
        territories = sorted(sat_terr_df["territory"].unique().tolist())
    elif adstock_territory:
        adstock_terr_df = pd.DataFrame(adstock_territory)
        territories = sorted(adstock_terr_df["territory"].unique().tolist())
    
    # View mode selector at page top
    if territories:
        col_mode, col_terr, col_spacer = st.columns([1, 1, 2])
        with col_mode:
            view_mode = st.radio("View Mode", ["Global", "By Territory"], horizontal=True, key="model_view")
        with col_terr:
            if view_mode == "By Territory":
                selected_terr = st.selectbox("Territory", territories, key="territory_main")
            else:
                selected_terr = None
        st.markdown("---")
    else:
        view_mode = "Global"
        selected_terr = None

    # ==========================================================================
    # MODEL PARAMETERS / METRICS (Always global)
    # ==========================================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Parameters")
        if run_id:
            try:
                params = get_run_params(run_id)
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": v}
                    for k, v in params.items()
                ])
                st.dataframe(params_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load parameters: {e}")

    with col2:
        st.subheader("Model Metrics")
        if run_id:
            try:
                metrics = get_run_metrics(run_id)
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else v}
                    for k, v in metrics.items()
                ])
                st.dataframe(metrics_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load metrics: {e}")

    st.markdown("---")

    # ==========================================================================
    # ADSTOCK DECAY PARAMETERS (Filtered by territory)
    # ==========================================================================
    context_label = f" ({selected_terr})" if selected_terr else " (Global)"
    st.subheader(f"Adstock Decay Parameters{context_label}")

    if view_mode == "By Territory" and adstock_territory and selected_terr:
        adstock_terr_df = pd.DataFrame(adstock_territory)
        adstock_data = adstock_terr_df[adstock_terr_df["territory"] == selected_terr].to_dict(orient="records")
    else:
        adstock_data = deliverables.get("adstock")

    if adstock_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            adstock_decay_chart(adstock_data)

        with col2:
            st.markdown("**Parameter Values**")
            adstock_df = pd.DataFrame(adstock_data)
            display_cols = ["channel", "alpha_mean", "half_life_weeks"]
            display_cols = [c for c in display_cols if c in adstock_df.columns]
            st.dataframe(adstock_df[display_cols], use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - **alpha**: Decay rate (0-1). Higher = longer carryover.
            - **half_life**: Weeks until 50% of effect remains.
            """)
    else:
        st.warning("Adstock data not available.")

    st.markdown("---")

    # ==========================================================================
    # SATURATION PARAMETERS (Filtered by territory)
    # ==========================================================================
    st.subheader(f"Saturation Parameters{context_label}")

    if view_mode == "By Territory" and saturation_territory and selected_terr:
        sat_terr_df = pd.DataFrame(saturation_territory)
        saturation_data = sat_terr_df[sat_terr_df["territory"] == selected_terr].to_dict(orient="records")
    else:
        saturation_data = deliverables.get("saturation")

    if saturation_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            saturation_curves_chart(saturation_data)

        with col2:
            st.markdown("**Parameter Values**")
            sat_df = pd.DataFrame(saturation_data)
            
            # Include max_spend if available
            display_cols = ["channel", "L_mean", "k_mean", "max_spend"]
            display_cols = [c for c in display_cols if c in sat_df.columns]
            
            # Format max_spend as currency
            if "max_spend" in sat_df.columns:
                sat_df["max_spend"] = sat_df["max_spend"].apply(
                    lambda x: f"${x:,.0f}" if x else "N/A"
                )
            
            st.dataframe(sat_df[display_cols], use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - **L**: Half-saturation point (normalized 0-1).
            - **k**: Steepness. Higher = faster saturation.
            - **max_spend**: Maximum weekly spend in training data.
            
            *To convert: Absolute Spend = Normalized × Max Spend*
            """)
    else:
        st.warning("Saturation data not available.")

    st.markdown("---")

    # ==========================================================================
    # MARGINAL ROAS (Global only - no territory-level data yet)
    # ==========================================================================
    st.subheader("Marginal ROAS Analysis (Global)")

    marginal = deliverables.get("marginal_roas")
    if marginal:
        import plotly.express as px

        marginal_df = pd.DataFrame(marginal)

        fig = px.line(
            marginal_df,
            x="spend_increase_pct",
            y="marginal_roas",
            color="channel",
            title="Marginal ROAS at Different Spend Levels",
            labels={
                "spend_increase_pct": "Spend Increase (%)",
                "marginal_roas": "Marginal ROAS",
            },
        )

        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "**Marginal ROAS data not available.**\n\n"
            "This analysis requires `marginal_roas.json` to be generated by the pipeline. "
            "Re-run the pipeline to generate this artifact."
        )


if __name__ == "__main__":
    main()
else:
    main()

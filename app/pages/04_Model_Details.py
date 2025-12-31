"""
Model Details page.
"""

import sys
from pathlib import Path

import streamlit as st

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

    # Model parameters from MLflow
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Parameters")
        if run_id:
            try:
                params = get_run_params(run_id)
                import pandas as pd
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": v}
                    for k, v in params.items()
                ])
                st.dataframe(params_df, width="stretch")
            except Exception as e:
                st.warning(f"Could not load parameters: {e}")

    with col2:
        st.subheader("Model Metrics")
        if run_id:
            try:
                metrics = get_run_metrics(run_id)
                import pandas as pd
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else v}
                    for k, v in metrics.items()
                ])
                st.dataframe(metrics_df, width="stretch")
            except Exception as e:
                st.warning(f"Could not load metrics: {e}")

    st.markdown("---")

    # Adstock parameters
    st.subheader("Adstock Decay Parameters")

    adstock = deliverables.get("adstock")
    if adstock:
        col1, col2 = st.columns([2, 1])

        with col1:
            adstock_decay_chart(adstock)

        with col2:
            st.markdown("**Parameter Values**")
            import pandas as pd
            adstock_df = pd.DataFrame(adstock)
            display_cols = ["channel", "alpha_mean", "half_life_weeks"]
            display_cols = [c for c in display_cols if c in adstock_df.columns]
            st.dataframe(adstock_df[display_cols], width="stretch")

            st.markdown("""
            **Interpretation:**
            - **alpha**: Decay rate (0-1). Higher = longer carryover.
            - **half_life**: Weeks until 50% of effect remains.
            """)
    else:
        st.warning("Adstock data not available.")

    st.markdown("---")

    # Saturation parameters
    st.subheader("Saturation Parameters")

    saturation = deliverables.get("saturation")
    if saturation:
        col1, col2 = st.columns([2, 1])

        with col1:
            saturation_curves_chart(saturation)

        with col2:
            st.markdown("**Parameter Values**")
            import pandas as pd
            sat_df = pd.DataFrame(saturation)
            display_cols = ["channel", "lam_mean"]
            display_cols = [c for c in display_cols if c in sat_df.columns]
            st.dataframe(sat_df[display_cols], width="stretch")

            st.markdown("""
            **Interpretation:**
            - **lambda**: Saturation steepness. Higher = faster saturation.
            - Curve shows diminishing returns as spend increases.
            """)
    else:
        st.warning("Saturation data not available.")

    st.markdown("---")

    # Marginal ROAS
    st.subheader("Marginal ROAS Analysis")

    marginal = deliverables.get("marginal_roas")
    if marginal:
        import pandas as pd
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

        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Marginal ROAS data not available.")


if __name__ == "__main__":
    main()
else:
    main()

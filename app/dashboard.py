"""
MMM Results Dashboard.

Streamlit dashboard for visualizing Marketing Mix Model results.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.mlflow_loader import (
    get_all_runs,
    get_latest_hierarchical_run,
    get_mlflow_client,
    load_all_deliverables,
)

# Page configuration
st.set_page_config(
    page_title="MMM Results Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "deliverables" not in st.session_state:
        st.session_state.deliverables = None


def sidebar():
    """Render sidebar with run selection."""
    st.sidebar.title("MMM Dashboard")
    st.sidebar.markdown("---")

    # Model run selection
    st.sidebar.subheader("Select Model Run")

    try:
        client = get_mlflow_client()
        runs = get_all_runs(client, model_type="hierarchical")

        if not runs:
            st.sidebar.warning("No hierarchical runs found.")
            return False

        # Create run options
        run_options = {
            f"{r['run_name']} (R2: {r['r2_test']:.3f})" if r['r2_test'] else r['run_name']: r['run_id']
            for r in runs
        }

        selected_label = st.sidebar.selectbox(
            "Model Run",
            options=list(run_options.keys()),
            index=0,
        )

        selected_run_id = run_options[selected_label]

        # Load deliverables if run changed
        if st.session_state.run_id != selected_run_id:
            st.session_state.run_id = selected_run_id
            with st.spinner("Loading deliverables..."):
                st.session_state.deliverables = load_all_deliverables(selected_run_id, client)

        # Show run info
        run_info = next(r for r in runs if r['run_id'] == selected_run_id)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Run Details**")
        st.sidebar.text(f"Type: {run_info['model_type']}")
        st.sidebar.text(f"Status: {run_info['status']}")

        return True

    except Exception as e:
        st.sidebar.error(f"Error loading runs: {e}")
        return False


def main_content():
    """Render main content area."""
    st.title("Marketing Mix Model Results")
    st.markdown("---")

    if st.session_state.deliverables is None:
        st.info("Select a model run from the sidebar to view results.")
        return

    deliverables = st.session_state.deliverables

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    # Revenue Lift
    lift = deliverables.get("revenue_lift")
    if lift:
        col1.metric(
            label="Revenue Lift",
            value=f"{lift['lift_pct']:.1f}%",
            delta=f"{lift['lift_absolute']:,.0f}",
        )

    # Best ROI Channel
    roi_data = deliverables.get("roi")
    if roi_data:
        best_roi = max(roi_data, key=lambda x: x.get("roi", 0))
        col2.metric(
            label="Best ROI Channel",
            value=best_roi["channel"],
            delta=f"{best_roi['roi']:.2f}x",
        )

    # Regions
    regional = deliverables.get("regional")
    if regional:
        col3.metric(
            label="Regions Analyzed",
            value=len(regional),
        )

    # Channels
    contributions = deliverables.get("contributions")
    if contributions:
        col4.metric(
            label="Channels",
            value=len(contributions),
        )

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "ROI Analysis",
        "Budget Optimization",
        "Regional Performance",
        "Model Parameters",
    ])

    with tab1:
        render_roi_tab(deliverables)

    with tab2:
        render_optimization_tab(deliverables)

    with tab3:
        render_regional_tab(deliverables)

    with tab4:
        render_params_tab(deliverables)


def render_roi_tab(deliverables: dict):
    """Render ROI analysis tab."""
    st.subheader("Channel ROI Analysis")

    roi_data = deliverables.get("roi")
    contributions = deliverables.get("contributions")

    if not roi_data:
        st.warning("ROI data not available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ROI by Channel**")
        import pandas as pd
        roi_df = pd.DataFrame(roi_data)
        if "region" in roi_df.columns:
            avg_roi = roi_df.groupby("channel")["roi"].mean().sort_values(ascending=False)
        else:
            avg_roi = roi_df.set_index("channel")["roi"].sort_values(ascending=False)
        st.bar_chart(avg_roi)

    with col2:
        if contributions:
            st.markdown("**Channel Contribution Share**")
            contrib_df = pd.DataFrame(contributions)
            contrib_df = contrib_df.set_index("channel")
            st.bar_chart(contrib_df["contribution_pct"])


def render_optimization_tab(deliverables: dict):
    """Render budget optimization tab."""
    st.subheader("Budget Optimization")

    optimization = deliverables.get("optimization")
    lift = deliverables.get("revenue_lift")

    if not optimization:
        st.warning("Optimization data not available.")
        return

    import pandas as pd
    opt_df = pd.DataFrame(optimization)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current vs Optimal Allocation**")
        chart_data = opt_df.set_index("channel")[["current_spend", "optimal_spend"]]
        st.bar_chart(chart_data)

    with col2:
        st.markdown("**Reallocation Recommendations**")
        for _, row in opt_df.iterrows():
            change = row["change_pct"]
            icon = "arrow_up" if change > 0 else "arrow_down" if change < 0 else "arrow_right"
            color = "green" if change > 0 else "red" if change < 0 else "gray"
            st.markdown(f"**{row['channel']}**: {change:+.1f}%")

    if lift:
        st.markdown("---")
        st.markdown("**Projected Impact**")
        st.metric(
            label="Revenue Lift from Optimal Allocation",
            value=f"{lift['lift_pct']:.1f}%",
            delta=f"{lift['lift_absolute']:,.0f}",
        )


def render_regional_tab(deliverables: dict):
    """Render regional performance tab."""
    st.subheader("Regional Performance")

    regional = deliverables.get("regional")
    roi_data = deliverables.get("roi")

    if not regional:
        st.warning("Regional data not available.")
        return

    import pandas as pd

    # Regional summary table
    st.markdown("**Regional Summary**")
    regional_df = pd.DataFrame(regional)
    st.dataframe(regional_df, use_container_width=True)

    # Heatmap data
    if roi_data:
        st.markdown("---")
        st.markdown("**ROI Heatmap (Channel x Region)**")
        roi_df = pd.DataFrame(roi_data)
        if "region" in roi_df.columns:
            pivot = roi_df.pivot(index="channel", columns="region", values="roi")
            st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=5))


def render_params_tab(deliverables: dict):
    """Render model parameters tab."""
    st.subheader("Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Adstock Decay Parameters**")
        adstock = deliverables.get("adstock")
        if adstock:
            import pandas as pd
            adstock_df = pd.DataFrame(adstock)
            st.dataframe(adstock_df, use_container_width=True)
        else:
            st.warning("Adstock data not available.")

    with col2:
        st.markdown("**Saturation Parameters**")
        saturation = deliverables.get("saturation")
        if saturation:
            import pandas as pd
            sat_df = pd.DataFrame(saturation)
            st.dataframe(sat_df, use_container_width=True)
        else:
            st.warning("Saturation data not available.")


def main():
    """Main application entry point."""
    init_session_state()

    if sidebar():
        main_content()
    else:
        st.title("MMM Results Dashboard")
        st.warning("No model runs available. Please run the hierarchical model first.")
        st.markdown("""
        To generate model results, run:
        ```bash
        python scripts/mmm_hierarchical.py
        ```
        """)


if __name__ == "__main__":
    main()

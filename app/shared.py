"""
Shared components for the MMM Dashboard.

Contains reusable UI elements used across all pages.
"""

import streamlit as st

from app.mlflow_loader import (
    get_all_runs,
    get_mlflow_client,
    load_all_deliverables,
)


def shared_sidebar() -> dict | None:
    """
    Render shared sidebar with run selector.
    
    All pages should call this function to maintain consistent
    navigation and run selection across the dashboard.
    
    Returns:
        Loaded deliverables dict, or None if no runs available.
    """
    st.sidebar.title("MMM Dashboard")
    
    try:
        client = get_mlflow_client()
        runs = get_all_runs(client, model_type="hierarchical")
        
        if not runs:
            st.sidebar.warning("No model runs found.")
            st.sidebar.info("Run `python scripts/mmm_hierarchical.py` first.")
            return None
        
        # Run selector
        run_options = {
            f"{r['run_name']} (R²: {r['r2_test']:.3f})": r['run_id'] for r in runs if r['r2_test']
        } or {r['run_name']: r['run_id'] for r in runs}
        
        selected_label = st.sidebar.selectbox(
            "Select Model Run",
            options=list(run_options.keys()),
        )
        
        run_id = run_options[selected_label]
        
        # Load deliverables if run changed
        if st.session_state.get("run_id") != run_id:
            st.session_state.run_id = run_id
            with st.spinner("Loading data..."):
                st.session_state.deliverables = load_all_deliverables(run_id, client)
        
        # Key Metrics (compact view)
        run_info = next(r for r in runs if r['run_id'] == run_id)
        st.sidebar.markdown("---")
        
        col1, col2 = st.sidebar.columns(2)
        r2 = run_info.get('r2_test', 0)
        mape = run_info.get('mape_test', 0)
        
        with col1:
            st.metric("R² Test", f"{r2:.2%}" if r2 else "N/A")
        with col2:
            st.metric("MAPE", f"{mape:.1f}%" if mape else "N/A")
        
        # Advanced Details (collapsible)
        with st.sidebar.expander("Run Details"):
            st.text(f"ID: {run_id[:8]}...")
            st.text(f"Type: {run_info.get('model_type', 'unknown')}")
            st.text(f"Status: {run_info.get('status', 'FINISHED')}")
            if run_info.get('training_time'):
                st.text(f"Training: {run_info['training_time']:.0f}s")
        
        return st.session_state.get("deliverables")
        
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return None


def page_header(title: str, subtitle: str = None) -> None:
    """
    Render consistent page header.
    
    Args:
        title: Page title (without emoji prefix).
        subtitle: Optional subtitle/description.
    """
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def init_page_config(page_title: str) -> None:
    """
    Initialize page configuration with consistent settings.
    
    Args:
        page_title: Title to show in browser tab.
    """
    st.set_page_config(
        page_title=f"MMM | {page_title}",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )

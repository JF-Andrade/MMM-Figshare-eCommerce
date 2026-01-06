"""
MMM Dashboard - Home Page.

Navigation hub for the Streamlit multipage app.
"""

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, init_page_config
from app.mlflow_loader import get_mlflow_client, get_all_runs

init_page_config("Home")


def navigation_card(title: str, description: str, page_path: str, icon: str = "") -> None:
    """Render a navigation card with link to a page."""
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        ">
            <h3 style="margin: 0 0 8px 0; color: #1f2937;">{title}</h3>
            <p style="margin: 0; color: #6b7280; font-size: 0.9em;">{description}</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    deliverables = shared_sidebar()
    
    # Header
    st.title("Marketing Mix Model Dashboard")
    
    # Last run info
    try:
        client = get_mlflow_client()
        runs = get_all_runs(client, model_type="hierarchical")
        if runs:
            latest = runs[0]
            run_date = datetime.fromtimestamp(latest["start_time"] / 1000)
            st.caption(f"Latest model run: {run_date.strftime('%Y-%m-%d %H:%M')} | R²: {latest.get('r2_test', 0):.3f}")
        else:
            st.warning("No model runs found. Run the pipeline first.")
            return
    except Exception:
        st.warning("Could not load model info.")
        return

    st.markdown("---")

    # Navigation Cards
    st.subheader("Dashboard Sections")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Performance Analysis
        
        View channel performance, regional insights, and efficiency metrics.
        
        - KPIs: Revenue Lift, Best Channel, Blended ROAS
        - Channel ROI and Contribution charts
        - Saturation and Anomaly alerts
        - Regional breakdown and heatmap
        """)
        st.page_link("pages/01_Performance_Analysis.py", label="Go to Performance Analysis")

        st.markdown("---")

        st.markdown("""
        ### What-If Simulator
        
        Interactive budget reallocation with real-time projections.
        
        - Drag sliders to adjust channel budgets
        - See projected contribution changes
        - Compare current vs simulated allocation
        """)
        st.page_link("pages/03_What_If_Simulator.py", label="Go to What-If Simulator")

    with col2:
        st.markdown("""
        ### Budget Optimization
        
        Optimal budget allocation recommendations.
        
        - Current vs Optimal contribution
        - Channel reallocation table
        - Actionable insights
        """)
        st.page_link("pages/02_Budget_Optimization.py", label="Go to Budget Optimization")

        st.markdown("---")

        st.markdown("""
        ### Technical Details
        
        Model parameters and diagnostics (for data scientists).
        
        - Adstock and Saturation parameters
        - Model comparison (Baseline vs Hierarchical)
        - Convergence diagnostics
        """)
        st.page_link("pages/04_Technical_Details.py", label="Go to Technical Details")

    st.markdown("---")

    st.markdown("""
    ### Historical Tracking
    
    Track model performance over multiple runs.
    
    - R² and MAPE trends over time
    - ROI evolution by channel
    - Historical benchmarking
    """)
    st.page_link("pages/05_Historical_Tracking.py", label="Go to Historical Tracking")

    st.markdown("---")

    # Quick stats
    if deliverables:
        st.subheader("Quick Stats")
        col1, col2, col3, col4 = st.columns(4)

        lift = deliverables.get("revenue_lift", {})
        roi = deliverables.get("roi", [])
        regional = deliverables.get("regional", [])
        contributions = deliverables.get("contributions", [])

        with col1:
            if lift:
                st.metric("Revenue Lift", f"{lift.get('lift_pct', 0):.1f}%")
            else:
                st.metric("Revenue Lift", "N/A")

        with col2:
            if roi:
                best = max(roi, key=lambda x: x.get("roi", 0))
                st.metric("Best Channel", best.get("channel", "N/A"))
            else:
                st.metric("Best Channel", "N/A")

        with col3:
            if regional:
                regions = set(r.get("region") for r in regional if r.get("region"))
                st.metric("Regions", len(regions))
            else:
                st.metric("Regions", "N/A")

        with col4:
            if contributions:
                st.metric("Channels", len(contributions))
            else:
                st.metric("Channels", "N/A")


if __name__ == "__main__":
    main()
else:
    main()

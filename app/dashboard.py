"""
MMM Dashboard - Home Page.

Simple navigation page that redirects to the main analysis sections.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config

init_page_config("Dashboard")


def main():
    """Main entry point."""
    deliverables = shared_sidebar()
    
    st.title("📊 Marketing Mix Model Dashboard")
    st.markdown("---")
    
    if not deliverables:
        st.warning("No model data available.")
        st.info("Run the hierarchical model first: `python scripts/mmm_hierarchical.py`")
        return
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    lift = deliverables.get("revenue_lift")
    if lift:
        col1.metric(
            "Revenue Lift",
            f"{lift.get('lift_pct', 0):.1f}%",
        )
    
    roi = deliverables.get("roi")
    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        col2.metric(
            "Best Channel",
            best["channel"],
            f"{best['roi']:.2f}x",
        )
    
    regional = deliverables.get("regional")
    if regional:
        # Count unique regions
        regions = set(r.get("region") for r in regional if r.get("region"))
        col3.metric("Regions", len(regions))
    
    contributions = deliverables.get("contributions")
    if contributions:
        col4.metric("Channels", len(contributions))
    
    st.markdown("---")
    
    # Navigation cards
    st.subheader("Explore Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Executive Summary
        Key metrics and insights from your MMM model.
        
        ### 💰 Budget Optimization
        Optimize budget allocation across channels.
        """)
    
    with col2:
        st.markdown("""
        ### 🌍 Regional Analysis
        Compare performance across regions.
        
        ### ⚙️ Model Details
        Technical parameters and diagnostics.
        """)
    
    st.markdown("---")
    st.caption("Use the sidebar to navigate between pages or change the selected model run.")


if __name__ == "__main__":
    main()
else:
    main()

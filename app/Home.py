"""
MMM Dashboard - Home Page.

Entry point for the Streamlit multipage app.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config

init_page_config("Home")


def main():
    deliverables = shared_sidebar()
    page_header("Marketing Mix Model", "Multi-region hierarchical Bayesian MMM analysis")

    if not deliverables:
        st.warning("No model data available.")
        st.info("Run `python scripts/mmm_hierarchical.py` to generate model results.")
        return

    # Quick overview
    st.subheader("Model Overview")
    st.caption("Showing aggregated global metrics across all territories")
    
    col1, col2, col3, col4 = st.columns(4)

    lift = deliverables.get("revenue_lift")
    if lift:
        lift_pct = lift.get('lift_pct', 0)
        lift_abs = lift.get('lift_absolute', 0)
        # Format small percentages properly
        if abs(lift_pct) < 0.01:
            pct_str = f"{lift_pct:.2e}%"
        else:
            pct_str = f"{lift_pct:.1f}%"
        col1.metric("Revenue Lift (Global)", pct_str, delta=f"+{lift_abs:,.0f}" if lift_abs else None)

    roi = deliverables.get("roi")
    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        col2.metric("Best Channel (iROAS)", best["channel"], f"{best['roi']:.2f}x")

    regional = deliverables.get("regional")
    if regional:
        regions = set(r.get("region") for r in regional if r.get("region"))
        col3.metric("Regions", len(regions))

    contributions = deliverables.get("contributions")
    if contributions:
        col4.metric("Channels", len(contributions))

    st.markdown("---")
    st.caption("Use the sidebar to navigate between analysis pages. Select 'Executive Summary' for territory-specific breakdowns.")


if __name__ == "__main__":
    main()
else:
    main()

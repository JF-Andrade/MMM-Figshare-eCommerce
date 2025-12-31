"""
Executive Summary page.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config
from app.components import insight_box, kpi_row, roi_bar_chart, contribution_pie_chart

init_page_config("Executive Summary")


def main():
    deliverables = shared_sidebar()
    page_header("Executive Summary", "Key metrics and insights from your MMM model")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    # KPI Row
    metrics = []

    lift = deliverables.get("revenue_lift")
    if lift:
        metrics.append({
            "label": "Projected Revenue Lift",
            "value": f"{lift['lift_pct']:.1f}%",
            "delta": f"{lift['lift_absolute']:,.0f}",
        })

    roi = deliverables.get("roi")
    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        metrics.append({
            "label": "Best Performing Channel",
            "value": best["channel"],
            "delta": f"ROI: {best['roi']:.2f}x",
        })

    regional = deliverables.get("regional")
    if regional:
        metrics.append({
            "label": "Regions Analyzed",
            "value": len(regional),
        })

    contributions = deliverables.get("contributions")
    if contributions:
        metrics.append({
            "label": "Active Channels",
            "value": len(contributions),
        })

    if metrics:
        kpi_row(metrics)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        if roi:
            roi_bar_chart(roi, title="Channel ROI (Average)")

    with col2:
        if contributions:
            contribution_pie_chart(contributions, title="Revenue Contribution by Channel")

    # Insights
    st.markdown("---")
    st.subheader("Key Insights")

    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        worst = min(roi, key=lambda x: x.get("roi", 0))
        insight_box(
            f"**{best['channel']}** has the highest ROI at {best['roi']:.2f}x. "
            f"Consider reallocating budget from **{worst['channel']}** (ROI: {worst['roi']:.2f}x).",
            insight_type="success",
        )

    if lift and lift["lift_pct"] > 0:
        insight_box(
            f"Optimal budget allocation could increase revenue by **{lift['lift_pct']:.1f}%** "
            f"({lift['lift_absolute']:,.0f} additional revenue).",
            insight_type="info",
        )


if __name__ == "__main__":
    main()
else:
    main()

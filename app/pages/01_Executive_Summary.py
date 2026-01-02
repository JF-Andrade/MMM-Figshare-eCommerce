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

    # Territory selector
    contributions_territory = deliverables.get("contributions_territory")
    
    if contributions_territory:
        import pandas as pd
        contrib_terr_df = pd.DataFrame(contributions_territory)
        territories = sorted(contrib_terr_df["territory"].unique().tolist())
        
        view_mode = st.radio("View Mode", ["Global (All Territories)", "By Territory"], horizontal=True)
        
        if view_mode == "By Territory":
            selected_territory = st.selectbox("Select Territory", territories)
            # Filter contributions and compute ROI for selected territory
            terr_data = contrib_terr_df[contrib_terr_df["territory"] == selected_territory]
            roi = terr_data.to_dict(orient="records")
            contributions = terr_data.to_dict(orient="records")
            
            # Get lift for selected territory
            lift_by_territory = deliverables.get("lift_by_territory", [])
            lift = next((l for l in lift_by_territory if l.get("territory") == selected_territory), None)
            context_label = f" ({selected_territory})"
        else:
            roi = deliverables.get("roi")
            contributions = deliverables.get("contributions")
            lift = deliverables.get("revenue_lift")
            context_label = " (Global)"
    else:
        roi = deliverables.get("roi")
        contributions = deliverables.get("contributions")
        lift = deliverables.get("revenue_lift")
        context_label = ""

    # KPI Row
    metrics = []

    if lift:
        lift_pct = lift.get('lift_pct', 0)
        lift_abs = lift.get('lift_absolute', lift.get('revenue_lift', 0))
        metrics.append({
            "label": f"Projected Revenue Lift{context_label}",
            "value": f"{lift_pct:.1f}%",
            "delta": f"{lift_abs:,.0f}" if lift_abs else None,
        })

    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        metrics.append({
            "label": f"Best Performing Channel{context_label}",
            "value": best["channel"],
            "delta": f"ROI: {best['roi']:.2f}x",
        })

    regional = deliverables.get("regional")
    if regional:
        metrics.append({
            "label": "Regions Analyzed",
            "value": len(set(r.get("region") for r in regional)),
        })

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
            roi_bar_chart(roi, title=f"Channel ROI{context_label}")

    with col2:
        if contributions:
            contribution_pie_chart(contributions, title=f"Revenue Contribution{context_label}")

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

    if lift and lift.get("lift_pct", 0) > 0:
        lift_pct = lift.get("lift_pct", 0)
        lift_abs = lift.get("lift_absolute", 0)
        pct_fmt = ".2f" if lift_pct < 0.1 else ".1f"
        
        # Adjust formatting for small currency values (normalized data or low absolute lift)
        if abs(lift_abs) < 1.0:
            abs_fmt = ",.4f"
        elif abs(lift_abs) < 100:
            abs_fmt = ",.2f"
        else:
            abs_fmt = ",.0f"
        
        insight_box(
            f"Optimal budget allocation could increase revenue by **{lift_pct:{pct_fmt}}%** "
            f"({lift_abs:{abs_fmt}} additional revenue).",
            insight_type="info",
        )


if __name__ == "__main__":
    main()
else:
    main()

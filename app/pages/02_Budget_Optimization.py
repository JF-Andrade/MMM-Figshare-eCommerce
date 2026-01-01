"""
Budget Optimization page.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config
from app.components import (
    insight_box,
    kpi_row,
    optimization_comparison_chart,
    reallocation_chart,
)

init_page_config("Budget Optimization")


def main():
    deliverables = shared_sidebar()
    page_header("Budget Optimization", "Optimize your marketing budget allocation")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    # Territory selector
    optimization_territory = deliverables.get("optimization_territory")
    
    if optimization_territory:
        territories = sorted(set(o["territory"] for o in optimization_territory))
        view_mode = st.radio("View Mode", ["Global", "By Territory"], horizontal=True)
        
        if view_mode == "By Territory":
            selected_territory = st.selectbox("Select Territory", territories)
            optimization = [o for o in optimization_territory if o["territory"] == selected_territory]
            
            # Get lift for selected territory
            lift_by_territory = deliverables.get("lift_by_territory", [])
            lift = next((l for l in lift_by_territory if l.get("territory") == selected_territory), None)
        else:
            optimization = deliverables.get("optimization")
            lift = deliverables.get("revenue_lift")
    else:
        optimization = deliverables.get("optimization")
        lift = deliverables.get("revenue_lift")

    if not optimization:
        st.warning("Optimization data not available.")
        return

    # KPI Row
    metrics = []

    if lift:
        # Handle both old format (current_revenue) and new format (current_contribution)
        current = lift.get('current_revenue') or lift.get('current_contribution', 0)
        projected = lift.get('optimal_revenue') or lift.get('projected_contribution', 0)
        lift_pct = lift.get('lift_pct', 0)
        lift_abs = lift.get('lift_absolute', 0)
        
        metrics.append({
            "label": "Current Contribution",
            "value": f"{current:,.0f}",
        })
        metrics.append({
            "label": "Projected Optimal Contribution",
            "value": f"{projected:,.0f}",
            "delta": f"+{lift_pct:.1f}%",
        })
        metrics.append({
            "label": "Contribution Lift",
            "value": f"{lift_abs:,.0f}",
            "delta": f"{lift_pct:.1f}%",
        })

    if metrics:
        kpi_row(metrics)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        optimization_comparison_chart(optimization)

    with col2:
        reallocation_chart(optimization)

    # Recommendations table
    st.markdown("---")
    st.subheader("Reallocation Recommendations")

    import pandas as pd
    opt_df = pd.DataFrame(optimization)

    # Color code the change
    def color_change(val):
        if val > 0:
            return "background-color: rgba(0, 255, 0, 0.2)"
        elif val < 0:
            return "background-color: rgba(255, 0, 0, 0.2)"
        return ""

    styled_df = opt_df.style.applymap(color_change, subset=["change_pct"])
    st.dataframe(styled_df, use_container_width=True)

    # Insights
    st.markdown("---")

    increases = [o for o in optimization if o["change_pct"] > 0]
    decreases = [o for o in optimization if o["change_pct"] < 0]

    if increases:
        top_increase = max(increases, key=lambda x: x["change_pct"])
        insight_box(
            f"**Increase** budget for **{top_increase['channel']}** by {top_increase['change_pct']:.1f}% "
            f"(from {top_increase['current_spend']:,.0f} to {top_increase['optimal_spend']:,.0f}).",
            insight_type="success",
        )

    if decreases:
        top_decrease = min(decreases, key=lambda x: x["change_pct"])
        insight_box(
            f"**Reduce** budget for **{top_decrease['channel']}** by {abs(top_decrease['change_pct']):.1f}% "
            f"(from {top_decrease['current_spend']:,.0f} to {top_decrease['optimal_spend']:,.0f}).",
            insight_type="warning",
        )


if __name__ == "__main__":
    main()
else:
    main()

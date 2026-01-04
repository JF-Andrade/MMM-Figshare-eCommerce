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
        # Calculate lift_abs if not provided (territory metrics don't include it)
        lift_abs = lift.get('lift_absolute') or (projected - current)
        
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
    
    # Normalize column names (schema differs between global and territory)
    pct_col = "delta_spend_pct" if "delta_spend_pct" in opt_df.columns else "change_pct"
    opt_col = "optimized_spend" if "optimized_spend" in opt_df.columns else "optimal_spend"

    # Color code the change
    def color_change(val):
        if val > 0:
            return "background-color: rgba(0, 255, 0, 0.2)"
        elif val < 0:
            return "background-color: rgba(255, 0, 0, 0.2)"
        return ""

    styled_df = opt_df.style.applymap(color_change, subset=[pct_col])
    st.dataframe(styled_df, use_container_width=True)

    # Insights
    st.markdown("---")

    increases = [o for o in optimization if o.get(pct_col, o.get("change_pct", 0)) > 0]
    decreases = [o for o in optimization if o.get(pct_col, o.get("change_pct", 0)) < 0]
    
    # Detect format: delta_spend_pct is decimal (0.30 = 30%), change_pct is percentage (30 = 30%)
    is_decimal_format = pct_col == "delta_spend_pct"

    if increases:
        top_increase = max(increases, key=lambda x: x.get(pct_col, x.get("change_pct", 0)))
        pct_val = top_increase.get(pct_col, top_increase.get("change_pct", 0))
        opt_val = top_increase.get(opt_col, top_increase.get("optimal_spend", 0))
        pct_display = pct_val * 100 if is_decimal_format else pct_val
        insight_box(
            f"**Increase** budget for **{top_increase['channel']}** by {pct_display:.1f}% "
            f"(from {top_increase['current_spend']:,.0f} to {opt_val:,.0f}).",
            insight_type="success",
        )

    if decreases:
        top_decrease = min(decreases, key=lambda x: x.get(pct_col, x.get("change_pct", 0)))
        pct_val = top_decrease.get(pct_col, top_decrease.get("change_pct", 0))
        opt_val = top_decrease.get(opt_col, top_decrease.get("optimal_spend", 0))
        pct_display = abs(pct_val) * 100 if is_decimal_format else abs(pct_val)
        insight_box(
            f"**Reduce** budget for **{top_decrease['channel']}** by {pct_display:.1f}% "
            f"(from {top_decrease['current_spend']:,.0f} to {opt_val:,.0f}).",
            insight_type="warning",
        )


if __name__ == "__main__":
    main()
else:
    main()

"""
Budget Optimization page.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.components import (
    insight_box,
    kpi_row,
    optimization_comparison_chart,
    reallocation_chart,
)
from app.mlflow_loader import get_mlflow_client, get_all_runs, load_all_deliverables

st.set_page_config(page_title="Budget Optimization", layout="wide")


def get_deliverables():
    """Get deliverables from session state or load them."""
    if "deliverables" not in st.session_state or st.session_state.deliverables is None:
        try:
            client = get_mlflow_client()
            runs = get_all_runs(client, model_type="hierarchical")
            if runs:
                run_id = runs[0]["run_id"]
                st.session_state.deliverables = load_all_deliverables(run_id, client)
                st.session_state.run_id = run_id
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return st.session_state.get("deliverables")


def main():
    st.title("Budget Optimization")
    st.markdown("---")

    deliverables = get_deliverables()
    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

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
    st.dataframe(styled_df, width="stretch")

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

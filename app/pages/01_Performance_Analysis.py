"""
Performance Analysis page.

Consolidated view of channel performance, regional analysis, and efficiency metrics.
Uses global territory filter from sidebar.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config, get_selected_territory
from app.components import (
    insight_box,
    kpi_row,
    roi_bar_chart,
    contribution_pie_chart,
    roi_heatmap,
    display_saturation_alerts,
    display_roi_anomalies,
    get_excel_download_button,
)

init_page_config("Performance Analysis")


def filter_by_territory(data: list[dict], territory: str | None, key: str = "territory") -> list[dict]:
    """Filter data list by territory. Returns all if territory is None."""
    if not territory or not data:
        return data
    return [d for d in data if d.get(key) == territory]


def main():
    deliverables = shared_sidebar()
    page_header("Performance Analysis", "Channel performance and regional insights")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    # Get global territory selection
    territory = get_selected_territory()
    context_label = f" ({territory})" if territory else " (All Territories)"

    # ==========================================================================
    # DATA PREPARATION (filter by territory if selected)
    # ==========================================================================
    
    if territory:
        # Use territory-specific data
        contributions_territory = deliverables.get("contributions_territory", [])
        contrib_df = pd.DataFrame(contributions_territory)
        
        if not contrib_df.empty and "territory" in contrib_df.columns:
            terr_data = contrib_df[contrib_df["territory"] == territory]
            roi = terr_data.to_dict(orient="records")
            contributions = terr_data.to_dict(orient="records")
        else:
            roi = deliverables.get("roi", [])
            contributions = deliverables.get("contributions", [])
        
        # Get territory-specific lift
        lift_by_territory = deliverables.get("lift_by_territory", [])
        lift = next((l for l in lift_by_territory if l.get("territory") == territory), None) or deliverables.get("revenue_lift")
    else:
        # Use global data
        roi = deliverables.get("roi", [])
        contributions = deliverables.get("contributions", [])
        lift = deliverables.get("revenue_lift")

    # ==========================================================================
    # KPI ROW
    # ==========================================================================
    metrics = []

    if lift:
        lift_pct = lift.get('lift_pct', 0)
        lift_abs = lift.get('lift_absolute', lift.get('revenue_lift', 0))
        metrics.append({
            "label": f"Revenue Lift{context_label}",
            "value": f"{lift_pct:.1f}%",
            "delta": f"+{lift_abs:,.0f}" if lift_abs else None,
        })

    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        metrics.append({
            "label": "Best Channel",
            "value": best.get("channel", "N/A"),
            "delta": f"iROAS: {best.get('roi', 0):.2f}x",
        })

    blended = deliverables.get("blended_metrics", {})
    if blended:
        metrics.append({
            "label": "Blended ROAS",
            "value": f"{blended.get('blended_roas', 0):.2f}x",
        })
        metrics.append({
            "label": "Blended CAC",
            "value": f"{blended.get('blended_cac', 0):.2f}",
        })

    if contributions:
        total_spend = sum(c.get("total_spend", 0) for c in contributions)
        total_contrib = sum(c.get("contribution", 0) for c in contributions)
        metrics.append({"label": "Total Spend", "value": f"{total_spend:,.0f}"})
        metrics.append({"label": "Total Contribution", "value": f"{total_contrib:,.0f}"})

    if metrics:
        kpi_row(metrics)

    # Export button
    get_excel_download_button(deliverables)

    st.markdown("---")

    # ==========================================================================
    # CHANNEL PERFORMANCE CHARTS
    # ==========================================================================
    st.subheader(f"Channel Performance{context_label}")

    col1, col2 = st.columns(2)

    with col1:
        if roi:
            roi_bar_chart(roi, title="Channel iROAS")

    with col2:
        if contributions:
            contribution_pie_chart(contributions, title="Contribution Share")

    st.markdown("---")

    # ==========================================================================
    # ALERTS SECTION
    # ==========================================================================
    col_sat, col_anom = st.columns(2)

    with col_sat:
        saturation = deliverables.get("saturation", [])
        if saturation and contributions:
            display_saturation_alerts(saturation, contributions)

    with col_anom:
        regional = deliverables.get("regional", [])
        if regional:
            regional_df = pd.DataFrame(regional)
            if not regional_df.empty and "region" in regional_df.columns:
                regional_summary = regional_df.groupby("region").agg({
                    "roi": "mean",
                }).reset_index()
                regional_summary.columns = ["region", "avg_iroas"]
                display_roi_anomalies(regional_summary.to_dict(orient="records"))

    st.markdown("---")

    # ==========================================================================
    # REGIONAL SUMMARY (collapsible)
    # ==========================================================================
    with st.expander("Regional Breakdown", expanded=False):
        regional = deliverables.get("regional", [])
        
        if regional:
            regional_df = pd.DataFrame(regional)
            
            if not regional_df.empty and "region" in regional_df.columns:
                # Summary table
                st.subheader("Regional Summary")
                summary = regional_df.groupby("region").agg({
                    "roi": "mean",
                    "total_spend": "sum",
                    "contribution": "sum",
                }).reset_index()
                summary.columns = ["Region", "Avg ROI", "Total Spend", "Total Contrib"]
                summary = summary.sort_values("Avg ROI", ascending=False)
                st.dataframe(summary.style.format({
                    "Avg ROI": "{:.2f}x",
                    "Total Spend": "{:,.0f}",
                    "Total Contrib": "{:,.0f}",
                }), use_container_width=True)

                # Heatmap
                st.subheader("ROI Heatmap (Channel × Region)")
                if "channel" in regional_df.columns:
                    pivot = regional_df.pivot(index="channel", columns="region", values="roi")
                    st.dataframe(
                        pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=5),
                        use_container_width=True
                    )
        else:
            st.info("Regional data not available.")

    st.markdown("---")

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================
    st.subheader("Key Insights")

    if roi:
        best = max(roi, key=lambda x: x.get("roi", 0))
        worst = min(roi, key=lambda x: x.get("roi", 0))
        insight_box(
            f"**{best.get('channel')}** has the highest iROAS at {best.get('roi', 0):.2f}x. "
            f"Consider reallocating budget from **{worst.get('channel')}** (iROAS: {worst.get('roi', 0):.2f}x).",
            insight_type="success",
        )

    if lift and lift.get("lift_pct", 0) > 0:
        insight_box(
            f"Optimal budget allocation could increase revenue by **{lift.get('lift_pct', 0):.1f}%**.",
            insight_type="info",
        )


if __name__ == "__main__":
    main()
else:
    main()

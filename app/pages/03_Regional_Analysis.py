"""
Regional Analysis page.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config
from app.components import insight_box, kpi_row, roi_heatmap

init_page_config("Regional Analysis")


def main():
    deliverables = shared_sidebar()
    page_header("Regional Analysis", "Compare performance across regions")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    regional = deliverables.get("regional")

    if not regional:
        st.warning("Regional data not available.")
        return

    # Regional data is now flat: each row has region + channel info
    regional_df = pd.DataFrame(regional)
    
    if regional_df.empty:
        st.warning("No regional data to display.")
        return

    # Get unique regions
    all_regions = regional_df["region"].unique().tolist()
    selected_regions = st.multiselect(
        "Select Regions",
        options=all_regions,
        default=all_regions[:5] if len(all_regions) > 5 else all_regions,
    )

    if not selected_regions:
        st.warning("Please select at least one region.")
        return

    # Filter data
    filtered_df = regional_df[regional_df["region"].isin(selected_regions)]

    # Compute regional summary
    regional_summary = filtered_df.groupby("region").agg({
        "roi": "mean",
        "total_spend": "sum",
        "contribution": "sum",
    }).reset_index()
    regional_summary.columns = ["region", "avg_iroas", "total_spend", "total_contribution"]
    regional_summary = regional_summary.sort_values("avg_iroas", ascending=False)

    # Find best channel per region
    best_channels = filtered_df.loc[filtered_df.groupby("region")["roi"].idxmax()][["region", "channel"]]
    best_channels.columns = ["region", "best_channel"]
    regional_summary = regional_summary.merge(best_channels, on="region", how="left")

    # KPIs
    if not regional_summary.empty:
        best_region = regional_summary.iloc[0]
        total_spend = regional_summary["total_spend"].sum()

        kpi_row([
            {"label": "Regions Selected", "value": len(selected_regions)},
            {"label": "Best Region", "value": best_region["region"], "delta": f"iROAS: {best_region['avg_iroas']:.2f}x"},
            {"label": "Total Spend", "value": f"{total_spend:,.0f}"},
        ])

    st.markdown("---")

    # Regional summary table
    st.subheader("Regional Summary")
    st.dataframe(regional_summary, use_container_width=True)

    # Heatmap
    st.markdown("---")
    st.subheader("ROI Heatmap (Channel x Region)")

    if "channel" in filtered_df.columns and "roi" in filtered_df.columns:
        pivot = filtered_df.pivot(index="channel", columns="region", values="roi")
        st.dataframe(
            pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=5),
            use_container_width=True
        )
    else:
        st.info("No channel ROI data available for heatmap.")

    # Top performers
    st.markdown("---")
    st.subheader("Top Regional Performers")

    for i, (_, region) in enumerate(regional_summary.head(3).iterrows(), 1):
        rank = ["1st", "2nd", "3rd"][i - 1]
        st.markdown(
            f"**{rank}: {region['region']}** - "
            f"Avg ROI: {region['avg_iroas']:.2f}x, "
            f"Best Channel: {region['best_channel']}"
        )


if __name__ == "__main__":
    main()
else:
    main()

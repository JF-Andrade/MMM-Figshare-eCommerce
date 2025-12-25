"""
Regional Analysis page.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.components import insight_box, kpi_row, roi_heatmap
from app.mlflow_loader import get_mlflow_client, get_all_runs, load_all_deliverables

st.set_page_config(page_title="Regional Analysis", layout="wide")


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
    st.title("Regional Performance Analysis")
    st.markdown("---")

    deliverables = get_deliverables()
    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    regional = deliverables.get("regional")
    roi = deliverables.get("roi")

    if not regional:
        st.warning("Regional data not available.")
        return

    # Region filter
    all_regions = [r["region"] for r in regional]
    selected_regions = st.multiselect(
        "Select Regions",
        options=all_regions,
        default=all_regions[:5] if len(all_regions) > 5 else all_regions,
    )

    if not selected_regions:
        st.warning("Please select at least one region.")
        return

    filtered_regional = [r for r in regional if r["region"] in selected_regions]
    
    # Compute summary metrics from channels data
    for region_data in filtered_regional:
        channels = region_data.get("channels", [])
        if channels:
            # Calculate avg_roi across channels
            rois = [c.get("roi", 0) for c in channels if c.get("roi", 0) > 0]
            region_data["avg_roi"] = sum(rois) / len(rois) if rois else 0
            
            # Calculate total_spend
            region_data["total_spend"] = sum(c.get("total_spend", 0) for c in channels)
            
            # Find best channel by ROI
            if rois:
                best = max(channels, key=lambda c: c.get("roi", 0))
                region_data["best_channel"] = best.get("channel", "N/A")
            else:
                region_data["best_channel"] = "N/A"
        else:
            region_data["avg_roi"] = 0
            region_data["total_spend"] = 0
            region_data["best_channel"] = "N/A"

    # KPIs
    if filtered_regional:
        best_region = max(filtered_regional, key=lambda x: x.get("avg_roi", 0))
        total_spend = sum(r.get("total_spend", 0) for r in filtered_regional)

        kpi_row([
            {"label": "Regions Selected", "value": len(selected_regions)},
            {"label": "Best Region", "value": best_region["region"], "delta": f"ROI: {best_region.get('avg_roi', 0):.2f}x"},
            {"label": "Total Spend", "value": f"{total_spend:,.0f}"},
        ])
    else:
        st.warning("No regional data to display.")
        return

    st.markdown("---")

    # Regional summary table
    st.subheader("Regional Summary")
    import pandas as pd
    
    # Create clean summary dataframe
    summary_data = [{
        "region": r["region"],
        "avg_roi": r.get("avg_roi", 0),
        "total_spend": r.get("total_spend", 0),
        "best_channel": r.get("best_channel", "N/A")
    } for r in filtered_regional]
    
    regional_df = pd.DataFrame(summary_data)
    regional_df = regional_df.sort_values("avg_roi", ascending=False)
    st.dataframe(regional_df, width="stretch")

    # Heatmap
    st.markdown("---")
    st.subheader("ROI Heatmap (Channel x Region)")

    # Build heatmap data from regional channels
    heatmap_data = []
    for region_data in filtered_regional:
        region_name = region_data["region"]
        for channel in region_data.get("channels", []):
            heatmap_data.append({
                "region": region_name,
                "channel": channel.get("channel", "Unknown"),
                "roi": channel.get("roi", 0),
            })
    
    if heatmap_data:
        roi_heatmap(heatmap_data)
    else:
        st.info("No channel ROI data available for heatmap.")


    # Top performers
    st.markdown("---")
    st.subheader("Top Regional Performers")

    sorted_regions = sorted(filtered_regional, key=lambda x: x.get("avg_roi", 0), reverse=True)

    for i, region in enumerate(sorted_regions[:3], 1):
        medal = ["1st", "2nd", "3rd"][i - 1]
        st.markdown(
            f"**{medal}:** {region['region']} - "
            f"Avg ROI: {region.get('avg_roi', 0):.2f}x, "
            f"Best Channel: {region.get('best_channel', 'N/A')}"
        )


if __name__ == "__main__":
    main()
else:
    main()

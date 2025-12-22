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

    # KPIs
    best_region = max(filtered_regional, key=lambda x: x["avg_roi"])
    total_spend = sum(r["total_spend"] for r in filtered_regional)

    kpi_row([
        {"label": "Regions Selected", "value": len(selected_regions)},
        {"label": "Best Region", "value": best_region["region"], "delta": f"ROI: {best_region['avg_roi']:.2f}x"},
        {"label": "Total Spend", "value": f"{total_spend:,.0f}"},
    ])

    st.markdown("---")

    # Regional summary table
    st.subheader("Regional Summary")
    import pandas as pd
    regional_df = pd.DataFrame(filtered_regional)
    regional_df = regional_df.sort_values("avg_roi", ascending=False)
    st.dataframe(regional_df, use_container_width=True)

    # Heatmap
    st.markdown("---")
    st.subheader("ROI Heatmap (Channel x Region)")

    if roi:
        filtered_roi = [r for r in roi if r.get("region") in selected_regions]
        if filtered_roi:
            roi_heatmap(filtered_roi)
        else:
            st.info("Heatmap requires regional ROI data.")

    # Top performers
    st.markdown("---")
    st.subheader("Top Regional Performers")

    sorted_regions = sorted(filtered_regional, key=lambda x: x["avg_roi"], reverse=True)

    for i, region in enumerate(sorted_regions[:3], 1):
        medal = ["1st", "2nd", "3rd"][i - 1]
        st.markdown(
            f"**{medal}:** {region['region']} - "
            f"Avg ROI: {region['avg_roi']:.2f}x, "
            f"Best Channel: {region['best_channel']}"
        )


if __name__ == "__main__":
    main()
else:
    main()

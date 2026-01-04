"""
Channel Efficiency Analysis page.
Focuses on CAC, ROAS, and acquisition efficiency.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config, format_currency
from app.components.charts import roi_with_uncertainty_chart

init_page_config("Channel Efficiency")


def main():
    deliverables = shared_sidebar()
    page_header("Channel Efficiency", "Deep dive into acquisition costs and return on spend")

    if not deliverables:
        st.warning("No model data available. Run the pipeline first.")
        return

    # Load Metrics
    blended_metrics = deliverables.get("blended_metrics")
    channel_metrics = deliverables.get("channel_metrics")
    roi_hdi = deliverables.get("roi_hdi")
    
    if not channel_metrics:
        st.error("Channel metrics not found. Please re-run the pipeline.")
        return

    metrics_df = pd.DataFrame(channel_metrics)

    # 1. High Level Efficiency
    st.subheader("Efficiency Overview")
    
    c1, c2, c3 = st.columns(3)
    
    if blended_metrics:
        c1.metric(
            "Blended CAC",
            format_currency(blended_metrics.get("blended_cac", 0)),
            help="Total Spend / Total New Customers"
        )
        c2.metric(
            "Blended ROAS",
            f"{blended_metrics.get('blended_roas', 0):.2f}x",
            help="Total Revenue / Total Spend (Marketing ROAS)"
        )
        c3.metric(
            "Marketing Spend",
            format_currency(blended_metrics.get("total_spend", 0))
        )
    
    st.markdown("---")

    # 2. Efficiency Matrix (The Star Chart)
    st.subheader("Efficiency Matrix")
    st.caption("Strategic view of channel performance: Cost vs. Return")
    
    # Scatter Plot: X=CAC (Log?), Y=iROAS
    # Quadrants:
    # - Low CAC, High ROAS (Stars)
    # - High CAC, High ROAS (Scale)
    # - Low CAC, Low ROAS (Potential)
    # - High CAC, Low ROAS (Concern)
    
    fig = px.scatter(
        metrics_df,
        x="cac",
        y="iroas",
        size="spend",
        color="channel",
        hover_name="channel",
        hover_data=["revenue_contribution", "attributed_conversions"],
        labels={
            "cac": "CAC (Customer Acquisition Cost)",
            "iroas": "iROAS (Incremental Return)",
            "spend": "Total Spend"
        },
        title="Efficiency Matrix (Size = Spend)",
        template="plotly_white"
    )
    
    # Add mean lines for context
    avg_cac = metrics_df["cac"].mean()
    avg_roas = metrics_df["iroas"].mean()
    
    fig.add_hline(y=avg_roas, line_dash="dot", annotation_text="Avg ROAS", line_color="gray")
    fig.add_vline(x=avg_cac, line_dash="dot", annotation_text="Avg CAC", line_color="gray")
    
    # Inverse X axis? Usually defined as:
    # Best is Top Left (High ROAS, Low CAC) -> standard Cartesian has 0 at left. 
    # So Low CAC is left. Correct.
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(
        "**Interpretation:**\n"
        "- **Top Left:** Efficient drivers (High Return, Low Cost).\n"
        "- **Top Right:** Scalable but expensive.\n"
        "- **Bottom Left:** Low cost but low return (Volume drivers?).\n"
        "- **Bottom Right:** Inefficient (High Cost, Low Return)."
    )
    
    st.markdown("---")

    # 3. ROI Uncertainty (New!)
    st.subheader("ROI Uncertainty Analysis")
    st.caption("94% High Density Interval (HDI) - Shows the probable range of ROI")
    
    if roi_hdi:
        roi_with_uncertainty_chart(roi_hdi)
        st.info(
            "The error bars show the 94% HDI (High Density Interval). "
            "A wide bar indicates higher uncertainty about the channel's true ROI."
        )
    else:
        st.info("Uncertainty data not available. Please re-run the pipeline with HDI computation.")

    st.markdown("---")

    # 4. Detailed Metrics Table
    st.subheader("Detailed Channel Metrics")
    
    # Format table for display
    display_df = metrics_df.copy()
    display_df = display_df[[
        "channel", "spend", "attributed_conversions", "cac", 
        "revenue_contribution", "iroas"
    ]]
    
    display_df.columns = [
        "Channel", "Spend", "Attributed Customers", "CAC", 
        "Revenue Contribution", "iROAS"
    ]
    
    st.dataframe(
        display_df.style.format({
            "Spend": "{:,.2f}",
            "Attributed Customers": "{:,.0f}",
            "CAC": "{:,.2f}",
            "Revenue Contribution": "{:,.2f}",
            "iROAS": "{:.2f}x"
        }),
        use_container_width=True
    )

if __name__ == "__main__":
    main()

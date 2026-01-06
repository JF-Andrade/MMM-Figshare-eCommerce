"""
Historical Tracking page.

Consolidated view of ROI trends and historical benchmarks over time.
Requires multiple pipeline runs to display meaningful data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config
from app.components import kpi_row
from app.mlflow_loader import get_mlflow_client, get_all_runs, load_deliverable

init_page_config("Historical Tracking")


def load_historical_data(n_runs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metrics and ROI data from multiple historical runs.
    
    Returns:
        Tuple of (run_metrics_df, roi_df)
    """
    client = get_mlflow_client()
    runs = get_all_runs(client, model_type="hierarchical")[:n_runs]
    
    metrics_list = []
    roi_list = []
    
    for run in runs:
        run_date = datetime.fromtimestamp(run["start_time"] / 1000)
        
        # Run metrics
        metrics_list.append({
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "run_date": run_date,
            "r2_test": run.get("r2_test", 0),
            "mape_test": run.get("mape_test", 0),
        })
        
        # ROI data
        roi_data = load_deliverable(run["run_id"], "roi", client)
        if roi_data:
            for item in roi_data:
                roi_list.append({
                    "run_id": run["run_id"],
                    "run_date": run_date,
                    "channel": item.get("channel", "Unknown"),
                    "roi": item.get("roi", 0),
                })
    
    return pd.DataFrame(metrics_list), pd.DataFrame(roi_list)


def main():
    deliverables = shared_sidebar()
    page_header("Historical Tracking", "Track model performance and ROI trends over time")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    # Load historical data
    n_runs = st.slider("Number of runs to analyze", min_value=2, max_value=20, value=10)
    
    with st.spinner("Loading historical data..."):
        metrics_df, roi_df = load_historical_data(n_runs)

    if metrics_df.empty or len(metrics_df) < 2:
        st.warning(
            "Not enough historical data to display trends. "
            "Run the pipeline multiple times to build history."
        )
        return

    # ==========================================================================
    # CURRENT VS BENCHMARK KPIs
    # ==========================================================================
    st.subheader("Current vs Historical Benchmark")

    # Calculate benchmarks (average of last 6 months)
    cutoff = datetime.now() - timedelta(days=180)
    benchmark_df = metrics_df[metrics_df["run_date"] >= cutoff]
    
    if benchmark_df.empty:
        benchmark_df = metrics_df  # Use all data if no recent runs
    
    benchmark_r2 = benchmark_df["r2_test"].mean()
    benchmark_mape = benchmark_df["mape_test"].mean()
    
    # Current run (most recent)
    current = metrics_df.sort_values("run_date", ascending=False).iloc[0]
    current_r2 = current["r2_test"]
    current_mape = current["mape_test"]
    
    # Determine trend
    r2_delta = current_r2 - benchmark_r2
    mape_delta = current_mape - benchmark_mape
    
    kpi_metrics = [
        {
            "label": "Current R² Test",
            "value": f"{current_r2:.3f}",
            "delta": f"{r2_delta:+.3f} vs benchmark",
        },
        {
            "label": "Current MAPE Test",
            "value": f"{current_mape:.1f}%",
            "delta": f"{mape_delta:+.1f}% vs benchmark",
        },
        {
            "label": "Benchmark R² (6mo avg)",
            "value": f"{benchmark_r2:.3f}",
        },
        {
            "label": "Benchmark MAPE (6mo avg)",
            "value": f"{benchmark_mape:.1f}%",
        },
    ]
    
    kpi_row(kpi_metrics)

    # Status indicator
    if r2_delta > 0.01 and mape_delta < 0:
        st.success("Model is improving")
    elif r2_delta < -0.01 or mape_delta > 5:
        st.error("Model performance is declining")
    else:
        st.info("Model performance is stable")

    st.markdown("---")

    # ==========================================================================
    # PERFORMANCE OVER TIME
    # ==========================================================================
    st.subheader("Performance Over Time")

    metrics_df = metrics_df.sort_values("run_date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df["run_date"],
        y=metrics_df["r2_test"],
        mode="lines+markers",
        name="R² Test",
        line=dict(color="steelblue"),
    ))

    fig.add_hline(
        y=benchmark_r2,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Benchmark: {benchmark_r2:.3f}",
    )

    fig.update_layout(
        xaxis_title="Run Date",
        yaxis_title="R² Test",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # ROI TRENDS BY CHANNEL
    # ==========================================================================
    st.subheader("ROI Trends by Channel")

    if not roi_df.empty:
        channels = roi_df["channel"].unique().tolist()
        selected_channels = st.multiselect("Select Channels", channels, default=channels[:5])

        if selected_channels:
            filtered_roi = roi_df[roi_df["channel"].isin(selected_channels)]
            filtered_roi = filtered_roi.sort_values("run_date")

            fig_roi = go.Figure()

            for channel in selected_channels:
                channel_data = filtered_roi[filtered_roi["channel"] == channel]
                fig_roi.add_trace(go.Scatter(
                    x=channel_data["run_date"],
                    y=channel_data["roi"],
                    mode="lines+markers",
                    name=channel,
                ))

            fig_roi.update_layout(
                xaxis_title="Run Date",
                yaxis_title="ROI",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )

            st.plotly_chart(fig_roi, use_container_width=True)

            # Trend indicators
            st.markdown("### Channel Trends")
            
            for channel in selected_channels:
                channel_data = filtered_roi[filtered_roi["channel"] == channel].sort_values("run_date")
                
                if len(channel_data) >= 2:
                    first_roi = channel_data.iloc[0]["roi"]
                    last_roi = channel_data.iloc[-1]["roi"]
                    trend = ((last_roi - first_roi) / first_roi * 100) if first_roi != 0 else 0
                    
                    if trend > 5:
                        st.success(f"**{channel}**: Improving (+{trend:.1f}%)")
                    elif trend < -5:
                        st.error(f"**{channel}**: Declining ({trend:.1f}%)")
                    else:
                        st.info(f"**{channel}**: Stable ({trend:+.1f}%)")
    else:
        st.info("No ROI data available across runs.")

    st.markdown("---")

    # ==========================================================================
    # RUN HISTORY TABLE
    # ==========================================================================
    st.subheader("Run History")

    display_df = metrics_df[["run_name", "run_date", "r2_test", "mape_test"]].copy()
    display_df.columns = ["Run Name", "Date", "R² Test", "MAPE Test"]
    display_df = display_df.sort_values("Date", ascending=False)

    st.dataframe(display_df.style.format({
        "R² Test": "{:.3f}",
        "MAPE Test": "{:.1f}%",
    }), use_container_width=True)


if __name__ == "__main__":
    main()
else:
    main()

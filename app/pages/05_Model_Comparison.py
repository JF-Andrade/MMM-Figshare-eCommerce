"""
Model Comparison page.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.mlflow_loader import get_mlflow_client, get_all_runs, get_run_metrics
from src.comparison import (
    compare_models,
    compute_improvement,
    format_metric_value,
    generate_comparison_insight,
)

st.set_page_config(page_title="Model Comparison", layout="wide")


def load_baseline_runs():
    """Load baseline model runs from MLflow."""
    try:
        client = get_mlflow_client()
        runs = get_all_runs(client)
        return [r for r in runs if r.get("model_type") == "ridge_baseline"]
    except Exception:
        return []


def load_hierarchical_runs():
    """Load hierarchical model runs from MLflow."""
    try:
        client = get_mlflow_client()
        runs = get_all_runs(client, model_type="hierarchical")
        return runs
    except Exception:
        return []


def main():
    st.title("Model Comparison")
    st.markdown("Compare Ridge Regression baseline vs Bayesian Hierarchical model")
    st.markdown("---")

    # Load runs
    baseline_runs = load_baseline_runs()
    hierarchical_runs = load_hierarchical_runs()

    if not baseline_runs:
        st.warning("No baseline runs found. Run `python scripts/mmm_baseline.py` first.")
        return

    if not hierarchical_runs:
        st.warning("No hierarchical runs found. Run `python scripts/mmm_hierarchical.py` first.")
        return

    # Run selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline (Ridge)")
        baseline_options = {
            f"{r['run_name']} (R2: {r['r2_test']:.3f})" if r['r2_test'] else r['run_name']: r['run_id']
            for r in baseline_runs
        }
        baseline_label = st.selectbox("Select Baseline", list(baseline_options.keys()))
        baseline_run_id = baseline_options[baseline_label]

    with col2:
        st.subheader("Hierarchical (Bayesian)")
        hier_options = {
            f"{r['run_name']} (R2: {r['r2_test']:.3f})" if r['r2_test'] else r['run_name']: r['run_id']
            for r in hierarchical_runs
        }
        hier_label = st.selectbox("Select Hierarchical", list(hier_options.keys()))
        hier_run_id = hier_options[hier_label]

    # Load metrics
    try:
        baseline_metrics = get_run_metrics(baseline_run_id)
        hierarchical_metrics = get_run_metrics(hier_run_id)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return

    st.markdown("---")

    # Comparison table
    st.subheader("Performance Comparison")

    comparison_df = compare_models(baseline_metrics, hierarchical_metrics)
    improvement = compute_improvement(baseline_metrics, hierarchical_metrics)

    # Display as formatted table
    display_data = []
    for _, row in comparison_df.iterrows():
        display_data.append({
            "Metric": row["metric"].replace("_", " ").title(),
            "Baseline": format_metric_value(row["metric"], row["baseline"]),
            "Hierarchical": format_metric_value(row["metric"], row["hierarchical"]),
            "Winner": row["winner"],
        })

    display_df = pd.DataFrame(display_data)

    # Color code winners
    def highlight_winner(row):
        if row["Winner"] == "Baseline":
            return ["", "background-color: rgba(0, 255, 0, 0.2)", "", ""]
        elif row["Winner"] == "Hierarchical":
            return ["", "", "background-color: rgba(0, 255, 0, 0.2)", ""]
        return ["", "", "", ""]

    styled_df = display_df.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, width="stretch")

    st.markdown("---")

    # Trade-off visualization
    st.subheader("Trade-offs")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart comparison
        fig = go.Figure()

        metrics = ["r2_test", "mape_test"]
        baseline_vals = [baseline_metrics.get(m, 0) for m in metrics]
        hier_vals = [hierarchical_metrics.get(m, 0) for m in metrics]

        fig.add_trace(go.Bar(
            name="Baseline",
            x=["R2 Test", "MAPE Test"],
            y=baseline_vals,
            marker_color="steelblue",
        ))

        fig.add_trace(go.Bar(
            name="Hierarchical",
            x=["R2 Test", "MAPE Test"],
            y=hier_vals,
            marker_color="coral",
        ))

        fig.update_layout(
            title="Accuracy Metrics",
            barmode="group",
            height=350,
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        # Training time comparison
        b_time = baseline_metrics.get("training_time", 1)
        h_time = hierarchical_metrics.get("training_time", 1)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["Baseline", "Hierarchical"],
            y=[b_time, h_time],
            marker_color=["steelblue", "coral"],
            text=[f"{b_time:.1f}s", f"{h_time:.1f}s"],
            textposition="outside",
        ))

        fig.update_layout(
            title="Training Time",
            height=350,
            yaxis_title="Seconds",
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Insight
    st.subheader("Summary")

    insight = generate_comparison_insight(comparison_df, improvement)
    st.info(insight)

    # Recommendation
    r2_winner = comparison_df[comparison_df["metric"] == "r2_test"].iloc[0]["winner"]
    if r2_winner == "Hierarchical":
        st.success(
            "Recommendation: Use the **Hierarchical model** for production. "
            "It provides better accuracy and uncertainty quantification."
        )
    else:
        st.warning(
            "Recommendation: Consider the **Baseline model** if speed is critical. "
            "The Hierarchical model provides uncertainty estimates but is slower."
        )


if __name__ == "__main__":
    main()
else:
    main()

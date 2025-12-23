"""
Chart components for dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def roi_bar_chart(roi_data: list[dict], title: str = "ROI by Channel") -> None:
    """
    Render horizontal bar chart of ROI by channel.

    Args:
        roi_data: List of ROI records.
        title: Chart title.
    """
    df = pd.DataFrame(roi_data)

    if "region" in df.columns:
        # Aggregate by channel
        df = df.groupby("channel")["roi"].mean().reset_index()

    df = df.sort_values("roi", ascending=True)

    fig = px.bar(
        df,
        x="roi",
        y="channel",
        orientation="h",
        title=title,
        labels={"roi": "ROI", "channel": "Channel"},
    )

    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even")

    fig.update_layout(
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")


def contribution_pie_chart(contributions: list[dict], title: str = "Channel Contribution") -> None:
    """
    Render pie chart of channel contributions.

    Args:
        contributions: List of contribution records.
        title: Chart title.
    """
    df = pd.DataFrame(contributions)

    fig = px.pie(
        df,
        values="contribution_pct",
        names="channel",
        title=title,
        hole=0.4,
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, width="stretch")


def optimization_comparison_chart(optimization: list[dict]) -> None:
    """
    Render comparison chart of current vs optimal allocation.

    Args:
        optimization: List of optimization records.
    """
    df = pd.DataFrame(optimization)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Current",
        x=df["channel"],
        y=df["current_spend"],
        marker_color="steelblue",
    ))

    fig.add_trace(go.Bar(
        name="Optimal",
        x=df["channel"],
        y=df["optimal_spend"],
        marker_color="coral",
    ))

    fig.update_layout(
        title="Current vs Optimal Budget Allocation",
        barmode="group",
        height=400,
        xaxis_title="Channel",
        yaxis_title="Spend",
    )

    st.plotly_chart(fig, width="stretch")


def reallocation_chart(optimization: list[dict]) -> None:
    """
    Render horizontal bar chart of budget reallocation percentages.

    Args:
        optimization: List of optimization records.
    """
    df = pd.DataFrame(optimization)
    df = df.sort_values("change_pct")

    colors = ["green" if x > 0 else "red" for x in df["change_pct"]]

    fig = px.bar(
        df,
        x="change_pct",
        y="channel",
        orientation="h",
        title="Budget Reallocation",
        labels={"change_pct": "Change (%)", "channel": "Channel"},
        color=df["change_pct"] > 0,
        color_discrete_map={True: "green", False: "red"},
    )

    fig.add_vline(x=0, line_color="gray")

    fig.update_layout(
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")


def roi_heatmap(roi_data: list[dict]) -> None:
    """
    Render heatmap of ROI by channel and region.

    Args:
        roi_data: List of ROI records with region.
    """
    df = pd.DataFrame(roi_data)

    if "region" not in df.columns:
        st.warning("Regional data not available for heatmap.")
        return

    pivot = df.pivot(index="channel", columns="region", values="roi")

    fig = px.imshow(
        pivot,
        title="ROI by Channel and Region",
        labels={"x": "Region", "y": "Channel", "color": "ROI"},
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=5,
        aspect="auto",
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, width="stretch")


def saturation_curves_chart(saturation: list[dict]) -> None:
    """
    Render saturation curves for each channel.

    Args:
        saturation: List of saturation parameter records.
    """
    import numpy as np

    fig = go.Figure()

    x_range = np.linspace(0, 1000, 100)

    for ch in saturation:
        lam = ch["lam_mean"]
        y = 1 - np.exp(-lam * x_range / 1000)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y,
            mode="lines",
            name=ch["channel"],
        ))

    fig.update_layout(
        title="Channel Saturation Curves",
        xaxis_title="Spend (normalized)",
        yaxis_title="Effect (0-1)",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")


def adstock_decay_chart(adstock: list[dict]) -> None:
    """
    Render adstock decay curves for each channel.

    Args:
        adstock: List of adstock parameter records.
    """
    import numpy as np

    fig = go.Figure()

    weeks = np.arange(0, 13)

    for ch in adstock:
        alpha = ch["alpha_mean"]
        decay = alpha ** weeks

        fig.add_trace(go.Scatter(
            x=weeks,
            y=decay,
            mode="lines+markers",
            name=ch["channel"],
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% retention")

    fig.update_layout(
        title="Adstock Decay Curves",
        xaxis_title="Weeks Since Spend",
        yaxis_title="Retention Rate",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

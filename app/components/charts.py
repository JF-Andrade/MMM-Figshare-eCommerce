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

    st.plotly_chart(fig, use_container_width=True)


def contribution_pie_chart(contributions: list[dict], title: str = "Channel Contribution") -> None:
    """
    Render pie chart of channel contributions.

    Args:
        contributions: List of contribution records.
        title: Chart title.
    """
    df = pd.DataFrame(contributions)

    # Fallback if contribution_pct is missing (use raw contribution)
    value_col = "contribution_pct" if "contribution_pct" in df.columns else "contribution"

    fig = px.pie(
        df,
        values=value_col,
        names="channel",
        title=title,
        hole=0.4,
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)


def saturation_curves_chart(saturation: list[dict]) -> None:
    """
    Render saturation curves for each channel using Hill function.

    Args:
        saturation: List of saturation parameter records with 'L_mean' and 'k_mean'.
    """
    import numpy as np

    fig = go.Figure()

    x_range = np.linspace(0.01, 1, 100)  # Normalized spend [0, 1]

    for ch in saturation:
        # Hill saturation: x^k / (L^k + x^k)
        L = ch.get("L_mean", 0.3)
        k = ch.get("k_mean", 2.0)
        
        y = (x_range ** k) / (L ** k + x_range ** k + 1e-8)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y,
            mode="lines",
            name=ch["channel"],
        ))

    fig.update_layout(
        title="Channel Saturation Curves (Hill Function)",
        xaxis_title="Spend (normalized 0-1)",
        yaxis_title="Saturation Effect (0-1)",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)


def contribution_waterfall(contributions: list[dict]) -> None:
    """
    Render waterfall chart showing contribution breakdown by channel.

    Args:
        contributions: List of contribution records.
    """
    df = pd.DataFrame(contributions).sort_values("contribution", ascending=False)

    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=df["channel"],
        y=df["contribution"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="Channel Contribution Waterfall",
        xaxis_title="Channel",
        yaxis_title="Contribution (log scale)",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


def roi_with_uncertainty_chart(roi_hdi: list[dict]) -> None:
    """
    Render ROI chart with uncertainty intervals (HDI).

    Args:
        roi_hdi: List of ROI records with 'roi_mean', 'roi_hdi_low', 'roi_hdi_high'.
    """
    if not roi_hdi:
        st.info("ROI HDI data not available.")
        return

    df = pd.DataFrame(roi_hdi)
    if "roi_mean" not in df.columns:
        st.warning("ROI HDI columns missing.")
        return

    df = df.sort_values("roi_mean", ascending=True)

    fig = go.Figure()

    # Error bars
    fig.add_trace(go.Bar(
        name="ROI",
        x=df["roi_mean"],
        y=df["channel"],
        orientation="h",
        error_x=dict(
            type="data",
            symmetric=False,
            array=df["roi_hdi_high"] - df["roi_mean"],
            arrayminus=df["roi_mean"] - df["roi_hdi_low"],
        ),
        marker_color="steelblue",
    ))

    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even")

    fig.update_layout(
        title="Channel ROI with 94% HDI Intervals",
        xaxis_title="ROI (with uncertainty)",
        yaxis_title="Channel",
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def spend_share_chart(contributions: list[dict]) -> None:
    """
    Render stacked bar chart comparing spend share vs contribution share.

    Args:
        contributions: List of contribution records with 'total_spend' and 'contribution'.
    """
    df = pd.DataFrame(contributions)
    
    if "total_spend" not in df.columns or "contribution" not in df.columns:
        st.warning("Spend and contribution data required.")
        return

    # Calculate shares
    df["spend_share"] = df["total_spend"] / df["total_spend"].sum() * 100
    df["contribution_share"] = df["contribution"] / df["contribution"].sum() * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Spend Share",
        x=df["channel"],
        y=df["spend_share"],
        marker_color="steelblue",
    ))

    fig.add_trace(go.Bar(
        name="Contribution Share",
        x=df["channel"],
        y=df["contribution_share"],
        marker_color="coral",
    ))

    fig.update_layout(
        title="Spend Share vs Contribution Share",
        xaxis_title="Channel",
        yaxis_title="Share (%)",
        barmode="group",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def response_curves_chart(contributions: list[dict], saturation: list[dict]) -> None:
    """
    Render response curves showing diminishing returns for each channel.

    Args:
        contributions: Contribution data with total_spend.
        saturation: Saturation parameters with L_mean and k_mean.
    """
    import numpy as np

    if not saturation:
        st.info("Saturation data not available.")
        return

    fig = go.Figure()

    # Get current spend levels for reference
    spend_map = {}
    if contributions:
        for c in contributions:
            spend_map[c["channel"]] = c.get("total_spend", 0)

    for ch in saturation:
        L = ch.get("L_mean", 0.3)
        k = ch.get("k_mean", 2.0)
        channel = ch["channel"]
        
        # Generate spend range [0, 2x current] normalized
        current_spend = spend_map.get(channel, 0.5)
        x_range = np.linspace(0.01, min(2.0, 1.0), 100)
        
        # Hill response: contribution = beta * hill(x)
        # Marginal response approximation
        y = (x_range ** k) / (L ** k + x_range ** k + 1e-8)

        fig.add_trace(go.Scatter(
            x=x_range * 100,  # Convert to percentage
            y=y,
            mode="lines",
            name=channel,
        ))

    fig.update_layout(
        title="Channel Response Curves (Normalized)",
        xaxis_title="Spend Level (% of max)",
        yaxis_title="Response (saturation effect)",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

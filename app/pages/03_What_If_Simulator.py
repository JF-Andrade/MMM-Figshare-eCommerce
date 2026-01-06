"""
What-If Budget Simulator page.

Interactive budget reallocation with real-time revenue projection using Hill saturation model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import shared_sidebar, page_header, init_page_config

init_page_config("What-If Simulator")


def hill_saturation(x: float, L: float, k: float) -> float:
    """Calculate Hill saturation response.
    
    Args:
        x: Normalized spend (0-1+)
        L: Half-saturation point
        k: Steepness parameter
        
    Returns:
        Saturated response (0-1)
    """
    if x <= 0 or L <= 0:
        return 0.0
    return (x ** k) / (L ** k + x ** k)


def simulate_budget(
    spend_dict: dict[str, float],
    saturation_params: list[dict],
    contributions: list[dict],
) -> dict:
    """Simulate total contribution for given budget allocation.
    
    Args:
        spend_dict: Dict of {channel: new_spend_absolute}
        saturation_params: List of {channel, L_mean, k_mean, max_spend}
        contributions: List of {channel, contribution, total_spend}
        
    Returns:
        Dict with projected_contribution and per-channel breakdown
    """
    # Create lookups
    sat_lookup = {p["channel"]: p for p in saturation_params}
    contrib_lookup = {c["channel"]: c for c in contributions}
    
    results = []
    total_projected = 0.0
    
    for channel, new_spend in spend_dict.items():
        sat = sat_lookup.get(channel, {})
        contrib = contrib_lookup.get(channel, {})
        
        L = sat.get("L_mean", 0.5)
        k = sat.get("k_mean", 2.0)
        max_spend = sat.get("max_spend", 1.0)
        current_contribution = contrib.get("contribution", 0)
        current_spend = contrib.get("total_spend", 1)
        
        # Normalize new spend
        new_normalized = new_spend / max_spend if max_spend > 0 else 0
        current_normalized = current_spend / max_spend if max_spend > 0 else 0
        
        # Calculate saturation responses
        current_response = hill_saturation(current_normalized, L, k)
        new_response = hill_saturation(new_normalized, L, k)
        
        # Estimate beta (contribution per unit saturation response)
        if current_response > 0:
            beta = current_contribution / current_response
        else:
            beta = 0
        
        # Project new contribution
        projected = beta * new_response
        
        results.append({
            "channel": channel,
            "current_spend": current_spend,
            "new_spend": new_spend,
            "current_contribution": current_contribution,
            "projected_contribution": projected,
            "saturation_current": current_response,
            "saturation_new": new_response,
        })
        
        total_projected += projected
    
    return {
        "total_projected": total_projected,
        "breakdown": results,
    }


def main():
    deliverables = shared_sidebar()
    page_header("What-If Budget Simulator", "Drag sliders to simulate budget reallocation")

    if not deliverables:
        st.warning("No model data available. Run the hierarchical model first.")
        return

    saturation = deliverables.get("saturation")
    contributions = deliverables.get("contributions")

    if not saturation or not contributions:
        st.warning("Saturation and contribution data required for simulation.")
        return

    # Get current spend per channel
    spend_dict = {}
    current_total = 0.0
    current_contribution_total = 0.0
    
    for c in contributions:
        channel = c["channel"]
        spend = c.get("total_spend", 0)
        contribution = c.get("contribution", 0)
        spend_dict[channel] = spend
        current_total += spend
        current_contribution_total += contribution

    # Build slider interface
    st.subheader("Adjust Budget Allocation")
    
    new_spend_dict = {}
    total_new_spend = 0.0
    
    cols = st.columns(len(spend_dict))
    
    for i, (channel, current_spend) in enumerate(spend_dict.items()):
        with cols[i]:
            max_slider = current_spend * 2 if current_spend > 0 else 10000
            new_spend = st.slider(
                channel,
                min_value=0.0,
                max_value=float(max_slider),
                value=float(current_spend),
                step=max_slider / 100,
                format="%.0f",
            )
            new_spend_dict[channel] = new_spend
            total_new_spend += new_spend

    # Simulate
    result = simulate_budget(new_spend_dict, saturation, contributions)
    projected_total = result["total_projected"]
    
    # Display results
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Total Spend", f"{current_total:,.0f}")
    
    with col2:
        delta_spend = total_new_spend - current_total
        st.metric("New Total Spend", f"{total_new_spend:,.0f}", delta=f"{delta_spend:+,.0f}")
    
    with col3:
        st.metric("Current Contribution", f"{current_contribution_total:,.0f}")
    
    with col4:
        delta_contribution = projected_total - current_contribution_total
        pct_change = (delta_contribution / current_contribution_total * 100) if current_contribution_total > 0 else 0
        st.metric(
            "Projected Contribution",
            f"{projected_total:,.0f}",
            delta=f"{pct_change:+.1f}%"
        )

    # Breakdown table
    st.markdown("---")
    st.subheader("Channel Breakdown")
    
    breakdown_df = pd.DataFrame(result["breakdown"])
    breakdown_df["delta"] = breakdown_df["projected_contribution"] - breakdown_df["current_contribution"]
    breakdown_df["saturation_change"] = (breakdown_df["saturation_new"] - breakdown_df["saturation_current"]) * 100
    
    # Format for display
    display_df = breakdown_df[["channel", "current_spend", "new_spend", "current_contribution", "projected_contribution", "delta"]].copy()
    display_df.columns = ["Channel", "Current Spend", "New Spend", "Current Contrib", "Projected Contrib", "Delta"]
    
    st.dataframe(display_df.style.format({
        "Current Spend": "{:,.0f}",
        "New Spend": "{:,.0f}",
        "Current Contrib": "{:,.0f}",
        "Projected Contrib": "{:,.0f}",
        "Delta": "{:+,.0f}",
    }), use_container_width=True)

    # Visualization
    st.markdown("---")
    st.subheader("Contribution Comparison")
    
    fig = go.Figure()
    
    channels = breakdown_df["channel"].tolist()
    
    fig.add_trace(go.Bar(
        name="Current",
        x=channels,
        y=breakdown_df["current_contribution"],
        marker_color="steelblue",
    ))
    
    fig.add_trace(go.Bar(
        name="Projected",
        x=channels,
        y=breakdown_df["projected_contribution"],
        marker_color="coral",
    ))
    
    fig.update_layout(
        barmode="group",
        xaxis_title="Channel",
        yaxis_title="Contribution",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
else:
    main()

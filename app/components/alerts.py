"""
Alert Components for MMM Dashboard.

Provides visual alerts for saturation levels and other important metrics.
"""

import streamlit as st
import pandas as pd


def saturation_alert_badge(channel: str, saturation_pct: float, threshold: float = 0.8) -> None:
    """Display saturation alert badge for a channel.
    
    Args:
        channel: Channel name
        saturation_pct: Current spend as % of half-saturation point (0-1+)
        threshold: Alert threshold (default 0.8 = 80% of L)
    """
    if saturation_pct >= threshold:
        severity = "high" if saturation_pct >= 1.0 else "medium"
        
        if severity == "high":
            st.error(f"**{channel}**: {saturation_pct:.0%} saturated - diminishing returns")
        else:
            st.warning(f"**{channel}**: {saturation_pct:.0%} saturated - approaching limit")


def display_saturation_alerts(
    saturation_params: list[dict],
    contributions: list[dict],
    threshold: float = 0.8
) -> None:
    """Display saturation alerts for all channels.
    
    Args:
        saturation_params: List of {channel, L_mean, k_mean, max_spend}
        contributions: List of {channel, total_spend, ...}
        threshold: Alert threshold (default 0.8)
    """
    if not saturation_params or not contributions:
        return
    
    # Create lookup dicts
    sat_dict = {p["channel"]: p for p in saturation_params}
    spend_dict = {c["channel"]: c.get("total_spend", 0) for c in contributions}
    
    alerts = []
    
    for channel, sat_params in sat_dict.items():
        L = sat_params.get("L_mean", 0.5)
        max_spend = sat_params.get("max_spend")
        current_spend = spend_dict.get(channel, 0)
        
        # Calculate normalized current spend
        if max_spend and max_spend > 0:
            current_normalized = current_spend / max_spend
        else:
            current_normalized = 0
        
        # Calculate saturation % relative to L (half-saturation point)
        if L > 0:
            saturation_pct = current_normalized / L
        else:
            saturation_pct = 0
        
        if saturation_pct >= threshold:
            alerts.append({
                "channel": channel,
                "saturation_pct": saturation_pct,
                "L": L,
                "current_normalized": current_normalized,
            })
    
    if alerts:
        st.subheader("Saturation Alerts")
        alerts_sorted = sorted(alerts, key=lambda x: x["saturation_pct"], reverse=True)
        
        for alert in alerts_sorted:
            saturation_alert_badge(alert["channel"], alert["saturation_pct"], threshold)
        
        with st.expander("Understanding Saturation"):
            st.markdown("""
            **Saturation** measures how close a channel is to diminishing returns.
            
            - **>100%**: Investing more yields minimal additional revenue
            - **80-100%**: Approaching saturation, consider reallocating
            - **<80%**: Room for growth in this channel
            
            *Based on Hill saturation model parameters (L = half-saturation point)*
            """)
    else:
        st.success("All channels are within efficient spending range.")


def display_roi_anomalies(regional_data: list[dict], z_threshold: float = 2.0) -> pd.DataFrame:
    """Detect and display ROI anomalies by territory.
    
    Args:
        regional_data: List of {region, avg_iroas, ...}
        z_threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly flags
    """
    if not regional_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(regional_data)
    
    if "avg_iroas" not in df.columns:
        return df
    
    # Calculate z-scores
    mean_roi = df["avg_iroas"].mean()
    std_roi = df["avg_iroas"].std()
    
    if std_roi > 0:
        df["z_score"] = (df["avg_iroas"] - mean_roi) / std_roi
    else:
        df["z_score"] = 0
    
    # Flag anomalies
    def classify_anomaly(z):
        if z > z_threshold:
            return "Outlier High"
        elif z < -z_threshold:
            return "Outlier Low"
        return "Normal"
    
    df["status"] = df["z_score"].apply(classify_anomaly)
    
    # Display summary
    outliers = df[df["status"] != "Normal"]
    
    if len(outliers) > 0:
        st.subheader("ROI Anomalies Detected")
        
        for _, row in outliers.iterrows():
            region = row.get("region", row.get("territory", "Unknown"))
            roi = row["avg_iroas"]
            status = row["status"]
            
            if status == "Outlier High":
                st.success(f"**{region}**: ROI {roi:.2f}x (above average)")
            else:
                st.error(f"**{region}**: ROI {roi:.2f}x (below average)")
    
    return df

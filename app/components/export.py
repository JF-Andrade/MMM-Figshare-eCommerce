"""
Export Components for MMM Dashboard.

Generates downloadable Excel reports from dashboard deliverables.
"""

from io import BytesIO
from datetime import datetime

import pandas as pd


def generate_excel_report(deliverables: dict, run_id: str = "") -> BytesIO:
    """Generate multi-sheet Excel report from deliverables.
    
    Args:
        deliverables: Dict of loaded deliverable data
        run_id: MLflow run ID for metadata
        
    Returns:
        BytesIO buffer containing Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Executive Summary
        summary_data = []
        
        lift = deliverables.get("revenue_lift", {})
        if lift:
            summary_data.append({
                "Metric": "Current Contribution",
                "Value": lift.get("current_contribution", 0),
            })
            summary_data.append({
                "Metric": "Projected Contribution",
                "Value": lift.get("projected_contribution", 0),
            })
            summary_data.append({
                "Metric": "Lift (%)",
                "Value": lift.get("lift_pct", 0),
            })
            summary_data.append({
                "Metric": "Lift (Absolute)",
                "Value": lift.get("lift_absolute", 0),
            })
        
        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        
        # Sheet 2: Channel ROI
        roi = deliverables.get("roi", [])
        if roi:
            roi_df = pd.DataFrame(roi)
            roi_df.to_excel(writer, sheet_name="Channel ROI", index=False)
        
        # Sheet 3: Contributions
        contributions = deliverables.get("contributions", [])
        if contributions:
            contrib_df = pd.DataFrame(contributions)
            contrib_df.to_excel(writer, sheet_name="Contributions", index=False)
        
        # Sheet 4: Optimization Recommendations
        optimization = deliverables.get("optimization", [])
        if optimization:
            opt_df = pd.DataFrame(optimization)
            opt_df.to_excel(writer, sheet_name="Optimization", index=False)
        
        # Sheet 5: Regional Analysis
        regional = deliverables.get("regional", [])
        if regional:
            regional_df = pd.DataFrame(regional)
            regional_df.to_excel(writer, sheet_name="Regional", index=False)
        
        # Sheet 6: Saturation Parameters
        saturation = deliverables.get("saturation", [])
        if saturation:
            sat_df = pd.DataFrame(saturation)
            sat_df.to_excel(writer, sheet_name="Saturation Params", index=False)
        
        # Sheet 7: Metadata
        metadata = pd.DataFrame([{
            "Report Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Run ID": run_id,
            "Model Type": "Hierarchical Bayesian MMM",
        }])
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
    
    output.seek(0)
    return output


def get_excel_download_button(deliverables: dict, run_id: str = "") -> None:
    """Display Streamlit download button for Excel report."""
    import streamlit as st
    
    excel_data = generate_excel_report(deliverables, run_id)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"mmm_report_{timestamp}.xlsx"
    
    st.download_button(
        label="Download Excel Report",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

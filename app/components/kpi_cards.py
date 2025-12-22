"""
KPI card components for dashboard.
"""

from __future__ import annotations

import streamlit as st


def kpi_card(
    label: str,
    value: str | float | int,
    delta: str | float | None = None,
    delta_color: str = "normal",
) -> None:
    """
    Render a KPI metric card.

    Args:
        label: Metric label.
        value: Main value to display.
        delta: Optional change indicator.
        delta_color: Color for delta ('normal', 'inverse', 'off').
    """
    if isinstance(value, float):
        value = f"{value:,.2f}"
    elif isinstance(value, int):
        value = f"{value:,}"

    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
    )


def kpi_row(metrics: list[dict]) -> None:
    """
    Render a row of KPI cards.

    Args:
        metrics: List of dicts with 'label', 'value', and optional 'delta'.
    """
    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):
        with col:
            kpi_card(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
            )


def info_card(title: str, content: str, icon: str | None = None) -> None:
    """
    Render an info card with title and content.

    Args:
        title: Card title.
        content: Card content.
        icon: Optional icon.
    """
    header = f"{icon} {title}" if icon else title
    st.markdown(f"**{header}**")
    st.markdown(content)


def insight_box(insight: str, insight_type: str = "info") -> None:
    """
    Render an insight callout box.

    Args:
        insight: Insight text.
        insight_type: Type ('info', 'success', 'warning', 'error').
    """
    if insight_type == "success":
        st.success(insight)
    elif insight_type == "warning":
        st.warning(insight)
    elif insight_type == "error":
        st.error(insight)
    else:
        st.info(insight)

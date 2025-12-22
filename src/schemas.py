"""
Pydantic schemas for MMM deliverables.

Defines data structures for all 8 business deliverables with validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChannelROI(BaseModel):
    """ROI data for a single channel (optionally by region)."""

    channel: str
    region: Optional[str] = None
    spend: float = Field(ge=0)
    contribution: float
    roi: float
    roi_hdi_3: Optional[float] = None
    roi_hdi_97: Optional[float] = None


class AdstockParams(BaseModel):
    """Adstock decay parameters for a channel."""

    channel: str
    alpha_mean: float = Field(ge=0, le=1)
    alpha_std: float = Field(ge=0)
    alpha_hdi_3: Optional[float] = None
    alpha_hdi_97: Optional[float] = None
    half_life_weeks: float = Field(ge=0)


class SaturationParams(BaseModel):
    """Saturation parameters for a channel."""

    channel: str
    lam_mean: float = Field(ge=0)
    lam_std: float = Field(ge=0)
    lam_hdi_3: Optional[float] = None
    lam_hdi_97: Optional[float] = None


class ChannelContribution(BaseModel):
    """Channel contribution to revenue."""

    channel: str
    contribution: float
    contribution_pct: float = Field(ge=0, le=100)


class BudgetOptimization(BaseModel):
    """Budget optimization result for a channel."""

    channel: str
    current_spend: float = Field(ge=0)
    optimal_spend: float = Field(ge=0)
    change_pct: float


class MarginalROAS(BaseModel):
    """Marginal ROAS at different spend levels."""

    channel: str
    spend_increase_pct: float
    marginal_roas: float


class RegionalPerformance(BaseModel):
    """Regional performance summary."""

    region: str
    total_spend: float = Field(ge=0)
    total_contribution: float
    avg_roi: float
    best_channel: str


class RevenueLift(BaseModel):
    """Projected revenue lift from optimal allocation."""

    current_revenue: float
    optimal_revenue: float
    lift_absolute: float
    lift_pct: float


class ModelMetadata(BaseModel):
    """Metadata for a model run."""

    run_id: str
    model_type: str = Field(pattern="^(baseline|hierarchical)$")
    timestamp: datetime
    n_regions: int = Field(ge=1)
    n_channels: int = Field(ge=1)
    r2_train: float
    r2_test: float
    mape_train: float
    mape_test: float
    max_rhat: float
    divergences: int = Field(ge=0)


class AllDeliverables(BaseModel):
    """Container for all 8 business deliverables."""

    metadata: ModelMetadata
    roi: list[ChannelROI]
    saturation: list[SaturationParams]
    adstock: list[AdstockParams]
    contributions: list[ChannelContribution]
    optimization: list[BudgetOptimization]
    marginal_roas: list[MarginalROAS]
    regional: list[RegionalPerformance]
    revenue_lift: RevenueLift

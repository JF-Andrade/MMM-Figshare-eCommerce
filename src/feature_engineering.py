"""
Feature Engineering Module.

Derives new features from raw MMM data to improve model performance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Channel spend columns
SPEND_COLS = [
    "GOOGLE_PAID_SEARCH_SPEND",
    "GOOGLE_SHOPPING_SPEND",
    "GOOGLE_PMAX_SPEND",
    "GOOGLE_DISPLAY_SPEND",
    "GOOGLE_VIDEO_SPEND",
    "META_FACEBOOK_SPEND",
    "META_INSTAGRAM_SPEND",
    "META_OTHER_SPEND",
    "TIKTOK_SPEND",
]


def compute_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CTR (Click-Through Rate) for each channel."""
    df = df.copy()
    
    channel_map = {
        "GOOGLE_PAID_SEARCH": "GOOGLE_PAID_SEARCH",
        "GOOGLE_SHOPPING": "GOOGLE_SHOPPING",
        "GOOGLE_PMAX": "GOOGLE_PMAX",
        "GOOGLE_DISPLAY": "GOOGLE_DISPLAY",
        "GOOGLE_VIDEO": "GOOGLE_VIDEO",
        "META_FACEBOOK": "META_FACEBOOK",
        "META_INSTAGRAM": "META_INSTAGRAM",
    }
    
    for channel in channel_map.values():
        click_col = f"{channel}_CLICKS"
        imp_col = f"{channel}_IMPRESSIONS"
        
        if click_col in df.columns and imp_col in df.columns:
            impressions = df[imp_col].replace(0, np.nan)
            df[f"{channel}_CTR"] = (df[click_col] / impressions).fillna(0)
    
    return df


def compute_cost_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CPC (Cost Per Click) for each channel."""
    df = df.copy()
    
    for spend_col in SPEND_COLS:
        if spend_col not in df.columns:
            continue
            
        click_col = spend_col.replace("_SPEND", "_CLICKS")
        
        if click_col in df.columns:
            clicks = df[click_col].replace(0, np.nan)
            cpc_col = spend_col.replace("_SPEND", "_CPC")
            df[cpc_col] = (df[spend_col] / clicks).fillna(0)
    
    return df


def compute_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute customer behavior metrics (handles missing columns gracefully)."""
    df = df.copy()
    
    # Net revenue (if not already computed and required columns exist)
    if "ALL_PURCHASES_NET_PRICE" not in df.columns:
        if "ALL_PURCHASES_ORIGINAL_PRICE" in df.columns and "ALL_PURCHASES_GROSS_DISCOUNT" in df.columns:
            df["ALL_PURCHASES_NET_PRICE"] = (
                df["ALL_PURCHASES_ORIGINAL_PRICE"]
                - df["ALL_PURCHASES_GROSS_DISCOUNT"].fillna(0)
            )
    
    # Average Order Value (requires purchases and price)
    if "ALL_PURCHASES" in df.columns:
        purchases = df["ALL_PURCHASES"].replace(0, np.nan)
        
        # Try NET_PRICE first, then ORIGINAL_PRICE
        price_col = None
        if "ALL_PURCHASES_NET_PRICE" in df.columns:
            price_col = "ALL_PURCHASES_NET_PRICE"
        elif "ALL_PURCHASES_ORIGINAL_PRICE" in df.columns:
            price_col = "ALL_PURCHASES_ORIGINAL_PRICE"
        
        if price_col:
            df["AVG_ORDER_VALUE"] = (df[price_col] / purchases).fillna(0)
        
        # Units per order
        if "ALL_PURCHASES_UNITS" in df.columns:
            df["UNITS_PER_ORDER"] = (df["ALL_PURCHASES_UNITS"] / purchases).fillna(0)
        
        # New customer ratio
        if "FIRST_PURCHASES" in df.columns:
            df["NEW_CUSTOMER_RATIO"] = (df["FIRST_PURCHASES"] / purchases).fillna(0)
    
    # Discount rate (only if both columns exist)
    if "ALL_PURCHASES_ORIGINAL_PRICE" in df.columns and "ALL_PURCHASES_GROSS_DISCOUNT" in df.columns:
        original = df["ALL_PURCHASES_ORIGINAL_PRICE"].replace(0, np.nan)
        df["DISCOUNT_RATE"] = (df["ALL_PURCHASES_GROSS_DISCOUNT"].fillna(0) / original).fillna(0)
    
    return df


def compute_temporal_features(df: pd.DataFrame, date_col: str = "DATE_DAY") -> pd.DataFrame:
    """Compute temporal features from date."""
    df = df.copy()
    
    if date_col in df.columns:
        date = pd.to_datetime(df[date_col])
        df["DAY_OF_WEEK"] = date.dt.dayofweek / 6  # Normalized 0-1
        df["QUARTER"] = date.dt.quarter / 4  # Normalized 0.25-1.0
        df["WEEK_OF_YEAR"] = date.dt.isocalendar().week / 52  # Normalized
    
    return df


def compute_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Compute rolling statistics for spend channels."""
    df = df.copy()
    
    for spend_col in SPEND_COLS:
        if spend_col not in df.columns:
            continue
        
        # Rolling mean (momentum indicator)
        df[f"{spend_col}_ROLLING_{window}D_MEAN"] = (
            df[spend_col].rolling(window, min_periods=1).mean()
        )
        
        # Rolling std (volatility indicator)
        df[f"{spend_col}_ROLLING_{window}D_STD"] = (
            df[spend_col].rolling(window, min_periods=1).std().fillna(0)
        )
    
    return df


def compute_spend_share(df: pd.DataFrame) -> pd.DataFrame:
    """Compute channel spend share (media mix)."""
    df = df.copy()
    
    available_spend = [c for c in SPEND_COLS if c in df.columns]
    
    if not available_spend:
        return df
    
    total_spend = df[available_spend].sum(axis=1).replace(0, np.nan)
    
    for spend_col in available_spend:
        share_col = spend_col.replace("_SPEND", "_SHARE")
        df[share_col] = (df[spend_col] / total_spend).fillna(0)
    
    return df


def engineer_features(
    df: pd.DataFrame,
    date_col: str = "DATE_DAY",
    rolling_window: int = 7,
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Features created:
    - Efficiency: CTR per channel
    - Cost: CPC per channel
    - Customer: AOV, units/order, discount rate, new customer ratio
    - Temporal: day of week, quarter, week of year
    - Rolling: 7-day mean and std of spend
    - Share: channel spend share
    """
    df = compute_efficiency_metrics(df)
    df = compute_cost_metrics(df)
    df = compute_customer_metrics(df)
    df = compute_temporal_features(df, date_col)
    df = compute_rolling_features(df, rolling_window)
    df = compute_spend_share(df)
    
    return df

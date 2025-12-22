"""
Model Insights Module.

Extracts learned parameters from fitted MMM and generates business-facing visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM


def extract_adstock_params(mmm: MMM) -> pd.DataFrame:
    """
    Extract adstock decay rates (alpha) from posterior.

    Args:
        mmm: Fitted MMM model.

    Returns:
        DataFrame with channel, alpha_mean, alpha_std, alpha_hdi_3%, alpha_hdi_97%.
    """
    posterior = mmm.idata.posterior

    # alpha is the adstock decay parameter
    if "alpha" not in posterior:
        raise ValueError("Model posterior does not contain 'alpha' parameter")

    alpha = posterior["alpha"]

    # Handle case where alpha is wrapped in a tuple
    if isinstance(alpha, tuple):
        alpha = alpha[0] if len(alpha) > 0 else None
        if alpha is None:
            raise ValueError("Could not extract alpha from tuple")

    results = []
    for i, ch in enumerate(mmm.channel_columns):
        try:
            if hasattr(alpha, 'dims') and "channel" in alpha.dims:
                alpha_ch = alpha.sel(channel=ch)
            elif hasattr(alpha, 'isel'):
                alpha_ch = alpha.isel(channel=i)
            else:
                # Fallback: assume flat array
                values = np.array(alpha).flatten()
                alpha_ch = None
        except Exception:
            # Fallback for unexpected data structures
            values = np.array(alpha).flatten()
            alpha_ch = None

        if alpha_ch is not None:
            values = alpha_ch.values.flatten()

        results.append({
            "channel": ch.replace("_SPEND", ""),
            "alpha_mean": float(values.mean()),
            "alpha_std": float(values.std()),
            "alpha_hdi_3%": float(np.percentile(values, 3)),
            "alpha_hdi_97%": float(np.percentile(values, 97)),
            "half_life_weeks": float(-np.log(2) / np.log(values.mean())) if values.mean() > 0 and values.mean() < 1 else np.inf,
        })

    return pd.DataFrame(results).sort_values("alpha_mean", ascending=False)


def extract_saturation_params(mmm: MMM) -> pd.DataFrame:
    """
    Extract saturation parameters (lambda) from posterior.

    Args:
        mmm: Fitted MMM model.

    Returns:
        DataFrame with channel, lam_mean, lam_std, lam_hdi_3%, lam_hdi_97%.
    """
    posterior = mmm.idata.posterior

    # lam is the saturation steepness parameter
    if "lam" not in posterior:
        raise ValueError("Model posterior does not contain 'lam' parameter")

    lam = posterior["lam"]

    # Handle case where lam is wrapped in a tuple
    if isinstance(lam, tuple):
        lam = lam[0] if len(lam) > 0 else None
        if lam is None:
            raise ValueError("Could not extract lam from tuple")

    results = []
    for i, ch in enumerate(mmm.channel_columns):
        try:
            if hasattr(lam, 'dims') and "channel" in lam.dims:
                lam_ch = lam.sel(channel=ch)
            elif hasattr(lam, 'isel'):
                lam_ch = lam.isel(channel=i)
            else:
                # Fallback: assume flat array
                values = np.array(lam).flatten()
                lam_ch = None
        except Exception:
            # Fallback for unexpected data structures
            values = np.array(lam).flatten()
            lam_ch = None

        if lam_ch is not None:
            values = lam_ch.values.flatten()

        results.append({
            "channel": ch.replace("_SPEND", ""),
            "lam_mean": float(values.mean()),
            "lam_std": float(values.std()),
            "lam_hdi_3%": float(np.percentile(values, 3)),
            "lam_hdi_97%": float(np.percentile(values, 97)),
        })

    return pd.DataFrame(results).sort_values("lam_mean", ascending=False)


def plot_saturation_curves(
    mmm: MMM,
    output_dir: Path,
    max_spend_multiplier: float = 2.0,
) -> None:
    """
    Plot saturation curves for each channel.

    Args:
        mmm: Fitted MMM model.
        output_dir: Directory to save plot.
        max_spend_multiplier: Maximum spend as multiple of observed max.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get saturation params
    saturation_df = extract_saturation_params(mmm)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate spend range
    X = mmm.X
    max_spend = max(X[ch].max() for ch in mmm.channel_columns)
    spend_range = np.linspace(0, max_spend * max_spend_multiplier, 200)

    for _, row in saturation_df.iterrows():
        lam = row["lam_mean"]
        # Logistic saturation: 1 - exp(-lam * x)
        saturation = 1 - np.exp(-lam * spend_range / max_spend)
        ax.plot(spend_range, saturation, label=row["channel"], linewidth=2)

    ax.set_xlabel("Spend")
    ax.set_ylabel("Saturation Effect (0-1)")
    ax.set_title("Channel Saturation Curves (Diminishing Returns)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "saturation_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_adstock_decay(
    mmm: MMM,
    output_dir: Path,
    max_weeks: int = 12,
) -> None:
    """
    Plot adstock decay curves for each channel.

    Args:
        mmm: Fitted MMM model.
        output_dir: Directory to save plot.
        max_weeks: Maximum weeks to show decay.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    adstock_df = extract_adstock_params(mmm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    weeks = np.arange(max_weeks + 1)

    # Decay curves
    ax = axes[0]
    for _, row in adstock_df.iterrows():
        alpha = row["alpha_mean"]
        decay = alpha ** weeks
        ax.plot(weeks, decay, label=row["channel"], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Weeks Since Spend")
    ax.set_ylabel("Retention Rate")
    ax.set_title("Adstock Decay (Carryover Effect)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% retention")

    # Half-life bar chart
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(adstock_df)))
    bars = ax.barh(adstock_df["channel"], adstock_df["half_life_weeks"], color=colors)
    ax.set_xlabel("Half-Life (weeks)")
    ax.set_title("Adstock Half-Life by Channel")

    # Add value labels
    for bar, val in zip(bars, adstock_df["half_life_weeks"]):
        if np.isfinite(val):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}w", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "adstock_decay.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_channel_contributions_waterfall(
    mmm: MMM,
    output_dir: Path,
) -> None:
    """
    Plot waterfall chart of channel contributions.

    Args:
        mmm: Fitted MMM model.
        output_dir: Directory to save plot.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get contributions
    contributions = mmm.compute_channel_contribution_original_scale()
    mean_contrib = contributions.mean(dim=["chain", "draw"])
    total_contrib = mean_contrib.sum(dim="date")

    # Build waterfall data
    channels = [ch.replace("_SPEND", "") for ch in mmm.channel_columns]
    values = [float(total_contrib.sel(channel=ch).values) for ch in mmm.channel_columns]

    # Sort by contribution
    sorted_data = sorted(zip(channels, values), key=lambda x: x[1], reverse=True)
    channels, values = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    cumsum = 0
    bars = []
    for i, (ch, val) in enumerate(zip(channels, values)):
        bar = ax.barh(ch, val, left=cumsum, color=plt.cm.viridis(i / len(channels)))
        bars.append(bar)
        cumsum += val

    ax.set_xlabel("Total Contribution")
    ax.set_title("Channel Contributions (Waterfall)")

    # Add value labels
    cumsum = 0
    for ch, val in zip(channels, values):
        ax.text(cumsum + val / 2, ch, f"{val:,.0f}", va="center", ha="center", fontsize=9, color="white")
        cumsum += val

    plt.tight_layout()
    plt.savefig(output_dir / "channel_contributions_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

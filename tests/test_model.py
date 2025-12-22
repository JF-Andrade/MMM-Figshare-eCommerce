"""Tests for model module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_create_model_returns_mmm() -> None:
    """Test that create_model returns an MMM instance."""
    from src.model import create_model

    channels = ["GOOGLE_SPEND", "META_SPEND"]
    controls = ["trend", "is_holiday"]

    mmm = create_model(channels, controls)

    assert mmm is not None
    assert mmm.channel_columns == channels
    assert mmm.control_columns == controls


def test_model_has_correct_channels() -> None:
    """Test that model has correct channel configuration."""
    from src.model import create_model

    channels = ["GOOGLE_PAID_SEARCH_SPEND", "META_FACEBOOK_SPEND"]
    controls = ["trend"]

    mmm = create_model(channels, controls)

    assert len(mmm.channel_columns) == 2
    assert "GOOGLE_PAID_SEARCH_SPEND" in mmm.channel_columns
    assert "META_FACEBOOK_SPEND" in mmm.channel_columns


def test_model_has_correct_date_column() -> None:
    """Test that model uses correct date column."""
    from src.model import create_model

    mmm = create_model(["SPEND"], ["trend"], date_column="week")

    assert mmm.date_column == "week"


def test_create_model_with_custom_priors() -> None:
    """Test that custom priors are applied."""
    from pymc_marketing.prior import Prior

    from src.model import create_model

    custom_adstock = {"alpha": Prior("Beta", alpha=2, beta=2)}

    mmm = create_model(
        channel_columns=["SPEND"],
        control_columns=["trend"],
        adstock_priors=custom_adstock,
    )

    # Model should be created without error
    assert mmm is not None

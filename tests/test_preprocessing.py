"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
from src.config import RAW_DATE_COL

def test_apply_adstock() -> None:
    """Test geometric adstock transformation."""
    from src.preprocessing import apply_adstock

    data = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    decay = 0.5

    result = apply_adstock(data, decay=decay)

    # First value should be 100
    assert result[0] == 100
    # Second value should be 0 + 0.5 * 100 = 50
    assert result[1] == 50
    # Third value should be 0 + 0.5 * 50 = 25
    assert result[2] == 25


def test_compute_temporal_features() -> None:
    """Test temporal feature generation (Seasonality, Holidays, Events)."""
    from src.preprocessing import add_seasonality_features, add_event_features
    
    # Create spanning dates including Black Friday
    df = pd.DataFrame({
        RAW_DATE_COL: pd.to_datetime([
            "2023-01-01", # Normal day
            "2023-11-24", # Black Friday 2023
        ]),
        "metric": [1, 1]
    })
    
    result = add_event_features(df, date_col=RAW_DATE_COL)
    result = add_seasonality_features(result, date_col=RAW_DATE_COL)
    
    # Verify new features exist
    expected_cols = [
        "WEEK_SIN", "WEEK_COS",
        "MONTH_SIN", "MONTH_COS",
        "is_q4", "is_black_friday",
        "is_holiday"
    ]
    
    for col in expected_cols:
        assert col in result.columns, f"Missing {col}"
    
    # Check Black Friday logic
    # Nov 24 is Black Friday behavior (month 11, day 24 >= 23)
    assert result.loc[1, "is_black_friday"] == 1
    assert result.loc[0, "is_black_friday"] == 0

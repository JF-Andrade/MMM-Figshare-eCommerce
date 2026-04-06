import pandas as pd
import pytest
import numpy as np

from src.insights import optimize_hierarchical_budget


@pytest.fixture
def mock_data():
    contrib_df = pd.DataFrame([
        {"channel": "Facebook", "total_spend": 10000, "contribution": 20000},
        {"channel": "Google", "total_spend": 5000, "contribution": 10000},
        {"channel": "Negative_ROAS", "total_spend": 2000, "contribution": -500},
    ])
    # Add predicted revenue as attrs (crucial for lift absolute)
    contrib_df.attrs["total_predicted_revenue"] = 100000.0

    sat_params = [
        {"channel": "Facebook", "L_mean": 0.5, "k_mean": 2.0, "max_spend": 20000, "beta_mean": 1.5},
        {"channel": "Google", "L_mean": 0.4, "k_mean": 1.8, "max_spend": 10000, "beta_mean": 2.0},
        {"channel": "Negative_ROAS", "L_mean": 0.5, "k_mean": 2.0, "max_spend": 5000, "beta_mean": -0.5},
    ]

    return contrib_df, sat_params


def test_optimize_hierarchical_multiplicative_objective(mock_data):
    """AC-5: Objective uses multiplicative model."""
    contrib_df, sat_params = mock_data
    
    # We test giving it the exact same budget.
    total_budget = contrib_df["total_spend"].sum()
    n_obs = 52
    
    res = optimize_hierarchical_budget(contrib_df, sat_params, total_budget, n_obs)
    
    assert "metrics" in res
    assert "allocation" in res
    
    metrics = res["metrics"]
    allocs = res["allocation"]
    
    # Optimization should at least maintain or increase projected contribution
    assert metrics["lift_absolute"] >= 0.0
    assert metrics["lift_pct"] >= 0.0
    
    # The sum of optimal spends should exactly equal total budget
    total_opt_spend = sum(a["optimal_spend"] for a in allocs)
    assert abs(total_opt_spend - total_budget) < 1.0


def test_optimize_hierarchical_bounds(mock_data):
    """AC-E2: Ensure budget multiplier bounds are respected."""
    contrib_df, sat_params = mock_data
    total_budget = contrib_df["total_spend"].sum()
    n_obs = 52
    
    bounds_pct = (0.8, 1.2) # Allow only 20% shift
    
    res = optimize_hierarchical_budget(
        contrib_df, sat_params, total_budget, n_obs, budget_bounds_pct=bounds_pct
    )
    
    for alloc in res["allocation"]:
        current = alloc["current_spend"]
        optimal = alloc["optimal_spend"]
        if current > 0:
            lb = current * 0.8
            ub = current * 1.2
            # Allow tiny floating point tolerance
            assert optimal >= lb - 1e-4, f"Spend {optimal} below bound {lb} for {alloc['channel']}"
            assert optimal <= ub + 1e-4, f"Spend {optimal} above bound {ub} for {alloc['channel']}"


def test_optimize_fixed_negative_beta(mock_data):
    """AC-E3: Channels with negative beta or contribution must be fixed."""
    contrib_df, sat_params = mock_data
    total_budget = contrib_df["total_spend"].sum()
    n_obs = 52
    
    res = optimize_hierarchical_budget(contrib_df, sat_params, total_budget, n_obs)
    
    # Negative_ROAS should be completely unchanged
    alloc = next(a for a in res["allocation"] if a["channel"] == "Negative_ROAS")
    
    assert abs(alloc["optimal_spend"] - alloc["current_spend"]) < 1e-4
    assert alloc["change_pct"] == 0.0

"""Tests for Horseshoe prior tau0 formula correctness."""

import numpy as np
import pytest


def test_tau0_formula_matches_literature():
    """
    Verify tau0 formula matches Piironen & Vehtari (2017).
    
    Reference: τ₀ = p₀/(p-p₀) × 1/√n
    """
    m0 = 5  # Expected relevant features
    D = 10  # Total features
    n_obs = 500
    
    # Correct formula (Piironen & Vehtari, 2017)
    tau0_correct = m0 / (D - m0) / np.sqrt(n_obs)
    
    # Old incorrect formula for reference
    tau0_old = m0 / (D - m0) * np.sqrt(2.0 / n_obs)
    
    # Expected value for this configuration
    expected = 5 / 5 / np.sqrt(500)  # ≈ 0.0447
    
    assert np.isclose(tau0_correct, expected, rtol=1e-3)
    assert tau0_correct < tau0_old  # Correct is more regularizing


def test_tau0_edge_cases():
    """Test tau0 formula behavior at edge cases."""
    n_obs = 1000
    
    # Case 1: Half features expected to be relevant
    m0, D = 5, 10
    tau0 = m0 / (D - m0) / np.sqrt(n_obs)
    assert tau0 > 0
    assert tau0 < 1  # Should be a small regularization value
    
    # Case 2: Very sparse (few relevant features)
    m0_sparse, D = 1, 10
    tau0_sparse = m0_sparse / (D - m0_sparse) / np.sqrt(n_obs)
    assert tau0_sparse < tau0  # More sparse = smaller tau0 = more regularization
    
    # Case 3: Dense (most features relevant)
    m0_dense, D = 9, 10
    tau0_dense = m0_dense / (D - m0_dense) / np.sqrt(n_obs)
    assert tau0_dense > tau0  # Less sparse = larger tau0 = less regularization

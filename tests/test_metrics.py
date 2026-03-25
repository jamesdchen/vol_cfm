"""Tests for cfm.evaluation.metrics."""

import numpy as np

from cfm.evaluation.metrics import (
    daily_consistency_error,
    diurnal_pattern_error,
    energy_distance,
    marginal_ks_test,
)


def test_daily_consistency_perfect():
    """If generated rows sum to daily_rv, error should be 0."""
    rng = np.random.default_rng(42)
    N = 50
    daily_rv = rng.random(N) + 0.1

    # Build generated_48 whose rows sum exactly to daily_rv
    generated_48 = rng.dirichlet(np.ones(48), size=N) * daily_rv[:, None]

    error = daily_consistency_error(generated_48, daily_rv)
    assert abs(error) < 1e-10


def test_diurnal_pattern_identical():
    """Same real and generated data should give l2_error = 0."""
    rng = np.random.default_rng(42)
    data = rng.random((100, 48))

    result = diurnal_pattern_error(data, data)
    assert abs(result["l2_error"]) < 1e-10
    assert abs(result["max_error"]) < 1e-10


def test_energy_distance_identical():
    """Same distribution should give energy distance approximately 0."""
    rng = np.random.default_rng(42)
    data = rng.random((200, 48))

    dist = energy_distance(data, data)
    assert abs(dist) < 1e-10


def test_marginal_ks_identical():
    """Same data should give all p_values > 0.05."""
    rng = np.random.default_rng(42)
    data = rng.random((200, 48))

    result = marginal_ks_test(data, data)
    assert all(p > 0.05 for p in result["p_values"]), (
        f"Expected all p > 0.05 for identical data, got min={min(result['p_values']):.4f}"
    )
    assert result["mean_ks"] == 0.0

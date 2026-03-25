"""Tests for cfm.data.transforms and cfm.data.loading."""

import numpy as np
import pandas as pd

from cfm.data.loading import build_cfm_pairs
from cfm.data.transforms import (
    apply_scaler,
    denormalize_from_proportions,
    fit_scaler,
    inverse_scaler,
    normalize_to_proportions,
    sqrt_transform,
)


def test_sqrt_transform():
    x = np.array([0.0, 1.0, 4.0, 9.0])
    result = sqrt_transform(x)
    np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0, 3.0])


def test_normalize_denormalize_roundtrip():
    rng = np.random.default_rng(42)
    intraday = rng.random(48).astype(np.float64)
    daily_rv = intraday.sum()

    proportions = normalize_to_proportions(intraday, daily_rv)
    recovered = denormalize_from_proportions(proportions, daily_rv)

    np.testing.assert_allclose(recovered, intraday, rtol=1e-7)
    np.testing.assert_allclose(proportions.sum(), 1.0, atol=1e-10)


def test_scaler_roundtrip():
    rng = np.random.default_rng(42)
    conditions = rng.standard_normal((100, 6)).astype(np.float64)

    stats = fit_scaler(conditions)
    scaled = apply_scaler(conditions, stats)
    recovered = inverse_scaler(scaled, stats)

    np.testing.assert_allclose(recovered, conditions, rtol=1e-6)


def test_build_cfm_pairs_shapes():
    """Mock a daily_df with 10 days, verify output shapes with context_days=5."""
    rng = np.random.default_rng(42)
    n_days = 10
    context_days = 5

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B").date
    daily_rv = rng.random(n_days) + 0.01  # ensure positive
    intraday_48 = []
    for rv in daily_rv:
        props = rng.dirichlet(np.ones(48))
        intraday_48.append(props * rv)

    daily_df = pd.DataFrame(
        {
            "date": dates,
            "daily_rv": daily_rv,
            "intraday_48": intraday_48,
        }
    )

    conditions, proportions, dates_out = build_cfm_pairs(daily_df, context_days)

    expected_n = n_days - context_days
    assert conditions.shape == (expected_n, context_days + 1)
    assert proportions.shape == (expected_n, 48)
    assert dates_out.shape == (expected_n,)

    # Proportions should sum to ~1
    np.testing.assert_allclose(proportions.sum(axis=1), 1.0, atol=1e-5)

"""Tests for cfm.data.transforms and cfm.data.loading."""

import numpy as np
import pandas as pd

from cfm.config import CFMConfig
from cfm.data.loading import build_cfm_pairs, compute_block_rv
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


def _make_daily_df(n_days=10, seed=42):
    """Helper: create a mock daily_df for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B").date
    daily_rv = rng.random(n_days) + 0.01
    intraday_48 = [rng.dirichlet(np.ones(48)) * rv for rv in daily_rv]
    return pd.DataFrame({"date": dates, "daily_rv": daily_rv, "intraday_48": intraday_48})


def test_compute_block_rv():
    intraday = np.arange(1.0, 49.0)  # 1..48
    block_rv = compute_block_rv(intraday, 12)
    assert block_rv.shape == (4,)
    np.testing.assert_allclose(block_rv[0], np.arange(1, 13).sum())
    np.testing.assert_allclose(block_rv.sum(), intraday.sum())


def test_compute_block_rv_full_day():
    intraday = np.ones(48) * 0.5
    block_rv = compute_block_rv(intraday, 48)
    assert block_rv.shape == (1,)
    np.testing.assert_allclose(block_rv[0], 24.0)


def test_build_cfm_pairs_with_intermediate_blocks():
    daily_df = _make_daily_df(10)
    context_days = 5
    conditions, proportions, dates = build_cfm_pairs(
        daily_df, context_days, intermediate_blocks=[12], intermediate_representation="sqrt",
    )
    expected_n = 10 - context_days
    # 6 daily + 4 blocks * 5 lags = 26
    assert conditions.shape == (expected_n, 26)
    assert proportions.shape == (expected_n, 48)
    # Intermediate features should be non-negative (sqrt of positive values)
    assert (conditions[:, 6:] >= 0).all()


def test_build_cfm_pairs_multiple_blocks():
    daily_df = _make_daily_df(10)
    conditions, _, _ = build_cfm_pairs(
        daily_df, 5, intermediate_blocks=[6, 12], intermediate_representation="sqrt",
    )
    # 6 daily + (8 + 4) blocks * 5 lags = 66
    assert conditions.shape[1] == 66


def test_build_cfm_pairs_proportion_representation():
    daily_df = _make_daily_df(10)
    conditions, _, _ = build_cfm_pairs(
        daily_df, 5, intermediate_blocks=[12], intermediate_representation="proportion",
    )
    assert conditions.shape[1] == 26
    # Proportion features for each lag should sum to ~1
    for lag_idx in range(5):
        start = 6 + lag_idx * 4
        end = start + 4
        np.testing.assert_allclose(conditions[:, start:end].sum(axis=1), 1.0, atol=1e-5)


def test_build_cfm_pairs_both_representation():
    daily_df = _make_daily_df(10)
    conditions, _, _ = build_cfm_pairs(
        daily_df, 5, intermediate_blocks=[12], intermediate_representation="both",
    )
    # 6 daily + 4 blocks * 2 (sqrt+prop) * 5 lags = 46
    assert conditions.shape[1] == 46


def test_build_cfm_pairs_backward_compatible():
    daily_df = _make_daily_df(10)
    cond_baseline, prop_baseline, dates_baseline = build_cfm_pairs(daily_df, 5)
    cond_empty, prop_empty, dates_empty = build_cfm_pairs(
        daily_df, 5, intermediate_blocks=[],
    )
    np.testing.assert_array_equal(cond_baseline, cond_empty)
    np.testing.assert_array_equal(prop_baseline, prop_empty)


def test_cond_dim_computation():
    cfg = CFMConfig(intermediate_blocks=[12])
    assert cfg.cond_dim == 26

    cfg2 = CFMConfig(intermediate_blocks=[])
    assert cfg2.cond_dim == 6

    cfg3 = CFMConfig(intermediate_blocks=[6, 12], intermediate_representation="both")
    # 6 + (8+4) * 2 * 5 = 126
    assert cfg3.cond_dim == 126


def test_cond_dim_invalid_block_raises():
    try:
        CFMConfig(intermediate_blocks=[7])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

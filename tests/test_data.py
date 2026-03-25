"""Tests for cfm.data.transforms and cfm.data.loading."""

import numpy as np
import pandas as pd
import pytest

from cfm.config import CFMConfig
from cfm.data.dataset import build_dataloaders
from cfm.data.loading import (
    build_cfm_pairs,
    compute_block_rv,
    compute_intraday_summary,
    load_rv_source,
)
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

    conditions, proportions, dates_out = build_cfm_pairs(
        daily_df,
        context_days,
        intraday_summary_features=False,
    )

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
        daily_df,
        context_days,
        intermediate_blocks=[12],
        intermediate_representation="sqrt",
        intraday_summary_features=False,
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
        daily_df,
        5,
        intermediate_blocks=[6, 12],
        intermediate_representation="sqrt",
        intraday_summary_features=False,
    )
    # 6 daily + (8 + 4) blocks * 5 lags = 66
    assert conditions.shape[1] == 66


def test_build_cfm_pairs_proportion_representation():
    daily_df = _make_daily_df(10)
    conditions, _, _ = build_cfm_pairs(
        daily_df,
        5,
        intermediate_blocks=[12],
        intermediate_representation="proportion",
        intraday_summary_features=False,
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
        daily_df,
        5,
        intermediate_blocks=[12],
        intermediate_representation="both",
        intraday_summary_features=False,
    )
    # 6 daily + 4 blocks * 2 (sqrt+prop) * 5 lags = 46
    assert conditions.shape[1] == 46


def test_build_cfm_pairs_backward_compatible():
    daily_df = _make_daily_df(10)
    cond_baseline, prop_baseline, dates_baseline = build_cfm_pairs(
        daily_df,
        5,
        intraday_summary_features=False,
    )
    cond_empty, prop_empty, dates_empty = build_cfm_pairs(
        daily_df,
        5,
        intermediate_blocks=[],
        intraday_summary_features=False,
    )
    np.testing.assert_array_equal(cond_baseline, cond_empty)
    np.testing.assert_array_equal(prop_baseline, prop_empty)


def test_cond_dim_computation():
    # Default has intraday_summary_features=True, so +20
    cfg = CFMConfig(intermediate_blocks=[12])
    # 6 + 4*5 + 4*5 = 6 + 20 + 20 = 46
    assert cfg.cond_dim == 46

    cfg2 = CFMConfig(intermediate_blocks=[])
    # 6 + 0 + 20 = 26
    assert cfg2.cond_dim == 26

    cfg3 = CFMConfig(intermediate_blocks=[6, 12], intermediate_representation="both")
    # 6 + (8+4)*2*5 + 20 = 6 + 120 + 20 = 146
    assert cfg3.cond_dim == 146

    # Backward-compat: no summary features
    cfg4 = CFMConfig(intermediate_blocks=[12], intraday_summary_features=False)
    assert cfg4.cond_dim == 26

    cfg5 = CFMConfig(intermediate_blocks=[], intraday_summary_features=False)
    assert cfg5.cond_dim == 6


def test_cond_dim_invalid_block_raises():
    try:
        CFMConfig(intermediate_blocks=[7])
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# New tests for load_rv_source, compute_intraday_summary, summary features,
# and multi-source dataloaders
# ---------------------------------------------------------------------------

HARXHAR_PATH = "data/all30min"

_SOURCE_PARAMS = [
    ("core_stats.parquet", "sumret2"),
    ("vwstock_stats.parquet", "sumret2_vwstock"),
    ("ewstock_stats.parquet", "sumret2_ewstock"),
]


@pytest.mark.parametrize("parquet_file,rv_column", _SOURCE_PARAMS)
def test_load_rv_source(parquet_file, rv_column):
    """load_rv_source returns a [t, RV] DataFrame with positive values."""
    df = load_rv_source(HARXHAR_PATH, parquet_file, rv_column)
    assert list(df.columns) == ["t", "RV"]
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["t"])
    assert (df["RV"] >= 0).all()


def test_compute_intraday_summary():
    """Known 48-element array -> [sqrt(max), sqrt(min), sqrt(first), sqrt(last)]."""
    intraday = np.arange(1.0, 49.0)  # 1..48
    result = compute_intraday_summary(intraday)

    assert result.shape == (4,)
    np.testing.assert_allclose(result[0], np.sqrt(48.0))  # max
    np.testing.assert_allclose(result[1], np.sqrt(1.0))  # min
    np.testing.assert_allclose(result[2], np.sqrt(1.0))  # first
    np.testing.assert_allclose(result[3], np.sqrt(48.0))  # last


def test_build_cfm_pairs_with_summary():
    """With intraday_summary_features=True, conditions gain 4*context_days cols."""
    daily_df = _make_daily_df(10)
    context_days = 5

    cond_no, _, _ = build_cfm_pairs(
        daily_df,
        context_days,
        intraday_summary_features=False,
    )
    cond_yes, _, _ = build_cfm_pairs(
        daily_df,
        context_days,
        intraday_summary_features=True,
    )

    expected_n = 10 - context_days
    # Without summary: context_days+1 = 6
    assert cond_no.shape == (expected_n, 6)
    # With summary: 6 + 4*5 = 26
    assert cond_yes.shape == (expected_n, 26)
    # First 6 columns should be identical
    np.testing.assert_array_equal(cond_no, cond_yes[:, :6])
    # Summary features should be non-negative (sqrt of positive values)
    assert (cond_yes[:, 6:] >= 0).all()


def test_build_cfm_pairs_summary_plus_blocks():
    """Summary features stack correctly with intermediate block features."""
    daily_df = _make_daily_df(10)
    context_days = 5

    cond, _, _ = build_cfm_pairs(
        daily_df,
        context_days,
        intermediate_blocks=[12],
        intermediate_representation="sqrt",
        intraday_summary_features=True,
    )
    # 6 daily + 4 blocks * 5 lags + 4 summary * 5 lags = 6 + 20 + 20 = 46
    assert cond.shape[1] == 46


def test_multisource_dataloaders():
    """build_dataloaders with multi-source params returns three non-empty loaders."""
    train_loader, val_loader, test_loader, scaler_stats = build_dataloaders(
        harxhar_path=HARXHAR_PATH,
        context_days=5,
        train_end="2020-12-31",
        val_end="2022-12-31",
        batch_size=64,
        intraday_summary_features=True,
        num_workers=0,
    )

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0

    # Verify condition dimension matches config
    cfg = CFMConfig(intraday_summary_features=True)
    x1, cond = next(iter(train_loader))
    assert cond.shape[1] == cfg.cond_dim
    assert x1.shape[1] == 48

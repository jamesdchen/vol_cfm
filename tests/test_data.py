"""Tests for cfm.data.transforms, cfm.data.loading, and cfm.data.dataset."""

import numpy as np
import pandas as pd
import pytest
import torch

from cfm.config import CFMConfig
from cfm.data.dataset import InpaintingDataset
from cfm.data.loading import (
    build_inpainting_pairs,
    compute_block_rv,
    compute_intraday_summary,
    load_rv_source,
)
from cfm.data.transforms import (
    apply_scaler,
    compute_diurnal_prior,
    denormalize_from_proportions,
    fit_scaler,
    inverse_scaler,
    normalize_to_proportions,
    sqrt_transform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_df(n_days=10, seed=42, include_zero_rv=False):
    """Helper: create a mock daily_df for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B").date
    daily_rv = rng.random(n_days) + 0.01
    if include_zero_rv and n_days > 2:
        daily_rv[1] = 0.0  # inject a zero-RV day
    intraday_48 = []
    for rv in daily_rv:
        if rv > 0:
            intraday_48.append(rng.dirichlet(np.ones(48)) * rv)
        else:
            intraday_48.append(np.zeros(48))
    return pd.DataFrame({"date": dates, "daily_rv": daily_rv, "intraday_48": intraday_48})


# ---------------------------------------------------------------------------
# Transform tests (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# compute_block_rv tests (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# compute_intraday_summary (unchanged)
# ---------------------------------------------------------------------------


def test_compute_intraday_summary():
    """Known 48-element array -> [sqrt(max), sqrt(min), sqrt(first), sqrt(last)]."""
    intraday = np.arange(1.0, 49.0)  # 1..48
    result = compute_intraday_summary(intraday)

    assert result.shape == (4,)
    np.testing.assert_allclose(result[0], np.sqrt(48.0))  # max
    np.testing.assert_allclose(result[1], np.sqrt(1.0))  # min
    np.testing.assert_allclose(result[2], np.sqrt(1.0))  # first
    np.testing.assert_allclose(result[3], np.sqrt(48.0))  # last


# ---------------------------------------------------------------------------
# load_rv_source (unchanged, requires data on disk)
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


# ---------------------------------------------------------------------------
# build_inpainting_pairs tests
# ---------------------------------------------------------------------------


def test_build_inpainting_pairs_shapes():
    """build_inpainting_pairs returns arrays with correct shapes."""
    daily_df = _make_daily_df(10)
    proportions, daily_rvs, intraday_raw, dates = build_inpainting_pairs(daily_df)

    n = len(daily_df)  # all days have positive RV by default
    assert proportions.shape == (n, 48)
    assert daily_rvs.shape == (n,)
    assert intraday_raw.shape == (n, 48)
    assert dates.shape == (n,)


def test_build_inpainting_pairs_proportions_sum_to_one():
    """Proportions should sum to ~1 for each day."""
    daily_df = _make_daily_df(10)
    proportions, _, _, _ = build_inpainting_pairs(daily_df)

    np.testing.assert_allclose(proportions.sum(axis=1), 1.0, atol=1e-5)


def test_build_inpainting_pairs_skips_zero_rv():
    """Days with zero daily RV should be excluded."""
    daily_df = _make_daily_df(10, include_zero_rv=True)
    n_total = len(daily_df)
    n_zero = (daily_df["daily_rv"].values == 0.0).sum()

    proportions, daily_rvs, intraday_raw, dates = build_inpainting_pairs(daily_df)

    assert len(proportions) == n_total - n_zero
    assert (daily_rvs > 0).all()


# ---------------------------------------------------------------------------
# InpaintingDataset tests
# ---------------------------------------------------------------------------


def test_inpainting_dataset_getitem_shapes():
    """__getitem__ returns (x_1(48,), cond(97,))."""
    daily_df = _make_daily_df(10)
    proportions, daily_rvs, intraday_raw, _ = build_inpainting_pairs(daily_df)
    config = CFMConfig()

    ds = InpaintingDataset(proportions, daily_rvs, intraday_raw, config, seed=42)
    x_1, cond = ds[0]

    assert x_1.shape == (48,)
    assert cond.shape == (97,)
    assert x_1.dtype == torch.float32
    assert cond.dtype == torch.float32


# ---------------------------------------------------------------------------
# compute_diurnal_prior
# ---------------------------------------------------------------------------


def test_compute_diurnal_prior():
    """Diurnal prior is the column-wise mean of proportions, shape (48,)."""
    rng = np.random.default_rng(42)
    proportions = rng.dirichlet(np.ones(48), size=100).astype(np.float32)

    prior = compute_diurnal_prior(proportions)

    assert prior.shape == (48,)
    np.testing.assert_allclose(prior, proportions.mean(axis=0), atol=1e-6)

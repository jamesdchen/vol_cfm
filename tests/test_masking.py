"""Tests for cfm.data.masking."""

import numpy as np

from cfm.data.masking import build_mask_and_known, sample_training_mask, spread_block_rv


def test_spread_block_rv_shape_and_sum():
    """block_size=12 -> 4 blocks; output shape (48,) and sum preserved."""
    block_rv = np.array([10.0, 20.0, 30.0, 40.0])
    out = spread_block_rv(block_rv, block_size=12)
    assert out.shape == (48,)
    np.testing.assert_allclose(out.sum(), block_rv.sum())


def test_spread_block_rv_uniform():
    """All bars within a block should receive the same value."""
    block_rv = np.array([12.0, 24.0, 36.0, 48.0])
    out = spread_block_rv(block_rv, block_size=12)
    # First block: 12/12 = 1.0 for bars 0-11
    np.testing.assert_allclose(out[:12], 1.0)
    # Second block: 24/12 = 2.0 for bars 12-23
    np.testing.assert_allclose(out[12:24], 2.0)


def test_build_mask_and_known_all_blocks():
    """With one block_size covering all bars, mask is all 1s and known sums to ~1."""
    rng = np.random.default_rng(42)
    intraday_48 = rng.random(48) + 0.01
    daily_rv = intraday_48.sum()

    mask, known_values = build_mask_and_known(intraday_48, daily_rv, block_sizes=[12])

    assert mask.shape == (48,)
    assert known_values.shape == (48,)
    # All bars are covered by the block
    np.testing.assert_array_equal(mask, np.ones(48, dtype=np.float32))
    # Known values are proportions that sum to ~1
    np.testing.assert_allclose(known_values.sum(), 1.0, atol=1e-5)


def test_build_mask_and_known_sparse_override():
    """Sparse bar observations override block-level values."""
    rng = np.random.default_rng(42)
    intraday_48 = rng.random(48) + 0.01
    daily_rv = intraday_48.sum()

    sparse_indices = np.array([0, 5, 47])
    mask, known_values = build_mask_and_known(intraday_48, daily_rv, block_sizes=[12], sparse_indices=sparse_indices)

    # Sparse indices should have exact bar proportions
    for idx in sparse_indices:
        expected = intraday_48[idx] / daily_rv
        np.testing.assert_allclose(known_values[idx], expected, atol=1e-6)


def test_sample_training_mask_shapes():
    """Output shapes should be (48,) each."""
    rng = np.random.default_rng(42)
    intraday_48 = rng.random(48) + 0.01
    daily_rv = intraday_48.sum()

    mask, known_values = sample_training_mask(
        intraday_48,
        daily_rv,
        block_granularities=[12],
        sparse_bar_prob=0.1,
        mask_schedule="random_blocks",
        rng=rng,
    )

    assert mask.shape == (48,)
    assert known_values.shape == (48,)


def test_sample_training_mask_all_blocks():
    """With mask_schedule='all_blocks', result is deterministic given same input."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(99)
    intraday_48 = np.arange(1.0, 49.0)
    daily_rv = intraday_48.sum()

    mask1, kv1 = sample_training_mask(
        intraday_48,
        daily_rv,
        block_granularities=[12],
        sparse_bar_prob=0.0,
        mask_schedule="all_blocks",
        rng=rng1,
    )
    mask2, kv2 = sample_training_mask(
        intraday_48,
        daily_rv,
        block_granularities=[12],
        sparse_bar_prob=0.0,
        mask_schedule="all_blocks",
        rng=rng2,
    )

    # With sparse_bar_prob=0 and all_blocks, both calls should give identical results
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(kv1, kv2)


def test_sample_training_mask_random_blocks():
    """With mask_schedule='random_blocks', the function runs without error."""
    rng = np.random.default_rng(42)
    intraday_48 = np.arange(1.0, 49.0)
    daily_rv = intraday_48.sum()

    mask, known_values = sample_training_mask(
        intraday_48,
        daily_rv,
        block_granularities=[6, 12],
        sparse_bar_prob=0.1,
        mask_schedule="random_blocks",
        rng=rng,
    )

    assert mask.shape == (48,)
    assert known_values.shape == (48,)

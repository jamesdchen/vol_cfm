"""Masking utilities for inpainting-style conditioning."""

from __future__ import annotations

import numpy as np

from cfm.config import PERIODS_PER_DAY
from cfm.data.loading import compute_block_rv


def spread_block_rv(block_rv: np.ndarray, block_size: int) -> np.ndarray:
    """Spread block-level RV uniformly across constituent 30-min bars.

    Parameters
    ----------
    block_rv : np.ndarray, shape (n_blocks,)
        RV summed within each block.
    block_size : int
        Number of 30-min bars per block.

    Returns
    -------
    np.ndarray, shape (48,)
        Each bar gets ``block_rv[k] / block_size`` for its block ``k``.
    """
    return np.repeat(block_rv / block_size, block_size)


def build_mask_and_known(
    intraday_48: np.ndarray,
    daily_rv: float,
    block_sizes: list[int],
    sparse_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mask and known_values arrays for one day.

    Known values are expressed as proportions (bar_rv / daily_rv) to match
    the target representation.

    Parameters
    ----------
    intraday_48 : np.ndarray, shape (48,)
        Raw 30-min RV bars.
    daily_rv : float
        Total daily RV (sum of intraday_48).
    block_sizes : list[int]
        Block granularities to include. Each block's RV is spread uniformly.
    sparse_indices : np.ndarray or None
        Indices of individually observed bars.

    Returns
    -------
    mask : np.ndarray, shape (48,), float32
        1.0 where known, 0.0 where to generate.
    known_values : np.ndarray, shape (48,), float32
        Proportion values at known positions, 0.0 elsewhere.
    """
    mask = np.zeros(PERIODS_PER_DAY, dtype=np.float32)
    known_values = np.zeros(PERIODS_PER_DAY, dtype=np.float32)

    # Coarse block observations: spread uniformly as proportions
    for bs in block_sizes:
        block_rv = compute_block_rv(intraday_48, bs)
        spread = spread_block_rv(block_rv, bs) / daily_rv  # proportions
        mask[:] = 1.0
        known_values[:] = spread

    # Sparse bar observations override block values
    if sparse_indices is not None and len(sparse_indices) > 0:
        mask[sparse_indices] = 1.0
        known_values[sparse_indices] = intraday_48[sparse_indices] / daily_rv

    return mask, known_values


def sample_training_mask(
    intraday_48: np.ndarray,
    daily_rv: float,
    block_granularities: list[int],
    sparse_bar_prob: float,
    mask_schedule: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a stochastic mask for training.

    Parameters
    ----------
    intraday_48 : np.ndarray, shape (48,)
    daily_rv : float
    block_granularities : list[int]
        Available block sizes to choose from.
    sparse_bar_prob : float
        Per-bar probability of being individually observed.
    mask_schedule : str
        ``"random_blocks"``: randomly choose a subset of block granularities.
        ``"all_blocks"``: always use all block granularities.
    rng : np.random.Generator

    Returns
    -------
    mask, known_values : tuple of np.ndarray, each shape (48,)
    """
    if mask_schedule == "all_blocks":
        chosen_blocks = list(block_granularities)
    else:  # random_blocks
        if len(block_granularities) > 0:
            n = rng.integers(1, len(block_granularities) + 1)
            chosen_blocks = list(rng.choice(block_granularities, size=n, replace=False))
        else:
            chosen_blocks = []

    # Sample sparse bar indices
    sparse_indices = None
    if sparse_bar_prob > 0:
        sparse_mask = rng.random(PERIODS_PER_DAY) < sparse_bar_prob
        if sparse_mask.any():
            sparse_indices = np.where(sparse_mask)[0]

    return build_mask_and_known(intraday_48, daily_rv, chosen_blocks, sparse_indices)

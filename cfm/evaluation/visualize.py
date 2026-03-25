"""Visualization utilities for intraday RV proportion evaluation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_sample_paths(
    real: np.ndarray,
    generated: np.ndarray,
    n_samples: int = 5,
    seed: int = 0,
) -> tuple[Figure, Axes]:
    """Overlay random real and generated 48-period proportion paths.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)
    n_samples : int
        Number of paths to draw from each set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (fig, ax)
    """
    rng = np.random.default_rng(seed)
    x = np.arange(48)

    fig, ax = plt.subplots(figsize=(10, 4))
    idx_real = rng.choice(len(real), min(n_samples, len(real)), replace=False)
    idx_gen = rng.choice(len(generated), min(n_samples, len(generated)), replace=False)

    for i, ri in enumerate(idx_real):
        ax.plot(
            x,
            real[ri],
            color="tab:blue",
            alpha=0.5,
            label="Real" if i == 0 else None,
        )
    for i, gi in enumerate(idx_gen):
        ax.plot(
            x,
            generated[gi],
            color="tab:red",
            alpha=0.5,
            label="Generated" if i == 0 else None,
        )

    ax.set_xlabel("Period")
    ax.set_ylabel("Proportion")
    ax.set_title("Sample Intraday Paths")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_diurnal_pattern(real: np.ndarray, generated: np.ndarray) -> tuple[Figure, Axes]:
    """Plot mean +/- 1 std diurnal pattern for real and generated data.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)

    Returns
    -------
    (fig, ax)
    """
    x = np.arange(48)

    real_mean, real_std = real.mean(axis=0), real.std(axis=0)
    gen_mean, gen_std = generated.mean(axis=0), generated.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, real_mean, color="tab:blue", label="Real")
    ax.fill_between(
        x,
        real_mean - real_std,
        real_mean + real_std,
        color="tab:blue",
        alpha=0.2,
    )
    ax.plot(x, gen_mean, color="tab:red", label="Generated")
    ax.fill_between(
        x,
        gen_mean - gen_std,
        gen_mean + gen_std,
        color="tab:red",
        alpha=0.2,
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Mean Proportion")
    ax.set_title("Diurnal Pattern (mean +/- 1 std)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_marginal_distributions(
    real: np.ndarray,
    generated: np.ndarray,
    periods: list[int] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Overlaid histograms of real vs generated for selected periods.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)
    periods : list[int]
        Period indices to plot. Defaults to [0, 12, 24, 36].

    Returns
    -------
    (fig, axes)
    """
    if periods is None:
        periods = [0, 12, 24, 36]

    n = len(periods)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for i, p in enumerate(periods):
        ax = axes_flat[i]
        ax.hist(real[:, p], bins=30, alpha=0.5, color="tab:blue", label="Real", density=True)
        ax.hist(generated[:, p], bins=30, alpha=0.5, color="tab:red", label="Generated", density=True)
        ax.set_title(f"Period {p}")
        ax.legend(fontsize="small")

    # hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Marginal Distributions")
    fig.tight_layout()
    return fig, axes


def plot_acf(real_acf: np.ndarray, gen_acf: np.ndarray) -> tuple[Figure, Axes]:
    """Side-by-side bar chart of ACF at each lag.

    Parameters
    ----------
    real_acf : np.ndarray, shape (L,)
    gen_acf : np.ndarray, shape (L,)

    Returns
    -------
    (fig, ax)
    """
    n_lags = len(real_acf)
    lags = np.arange(1, n_lags + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags - width / 2, real_acf, width, label="Real", color="tab:blue")
    ax.bar(lags + width / 2, gen_acf, width, label="Generated", color="tab:red")

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF Comparison")
    ax.set_xticks(lags)
    ax.legend()
    fig.tight_layout()
    return fig, ax
